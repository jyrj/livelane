"""Optimisation-shaped rewrites of a whole ibex_core.

The question: when an agent rewrites a real core to make it faster or smaller,
how often is the rewrite broken, how often do the cheap evaluators REWARD the
broken ones, and does the formal gate refuse them while still accepting the
rewrites that are genuinely equivalent?

The paper's thesis, simulation passes broken designs, synthesis rewards them,
only formal refuses, rested on one injected bug and one loop observation.
This turns it into a rate on a whole RISC-V core.

Two families of rewrite, both shaped like what a QoR-seeking agent proposes:

  EQUIVALENT (controls). Commute the operands of a commutative operator inside
  parentheses. The design's behaviour cannot change. The gate MUST prove every
  one; a refutation here is a false rejection, and a gate that rejects
  everything is useless however safe it looks.

  SIMPLIFYING. Drop a conjunct, drop a disjunct, remove an inversion, fold a
  guard to true, weaken a comparison, swap a mux's arms. Each removes or
  rearranges logic, which is exactly why synthesis tends to score it better.
  Most change behaviour. Some do not, the edited logic is dead or masked,
  and formal proves those equivalent, which a simulation-based mutation score
  would miscount as a coverage gap.

Every operator matches only a PARENTHESISED binary expression of two simple
operands, or a whole ternary right-hand side. Commuting `a && b` found inside
`c == a && b` would bind as `(c == a) && b` and turn a "control" into a real
change; parentheses make the operand boundaries unambiguous. Sites inside
comments, strings, or preprocessor branches excluded by the defines in use are
skipped, so a rewrite of stripped assertion code is never counted as anything.

Verdicts and QoR are deterministic (QoR is byte-identical over 10 repeats), so
machine load does not affect any number this script reports.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from hwebench import cfg_text, sh  # noqa: E402
from localisation import decomment  # noqa: E402
from compare_frontend import instance_names  # noqa: E402

ID = r"[A-Za-z_]\w*(?:\[[^\]\[]*\])?"

# (name, family, pattern, rewrite) , rewrite(match) -> replacement text
OPERATORS = [
    ("commute_land", "equivalent",
     re.compile(rf"\(\s*({ID})\s*&&\s*({ID})\s*\)"),
     lambda m: f"({m.group(2)} && {m.group(1)})"),
    ("commute_lor", "equivalent",
     re.compile(rf"\(\s*({ID})\s*\|\|\s*({ID})\s*\)"),
     lambda m: f"({m.group(2)} || {m.group(1)})"),
    ("commute_band", "equivalent",
     re.compile(rf"\(\s*({ID})\s*&(?!&)\s*({ID})\s*\)"),
     lambda m: f"({m.group(2)} & {m.group(1)})"),
    ("commute_eq", "equivalent",
     re.compile(rf"\(\s*({ID})\s*==\s*({ID})\s*\)"),
     lambda m: f"({m.group(2)} == {m.group(1)})"),

    ("drop_conjunct", "simplifying",
     re.compile(rf"\(\s*({ID})\s*&&\s*({ID})\s*\)"),
     lambda m: f"({m.group(1)})"),
    ("drop_disjunct", "simplifying",
     re.compile(rf"\(\s*({ID})\s*\|\|\s*({ID})\s*\)"),
     lambda m: f"({m.group(1)})"),
    ("remove_inversion", "simplifying",
     re.compile(rf"(?<![!=<>~&|^])!(?!=)\s*({ID})"),
     lambda m: m.group(1)),
    ("weaken_compare", "simplifying",
     re.compile(rf"\(\s*({ID})\s*<(?![<=])\s*({ID})\s*\)"),
     lambda m: f"({m.group(1)} <= {m.group(2)})"),
    ("fold_guard", "simplifying",
     re.compile(rf"\bif\s*\(\s*(!?\s*{ID})\s*\)"),
     lambda m: "if (1'b1)"),
    ("swap_mux", "simplifying",
     re.compile(rf"=\s*({ID})\s*\?\s*({ID})\s*:\s*({ID})\s*;"),
     lambda m: f"= {m.group(1)} ? {m.group(3)} : {m.group(2)};"),
]

_PP = re.compile(r"^\s*`(ifdef|ifndef|elsif|else|endif)\b\s*(\w*)", re.M)


def excluded_ranges(text: str, defines: set[str]) -> list[tuple[int, int]]:
    """Byte ranges the preprocessor drops under `defines`.

    ibex keeps its SVA under `ifndef SYNTHESIS` / `ifndef VERILATOR`, and those
    are defined so the core elaborates. A rewrite there changes nothing that is
    checked, and would be counted as an "equivalent" simplification it is not.
    """
    out, stack = [], []   # stack of (active, taken_any)
    start_off = None

    def active() -> bool:
        return all(a for a, _ in stack)

    for m in _PP.finditer(text):
        kind, name = m.group(1), m.group(2)
        was = active()
        if kind in ("ifdef", "ifndef"):
            cond = (name in defines) if kind == "ifdef" else (name not in defines)
            stack.append((cond, cond))
        elif kind == "elsif" and stack:
            _, taken = stack[-1]
            cond = (not taken) and (name in defines)
            stack[-1] = (cond, taken or cond)
        elif kind == "else" and stack:
            _, taken = stack[-1]
            stack[-1] = (not taken, True)
        elif kind == "endif" and stack:
            stack.pop()
        now = active()
        if was and not now:
            start_off = m.end()
        elif not was and now and start_off is not None:
            out.append((start_off, m.start()))
            start_off = None
    if start_off is not None:
        out.append((start_off, len(text)))
    return out


def sites(path: Path, defines: set[str]):
    """Every (operator, family, start, end, replacement, original) in a file."""
    raw = path.read_text(errors="ignore")
    clean = decomment(raw)
    drop = excluded_ranges(clean, defines)
    out = []
    for name, fam, pat, fn in OPERATORS:
        for m in pat.finditer(clean):
            s, e = m.span()
            if any(a <= s < b for a, b in drop):
                continue
            # skip assertion / system-task lines even outside a guarded region
            line = clean[clean.rfind("\n", 0, s) + 1: clean.find("\n", e)]
            if re.search(r"`ASSERT|\bassert\b|\bassume\b|\bcover\b|\$\w+", line):
                continue
            # A `for` header's bounds are genvars and parameters: rewriting
            # them changes how much hardware is generated, not the logic.
            if re.search(r"\bfor\s*\(", line):
                continue
            # A simplifying rewrite whose operand is a PARAMETER or enum
            # constant is structural, not an optimisation: `if (BranchPredictor)`
            # is a generate-if, `!RV32E` selects an ISA variant. No QoR-seeking
            # agent proposes those, and counting them would skew the rate. ibex
            # names parameters and constants CamelCase/UPPER and signals
            # snake_case, which makes them detectable. Controls keep them:
            # commuting `(opcode == OPCODE_LOAD)` is equivalent regardless.
            if fam == "simplifying" and re.search(
                    r"(?<![\w'])[A-Z][A-Za-z0-9_]*", m.group(0).replace("1'b1", "")):
                continue
            # Reset and clock are not logic an optimiser rewrites.
            # `if (!rst_ni)` -> `if (1'b1)` pins every flop in reset and
            # `!rst_ni` -> `rst_ni` inverts reset polarity; synthesis then
            # collapses the core to constants and "scores" it enormously
            # better. That would manufacture exactly the headline this script
            # measures, out of an edit nobody would make.
            if fam == "simplifying" and re.search(
                    r"\b\w*(rst|reset|clk|clock)\w*\b", m.group(0), re.I):
                continue
            orig = raw[s:e]
            new = fn(m)
            if new.replace(" ", "") == orig.replace(" ", ""):
                continue            # a no-op, e.g. commuting `(a && a)`
            out.append((name, fam, s, e, new, orig))
    return out


def run(cmd, cwd=None, timeout=900):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sha", default="23806e2ad7d0",
                    help="ibex commit to rewrite (default: corpus #1735's base)")
    ap.add_argument("--per-op", type=int, default=7,
                    help="sites sampled per operator")
    ap.add_argument("--per-file", type=int, default=2,
                    help="at most this many sites per (operator, file)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--synth-parallel", type=int, default=6)
    ap.add_argument("--clock-ns", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=None, help="probe: first N")
    ap.add_argument("--workroot", default=str(ROOT / "var" / "rewrites"))
    ap.add_argument("--out", default=str(ROOT / "var" / "final" / "rewrites.json"))
    ap.add_argument("--redo-errors", action="store_true",
                    help="re-run only the rows of --out whose verdict was "
                         "'error', merging the results back in. The site "
                         "sample is seeded, so the same rewrites are chosen.")
    a = ap.parse_args()
    prior = (json.loads(Path(a.out).read_text())
             if a.redo_errors and Path(a.out).exists() else None)

    import os
    from chia_livelane.formal.proof_cache import (PartitionProofCache,
                                                  incremental_check)
    from chia_livelane.vlsi.elaborate import resolve, seed_sources
    from chia_livelane.vlsi.yosys_sta import yosys_sta_qor

    yosys = os.environ["LIVELANE_TOOLS"] + "/bin/yosys"
    sta = os.environ["LIVELANE_TOOLS"] + "/bin/sta"
    liberty = os.environ["LIVELANE_LIBERTY"]
    checkout = ROOT / "thirdparty" / "ibex"
    wr = Path(a.workroot).resolve()
    shutil.rmtree(wr, ignore_errors=True)
    wr.mkdir(parents=True)
    tree = wr / "src"
    r = run(["git", "worktree", "add", "--detach", "--force", str(tree), a.sha],
            cwd=checkout)
    if r.returncode:
        print(r.stderr)
        return 1
    try:
        seed = seed_sources(tree, "rtl/ibex_core.f", "rtl/*.sv")
        elab = resolve(tree, seed, "ibex_core", yosys, stash=wr / "stubs")
        if not elab.ok:
            print("base does not elaborate:", elab.error)
            return 1
        srcs, incs, defs = elab.sources, elab.includes, elab.defines
        # A rewritten file is written OUTSIDE rtl/, so its relative
        # `include`s stop resolving: ibex_cs_registers.sv:1030 includes
        # "ibex_pmp_reset_default.svh" and every rewrite of that file failed
        # to elaborate, dropping silently out of the sample as "error".
        rtl_dir = tree / "rtl"
        if rtl_dir not in incs:
            incs = list(incs) + [rtl_dir]
        read_cmd = ("read_slang --keep-hierarchy "
                    + " ".join(f"-I {i}" for i in incs) + " "
                    + " ".join(f"-D {d}" for d in defs))
        print(f"base: {len(srcs)} sources, defines={defs}", flush=True)

        # ---- base QoR -----------------------------------------------------
        t0 = time.time()
        base_q = yosys_sta_qor([str(s) for s in srcs], "ibex_core", liberty,
                               workdir=str(wr / "synth-base"),
                               clock_period_ns=a.clock_ns, clock_port="clk_i",
                               yosys=yosys, sta=sta, read_cmd=read_cmd)
        print(f"base QoR: area={base_q.get('area_um2')} "
              f"delay={base_q.get('max_delay_ns')} cells={base_q.get('cells')} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if not base_q.get("success"):
            print("base synthesis failed")
            return 1

        # ---- warm the proof cache on base vs base ---------------------------
        cache = PartitionProofCache(wr / "cache")
        warm_cfg = wr / "warm.eqy"
        warm_cfg.write_text(cfg_text(srcs, srcs, incs, defs, "ibex_core",
                                     "sat-first", ["--keep-hierarchy"]))
        t0 = time.time()
        warm = incremental_check(warm_cfg, wr / "wd-warm", cache,
                                 jobs=a.jobs, timeout_s=3600)
        print(f"warm: {warm.verdict} {warm.partitions_total} partitions "
              f"({time.time() - t0:.0f}s)", flush=True)
        if warm.verdict != "proven":
            print("base does not prove against itself -- apparatus failure")
            return 1

        inst = instance_names(checkout / "rtl", a.sha, checkout)

        # ---- candidate sites ----------------------------------------------
        targets = [s for s in srcs
                   if s.parent.name == "rtl" and not s.name.endswith("_pkg.sv")]
        dset = set(defs)
        by_op: dict[str, list] = {}
        for f in targets:
            for site in sites(f, dset):
                by_op.setdefault(site[0], []).append((f, *site))
        rng = random.Random(a.seed)
        chosen = []
        print("sites available per operator:", flush=True)
        for name, fam, _, _ in OPERATORS:
            pool = by_op.get(name, [])
            rng.shuffle(pool)
            per_file: dict = {}
            picked = []
            for c in pool:
                k = c[0].name
                if per_file.get(k, 0) >= a.per_file:
                    continue
                per_file[k] = per_file.get(k, 0) + 1
                picked.append(c)
                if len(picked) >= a.per_op:
                    break
            print(f"  {name:18s} {fam:12s} {len(pool):5d} available, "
                  f"{len(picked)} sampled", flush=True)
            chosen += picked
        if a.limit:
            chosen = chosen[:a.limit]
        redo = None
        if prior is not None:
            redo = {r["i"] for r in prior["rows"] if r["verdict"] == "error"}
            print(f"redo: {len(redo)} errored row(s): {sorted(redo)}",
                  flush=True)

        # ---- mutate, prove, score -------------------------------------------
        rows = []
        pool = ThreadPoolExecutor(max_workers=a.synth_parallel)
        futs = {}
        for i, (f, name, fam, s, e, new, orig) in enumerate(chosen):
            if redo is not None and i not in redo:
                continue
            if prior is not None:
                p_row = next(r for r in prior["rows"] if r["i"] == i)
                assert (p_row["operator"], p_row["file"]) == \
                    (name, f"rtl/{f.name}"), \
                    f"seeded sample drifted at {i}: rerun would not match"
            mdir = wr / f"m{i:03d}"
            mdir.mkdir()
            text = f.read_text(errors="ignore")
            mfile = mdir / f.name
            mfile.write_text(text[:s] + new + text[e:])
            gate_srcs = [mfile if q == f else q for q in srcs]
            line = text.count("\n", 0, s) + 1
            row = {"i": i, "operator": name, "family": fam,
                   "file": f"rtl/{f.name}", "line": line,
                   "module": f.stem, "original": orig, "rewrite": new}
            futs[i] = pool.submit(
                yosys_sta_qor, [str(q) for q in gate_srcs], "ibex_core",
                liberty, workdir=str(mdir / "synth"),
                clock_period_ns=a.clock_ns, clock_port="clk_i",
                yosys=yosys, sta=sta, read_cmd=read_cmd)
            cfg = mdir / "m.eqy"
            cfg.write_text(cfg_text(srcs, gate_srcs, incs, defs, "ibex_core",
                                    "sat-first", ["--keep-hierarchy"]))
            t0 = time.time()
            got = incremental_check(cfg, mdir / "wd", cache, jobs=a.jobs,
                                    timeout_s=1800)
            row["verdict"] = got.verdict
            row["gate_s"] = round(time.time() - t0, 2)
            row["failed"] = list(got.failed_partitions)
            names = inst.get(f.stem, set())
            row["localised"] = (any(f".{n}." in p for p in row["failed"]
                                    for n in names)
                                if row["failed"] and names else None)
            rows.append(row)
            print(f"[{i + 1:>3}/{len(chosen)}] {name:18s} {f.name:28s} "
                  f"L{line:<5} {got.verdict:8s} {row['gate_s']:6.1f}s "
                  f"loc={row['localised']}", flush=True)

        print("waiting for synthesis...", flush=True)
        for row in rows:
            q = futs[row["i"]].result()
            row["synth_ok"] = bool(q.get("success"))
            row["area_um2"] = q.get("area_um2")
            row["max_delay_ns"] = q.get("max_delay_ns")
            row["cells"] = q.get("cells")
        pool.shutdown()

        if prior is not None:
            fixed = {r["i"]: r for r in rows}
            rows = [fixed.get(r["i"], r) for r in prior["rows"]]
        out = {"sha": a.sha, "seed": a.seed, "clock_ns": a.clock_ns,
               "base": {"area_um2": base_q.get("area_um2"),
                        "max_delay_ns": base_q.get("max_delay_ns"),
                        "cells": base_q.get("cells")},
               "warm_partitions": warm.partitions_total,
               "rows": rows}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {a.out}")
        summarise(out)
        return 0
    finally:
        run(["git", "worktree", "remove", "--force", str(tree)], cwd=checkout)


def summarise(out: dict) -> None:
    base = out["base"]
    rows = out["rows"]
    eps = 1e-6

    def better(r, key):
        return (r.get(key) is not None and base.get(key) is not None
                and r[key] < base[key] - eps)

    eq = [r for r in rows if r["family"] == "equivalent"]
    sm = [r for r in rows if r["family"] == "simplifying"]
    settled = lambda rs: [r for r in rs if r["verdict"] in ("proven", "refuted")]
    print("\n=== equivalent rewrites (controls) ===")
    es = settled(eq)
    print(f"  {len(es)} settled of {len(eq)};  proven {sum(r['verdict'] == 'proven' for r in es)}"
          f"  refuted {sum(r['verdict'] == 'refuted' for r in es)}  (a refutation here is a FALSE REJECTION)")
    print("\n=== simplifying rewrites ===")
    ss = settled(sm)
    broken = [r for r in ss if r["verdict"] == "refuted"]
    equivm = [r for r in ss if r["verdict"] == "proven"]
    print(f"  {len(ss)} settled of {len(sm)}: {len(broken)} BROKEN, "
          f"{len(equivm)} equivalent (edited logic dead or masked)")
    ok = [r for r in broken if r.get("synth_ok")]
    d = [r for r in ok if better(r, "max_delay_ns")]
    ar = [r for r in ok if better(r, "area_um2")]
    either = [r for r in ok if better(r, "max_delay_ns") or better(r, "area_um2")]
    if ok:
        print(f"  of {len(ok)} broken rewrites that synthesise:")
        print(f"    FASTER critical path than the original : {len(d):3d}  "
              f"({100 * len(d) / len(ok):.0f}%)")
        print(f"    SMALLER area than the original         : {len(ar):3d}  "
              f"({100 * len(ar) / len(ok):.0f}%)")
        print(f"    better on EITHER -- a QoR loop keeps it : {len(either):3d}  "
              f"({100 * len(either) / len(ok):.0f}%)")
    loc = [r for r in broken if r.get("localised") is not None]
    if loc:
        h = sum(1 for r in loc if r["localised"])
        print(f"  localised to the rewritten module: {h}/{len(loc)}")
    uns = [r for r in rows if r["verdict"] not in ("proven", "refuted")]
    if uns:
        print(f"  !! {len(uns)} did not settle: "
              + ", ".join(f"#{r['i']} {r['verdict']}" for r in uns))
    gs = sorted(r["gate_s"] for r in rows)
    if gs:
        print(f"\n  gate per rewrite: median {gs[len(gs) // 2]:.1f}s  "
              f"max {gs[-1]:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summarise":
        summarise(json.loads(Path(sys.argv[2]).read_text()))
        raise SystemExit(0)
    raise SystemExit(main())
