"""The proposed loop, end to end, as one CHIA graph.

    agent edits one module
      -> simulate        (cheap, coverage-limited: reject on it, never accept)
      -> score           (synthesis + timing: says FASTER, never says CORRECT)
      -> prove           (the only stage that can refute; incremental)
      -> accept/reject, iterate from the best

Each stage is a ChiaFunction with its own resource token, so the graph is what
schedules them, not this script.

Three properties this encodes, each measured rather than assumed:

* **Simulation cannot gate.** picorv32 with an inverted BGE produces a trace
  byte-identical to the correct design (same sha256 over 274 lines), because the
  testbench's six-instruction program contains no conditional branch. Passing it
  bounds nothing.
* **Score cannot gate either.** That same bug synthesises to 6456 cells against
  6691 and 75,074 um2 against 75,663, it is scored BETTER than correct.
* **So the proof is the gate**, and it is affordable only because it is
  incremental: `eqy` has no reuse path, and re-proves every partition of the
  design for a one-line edit, 16,065 of them on a XiangShan block. With a
  partition cache that becomes only what the edit touched: **95.4-98.3% of
  PROVE time saved** on live semantic edits, measured on three XiangShan
  blocks and a CVA6 cache-tag comparator cone (7 files, not the
  application-class core, whole-core CVA6 does not partition at all).

  Two things that number is not. It is prove-phase only: the partition step
  is re-run in full every time and the cache never touches it, so end-to-end
  is far smaller and on some instances below 1x. And it excludes the
  comment-only probes, which reach 98.5% precisely because they change no
  behaviour.

The agent is blind by construction: it is handed the design, its QoR and its
worst path, and nothing that names a lane, an arm, an injected delay or a
wall-clock. Its timing view now includes the RTL line the critical path ends at
-- before this it was given only post-synthesis cell names like `_10915_`,
which cannot be edited.

Run:  python scripts/chia/loop.py --iterations 3 --model gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DESIGNS = {
    # Single-file design: `rtl` is both the whole design and the only editable
    # file, and the front-end command line is fixed.
    "picorv32": {
        "rtl": "designs/picorv32/picorv32.v",
        "top": "picorv32",
        "tb": "thirdparty/picorv32/testbench_ez.v",
        "tb_top": "testbench",
        "clk": "clk",
        "period": 10.0,
        "read_cmd": "read_slang",
    },
    # Multi-file design: the source list, include paths and defines are all
    # resolved from the checkout at start-up rather than written down here,
    # because they move between commits. `editable_prefix` keeps the agent
    # inside the design's own RTL, vendored primitives are context, not
    # something it may rewrite.
    "ibex": {
        "tree": "thirdparty/ibex",
        "filelist": "rtl/ibex_core.f",
        "rtl_glob": "rtl/*.sv",
        "editable_prefix": "rtl/",
        "top": "ibex_core",
        "clk": "clk_i",
        "period": 10.0,
    },
    # An application-class core, and a third design family. `prefer` names the
    # build configuration: 21 files declare `cva6_config_pkg` and they are not
    # interchangeable, so which one is used is stated here rather than decided
    # by sort order.
    "cva6": {
        "tree": "thirdparty/cva6",
        "filelist": "core/Flist.cva6",
        "rtl_glob": "core/*.sv",
        "editable_prefix": "core/",
        "prefer": {"cva6_config_pkg": "cv64a6_imafdc_sv39_config_pkg"},
        "top": "cva6",
        "clk": "clk_i",
        "period": 10.0,
    },
}


def _file_of(loc: object) -> str | None:
    """The file part of an STA location like `../parent.v:1402.2`."""
    if not loc:
        return None
    return Path(str(loc).split(":")[0]).name


def _timing_view(q: dict, withhold_rtl: bool) -> dict:
    """What the agent is shown about its worst path.

    The treatment is a single field. With ``withhold_rtl`` the agent sees only
    post-synthesis cell names, ``_10915_``, which name nothing it can edit;
    that is the state before ``endpoint_rtl`` existed, and the control arm here.
    Without it, the same view carries the RTL location the path ends at.

    The path summary is stripped of its inline annotations in the control, or
    the hint would leak back in through it.
    """
    keys = ["slack_ns", "endpoint", "startpoint", "path_summary"]
    view = {k: q.get(k) for k in keys}
    if withhold_rtl:
        ps = view.get("path_summary") or ""
        view["path_summary"] = "\n".join(
            ln.split("   <- ")[0] for ln in ps.splitlines())
        return view
    view["endpoint_rtl"] = q.get("endpoint_rtl")
    view["startpoint_rtl"] = q.get("startpoint_rtl")
    return view


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", default="picorv32", choices=sorted(DESIGNS))
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--model", default="gemini-2.5-flash",
                    help="use a SKU the price table knows, or cost stays None")
    ap.add_argument("--workdir", default=str(ROOT / "var" / "loop"))
    ap.add_argument("--cache", default=str(ROOT / "var" / "proofcache"),
                    help="shared partition-proof store; reuse crosses runs")
    ap.add_argument("--max-usd", type=float, default=2.0)
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel proof jobs; default is CHIA's share of the "
                         "machine. Cap it when other work is running, or the "
                         "runs oversubscribe and every timing becomes an "
                         "upper bound with no way to tell by how much.")
    ap.add_argument("--no-sim", action="store_true")
    ap.add_argument("--no-rtl-hint", action="store_true",
                    help="CONTROL ARM: withhold endpoint_rtl, so the agent sees "
                         "only post-synthesis cell names (_10915_) as it did "
                         "before that field existed. Everything else identical.")
    ap.add_argument("--second-stage", action="store_true",
                    help="on a refusal, prove the edited instance (unbounded, "
                         "at the design's real parameters) before believing "
                         "it; accept on that proof")
    ap.add_argument("--init", choices=["reset", "zero"], default="reset",
                    help="second stage's initial state: 'reset' -- both copies "
                         "power up to the same arbitrary state, reset asserted "
                         "in the first cycle; package edits proven where their "
                         "types reach. 'zero' -- both from all-zero, as in the "
                         "runs reported with the release.")
    ap.add_argument("--seed", type=int, default=0,
                    help="recorded with the run; the model is still sampled")
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from chia_livelane.base.hostload import host_load, load_warning
    warn = load_warning()
    if warn:
        log(f"!! {warn}")

    d = DESIGNS[a.design]
    liberty = os.environ.get("LIVELANE_LIBERTY")
    if not liberty or not Path(liberty).exists():
        log(f"FATAL: LIVELANE_LIBERTY unset/missing ({liberty!r}); source env.sh")
        return 2

    wd = Path(a.workdir)
    shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True, exist_ok=True)

    # The loop edits a working COPY, never the checkout. `work` is the full
    # source list handed to every tool; `editable` is the subset the agent may
    # rewrite. For a single-file design the two coincide.
    if d.get("tree"):
        from chia_livelane.vlsi.elaborate import resolve, seed_sources
        tree = (ROOT / d["tree"]).resolve()
        yosys = str(ROOT / "tools" / "bin" / "yosys")
        elab = resolve(tree, seed_sources(tree, d.get("filelist"),
                                          d.get("rtl_glob")),
                       d["top"], yosys, prefer=d.get("prefer"),
                       stash=wd / "resolved")
        if not elab.ok:
            log(f"FATAL: {a.design} does not elaborate: {elab.error}")
            return 2
        for n in elab.notes:
            log(f"  elaborate: {n}")
        # Hierarchy kept: partition cuts land on module ports, so a refusal
        # names the module that changed, and there are 1.51x fewer partitions
        # (12/12 verdict agreement with flat on the ibex corpus).
        read_cmd = elab.read_cmd()
        gate_read_cmd = elab.read_cmd("read_slang --keep-hierarchy")
        srcdir = wd / "src"
        srcdir.mkdir(parents=True, exist_ok=True)
        prefix = d.get("editable_prefix", "")
        work: list[Path] = []
        editable: list[Path] = []
        for s in elab.sources:
            try:
                rel = str(s.relative_to(tree))
            except ValueError:
                rel = s.name          # copied in from outside the checkout
            if prefix and rel.startswith(prefix):
                dst = srcdir / Path(rel).name
                dst.write_text(s.read_text())
                work.append(dst)
                editable.append(dst)
            else:
                work.append(s)
        if not editable:
            log(f"FATAL: no editable sources under {prefix!r}")
            return 2
        parent = editable[0]
        tb = None
        log(f"sources: {len(work)} ({len(editable)} editable)")
    else:
        read_cmd = d["read_cmd"]
        gate_read_cmd = read_cmd
        parent = wd / "parent.v"
        parent.write_text((ROOT / d["rtl"]).read_text())
        work = [parent]
        editable = [parent]
        tb = ROOT / d["tb"]

    from chia_livelane.formal.lec_gate import (DEFAULT_STRATEGIES, auto_jobs,
                                               lec_gate)
    from chia_livelane.sim.verilator import verilator_sim
    from chia_livelane.vlsi.yosys_sta import yosys_sta_qor
    from livelane.agent.llm import build_user_prompt  # noqa: F401  (shape only)

    sys.path.insert(0, str(ROOT / "scripts" / "chia"))
    from population import _make_propose
    propose_edit = _make_propose()

    import ray
    from chia.base.ChiaFunction import get

    import ray

    def rget(make, tries: int = 6):
        """Submit and fetch a CHIA task, resubmitting if its worker is lost.

        On a machine shared with other work, Ray's memory monitor kills the
        most recent worker whenever the NODE crosses 95%, a momentary spike
        from someone else's job is enough, and a bare get() then raises and
        ends the whole run. 13 of 13 runs were lost that way in one evening.
        A lost worker is not a result; the step is simply run again.
        """
        for k in range(tries):
            try:
                return get(make())
            except ray.exceptions.RayError as e:
                if k == tries - 1:
                    raise
                wait = 20 * (k + 1)
                log(f"  !! task lost ({type(e).__name__}); resubmitting in {wait}s")
                time.sleep(wait)
    ray.init(resources={"vertex_creds": 1.0, "yosys_sta": 2.0, "eqy": 1.0,
                        "verilator_run": 1.0}, log_to_driver=False)
    log(f"graph up: design={a.design} model={a.model} iterations={a.iterations}")
    log(f"proof cache: {a.cache}")

    # --- seed -------------------------------------------------------------
    if tb is None and not a.no_sim:
        # A design with no testbench cannot be simulated. Say so once rather
        # than failing an iteration at a time, and carry on: simulation is a
        # cheap pre-filter here, never the thing that accepts an edit.
        log("no testbench for this design -- simulation stage disabled")
        a.no_sim = True

    srcs = [str(x) for x in work]
    seed = rget(lambda: yosys_sta_qor.chia_remote(
        srcs, d["top"], liberty, workdir=str(wd / "qor-seed"),
        clock_port=d["clk"], clock_period_ns=d["period"], read_cmd=read_cmd))
    best = seed["max_delay_ns"]
    # The view the agent is shown must track the design it is editing. `seed`
    # is the measurement of the ORIGINAL netlist; once an edit is accepted the
    # critical path has moved, and continuing to show the seed's path asks the
    # agent to optimise a path that no longer dominates. `cur` is re-pointed at
    # the accepted candidate's own QoR so every iteration sees the design as it
    # now is.
    cur = seed
    log(f"seed: cells={seed['cells']} delay={best} ns  "
        f"endpoint_rtl={seed.get('endpoint_rtl')}")

    history: list[dict] = []
    spent = 0.0
    rows: list[dict] = []
    accepted = 0

    for i in range(a.iterations):
        if spent >= a.max_usd:
            log(f"STOP: ${spent:.3f} hit the ${a.max_usd:.2f} cap")
            break
        log(f"--- iteration {i} ---")
        t0 = time.monotonic()

        # The file the agent is shown follows the critical path. On a
        # multi-file design the dominant path moves between modules as edits
        # land, so pinning one file would send the agent to the wrong module
        # as soon as it succeeded. Endpoint first, then startpoint, then the
        # design's own first file.
        by_name = {f.name: f for f in editable}
        focus = (by_name.get(_file_of(cur.get("endpoint_rtl")))
                 or by_name.get(_file_of(cur.get("startpoint_rtl")))
                 or editable[0])
        rtl = focus.read_text()

        # 1. propose --------------------------------------------------------
        p = rget(lambda: propose_edit.chia_remote(
            rtl, str(focus), d["top"], a.design, i,
            {k: cur.get(k) for k in ("cells", "area_um2", "max_delay_ns")},
            # the agent's timing view: now carries the RTL line, not just a cell
            _timing_view(cur, a.no_rtl_hint),
            history[-5:], a.model, None))
        spent += p.get("cost_usd") or 0.0
        if not p.get("ok"):
            log(f"  no usable proposal ({p.get('error')})")
            rows.append({"iteration": i, "outcome": "no-proposal"})
            continue
        # Which file does the edit land in? The focus file is the intended
        # target, but the agent may quote a line that is not unique there, or
        # that belongs to another editable file. Require exactly one owner: an
        # ambiguous site would be applied somewhere we did not choose.
        owners = [f for f in ([focus] if p["old"] in rtl else [])] or \
                 [f for f in editable if p["old"] in f.read_text()]
        if not owners:
            log("  edit text not found -- rejected before any tool ran")
            rows.append({"iteration": i, "outcome": "unmatched-edit"})
            history.append({"note": p["note"], "result": "edit did not apply"})
            continue
        if len(owners) > 1:
            log(f"  edit text occurs in {len(owners)} files -- ambiguous, "
                f"rejected before any tool ran")
            rows.append({"iteration": i, "outcome": "ambiguous-edit"})
            history.append({"note": p["note"],
                            "result": "edit site was not unique"})
            continue
        target = owners[0]
        cand = wd / f"cand{i}_{target.name}"
        cand.write_text(target.read_text().replace(p["old"], p["new"], 1))
        cand_srcs = [str(cand if x == target else x) for x in work]
        log(f"  proposed: {p['note'][:70]}"
            + (f"  [{target.name}]" if len(editable) > 1 else ""))

        # 2. simulate (cheap filter; may only REJECT) ------------------------
        if not a.no_sim:
            s = rget(lambda: verilator_sim.chia_remote(
                [str(tb), str(cand)], str(wd / "sim"), top=d["tb_top"],
                jobs=8, name=f"i{i}"))
            if not s["passed"]:
                log(f"  simulate: FAIL ({s['message'][:60]}) -- rejected, "
                    f"no proof paid for")
                rows.append({"iteration": i, "outcome": "sim-rejected",
                             "sim_s": s["wall_s"]})
                history.append({"note": p["note"], "result": "broke simulation"})
                continue
            log(f"  simulate: pass ({s['wall_s']:.2f}s)  [evidence, not proof]")

        # 3. score ----------------------------------------------------------
        q = rget(lambda: yosys_sta_qor.chia_remote(
            cand_srcs, d["top"], liberty, workdir=str(wd / f"qor{i}"),
            clock_port=d["clk"], clock_period_ns=d["period"],
            read_cmd=read_cmd))
        if q.get("max_delay_ns") is None:
            log("  score: failed to synthesise -- rejected")
            rows.append({"iteration": i, "outcome": "synth-failed"})
            continue
        faster = q["max_delay_ns"] < best
        log(f"  score: {q['max_delay_ns']:.4f} ns vs {best:.4f} "
            f"({'better' if faster else 'not better'})")

        # 4. prove (incremental; the only stage that may ACCEPT) -------------
        v = rget(lambda: lec_gate.chia_remote(
            srcs, cand_srcs, d["top"], workdir=str(wd / f"lec{i}"),
            read_cmd=gate_read_cmd, jobs=a.jobs or auto_jobs(),
            strategies=[(n, list(v)) for n, v in DEFAULT_STRATEGIES],
            cache_dir=a.cache))
        log(f"  prove: {v['verdict']} ({v['wall_s']:.2f}s) {v['message'][:60]}")
        verdict_eqy = v["verdict"]

        # 4b. second stage: settle a refusal before believing it ------------
        # The partitioned gate can refuse a CORRECT edit that redefines a net
        # it cuts at. Proving the edited instance (at the design's real
        # parameters, unbounded) either shows the edit correct, accept on
        # that proof, or finds a counterexample, or cannot decide and the
        # refusal stands. The agent then hears the truth about its own edit.
        s2 = None
        if a.second_stage and v.get("refuted") and d.get("tree"):
            from chia_livelane.formal.second_stage import (PROVEN, second_stage,
                                                            settle)
            absolute = lambda xs: [str(Path(x).resolve()) for x in xs]
            s2_args = (absolute(srcs), absolute(cand_srcs), d["top"], target.stem,
                       [str(x) for x in elab.includes] + [str(tree / "rtl")],
                       list(elab.defines), str((wd / f"s2_{i}").resolve()))
            if a.init == "reset":
                s2 = getattr(second_stage, "__wrapped__", second_stage)(
                    *s2_args, init="reset")
            else:
                s2 = settle(*s2_args)
            log(f"  second stage: {s2['verdict']} ({s2['wall_s']:.2f}s) "
                + ", ".join(f"{x['instance'].split('.')[-1]}="
                            f"{x.get('result', x.get('reset'))}"
                            for x in s2["instances"]))
            if s2["verdict"] == PROVEN:
                v = dict(v, equivalent=True, refuted=False,
                         verdict="proven-seq")

        outcome = "rejected"
        parent_ns = best                  # what this proposal was scored against
        if v["equivalent"] and faster:
            target.write_text(cand.read_text())
            best = q["max_delay_ns"]
            cur = q                       # the design moved; so does the view
            accepted += 1
            outcome = "accepted"
            log(f"  ACCEPTED -> {best:.4f} ns")
        elif v["refuted"]:
            outcome = "refuted"
        history.append({"note": p["note"], "result": outcome})
        # Every proposal that reached the gate is recorded with what the loop
        # would need to judge it WITHOUT the gate: its QoR against the parent
        # it was scored against, the file it edited, the raw verdict, and the
        # candidate file itself so a refusal can be re-checked independently.
        rows.append({"iteration": i, "outcome": outcome,
                     "verdict": v["verdict"], "verdict_eqy": verdict_eqy,
                     "second_stage": (s2 or {}).get("verdict"),
                     "second_stage_s": (s2 or {}).get("wall_s"),
                     "parent_ns": parent_ns,
                     "faster": faster, "area_um2": q.get("area_um2"),
                     "file": target.name, "note": p["note"][:160],
                     "cand": str(cand),
                     "delay_ns": q["max_delay_ns"], "cells": q["cells"],
                     "prove_s": v["wall_s"], "prove_msg": v["message"][:80],
                     "load": host_load().as_dict(),
                     "iter_s": round(time.monotonic() - t0, 2),
                     "cost_usd": p.get("cost_usd")})

    base = seed["max_delay_ns"]
    summary = {
        "design": a.design, "model": a.model, "iterations": len(rows),
        "arm": "no-rtl-hint" if a.no_rtl_hint else "rtl-hint", "seed": a.seed,
        "second_stage": a.second_stage,
        "second_stage_init": a.init if a.second_stage else None,
        "accepted": accepted, "seed_ns": base, "best_ns": best,
        "improvement_pct": round(100 * (base - best) / base, 3) if base else None,
        "cost_usd": round(spent, 5) if spent else None,
        "proof_cache": a.cache, "load_at_end": host_load().as_dict(),
        "ladder": [n for n, _ in DEFAULT_STRATEGIES], "gate_read_cmd_base":
        gate_read_cmd.split(" -I")[0].split(" -D")[0], "rows": rows,
    }
    print("\n=== loop ===")
    for k in ("iterations", "accepted", "seed_ns", "best_ns",
              "improvement_pct", "cost_usd"):
        print(f"  {k:16s} {summary[k]}")
    out = wd / "loop-summary.json"
    out.write_text(json.dumps(summary, indent=2))
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
