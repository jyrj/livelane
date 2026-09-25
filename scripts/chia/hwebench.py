"""Measure incremental proof reuse on real, human-authored RTL bug fixes.

Every prior locality number came from an edit *we* wrote. HWE-Bench supplies
edits nobody on this project chose: merged bug-fix PRs from open-source cores,
each pinned to the commit it was written against.

Per instance the protocol is:

  warm    prove the buggy design against itself      -> fills a FRESH cache
  measure prove the buggy design against the fixed   -> reuse% and verdict

Isolation is per instance and total: its own git worktree checked out at the
instance's own ``base.sha``, its own cache directory, its own eqy workdir.
Nothing carries across instances. A cache shared between two unrelated PRs
would hand instance N spurious hits from instance N-1 and the reuse number
would be fiction.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from chia_livelane.base.hostload import host_load, load_warning  # noqa: E402
from chia_livelane.vlsi.elaborate import resolve, seed_sources  # noqa: E402

DATASETS = ROOT / "thirdparty" / "hwe-bench" / "datasets"

# Per-repo elaboration entry points. This is configuration, not design
# knowledge: the top module and the directory the RTL lives in. Include paths
# are NOT listed, they are discovered from the error stream (see
# `resolve_includes`), because they differ across the pinned SHAs.
REPOS: dict[str, dict] = {
    "lowRISC__ibex": {
        "checkout": ROOT / "thirdparty" / "ibex",
        "top": "ibex_core",
        "rtl_glob": "rtl/*.sv",
        "rtl_prefix": "rtl/",
        "filelist": "rtl/ibex_core.f",
    },
    "openhwgroup__cva6": {
        "checkout": ROOT / "thirdparty" / "cva6",
        "top": "cva6",
        "rtl_glob": "core/*.sv",
        "rtl_prefix": "core/",
        "filelist": "core/Flist.cva6",
        # 21 files declare `cva6_config_pkg`; they are mutually exclusive build
        # configurations, so the one used is named rather than inferred.
        "prefer": {"cva6_config_pkg": "cv64a6_imafdc_sv39_config_pkg"},
        # The only two submodules carrying RTL in the `cva6` cone. They are
        # initialised inside each instance's worktree, so each instance gets
        # the submodule commit ITS base.sha pinned, not whatever HEAD points
        # at, which would silently mix versions across instances.
        "submodules": ["core/cvfpu", "core/cache_subsystem/hpdcache"],
        # Whole-core scope is not reachable for CVA6: `eqy_partition` on
        # `--top cva6` ran 17+ minutes without emitting a single partition
        # (2.2 GB of progress output, empty partition.list). Each instance is
        # therefore checked at the scope of the module its own fix modifies.
        #
        # This claims LESS than the ibex rows, and the results must say so: it
        # states that the buggy and fixed versions of THAT MODULE differ, not
        # that the core does. `read_slang` flattens, so a module-scope result
        # does not compose into a whole-core one.
        "top_mode": "module-of-fix",
        # Measured, not assumed. Both routes are closed:
        #  - whole core: `eqy_partition` on `--top cva6` ran 17m25s and emitted
        #    ZERO partitions (2.2 GB of progress output). Not a timeout to
        #    raise, it had not begun producing output.
        #  - module scope: CVA6 modules are not independently elaborable. Their
        #    struct types are TYPE PARAMETERS defaulting to `logic`, bound by
        #    the instantiating parent, so standalone elaboration fails on every
        #    field access:
        #      core/decoder.sv:183:18: error: invalid member access for type
        #                              'scoreboard_entry_t' (aka 'logic')
        #    Checked on decoder, csr_regfile and cache_ctrl, all three, the
        #    same way. `read_slang -G` could supply the types, but only from a
        #    hand-written per-module parameter map, redone at each instance's
        #    own base commit where the config package differs. That is the
        #    per-design knowledge this harness exists to avoid.
        "excluded": ("whole-core partitioning does not terminate, and modules "
                     "are not independently elaborable (type parameters "
                     "default to `logic`)"),
    },
}

def sh(cmd: list[str], cwd: Path | None = None, timeout: int = 900):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout)


def usable_instances(repo_key: str, limit: int | None = None) -> list[dict]:
    """Instances whose fix touches exactly one RTL file in the elaborated tree.

    Multi-file fixes are not excluded because they are hard, they are
    excluded because attributing reuse to a change means knowing what changed.
    Documentation-only hunks in the same PR are filtered out by prefix, so a PR
    that edits one module plus three .rst files still counts.
    """
    cfg = REPOS[repo_key]
    out = []
    path = DATASETS / f"{repo_key}.jsonl"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("level1") != "RTL_BUG_FIX":
            continue
        if not r.get("fix_patch") or not r.get("base", {}).get("sha"):
            continue
        rtl = [f for f in (r.get("modified_files") or [])
               if f.startswith(cfg["rtl_prefix"])
               and f.endswith((".sv", ".v"))]
        if len(rtl) != 1:
            continue
        out.append({"number": r["number"], "sha": r["base"]["sha"],
                    "file": rtl[0], "title": (r.get("title") or "")[:90],
                    "fix_patch": r["fix_patch"],
                    "added": r.get("lines_added"),
                    "removed": r.get("lines_removed")})
    out.sort(key=lambda d: d["number"])
    return out[:limit] if limit else out


#: Strategy ladders the corpus can be run under.
#:
#: `smt-first` is the safe default. Leading with `sat` is roughly 9x faster on
#: proving, and the false accept that ordering was chosen to avoid is
#: independently closed by `setundef -zero -init`, which both ladders emit,
#: but that was shown on two fixtures, and two fixtures is how the original bug
#: hid. This corpus is the test that settles it: every instance here is a real
#: bug, authored by someone else, and every one must still be refuted.
#: `smt-only` is the default because it is what every result recorded so far
#: was measured under, and because it is the most conservative: a partition
#: `sby` leaves undecided stays undecided. The ladders that append `induct`
#: let yosys's `sat` PASS such a partition, which is the residual risk
#: `lec_gate.py` documents and declines to hide. Changing the default would
#: silently make new numbers incomparable with old ones AND weaken the check.
LADDERS: dict[str, str] = {
    "smt-only": "[strategy smt]\nuse sby\nengine smtbmc yices\ndepth 10\n",
    "smt-first": ("[strategy smt]\nuse sby\nengine smtbmc yices\ndepth 10\n\n"
                  "[strategy induct]\nuse sat\ndepth 10\n"),
    "sat-first": ("[strategy simple]\nuse sat\ndepth 3\n\n"
                  "[strategy smt]\nuse sby\nengine smtbmc yices\ndepth 10\n"),
}


def cfg_text(gold_srcs: list[Path], gate_srcs: list[Path], incs: list[Path],
             defs: list[str], top: str, ladder: str = "smt-only",
             flags: list[str] | None = None) -> str:
    inc = " ".join(f"-I {q}" for q in incs)
    dfs = " ".join(f"-D {d}" for d in defs)
    fl = " ".join(flags or [])

    def block(srcs):
        files = " ".join(str(s) for s in srcs)
        return (f"read_slang {fl} {inc} {dfs} {files} --top {top}\n"
                f"prep -top {top}\nsetundef -zero -init\nmemory_map")

    return (f"[gold]\n{block(gold_srcs)}\n\n[gate]\n{block(gate_srcs)}\n\n"
            f"[collect *]\n\n" + LADDERS[ladder])


_MODULE_DECL = re.compile(r"^\s*module\s+(\w+)", re.M)


def top_for(cfgr: dict, tree: Path, rel_file: str) -> tuple[str, str]:
    """(top module, scope label) for this instance.

    Default is the repo's whole-design top. `module-of-fix` instead takes the
    first module the edited file declares, which is the only tractable scope
    for a design whose whole-core partitioning does not terminate.
    """
    if cfgr.get("top_mode") != "module-of-fix":
        return cfgr["top"], "whole-design"
    try:
        text = (tree / rel_file).read_text(errors="ignore")
    except OSError:
        # The fix's file need not exist in every checkout this is asked
        # about, a file renamed or deleted since the instance's base commit
        # is a missing path, not a crash. Fall back and let elaboration report
        # the real problem.
        return cfgr["top"], "whole-design"
    m = _MODULE_DECL.search(text)
    if not m:
        return cfgr["top"], "whole-design"
    return m.group(1), "module-of-fix"


def run_instance(repo_key: str, inst: dict, workroot: Path, jobs: int,
                 yosys: str, timeout_s: int,
                 verify_uncached: bool = False,
                 ladder: str = "smt-only",
                 front_end_flags: list[str] | None = None) -> dict:
    """One instance, start to finish, sharing nothing with any other."""
    from chia_livelane.formal.proof_cache import (PartitionProofCache,
                                                  incremental_check)
    cfgr = REPOS[repo_key]
    tag = f"{repo_key}#{inst['number']}"
    sc = workroot / repo_key / str(inst["number"])
    shutil.rmtree(sc, ignore_errors=True)
    sc.mkdir(parents=True)
    rec: dict = {"load_at_start": host_load().as_dict(), "ladder": ladder,
                 "front_end_flags": list(front_end_flags or []),
                 "repo": repo_key, "number": inst["number"],
                 "sha": inst["sha"], "file": inst["file"],
                 "title": inst["title"], "lines_added": inst["added"],
                 "lines_removed": inst["removed"]}

    # --- isolated checkout at this instance's own base commit ---------------
    tree = sc / "src"
    r = sh(["git", "worktree", "add", "--detach", "--force", str(tree),
            inst["sha"]], cwd=cfgr["checkout"])
    if r.returncode != 0:
        rec["status"] = "checkout-failed"
        rec["detail"] = (r.stderr or r.stdout)[-300:]
        return rec

    try:
        for sub in cfgr.get("submodules", []):
            if (tree / sub).exists() and any((tree / sub).iterdir()):
                continue
            r = sh(["git", "submodule", "update", "--init", "--depth", "1",
                    "--filter=blob:none", sub], cwd=tree, timeout=1800)
            if r.returncode != 0:
                rec["status"] = "submodule-failed"
                rec["detail"] = f"{sub}: {(r.stderr or r.stdout)[-200:]}"
                return rec

        seed = seed_sources(tree, cfgr.get("filelist"), cfgr.get("rtl_glob"))
        if not seed:
            rec["status"] = "no-sources"
            return rec
        top, scope = top_for(cfgr, tree, inst["file"])
        rec["top"] = top
        rec["scope"] = scope
        elab = resolve(tree, seed, top, yosys,
                       fallback=cfgr["checkout"], stash=sc / "stubs",
                       prefer=cfgr.get("prefer"),
                       front_end_flags=front_end_flags)
        rec["resolution"] = elab.notes
        if not elab.ok:
            rec["status"] = "elaborate-failed"
            rec["detail"] = elab.error or ""
            return rec
        sources, incs, defs = elab.sources, elab.includes, elab.defines
        rec["include_dirs"] = [str(q) if not q.is_relative_to(tree)
                               else str(q.relative_to(tree)) for q in incs]
        rec["defines"] = list(defs)
        rec["n_sources"] = len(sources)

        # --- buggy (as committed) vs fixed (that one file, patched) --------
        buggy_file = (tree / inst["file"]).resolve()
        if buggy_file not in {q.resolve() for q in sources}:
            # The fix touches a file outside the elaborated cone, so no proof
            # over `top` can see it. Skipping is the honest outcome: a
            # "proven" here would say nothing about the change.
            rec["status"] = "file-outside-cone"
            return rec
        buggy_text = buggy_file.read_text()
        patch = sc / "fix.patch"
        patch.write_text(inst["fix_patch"])
        ap = sh(["git", "apply", "--include", inst["file"], str(patch)],
                cwd=tree)
        if ap.returncode != 0:
            rec["status"] = "patch-failed"
            rec["detail"] = (ap.stderr or ap.stdout)[-300:]
            return rec
        fixed_text = buggy_file.read_text()
        if fixed_text == buggy_text:
            rec["status"] = "patch-noop"
            return rec
        buggy_file.write_text(buggy_text)          # restore: tree stays buggy

        # gold = fixed, gate = buggy. Swap in a sibling copy of the one file
        # so both sides elaborate from the same source list.
        fixed_copy = sc / ("fixed_" + buggy_file.name)
        fixed_copy.write_text(fixed_text)
        gold_srcs = [fixed_copy if s.resolve() == buggy_file else s
                     for s in sources]
        gate_srcs = list(sources)

        # Resolve the FIXED side too, and union.
        #
        # The resolver above ran against the buggy sources only. A fix may
        # reference something the buggy version never did, ibex #1780 adds a
        # function that pulls in `ibex_pmp_reset_default.svh`, an include the
        # buggy file has no reason to mention. Elaborating gold with a
        # configuration derived from gate then fails with "Reading sources
        # failed", the measure phase returns `error`, and the instance is
        # counted as measured with 0 partitions and 0% reuse.
        #
        # Both sides must end up with the IDENTICAL union, because giving gold
        # and gate different includes or defines would break the symmetry the
        # soundness argument rests on.
        gold_elab = resolve(tree, gold_srcs, top, yosys,
                            fallback=cfgr["checkout"], stash=sc / "stubs",
                            prefer=cfgr.get("prefer"),
                            front_end_flags=front_end_flags)
        if not gold_elab.ok:
            rec["status"] = "elaborate-failed-fixed-side"
            rec["detail"] = gold_elab.error or ""
            rec["resolution"] = list(elab.notes) + list(gold_elab.notes)
            return rec
        if gold_elab.notes:
            rec["resolution"] = list(elab.notes) + [
                f"[fixed side] {n}" for n in gold_elab.notes]

        def _union(a, b):
            seen, out = set(), []
            for x in list(a) + list(b):
                k = str(x)
                if k not in seen:
                    seen.add(k)
                    out.append(x)
            return out

        incs = _union(incs, gold_elab.includes)
        defs = _union(defs, gold_elab.defines)
        rec["include_dirs"] = [str(q) if not q.is_relative_to(tree)
                               else str(q.relative_to(tree)) for q in incs]
        rec["defines"] = list(defs)
        # Any source the fixed side needed and the buggy side did not goes on
        # BOTH lists, so the two remain a valid miter.
        extra = [s for s in gold_elab.sources
                 if s not in gold_srcs and s != fixed_copy]
        if extra:
            gold_srcs = gold_srcs + extra
            gate_srcs = gate_srcs + extra
            rec["extra_sources_from_fixed_side"] = [str(s) for s in extra]
        rec["n_sources"] = len(gate_srcs)

        cache = PartitionProofCache(sc / "cache")   # FRESH, per instance
        assert len(cache) == 0, "cache must start empty for every instance"

        # warm: buggy vs buggy. Must prove; anything else is apparatus failure.
        warm_cfg = sc / "warm.eqy"
        warm_cfg.write_text(cfg_text(gate_srcs, gate_srcs, incs, defs,
                                     top, ladder, front_end_flags))
        t0 = time.time()
        warm = incremental_check(warm_cfg, sc / "wd", cache, jobs=jobs,
                                 timeout_s=timeout_s)
        rec["warm"] = {"verdict": warm.verdict, "partitions": warm.partitions_total,
                       "reuse_pct": warm.reuse_pct, "prove_s": warm.prove_s,
                       "setup_s": warm.setup_s,
                       "wall_s": round(time.time() - t0, 2)}
        print(f"      warm  {warm.verdict:9s} {warm.partitions_total:5d} parts "
              f"prove={warm.prove_s:7.2f}s", flush=True)
        if warm.verdict != "proven":
            rec["status"] = "warm-not-proven"
            rec["detail"] = str(warm.failed_partitions[:3])
            return rec

        # measure: buggy vs fixed, against the warm cache.
        meas_cfg = sc / "measure.eqy"
        meas_cfg.write_text(cfg_text(gold_srcs, gate_srcs, incs, defs,
                                     top, ladder, front_end_flags))
        t1 = time.time()
        got = incremental_check(meas_cfg, sc / "wd", cache, jobs=jobs,
                                timeout_s=timeout_s)
        rec["measure"] = {"verdict": got.verdict,
                          "partitions": got.partitions_total,
                          "reuse_pct": got.reuse_pct,
                          "hits": got.hits, "misses": got.misses,
                          "prove_s": got.prove_s, "setup_s": got.setup_s,
                          "wall_s": round(time.time() - t1, 2),
                          # NOT truncated. It was [:5], and seven of twelve
                          # flat instances came back with exactly 5 failing
                          # partitions, i.e. an unknown number, silently
                          # capped. Every localisation figure computed from
                          # this field was therefore computed over an
                          # arbitrary prefix, and the ladder comparison's
                          # "same failing partitions" control was comparing
                          # two truncated prefixes and calling them sets.
                          "failed": list(got.failed_partitions),
                          "n_failed": len(got.failed_partitions)}
        if warm.prove_s:
            rec["prove_time_saved_pct"] = round(
                (1 - got.prove_s / warm.prove_s) * 100, 1)
        # Optional soundness control. Everything above says the cache makes the
        # check FAST. It says nothing about whether the cache makes it RIGHT.
        # Re-run the identical comparison against a cache that has never seen
        # this design and require the same answer, same verdict, and the same
        # set of failing partitions. A cache that returned a stale proof for a
        # partition the edit touched would show up here as a disagreement, and
        # nowhere else.
        if verify_uncached:
            fresh = PartitionProofCache(sc / "cache-control")
            assert len(fresh) == 0
            t2 = time.time()
            ctl = incremental_check(meas_cfg, sc / "wd-control", fresh,
                                    jobs=jobs, timeout_s=timeout_s)
            same_verdict = ctl.verdict == got.verdict
            same_failed = set(ctl.failed_partitions) == set(got.failed_partitions)
            rec["control"] = {
                "verdict": ctl.verdict, "reuse_pct": ctl.reuse_pct,
                "prove_s": ctl.prove_s, "wall_s": round(time.time() - t2, 2),
                "failed": list(ctl.failed_partitions),
                "n_failed": len(ctl.failed_partitions),
                "agrees_verdict": same_verdict,
                "agrees_failed_partitions": same_failed,
            }
            if not (same_verdict and same_failed):
                rec["status"] = "CACHE-DISAGREES-WITH-UNCACHED"
                print(f"      !! cached={got.verdict} uncached={ctl.verdict} "
                      f"failed_match={same_failed}", flush=True)
                return rec
            print(f"      ctrl  {ctl.verdict:9s} uncached prove="
                  f"{ctl.prove_s:7.2f}s  AGREES", flush=True)

        # A `proven` verdict on a real merged bug fix always needs explaining.
        # Two very different causes, and they must not be conflated:
        #   (a) the checker missed the bug          , a soundness failure
        #   (b) the fix changes no OUTPUT behaviour , nothing to find
        #
        # ibex #1780 is (b): it fixes PMP handling, and `ibex_core` defaults to
        # `parameter bit PMPEnable = 1'b0`, so the added logic is dead and
        # cannot affect an output.
        #
        # NOTE: an earlier version tried to detect this by testing whether the
        # two elaborated netlists were byte-identical. They are NOT, #1780's
        # gold.il is 876,439 bytes against gate.il's 877,503: the fix really
        # does add cells, they are simply unobservable. Structural identity is
        # the wrong test for behavioural equivalence, so the netlist digests are
        # recorded as evidence and the instance is FLAGGED for a human rather
        # than auto-classified.
        gold_il, gate_il = sc / "wd" / "gold.il", sc / "wd" / "gate.il"
        if got.verdict == "proven" and gold_il.exists() and gate_il.exists():
            rec["elaborated_identical"] = (
                PartitionProofCache.key_for(gold_il)
                == PartitionProofCache.key_for(gate_il))
            rec["needs_explanation"] = (
                "PROVEN on a real bug fix: either the fix is unobservable in "
                "this parameterisation, or the check missed it. Netlists "
                + ("are byte-identical." if rec["elaborated_identical"]
                   else "DIFFER, so the fix adds logic that changes no output."))
            print(f"      !! PROVEN on a real bug fix -- needs explanation "
                  f"(netlists {'identical' if rec['elaborated_identical'] else 'differ'})",
                  flush=True)

        if got.verdict == "error":
            # A check that errored produced no verdict, no partitions and no
            # reuse. Calling it "ok" inflates the measured count and drags a
            # 0.00% reuse into the distribution as though it were a result.
            rec["status"] = "measure-failed"
            rec["load_at_end"] = host_load().as_dict()
            return rec
        rec["status"] = "ok"
        rec["cache_entries"] = len(cache)
        rec["load_at_end"] = host_load().as_dict()
        return rec
    finally:
        sh(["git", "worktree", "remove", "--force", str(tree)],
           cwd=cfgr["checkout"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="lowRISC__ibex", choices=sorted(REPOS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", type=int, default=None, help="one PR number")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--workroot", default=str(ROOT / "var" / "hwebench"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="run a repo that has been excluded from the corpus")
    ap.add_argument("--ladder", default="smt-only", choices=sorted(LADDERS),
                    help="strategy ladder; see LADDERS. Running the corpus "
                         "under both is what decides whether the cheap "
                         "ordering is safe.")
    ap.add_argument("--keep-hierarchy", action="store_true",
                    help="elaborate with read_slang --keep-hierarchy, so module "
                         "boundaries survive. EXPERIMENTAL in slang, and "
                         "hierarchical equivalence is an assume-guarantee "
                         "argument: run it against the flat results and require "
                         "verdict agreement on every instance before trusting "
                         "it.")
    ap.add_argument("--verify-uncached", action="store_true",
                    help="re-run each measurement against a cache that has "
                         "never seen the design, and require the same verdict "
                         "and the same failing partitions")
    ap.add_argument("--summarise", action="store_true",
                    help="re-report an existing results file; runs nothing")
    a = ap.parse_args()

    yosys = str(ROOT / "tools" / "bin" / "yosys")
    why = REPOS[a.repo].get("excluded")
    if why and not a.force:
        print(f"{a.repo} is excluded from the corpus: {why}")
        print("See the comment on its REPOS entry for the measurements. "
              "Pass --force to run it anyway.")
        return 2
    insts = usable_instances(a.repo, a.limit)
    if a.only is not None:
        insts = [i for i in insts if i["number"] == a.only]
    if a.list:
        for i in insts:
            print(f"#{i['number']:<6} +{i['added']:<4}/-{i['removed']:<4} "
                  f"{i['file']:<46} {i['title']}")
        print(f"\n{len(insts)} single-RTL-file bug fixes in {a.repo}")
        return 0

    # ABSOLUTE, always. `git worktree add` runs with cwd=<checkout>, so a
    # RELATIVE workroot makes git create the worktrees INSIDE the checkout
    # while everything else resolves them against the process CWD. The result
    # is not an error: every instance reports `no-sources`, because the path
    # that gets globbed is an empty directory that was never populated. The
    # default workroot is absolute, which is why this only appeared the first
    # time --workroot was passed by hand.
    workroot = Path(a.workroot).resolve()
    out = Path(a.out or workroot / f"{a.repo}.json").resolve()
    if a.summarise:
        if not out.exists():
            print(f"no results at {out}")
            return 1
        data = json.loads(out.read_text())
        print(f"{data.get('repo', a.repo)}  jobs={data.get('jobs')}")
        for r in data["rows"]:
            if r.get("status") == "ok":
                m, w = r["measure"], r["warm"]
                print(f"  #{r['number']:<6} {Path(r['file']).name:<34} "
                      f"{m['verdict']:9s} reuse={m['reuse_pct']:6.2f}%  "
                      f"{w['prove_s']:7.1f}s -> {m['prove_s']:6.2f}s")
            else:
                print(f"  #{r['number']:<6} {Path(r.get('file','?')).name:<34} "
                      f"{r.get('status')}")
        summarise(data["rows"])
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    warn = load_warning()
    if warn:
        print(f"!! {warn}\n", flush=True)
    print(f"{a.repo}: {len(insts)} instances  jobs={a.jobs}  "
          f"ladder={a.ladder}"
          + ("  --keep-hierarchy" if a.keep_hierarchy else "") + "\n",
          flush=True)

    rows = []
    for n, inst in enumerate(insts, 1):
        print(f"[{n}/{len(insts)}] #{inst['number']} {inst['file']} "
              f"(+{inst['added']}/-{inst['removed']})", flush=True)
        try:
            rec = run_instance(a.repo, inst, workroot, a.jobs, yosys,
                               a.timeout, verify_uncached=a.verify_uncached,
                               ladder=a.ladder,
                               front_end_flags=(["--keep-hierarchy"]
                                                if a.keep_hierarchy else []))
        except Exception as e:                    # one bad PR must not end the run
            rec = {"repo": a.repo, "number": inst["number"],
                   "status": "exception", "detail": f"{type(e).__name__}: {e}"}
        rows.append(rec)
        if rec.get("status") == "ok":
            m = rec["measure"]
            print(f"      fix   {m['verdict']:9s} "
                  f"reuse={m['reuse_pct']:6.2f}% ({m['hits']}/{m['partitions']}) "
                  f"prove={m['prove_s']:7.2f}s  "
                  f"saved={rec.get('prove_time_saved_pct')}%", flush=True)
        else:
            print(f"      {rec['status']}: {rec.get('detail','')[:160]}",
                  flush=True)
        out.write_text(json.dumps({"repo": a.repo, "jobs": a.jobs,
                                   "ladder": a.ladder, "rows": rows}, indent=2))

    summarise(rows)
    print(f"wrote {out}")
    return 0


def summarise(rows: list[dict]) -> None:
    """Report the distribution, and name what was NOT measured.

    An average over only the instances that worked, with the failures left
    unmentioned, reads as coverage the run does not have.
    """
    from collections import Counter
    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"\n{len(ok)}/{len(rows)} instances measured")
    skipped = Counter(r.get("status") for r in rows if r.get("status") != "ok")
    for status, n in skipped.most_common():
        print(f"  not measured: {status:22s} {n}")
    # Load that CHANGED during an instance means something else started running
    # on the box mid-measurement, and that instance's timings are not comparable
    # with its neighbours. A single sample at start cannot see this: instance
    # #167 sampled 0.47/core at start and 29.22 total (1.22/core) at end, and
    # its fix phase came out 3x slower than every neighbour.
    #
    # Two tests, because the first one alone let a contaminated instance
    # through. A recursive grep over var/ during the hierarchical arm pushed
    # #167 from load 12.0 to 22.2, a 1.85x rise, UNDER the 2x within-instance
    # threshold, and its wall came out 45.0s against 21-22s for its
    # neighbours. Its own start/end ratio could not see that, because it was
    # already running slow when it started.
    #
    # Which timings the box disqualifies. The rule and the reasoning behind
    # it live in loadguard.py, because the front-end comparator divides the
    # same wall-clocks and must disqualify the same instances.
    from loadguard import (drifted as _drift, MIN_BASELINE,
                           baseline_is_suspect as _suspect)
    drift_map, base = _drift(rows)
    suspect = _suspect(rows)
    if suspect:
        print(f"  !! {suspect}")
    drifted = sorted(drift_map.items(), key=lambda kv: str(kv[0]))
    drift_nums = set(drift_map)
    if base is None:
        print(f"  (fewer than {MIN_BASELINE} load samples: too few to "
              f"establish a baseline, so NO timing is disqualified here. "
              f"Treat the timings as unvetted.)")
    if drifted:
        print(f"  !! load drift on {len(drifted)} instance(s). Their TIMINGS "
              f"are EXCLUDED from the medians below:")
        for n, why in drifted:
            print(f"       #{n}: {why}")
        print("       (verdicts, partition counts and reuse are unaffected "
              "and are still counted)")

    bad = [r for r in rows if r.get("status") == "CACHE-DISAGREES-WITH-UNCACHED"]
    if bad:
        print(f"  !! {len(bad)} instance(s) where the cached verdict did NOT "
              f"match the uncached one: "
              f"{[r.get('number') for r in bad]}")
    checked = [r for r in ok if "control" in r]
    if checked:
        print(f"  cache-vs-uncached control: {len(checked)}/{len(ok)} "
              f"instances re-verified, all agreeing")
    if not ok:
        return
    flagged = [r for r in ok if r.get("needs_explanation")]
    if flagged:
        print(f"  !! {len(flagged)} instance(s) returned PROVEN on a real bug "
              f"fix and need explanation: {[r['number'] for r in flagged]}")
        for r in flagged:
            print(f"       #{r['number']}: {r['needs_explanation']}")

    verdicts = Counter(r["measure"]["verdict"] for r in ok)
    for v, n in verdicts.most_common():
        print(f"  verdict {v:11s} {n}")

    # Group by SCOPE. A whole-design result and a module-scope result are
    # different claims, the first says the core differs, the second only
    # that one module does, and pooling them into one median would let the
    # weaker claim inherit the stronger one's headline.
    def mid(xs):
        """True median; the upper-middle value would bias every figure up."""
        n = len(xs)
        return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0
    scopes: dict[str, list[dict]] = {}
    for r in ok:
        scopes.setdefault(r.get("scope", "whole-design"), []).append(r)
    for scope, rows in sorted(scopes.items()):
        # Reuse is decided by which obligations hash the same, so a busy box
        # does not change it: every instance counts. Every ratio of two
        # wall-clocks does change, so drifted instances are dropped from
        # those and the drop is stated rather than left to the warning above.
        rs = sorted(x["measure"]["reuse_pct"] for x in rows)
        clean = [x for x in rows if x.get("number") not in drift_nums]
        dropped = len(rows) - len(clean)
        sp = sorted(x["warm"]["prove_s"] / x["measure"]["prove_s"]
                    for x in clean if x["measure"]["prove_s"] > 0)
        e2e = sorted(x["warm"]["wall_s"] / x["measure"]["wall_s"]
                     for x in clean if x["measure"]["wall_s"] > 0)
        label = f"  [{scope}, n={len(rows)}]"
        print(label)
        print(f"    reuse        median {mid(rs):6.2f}%  min {rs[0]:6.2f}%  "
              f"max {rs[-1]:6.2f}%   (all {len(rows)} instances)")
        def _n(k, xs):
            # Each line states its OWN n. They differ: an instance at 100%
            # reuse has prove_s == 0 and no finite prove speedup, so it counts
            # end-to-end and not in the prove column.
            note = f"{len(xs)} of {len(rows)}"
            if dropped:
                note += f", {dropped} dropped for load drift"
            return f"   ({note})"
        if e2e:
            print(f"    end-to-end   median {mid(e2e):6.2f}x  "
                  f"min {e2e[0]:6.2f}x  max {e2e[-1]:6.2f}x{_n('e2e', e2e)}")
        if sp:
            print(f"    prove speedup median {mid(sp):6.1f}x  "
                  f"min {sp[0]:6.1f}x  max {sp[-1]:6.1f}x{_n('sp', sp)}")
        elif dropped:
            print(f"    no timing-clean instance in this scope "
                  f"({dropped} dropped for load drift) -- no speedup reported")
        if scope == "module-of-fix":
            print("    NOTE: module scope. These say the buggy and fixed "
                  "versions of that MODULE differ, not that the core does.")
    if len(scopes) > 1:
        print("  (scopes reported separately and deliberately not pooled)")


if __name__ == "__main__":
    raise SystemExit(main())
