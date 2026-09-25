"""Re-decide refusals with the initial state stated explicitly.

Every refusal (and gate error) in a results file is proven again, at
instance scope, twice
(``second_stage.settle_explicit``):

  from_reset  both copies power up to the same arbitrary state (registers
              matched by name and width), reset is asserted in the first
              cycle, and PDR proves they never diverge, or finds a
              counterexample reachable from reset;
  any_state   the same, without reset: they never diverge from ANY common
              state.

The earlier checks started both copies from all-zero. That is not the reset
state of an edit that changes a reset value (a one-hot state machine resets to
'h001, and all-zero is illegal), and it hides un-reset flops powering up
nonzero. These verdicts replace both.

An edit to a package has no instance. Its scope is the modules that use the
types whose definitions changed: one such module is proven at its instances;
types that cross module boundaries are proven at the top.

Rows are updated in place under ``explicit``; nothing else changes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--rows", choices=["refused", "controls", "accepted"],
                    default="refused",
                    help="refused: every refusal and gate error; controls: the "
                         "equivalence-preserving rewrites; accepted: every edit "
                         "the loop accepted (the partitioned gate proves "
                         "sequential partitions from all-zero, so re-prove "
                         "them from reset)")
    ap.add_argument("--rewrites-dir", default=str(ROOT / "var/rewrites"))
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--pdr-timeout", type=float, default=240)
    ap.add_argument("--top-timeout", type=float, default=1800,
                    help="PDR budget when a package edit is proven at the top")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--workroot", default=str(ROOT / "var/resettle"))
    a = ap.parse_args()

    from chia_livelane.formal.second_stage import edit_scope, settle_explicit
    from chia_livelane.vlsi.elaborate import resolve, seed_sources
    res_path = Path(a.results)
    res = json.loads(res_path.read_text())
    wr = Path(a.workroot).resolve() / f"{res_path.stem}-{a.rows}"
    wr.mkdir(parents=True, exist_ok=True)
    checkout = ROOT / "thirdparty/ibex"
    sha = res.get("sha")
    if sha:
        tree = wr / "src"
        if tree.exists():
            subprocess.run(["git", "worktree", "remove", "--force", str(tree)],
                           cwd=checkout, capture_output=True)
            shutil.rmtree(tree, ignore_errors=True)
        r = subprocess.run(["git", "worktree", "add", "--detach", "--force",
                            str(tree), sha], cwd=checkout, capture_output=True,
                           text=True)
        if r.returncode:
            print(r.stderr)
            return 1
    else:
        tree = checkout
    e = resolve(tree, seed_sources(tree, "rtl/ibex_core.f", "rtl/*.sv"),
                "ibex_core", os.environ["LIVELANE_TOOLS"] + "/bin/yosys",
                stash=wr / "stubs")
    srcs = [Path(s) for s in e.sources]
    incs = [str(i) for i in e.includes] + [str(tree / "rtl")]
    defs = list(e.defines)

    def refused(r):
        # a gate error is not an acceptance either: settle it the same way
        return r.get("verdict_eqy", r["verdict"]) in ("refuted", "error")

    if a.rows == "refused":
        todo = [r for r in res["rows"] if refused(r)]
    elif a.rows == "accepted":
        todo = [r for r in res["rows"] if r.get("outcome") == "accepted"]
    else:
        todo = [r for r in res["rows"] if r.get("family") == "equivalent"]
    if not a.redo:
        todo = [r for r in todo if not r.get("explicit")]
    print(f"{len(todo)} row(s) to decide from reset and from any state",
          flush=True)

    def ab(p):
        q = Path(p)
        return q if q.is_absolute() else (ROOT / q).resolve()

    def one(r):
        fname = Path(r["file"]).name
        cand = ab(r["cand_file"]) if r.get("cand_file") else \
            Path(a.rewrites_dir) / f"m{r['i']:03d}" / fname
        gold_file = ab(r["gold_file"]) if r.get("gold_file") else None
        gold = [str(gold_file) if (gold_file and s.name == fname) else str(s)
                for s in srcs]
        gate = [str(cand) if s.name == fname else str(s) for s in srcs]
        module, changed = r["module"], None
        timeout = a.pdr_timeout
        text = cand.read_text(errors="ignore")
        if re.search(r"^\s*package\s+\w+", text, re.M) and \
                not re.search(r"^\s*module\s+", text, re.M):
            base = gold_file or next(s for s in srcs if s.name == fname)
            module, changed = edit_scope(Path(base), cand, srcs, "ibex_core")
            if module == "ibex_core":
                timeout = a.top_timeout
        out = settle_explicit(gold, gate, "ibex_core", module, incs, defs,
                              str(wr / f"{r['i']:03d}"), pdr_timeout_s=timeout,
                              bmc_timeout_s=timeout)
        out["scope"] = module
        if changed is not None:
            out["package_changed"] = changed
        return r["i"], out

    with ThreadPoolExecutor(a.parallel) as ex:
        # save each row as it finishes: one slow proof must not hold the rest
        for fut in as_completed([ex.submit(one, r) for r in todo]):
            i, out = fut.result()
            row = next(x for x in res["rows"] if x["i"] == i)
            row["explicit"] = {k: out[k] for k in
                               ("from_reset", "any_state", "scope", "wall_s")}
            if "package_changed" in out:
                row["explicit"]["package_changed"] = out["package_changed"]
            row["explicit"]["instances"] = out["instances"]
            old = row.get("second_stage") or row.get("bmc")
            print(f"  #{i:<4} {out['scope']:24s} reset={out['from_reset']:<10} "
                  f"any={out['any_state']:<14} was={old}  {out['wall_s']}s",
                  flush=True)
            res_path.write_text(json.dumps(res, indent=1))
    res_path.write_text(json.dumps(res, indent=1))
    if sha:
        subprocess.run(["git", "worktree", "remove", "--force", str(tree)],
                       cwd=checkout, capture_output=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
