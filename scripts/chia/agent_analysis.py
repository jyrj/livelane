"""What a real agent proposes when it optimises a whole ibex_core.

rewrites.py measures a GENERATED distribution of edits. This reads the loop's
own records, a production model optimising the core's critical path, gated on
every iteration, and asks the same three questions of the agent's proposals:

  * how many did the gate refuse?
  * of those, how many did synthesis score FASTER than the design they were
    proposed against, i.e. how many would a QoR-only loop have accepted?
  * are the refusals real? Each refused edit is re-checked by module-level
    bounded model checking against its PARENT (the file as it stood when the
    edit was proposed, after any earlier accepted edits), not against the
    checkout's original.

Writes a results file that scripts/chia/resettle.py re-decides row by row.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _abs(path: str) -> Path:
    """The loop records paths relative to where it was launched. sby runs each
    job in its own directory, so a relative path there names nothing, every
    confirmation errored with 'No such file' until this was made absolute."""
    q = Path(path)
    return q if q.is_absolute() else (ROOT / q).resolve()


def parent_of(run_dir: Path, rows: list[dict], i: int, fname: str,
              checkout: Path) -> Path:
    """The file an iteration-i edit was applied to."""
    prior = [r for r in rows if r.get("iteration", -1) < i
             and r.get("outcome") == "accepted" and r.get("file") == fname]
    if prior:
        last = max(prior, key=lambda r: r["iteration"])
        return _abs(last["cand"])
    return checkout / "rtl" / fname


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+",
                    default=sorted(str(p) for p in (ROOT / "var/agent").glob("*-s*")
                                   if (p / "loop-summary.json").exists()))
    ap.add_argument("--out", default=str(ROOT / "var/final/agent.json"))
    a = ap.parse_args()

    checkout = ROOT / "thirdparty" / "ibex"
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout,
                         capture_output=True, text=True).stdout.strip()
    allrows, runs, k = [], [], 0
    for rd in map(Path, a.runs):
        s = json.loads((rd / "loop-summary.json").read_text())
        runs.append({"run": rd.name, "model": s["model"], "seed": s.get("seed"),
                     "iterations": s["iterations"], "accepted": s["accepted"],
                     "seed_ns": s["seed_ns"], "best_ns": s["best_ns"],
                     "improvement_pct": s["improvement_pct"],
                     "cost_usd": s.get("cost_usd"),
                     "ladder": s.get("ladder"),
                     "second_stage": bool(s.get("second_stage"))})
        rows = s["rows"]
        for r in rows:
            if "verdict" not in r:          # never reached the gate
                continue
            row = {"i": k, "run": rd.name, "model": s["model"],
                   "iteration": r["iteration"], "verdict": r["verdict"],
                   "outcome": r["outcome"], "faster": r["faster"],
                   "parent_ns": r["parent_ns"], "delay_ns": r["delay_ns"],
                   "area_um2": r.get("area_um2"), "file": f"rtl/{r['file']}",
                   "gate_s": r.get("prove_s"), "iter_s": r.get("iter_s"),
                   # treatment arm: the partitioned gate's own verdict and the
                   # second stage's, recorded separately so neither is lost
                   "verdict_eqy": r.get("verdict_eqy", r["verdict"]),
                   "second_stage": r.get("second_stage"),
                   "second_stage_s": r.get("second_stage_s"),
                   "module": Path(r["file"]).stem, "note": r["note"],
                   "cand_file": str(_abs(r["cand"])),
                   "gold_file": str(parent_of(rd, rows, r["iteration"],
                                              r["file"], checkout)),
                   "operator": "agent", "family": "simplifying"}
            allrows.append(row)
            k += 1

    out = {"sha": sha, "base": {}, "runs": runs, "rows": allrows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    summarise(out)
    print(f"\nwrote {a.out}")
    return 0


def summarise(out: dict) -> None:
    rows = out["rows"]
    print(f"{len(out['runs'])} runs:")
    for r in out["runs"]:
        print(f"  {r['run']:14s} {r['model']:24s} iters={r['iterations']:>3} "
              f"accepted={r['accepted']:>2} {r['seed_ns']:.2f}->{r['best_ns']:.2f} ns "
              f"({r['improvement_pct']}%)  ${r['cost_usd']}")
    gated = [r for r in rows if r["verdict"] in ("proven", "refuted",
                                                   "proven-seq")]
    ref = [r for r in gated if r["verdict"] == "refuted"]
    prv = [r for r in gated if r["verdict"] in ("proven", "proven-seq")]
    rescued = [r for r in gated if r["verdict"] == "proven-seq"]
    if any(r.get("second_stage") for r in gated):
        eqy_ref = [r for r in gated if r["verdict_eqy"] == "refuted"]
        from collections import Counter
        c = Counter(r["second_stage"] for r in eqy_ref)
        print(f"  second stage on {len(eqy_ref)} partitioned refusals: "
              + ", ".join(f"{k}={v}" for k, v in c.items()))
        print(f"  rescued (proven by the second stage): {len(rescued)}; "
              f"accepted among them: "
              f"{sum(r['outcome'] == 'accepted' for r in rescued)}")
    ref_fast = [r for r in ref if r["faster"]]
    prv_fast = [r for r in prv if r["faster"]]
    print(f"\nproposals that reached the gate: {len(rows)} "
          f"({len(gated)} settled)")
    print(f"  proven equivalent : {len(prv):3d}   of which faster: {len(prv_fast)}"
          f"  (these are what the loop may accept)")
    print(f"  REFUSED           : {len(ref):3d}   of which FASTER than their "
          f"parent: {len(ref_fast)}")
    fast = [r for r in gated if r["faster"]]
    if fast:
        print(f"\n  of the {len(fast)} proposals synthesis scored faster, "
              f"{len(ref_fast)} ({100 * len(ref_fast) / len(fast):.0f}%) were "
              f"refused -- a QoR-only loop would have accepted every one")
    conf = [r for r in ref if r.get("bmc") == "CONFIRMED"]
    if any("bmc" in r for r in ref):
        print(f"  refusals confirmed by module-level counterexample: "
              f"{len(conf)}/{sum('bmc' in r for r in ref)}")
        cf = [r for r in ref_fast if r.get("bmc") == "CONFIRMED"]
        print(f"  faster AND confirmed broken: {len(cf)}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--summarise":
        summarise(json.loads(Path(sys.argv[2]).read_text()))
        raise SystemExit(0)
    raise SystemExit(main())
