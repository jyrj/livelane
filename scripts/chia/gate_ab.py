"""Standard gate vs gate + second stage, same model, whole ibex_core.

Arms are identical except for ``--second-stage``: same model, same 15
iterations, same core, same prompt. In the control arm the partitioned gate's
refusal is final and the agent is told its edit was refuted. In the treatment
arm a refusal is settled by an unbounded proof of the edited instance, and an
edit it proves is treated as proven.

For a like-for-like count, EVERY refusal in EVERY arm is decided offline by
one procedure (resettle.py): PDR at instance scope, both copies powered up to
the same arbitrary state with reset asserted in the first cycle, and again
without reset. The treatment arm's live verdicts record only what the agent
was TOLD; classification never uses them.

Reports, per arm: runs, iterations, improvement, accepted edits, how many
iterations went to edits the partitioned gate refused but were correct, and
(treatment) how many of those the agent was told were correct.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ARMS = {
    "pro / standard gate": ["var/agent/pro-s*", "var/agent3/ctl-pro-s*"],
    "pro / + second stage": ["var/agent2/pro-s*", "var/agent3/s2-pro-s*"],
    # the released second stage: from reset, package edits scoped (--init reset)
    "pro / + scoped second stage": ["var/agent4/s3-pro-s*"],
    "2.5-flash / standard gate": ["var/agent/flash-s*"],
    "2.5-flash / + second stage": ["var/agent2/flash-s*"],
    "2.5-pro / standard gate": ["var/agent3/ctl-g25pro-s*"],
    "3.8-flash / standard gate": ["var/agent3/ctl-g38flash-s*"],
    "3.8-flash / + scoped second stage": ["var/agent5/s3-g38flash-s*"],
    "2.5-pro / + scoped second stage": ["var/agent5/s3-g25pro-s*"],
}


def runs(patterns):
    out = []
    for pat in patterns:
        for d in sorted(glob.glob(str(ROOT / pat))):
            if d.endswith("-cache"):
                continue
            if (Path(d) / "loop-summary.json").exists():
                out.append(d)
    return out


def ship(d: dict) -> None:
    """Copy every candidate and parent file a row names into the artifact.

    Run directories (var/) are not shipped; the files a verdict was decided on
    are, under measurements/data/<batch>/<run>/, and rows name them repo-relative
    so any checkout can re-decide them.
    """
    dest_root = ROOT / "measurements/data/agent"
    for r in d["rows"]:
        for key in ("cand_file", "gold_file"):
            f = r.get(key)
            if not f:
                continue
            src = Path(f) if Path(f).is_absolute() else ROOT / f
            try:
                rel = src.resolve().relative_to((ROOT / "var").resolve())
            except ValueError:
                # a pinned source (thirdparty/, recreated by the setup script)
                # or an already-shipped file: name it repo-relative
                try:
                    r[key] = str(src.resolve().relative_to(ROOT.resolve()))
                except ValueError:
                    pass
                continue
            # var/<batch>/<run>/<file> -> measurements/data/<batch>/<run>/<file>
            # (batch-1 runs already live under measurements/data/agent/)
            dest = dest_root.parent.joinpath(*rel.parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src.exists() and not dest.exists():
                shutil.copy2(src, dest)
            r[key] = str(dest.relative_to(ROOT))


def mid(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirm", action="store_true",
                    help="decide every refusal offline (resettle.py)")
    ap.add_argument("--out", default=str(ROOT / "measurements/data/gate_ab.json"))
    a = ap.parse_args()
    py = str(ROOT / ".venv/bin/python")
    summary = {}
    for arm, pats in ARMS.items():
        rs = runs(pats)
        if not rs:
            continue
        tag = arm.replace(" / ", "_").replace(" ", "").replace(".", "")
        j = ROOT / f"measurements/data/arm_{tag}.json"
        subprocess.run([py, str(ROOT / "scripts/chia/agent_analysis.py"),
                        "--runs", *rs, "--out", str(j)],
                       capture_output=True, text=True)
        d = json.loads(j.read_text())
        ship(d)
        j.write_text(json.dumps(d, indent=1))
        refused = [r for r in d["rows"] if r.get("verdict_eqy", r["verdict"]) == "refuted"]
        if a.confirm and refused:
            subprocess.run([py, str(ROOT / "scripts/chia/resettle.py"),
                            "--results", str(j), "--parallel", "6",
                            "--workroot", str(ROOT / "var/resettle-ab")],
                           capture_output=True, text=True)
            d = json.loads(j.read_text())
            refused = [r for r in d["rows"]
                       if r.get("verdict_eqy", r["verdict"]) == "refuted"]
        if a.confirm:
            subprocess.run([py, str(ROOT / "scripts/chia/resettle.py"),
                            "--results", str(j), "--rows", "accepted",
                            "--parallel", "6",
                            "--workroot", str(ROOT / "var/resettle-ab")],
                           capture_output=True, text=True)
            d = json.loads(j.read_text())
            refused = [r for r in d["rows"]
                       if r.get("verdict_eqy", r["verdict"]) == "refuted"]
        verdict = lambda r: (r.get("explicit") or {}).get("from_reset")
        acc_rows = [r for r in d["rows"] if r.get("outcome") == "accepted"]
        false_ref = [r for r in refused if verdict(r) == "PROVEN-SEQ"]
        real = [r for r in refused if verdict(r) == "CONFIRMED"]
        iters = sum(x["iterations"] for x in d["runs"])
        imp = [x["improvement_pct"] for x in d["runs"]]
        summary[arm] = {
            "runs": len(d["runs"]), "iterations": iters,
            "improvement_median": mid(imp), "improvement_mean":
                statistics.mean(imp) if imp else None,
            "improvement_all": imp,
            "accepted": sum(x["accepted"] for x in d["runs"]),
            "refused_by_partitioned_gate": len(refused),
            "false_refusals": len(false_ref), "real_bugs": len(real),
            "false_refusal_iteration_share": len(false_ref) / iters if iters else None,
            "told_correct": sum(1 for r in d["rows"] if r["verdict"] == "proven-seq"),
            "false_any_state": sum(1 for r in false_ref
                                   if r["explicit"].get("any_state") == "ANY-STATE"),
            "undecided": len(refused) - len(false_ref) - len(real),
            # every edit the live second stage let through, re-proven
            "told_correct_reproven": sum(
                1 for r in d["rows"] if r["verdict"] == "proven-seq"
                and verdict(r) == "PROVEN-SEQ"),
            # every accepted edit, re-proven from reset (shared power-up)
            "accepted_reproven": sum(verdict(r) == "PROVEN-SEQ" for r in acc_rows),
            "accepted_any_state": sum((r.get("explicit") or {}).get("any_state")
                                      == "ANY-STATE" for r in acc_rows),
            "cost_usd": round(sum(x["cost_usd"] or 0 for x in d["runs"]), 2),
        }
    Path(a.out).write_text(json.dumps(summary, indent=1))
    print(f"{'arm':30s} {'runs':>4} {'iters':>5} {'imp med':>8} {'imp mean':>8} "
          f"{'acc':>4} {'refused':>7} {'false':>5} {'real':>4} {'told ok':>7} {'$':>7}")
    for arm, s in summary.items():
        print(f"{arm:30s} {s['runs']:>4} {s['iterations']:>5} "
              f"{s['improvement_median'] or 0:>7.2f}% {s['improvement_mean'] or 0:>7.2f}% "
              f"{s['accepted']:>4} {s['refused_by_partitioned_gate']:>7} "
              f"{s['false_refusals']:>5} {s['real_bugs']:>4} {s['told_correct']:>7} "
              f"{s['cost_usd']:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
