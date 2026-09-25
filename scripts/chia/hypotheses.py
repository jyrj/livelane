"""The latency study's five questions, H1-H5, computed from the run databases.

This reads every sweep database on disk and reports all five in one place:

  H1  lower evaluator latency raises verified improvement (binding budget)
  H2  at EQUAL iteration count, latency does not change quality
  H3  latency interacts with model speed: a fast seat is hurt differently
  H4  the lane-L (LiveHD) proxy ranks candidates like the neutral judge
      (Spearman rho >= 0.8 over >= 30 candidates)
  H5  faster arms are not sloppier: equal gate rejection rate across arms

Metric definitions follow the H1 measurement exactly: a cell's improvement is
(seed delay - best accepted delay) / seed delay, where only variants the loop
ACCEPTED (proven equivalent and better) count, and the seed itself is the floor,
so a cell that found nothing scores 0 rather than a negative number.
"""
from __future__ import annotations

import argparse
import glob
import json
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from livelane.analysis.metrics import bootstrap, spearman  # noqa: E402


def _cells(db: str) -> list[dict]:
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    out = []
    for r in c.execute("SELECT * FROM runs"):
        vs = c.execute("SELECT * FROM variants WHERE run_id=? ORDER BY iteration_index",
                       (r["run_id"],)).fetchall()
        seed = next((v["qor_max_delay_ns"] for v in vs if v["iteration_index"] == 0), None)
        acc = [v["qor_max_delay_ns"] for v in vs
               if v["accepted"] and v["qor_max_delay_ns"] is not None]
        best = min(acc) if acc else seed
        # the seed evaluation is iteration 0, not an agent iteration
        its = c.execute("SELECT COUNT(*) FROM iterations WHERE run_id=? "
                        "AND iteration_index > 0", (r["run_id"],)).fetchone()[0]
        decided = [v for v in vs if v["iteration_index"] > 0
                   and v["lec_verdict"] in ("proven", "refuted")]
        out.append({
            "db": db, "run": r["run_id"], "design": r["design"], "lane": r["lane"],
            "arm": r["arm"], "delay": r["arm_delay_s"], "model": r["model"],
            "seed": r["seed"], "status": r["status"], "budget_iters": r["budget_iters"],
            "seed_ns": seed, "best_ns": best,
            "improvement": (seed - best) / seed if seed and best is not None else None,
            "iterations": its,
            "decided": len(decided),
            "refused": sum(1 for v in decided if v["lec_verdict"] == "refuted"),
            "candidates": [(v["qor_max_delay_ns"], v["judge_max_delay_ns"])
                           for v in vs if v["iteration_index"] > 0],
        })
    return out


def _mean_ci(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    if len(xs) == 1:
        return {"mean": xs[0], "lo": None, "hi": None, "n": 1}
    b = bootstrap(xs)
    lo, hi = getattr(b, "lo", None), getattr(b, "hi", None)
    return {"mean": statistics.mean(xs), "lo": lo, "hi": hi, "n": len(xs)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "measurements/data/hypotheses.json"))
    a = ap.parse_args()

    h1 = _cells(str(ROOT / "var/livelane-h1.db"))
    h3 = [c for db in sorted(glob.glob(str(ROOT / "var/h3/*.db"))) for c in _cells(db)]
    h2 = ([c for c in _cells(str(ROOT / "var/livelane-h2-iso-iteration.db"))]
          + [c for db in sorted(glob.glob(str(ROOT / "var/h2/*.db"))) for c in _cells(db)])
    h4 = [c for db in sorted(glob.glob(str(ROOT / "var/h4/laneL-*.db"))) for c in _cells(db)
          if c["lane"] == "L"]
    done = lambda cs: [c for c in cs if c["status"] == "done"]
    arms = ["I(0)", "I(30)", "I(120)", "I(600)"]
    res: dict = {}

    # ---- H1 + H3: binding 1800 s budget, two models ---------------------
    res["latency"] = {}
    print("=== H1 / H3: improvement under a fixed 1800 s budget ===")
    print(f"{'model':26s} " + " ".join(f"{a_:>16s}" for a_ in arms))
    for model, cells in (("gemini-3.1-pro-preview", done(h1)),
                         ("gemini-2.5-flash", done(h3))):
        row = {}
        for arm in arms:
            cs = [c for c in cells if c["arm"] == arm and c["model"] == model]
            row[arm] = {"improvement": _mean_ci([c["improvement"] for c in cs]),
                        "iterations": _mean_ci([c["iterations"] for c in cs]),
                        "n": len(cs)}
        res["latency"][model] = row
        print(f"{model:26s} " + " ".join(
            f"{100 * row[a_]['improvement']['mean']:5.1f}% it{row[a_]['iterations']['mean']:4.1f} n{row[a_]['n']}"
            if row[a_]["improvement"] else f"{'--':>16s}" for a_ in arms))

    # ---- the mechanism: each model's own turn time at I(0) ----------------
    def turn(dbs, model):
        xs = []
        for db in dbs:
            c = sqlite3.connect(db)
            xs += [r[0] / 1000 for r in c.execute(
                "SELECT i.t_llm_ms FROM iterations i JOIN runs r ON r.run_id=i.run_id "
                "WHERE r.arm='I(0)' AND r.model=? AND i.iteration_index>0 "
                "AND i.t_llm_ms>0", (model,))]
        return {"mean": statistics.mean(xs), "sd": statistics.pstdev(xs),
                "n": len(xs)} if xs else None
    res["turn"] = {
        "gemini-3.1-pro-preview": turn([str(ROOT / "var/livelane-h1.db")],
                                       "gemini-3.1-pro-preview"),
        "gemini-2.5-flash": turn(sorted(glob.glob(str(ROOT / "var/h3/*.db"))),
                                 "gemini-2.5-flash")}
    print("\nmodel turn at I(0): " + ", ".join(
        f"{k} {v['mean']:.0f}s (sd {v['sd']:.0f})" for k, v in res["turn"].items() if v))

    # ---- H2: iso-iteration ---------------------------------------------
    res["iso"] = {}
    print("\n=== H2: quality at equal iteration count (8) ===")
    for arm in arms:
        cs = [c for c in done(h2) if c["arm"] == arm]
        res["iso"][arm] = {"improvement": _mean_ci([c["improvement"] for c in cs]),
                           "n": len(cs)}
        res["iso"][arm]["values"] = [c["improvement"] for c in cs]
        m = res["iso"][arm]["improvement"]
        print(f"  {arm:7s} n={len(cs)}  mean improvement "
              + (f"{100 * m['mean']:.1f}%" if m else "--"))
    # the test: is QoR at equal iterations distinguishable
    # across delay arms? Kruskal-Wallis, no normality assumed.
    groups = [res["iso"][a_]["values"] for a_ in arms if res["iso"][a_]["values"]]
    try:
        from scipy.stats import kruskal
        stat, pval = kruskal(*groups) if len(groups) >= 2 else (None, None)
    except Exception:
        stat, pval = None, None
    res["iso_test"] = {"test": "kruskal-wallis", "H": stat, "p": pval,
                       "arms": len(groups),
                       "verdict": None if pval is None else
                       ("holds" if pval >= 0.05 else "refuted")}
    if pval is not None:
        print(f"  Kruskal-Wallis across {len(groups)} arms: H = {stat:.2f}, "
              f"p = {pval:.3f} -> H2 {res['iso_test']['verdict']}")

    # ---- H4: lane-L proxy vs neutral judge -------------------------------
    pairs = [(p, j) for c in h4 for p, j in c["candidates"]
             if p is not None and j is not None]
    rho = spearman([p for p, _ in pairs], [j for _, j in pairs]) if len(pairs) >= 3 else None
    per_design = {}
    for d in sorted({c["design"] for c in h4}):
        pp = [(p, j) for c in h4 if c["design"] == d for p, j in c["candidates"]
              if p is not None and j is not None]
        per_design[d] = {"n": len(pp),
                         "rho": spearman([p for p, _ in pp], [j for _, j in pp])
                         if len(pp) >= 3 else None}
    res["h4"] = {"n": len(pairs), "rho": rho, "per_design": per_design,
                 "threshold": 0.8, "min_candidates": 30}
    print(f"\n=== H4: lane-L proxy vs judge (critical-path delay) ===\n"
          f"  {len(pairs)} candidates with both scores; Spearman rho = {rho}")
    for d, v in per_design.items():
        print(f"    {d:12s} n={v['n']:3d} rho={v['rho']}")

    # ---- H5: rejection rate per arm --------------------------------------
    res["h5"] = {}
    print("\n=== H5: gate rejection rate per arm (both models) ===")
    for arm in arms:
        cs = [c for c in h1 + h3 if c["arm"] == arm]
        dec = sum(c["decided"] for c in cs)
        ref = sum(c["refused"] for c in cs)
        res["h5"][arm] = {"decided": dec, "refused": ref,
                          "rate": ref / dec if dec else None}
        print(f"  {arm:7s} refused {ref:3d} / {dec:3d} decided"
              + (f"  ({100 * ref / dec:.1f}%)" if dec else ""))

    res["cells"] = {"h1": len(done(h1)), "h3": len(done(h3)), "h2": len(done(h2)),
                    "h4": len(h4)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
