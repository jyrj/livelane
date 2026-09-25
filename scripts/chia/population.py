"""Parallel population search as a CHIA graph, the throughput lane.

Why this exists
---------------
A serial agentic loop's iteration rate is fixed by arithmetic:

    iterations = budget / (T_model + T_evaluator + T_gate)

and on picorv32 that is about 114 + 2.6 + 15 = 131 s, of which the MODEL is 87%.
No evaluator speedup can move it: the measured Amdahl ceiling on making the
evaluator infinitely fast is about 1.02x. Latency is simply the wrong lever.

Throughput is the right one. k agents proposing concurrently give k times the
candidates per hour at the same wall-clock, because each one's 114 s of
deliberation overlaps every other's. The LLM call is network-bound, so k is
limited by rate limits and budget rather than by the 24 cores on the host; the
evaluator and the gate, which ARE cpu-bound, are the 13% and are dispatched
against their own resource tokens.

This is the "best variants breed" stage: propose k, evaluate k, keep the best
proven improvement, repeat.

What it measures
----------------
Candidates per hour, dollars per accepted improvement, and best achieved
critical path, the same three axes at k=1 and k>1, so the comparison is the
loop's own serial baseline rather than a remembered number.

Run the control and the treatment back to back::

    python scripts/chia/population.py --workers 1 --generations 4
    python scripts/chia/population.py --workers 8 --generations 4

Pricing note: pass a model that the price table actually knows
(``gemini-2.5-flash``), or ``cost_usd`` comes back None and the dollars axis is
empty. The table refuses to price an unlisted or preview SKU rather than fuzzy-
matching it onto a GA rate, which is why every recorded cost in the earlier
sweeps is NULL.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DESIGNS = {
    "picorv32": ("designs/picorv32/picorv32.v", "picorv32", "clk", 10.0,
                 "read_verilog -sv"),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _make_propose():
    """Build the proposal node. Imported lazily so --help needs no CHIA."""
    from livelane.agent.base import AgentContext
    from livelane.agent.llm import LLMAgent
    from livelane.agent.vertex import VertexSeat, estimate_cost_usd

    try:
        from chia.base.ChiaFunction import ChiaFunction
    except Exception:  # pragma: no cover
        def ChiaFunction(**_kw):
            def deco(fn):
                return fn
            return deco

    # A FRACTIONAL token, which is CHIA's own convention for LLM nodes
    # (chia/models/*.py use {"<provider>_creds": 0.01}). A whole token would
    # serialise the population onto one worker and defeat the entire point.
    @ChiaFunction(resources={"vertex_creds": 0.01})
    def propose_edit(rtl_text: str, path: str, top: str, design: str,
                     iteration: int, qor: dict, timing: dict, history: list,
                     model: str, thinking_budget: int | None,
                     timeout_s: float = 300.0) -> dict:
        """One independent edit proposal. Returns the edit plus its usage."""
        t0 = time.monotonic()
        seat = VertexSeat(model=model, thinking_budget=thinking_budget)
        agent = LLMAgent(seat=seat)
        ctx = AgentContext(design=design, top=top, iteration=iteration,
                           files=[path], qor=qor, timing=timing,
                           history=history, read_file=lambda _p: rtl_text,
                           primary_file=path)
        try:
            p = agent.propose(ctx)
        except Exception as e:  # a dead proposal must not kill the generation
            return {"ok": False, "error": f"{type(e).__name__}: {e}",
                    "wall_s": time.monotonic() - t0}
        # LLMAgent records every turn in `usages`; there is no `last_usage`.
        # Read the LAST one, and read it through billed_input/billed_output:
        # `thoughts` are billed as output by Google and DOMINATE it on
        # gemini-3.x, so tokens_in+tokens_out alone undercounts billed output by
        # roughly 45x and would corrupt both the cost table and any token cap.
        u = agent.usages[-1] if agent.usages else None
        bi = u.billed_input if u else 0
        bo = u.billed_output if u else 0
        # The seat prices its own turn; fall back to the list-price table only
        # if it could not, and carry the source either way so an unpriced SKU is
        # visible as "unknown" rather than as a confident zero.
        cost = u.cost_usd if u else None
        src = u.cost_source if u else "unknown"
        if cost is None and u is not None:
            cost, src = estimate_cost_usd(model, u.tokens_in, u.tokens_out,
                                          cache_read=u.cache_read,
                                          thoughts=u.thoughts)
        common = {"billed_in": bi, "billed_out": bo, "thoughts": (u.thoughts if u else 0),
                  "cost_usd": cost, "cost_source": src,
                  "wall_s": time.monotonic() - t0}
        if p is None or p.is_noop():
            return {"ok": False, "error": "no proposal", **common}
        return {"ok": True, "path": p.path, "old": p.old, "new": p.new,
                "note": p.note, **common}

    return propose_edit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", default="picorv32", choices=sorted(DESIGNS))
    ap.add_argument("--workers", type=int, default=4,
                    help="k: candidates proposed and evaluated per generation")
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--model", default="gemini-2.5-flash",
                    help="use a model the price table knows, or dollars stay None")
    ap.add_argument("--thinking-budget", type=int, default=-1,
                    help="-1 leaves the model default")
    ap.add_argument("--workdir", default=str(ROOT / "var" / "population"))
    ap.add_argument("--max-usd", type=float, default=10.0)
    a = ap.parse_args()
    if a.workers < 1:
        ap.error("--workers must be >= 1")

    rel, top, clk, period, read_cmd = DESIGNS[a.design]
    rtl = ROOT / rel
    liberty = os.environ.get("LIVELANE_LIBERTY")
    if not liberty or not Path(liberty).exists():
        log(f"FATAL: LIVELANE_LIBERTY unset or missing ({liberty!r}); source env.sh")
        return 2

    wd = Path(a.workdir) / f"k{a.workers}"
    shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True, exist_ok=True)

    from chia_livelane.formal.lec_gate import auto_jobs, lec_gate
    from chia_livelane.vlsi.yosys_sta import yosys_sta_qor
    propose_edit = _make_propose()

    import ray
    from chia.base.ChiaFunction import get
    # vertex_creds is fractional per call, so k concurrent proposals need only
    # k*0.01 of it; declare 1.0 and the population is limited by the API, not
    # by a token we invented.
    ray.init(resources={"vertex_creds": 1.0, "yosys_sta": 4.0, "eqy": 2.0},
             log_to_driver=False)
    log(f"ray up: {a.workers} workers, model={a.model}, "
        f"thinking={'default' if a.thinking_budget < 0 else a.thinking_budget}")

    parent = wd / "parent.v"
    parent.write_text(rtl.read_text())
    tb = None  # cheap stage is optional; the gate is what decides

    log("scoring the seed ...")
    seed_qor = get(yosys_sta_qor.chia_remote(
        [str(parent)], top, liberty, workdir=str(wd / "seed"),
        clock_port=clk, clock_period_ns=period, read_cmd=read_cmd))
    base = seed_qor["max_delay_ns"]
    log(f"  seed: cells={seed_qor['cells']} delay={base} ns")

    history: list[dict] = []
    spent = 0.0
    tokens = 0
    accepted: list[dict] = []
    best = base
    t_start = time.monotonic()
    candidates = 0

    tb_i = -1 if a.thinking_budget < 0 else a.thinking_budget
    for gen in range(a.generations):
        if spent >= a.max_usd:
            log(f"STOP: ${spent:.2f} reached the ${a.max_usd:.2f} cap")
            break
        log(f"--- generation {gen}: proposing {a.workers} candidates in parallel ---")
        t_gen = time.monotonic()
        rtl_text = parent.read_text()
        refs = [propose_edit.chia_remote(
            rtl_text, str(parent), top, a.design, gen,
            {k: seed_qor.get(k) for k in ("cells", "area_um2", "max_delay_ns")},
            {}, history[-5:], a.model, None if tb_i < 0 else tb_i)
            for _ in range(a.workers)]
        props = [get(r) for r in refs]
        t_prop = time.monotonic() - t_gen
        good = [p for p in props if p.get("ok")]
        for p in props:
            tokens += (p.get("billed_in") or 0) + (p.get("billed_out") or 0)
            spent += p.get("cost_usd") or 0.0
        unpriced = [p for p in props if p.get("cost_source") == "unknown"]
        if unpriced:
            log(f"  WARNING: {len(unpriced)}/{len(props)} turns could not be "
                f"priced (model not in the price table); dollars are incomplete")
        distinct = len({(p["old"], p["new"]) for p in good})
        log(f"  {len(good)}/{a.workers} proposals in {t_prop:.1f}s "
            f"({distinct} distinct edits); spent ${spent:.4f}")
        if not good:
            log("  no usable proposal this generation")
            continue

        # Materialise every candidate, then evaluate them concurrently.
        cand_paths = []
        for j, p in enumerate(good):
            txt = rtl_text
            if p["old"] not in txt:
                log(f"  candidate {j}: old text not found -- rejected before evaluation")
                continue
            cp = wd / f"gen{gen}-cand{j}.v"
            cp.write_text(txt.replace(p["old"], p["new"], 1))
            cand_paths.append((j, p, cp))
        candidates += len(cand_paths)
        if not cand_paths:
            continue

        t_eval = time.monotonic()
        qrefs = [yosys_sta_qor.chia_remote(
            [str(cp)], top, liberty, workdir=str(wd / f"qor-g{gen}-c{j}"),
            clock_port=clk, clock_period_ns=period, read_cmd=read_cmd)
            for j, _, cp in cand_paths]
        qors = [get(r) for r in qrefs]
        # Only pay for a proof on candidates that actually improved the metric.
        improved = [(j, p, cp, q) for (j, p, cp), q in zip(cand_paths, qors)
                    if q.get("max_delay_ns") is not None
                    and q["max_delay_ns"] < best]
        log(f"  scored {len(qors)} in {time.monotonic() - t_eval:.1f}s; "
            f"{len(improved)} beat the incumbent ({best:.4f} ns)")
        if not improved:
            history.extend({"note": p["note"], "result": "no improvement"}
                           for _, p, _ in cand_paths)
            continue

        lrefs = [lec_gate.chia_remote([str(parent)], [str(cp)], top,
                                      workdir=str(wd / f"lec-g{gen}-c{j}"),
                                      read_cmd=read_cmd, jobs=auto_jobs())
                 for j, _, cp, _ in improved]
        lecs = [get(r) for r in lrefs]
        proven = [(j, p, cp, q) for (j, p, cp, q), l in zip(improved, lecs)
                  if l.get("equivalent")]
        log(f"  proved {len(proven)}/{len(improved)} equivalent")
        if not proven:
            history.extend({"note": p["note"], "result": "refuted by the gate"}
                           for _, p, _, _ in improved)
            continue

        j, p, cp, q = min(proven, key=lambda x: x[3]["max_delay_ns"])
        best = q["max_delay_ns"]
        parent.write_text(cp.read_text())
        accepted.append({"generation": gen, "note": p["note"],
                         "max_delay_ns": best, "cells": q["cells"]})
        history.append({"note": p["note"], "result": f"accepted, {best:.4f} ns"})
        log(f"  ACCEPTED: {best:.4f} ns  ({p['note'][:60]})")

    wall = time.monotonic() - t_start
    per_hour = candidates / (wall / 3600) if wall else 0.0
    n_imp = len(accepted)
    summary = {
        "design": a.design, "workers": a.workers, "generations": a.generations,
        "model": a.model, "wall_s": round(wall, 1),
        "candidates": candidates, "candidates_per_hour": round(per_hour, 1),
        "accepted_improvements": n_imp,
        "tokens": tokens, "cost_usd": round(spent, 4) if spent else None,
        "usd_per_improvement": round(spent / n_imp, 4) if (spent and n_imp) else None,
        "tokens_per_improvement": round(tokens / n_imp) if n_imp else None,
        "seed_ns": base, "best_ns": best,
        "improvement_pct": round(100 * (base - best) / base, 2),
        "accepted": accepted,
    }
    print("\n=== population search ===")
    for k in ("workers", "wall_s", "candidates", "candidates_per_hour",
              "accepted_improvements", "cost_usd", "usd_per_improvement",
              "tokens_per_improvement", "best_ns", "improvement_pct"):
        print(f"  {k:24s} {summary[k]}")
    out = wd / "population-summary.json"
    out.write_text(json.dumps(summary, indent=2))
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
