"""The experiment driver: run the experiment matrix, resumably and safely.

One "cell" is one (design x arm x model x seed) run.  The sweep walks the matrix,
skips cells already present in the database, and writes everything to one store
so analysis never has to stitch files together.

Three safety properties, in order of how much they matter:

**A hard dollar cap that actually stops.**  The billing account backing this
project is a university org account, not a free trial, so it has no spend-stop of
its own; a GCP budget only *alerts*.  :class:`SweepBudget` is therefore the real
brake, it is checked before every cell and again after every iteration, and it
aborts rather than warns.

**Arm order is interleaved, not blocked.**  Running all of ``I(0)`` then all of
``I(600)`` would confound the treatment with anything that drifts over the
session (thermal state, a background job, a model-side deployment change).  Cells
are emitted round-robin across arms so drift hits every arm equally.

**Resume is by identity, not by position.**  A cell is identified by its
(design, arm, model, seed) tuple looked up in the store, so an interrupted sweep
resumes correctly even if the matrix definition changed between invocations.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from chia_livelane.delay_node import DelayMode, DelayNode
from chia_livelane.lec_gate import DEFAULT_STRATEGIES, LecGateNode
from livelane.db.store import VariantStore
from livelane.evaluators import LaneLEvaluator, LaneSEvaluator
from livelane.harness.provenance import capture
from livelane.loop import LiveLaneLoop, LoopConfig
from livelane.nodes.lhd import find_lhd
from livelane.nodes.yosys_sta import SCRIPTS

#: The fixed ladder.
LADDER_S: tuple[float, ...] = (0.0, 30.0, 120.0, 600.0)


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class SweepBudget:
    """A hard stop. Not an alert.

    The GCP budget on this project alerts at 10/25/50/75/90/100% of $150 but does
    not halt anything; the billing account is a UCSC org account with no
    free-trial cutoff. This class is the only thing that actually stops spending.

    **Why there is a token cap as well as a dollar cap.** Vertex does not return
    a billed cost, so :class:`~livelane.agent.vertex.VertexSeat` prices a call
    from a hand-typed list-price table and reports ``cost_usd=None`` for any
    model absent from it, correctly, because recording an unpriced call as
    ``0.0`` would quietly deflate $/improvement. But that also means a dollar cap
    fed by ``cost_usd`` is INERT for an unpriced model: it would sit at $0.00
    forever while the sweep spent real money. The token cap does not depend on
    anyone's price table, so it is the backstop that always works, and
    :meth:`warn_if_dollar_cap_inert` says so out loud rather than letting a
    silent $0.00 look like thrift.
    """

    max_usd: float = 25.0
    max_wall_s: float = 6 * 3600.0
    #: Total billed tokens (in + out + thinking + cache). Enforced regardless of
    #: whether anything could be priced.
    max_tokens: int = 20_000_000
    spent_usd: float = 0.0
    spent_tokens: int = 0
    priced_turns: int = 0
    unpriced_turns: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def remaining_usd(self) -> float:
        return max(0.0, self.max_usd - self.spent_usd)

    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.spent_tokens)

    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def dollar_cap_is_inert(self) -> bool:
        """True when nothing has been priceable, so the $ cap cannot fire."""
        return self.unpriced_turns > 0 and self.priced_turns == 0

    def check(self, note: str = "") -> None:
        if self.spent_usd >= self.max_usd:
            raise BudgetExhausted(
                f"dollar cap reached: ${self.spent_usd:.4f} >= ${self.max_usd:.2f} {note}")
        if self.spent_tokens >= self.max_tokens:
            raise BudgetExhausted(
                f"token cap reached: {self.spent_tokens:,} >= {self.max_tokens:,} tokens {note}")
        if self.elapsed_s() >= self.max_wall_s:
            raise BudgetExhausted(
                f"wall-clock cap reached: {self.elapsed_s():.0f}s >= {self.max_wall_s:.0f}s {note}")

    def add(self, usd: float) -> None:
        self.spent_usd += max(0.0, usd or 0.0)

    def add_usage(self, usage: dict[str, Any]) -> None:
        """Charge one cell's usage. Tracks tokens even when dollars are unknown."""
        cost = usage.get("cost_usd")
        turns = int(usage.get("turns", 0) or 0)
        if cost is None:
            self.unpriced_turns += turns
        else:
            self.priced_turns += turns
            self.spent_usd += max(0.0, cost)
        for k in ("tokens_in", "tokens_out", "cache_read", "cache_write", "thoughts"):
            self.spent_tokens += int(usage.get(k, 0) or 0)

    def warn_if_dollar_cap_inert(self) -> str | None:
        if not self.dollar_cap_is_inert:
            return None
        return (f"the ${self.max_usd:.2f} cap is INERT -- this seat reports no "
                f"billed cost, so spending is bounded ONLY by the "
                f"{self.max_tokens:,}-token cap and the "
                f"{self.max_wall_s / 3600:.1f}h wall-clock cap")

    def summary(self) -> str:
        d = (f"${self.spent_usd:.4f}/${self.max_usd:.2f}" if not self.dollar_cap_is_inert
             else "$ unpriced")
        return (f"{d}, {self.spent_tokens:,}/{self.max_tokens:,} tokens, "
                f"{self.elapsed_s() / 60:.1f} min")


@dataclass(frozen=True)
class Cell:
    design: str
    arm: str
    delay_s: float
    lane: str
    model: str
    seed: int

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.design, self.arm, self.model, self.seed)


@dataclass
class SweepConfig:
    designs: Sequence[str] = ("Alu", "DecodeUnit")
    delays_s: Sequence[float] = LADDER_S
    models: Sequence[str] = ("gemini-3.8-flash",)
    seeds: Sequence[int] = (0, 1, 2)
    #: Include the LiveHD lane as an extra arm alongside the injection ladder.
    include_lane_l: bool = False
    budget_wall_s: float = 1800.0
    budget_iters: int = 20
    #: Equivalence gating. NEVER disable for a reported run.
    skip_lec: bool = False
    #: Extended-thinking budget for the seat. An EXPERIMENTAL VARIABLE, not a
    #: tuning knob: it changes T_llm and therefore the ladder's dynamic range
    #: (picorv32 spans 8.2x at 0, 3.7x at the model default). It is identical
    #: across every arm of one sweep and recorded on every run.
    thinking_budget: int | None = 0

    def cells(self) -> list[Cell]:
        """Matrix in interleaved-by-arm order, so drift hits every arm equally."""
        by_arm: dict[str, list[Cell]] = {}
        for design, model, seed in itertools.product(self.designs, self.models, self.seeds):
            for d in self.delays_s:
                arm = f"I({int(d)})"
                by_arm.setdefault(arm, []).append(
                    Cell(design, arm, d, "S", model, seed))
            if self.include_lane_l:
                by_arm.setdefault("L", []).append(
                    Cell(design, "L", 0.0, "L", model, seed))
        out: list[Cell] = []
        for group in itertools.zip_longest(*by_arm.values()):
            out.extend(c for c in group if c is not None)
        return out


def completed_cells(store: VariantStore) -> set[tuple[str, str, str, int]]:
    rows = store.query(
        "SELECT design, arm, model, seed FROM runs WHERE status='done'")
    return {(r["design"], r["arm"], r["model"], r["seed"]) for r in rows}


def _make_agent(model: str, seat_kind: str, thinking_budget: int | None = 0):
    """Build the agent seat. Imported lazily so a missing SDK is not fatal."""
    from livelane.agent.llm import LLMAgent
    if seat_kind == "vertex":
        from livelane.agent.vertex import VertexSeat
        return LLMAgent(seat=VertexSeat(model=model, thinking_budget=thinking_budget))
    if seat_kind == "claude":
        from livelane.agent.llm import ClaudeCliSeat
        return LLMAgent(seat=ClaudeCliSeat(model=model))
    if seat_kind == "recorded":
        from livelane.agent.llm import RecordedSeat
        return LLMAgent(seat=RecordedSeat(transcript=[]))
    raise ValueError(f"unknown seat kind {seat_kind!r}")


def run_sweep(root: Path, cfg: SweepConfig, budget: SweepBudget, *,
              seat_kind: str = "vertex", db_path: Path | None = None,
              dry_run: bool = False, judge_finals: bool = True,
              verbose: bool = True) -> dict[str, Any]:
    liberty = os.environ.get("LIVELANE_LIBERTY")
    if not liberty or not Path(liberty).exists():
        raise SystemExit("LIVELANE_LIBERTY unset/missing; `source env.sh` first.")

    yosys = str(root / "tools" / "bin" / "yosys")
    sta = str(root / "tools" / "bin" / "sta")
    eqy = str(root / "tools" / "bin" / "eqy")
    lhd_bin = find_lhd(root)
    suite = root / "thirdparty" / "lhdsuite"
    xs = suite / "xiangshan" / "Backend" / "verilog"

    # design -> (top, source_dir, filelist_name, clock_port, clock_period_ns)
    #
    # The clock is not optional bookkeeping. Without it OpenSTA reports the
    # longest unconstrained combinational I/O path rather than the
    # register-to-register critical path, and on picorv32 that is 0.196 ns
    # against a real 12.76 ns, an objective the agent cannot improve because it
    # does not measure the design. XiangShan's Alu and DecodeUnit are 0%
    # sequential, so for them the unconstrained path IS the critical path and the
    # clock is correctly None.
    DESIGNS = {
        "Alu": ("Alu", xs, "filelist.f", None, None),
        "DivUnit": ("DivUnit", xs, "filelist.f", None, None),
        "DecodeUnit": ("DecodeUnit", xs, "filelist.f", None, None),
        "RenameTableWrapper": ("RenameTableWrapper", xs, "filelist.f", None, None),
        "picorv32": ("picorv32", root / "designs" / "picorv32", "filelist.f",
                     "clk", 10.0),
    }

    db = db_path or (root / "var" / "livelane.db")
    store = VariantStore(db, verbose=False)
    done = completed_cells(store)
    cells = cfg.cells()
    todo = [c for c in cells if c.key not in done]

    if verbose:
        print(f"=== LiveLane sweep ===")
        print(f"    db          : {db}")
        print(f"    matrix      : {len(cells)} cells "
              f"({len(cfg.designs)} designs x {len(cfg.delays_s)}"
              f"{'+1' if cfg.include_lane_l else ''} arms x "
              f"{len(cfg.models)} models x {len(cfg.seeds)} seeds)")
        print(f"    already done: {len(cells) - len(todo)}")
        print(f"    to run      : {len(todo)}")
        print(f"    seat        : {seat_kind}")
        print(f"    caps        : ${budget.max_usd:.2f}, "
              f"{budget.max_tokens:,} tokens, "
              f"{budget.max_wall_s / 3600:.1f}h wall  <- HARD STOP")
        if seat_kind == "vertex":
            print(f"    NOTE        : Vertex reports no billed cost, so the $ cap "
                  f"is inert and the TOKEN cap is the real brake")
        print(f"    per-cell    : {cfg.budget_iters} iters / "
              f"{cfg.budget_wall_s:.0f}s")
        print(f"    lec gating  : {'ON' if not cfg.skip_lec else '*** OFF ***'}")
        tb = cfg.thinking_budget
        print(f"    thinking    : "
              f"{'model default' if tb is None else tb}"
              f"   (experimental variable: it changes T_llm and so the "
              f"ladder's spread; identical across every arm)")

    if dry_run:
        print("\n--- dry run: cell order (interleaved by arm) ---")
        for i, c in enumerate(todo[:40]):
            print(f"  {i:>3}  {c.design:<20} {c.arm:<8} {c.model:<20} seed={c.seed}")
        if len(todo) > 40:
            print(f"  ... and {len(todo) - 40} more")
        store.close()
        return {"dry_run": True, "cells_total": len(cells), "cells_todo": len(todo)}

    prov = json.loads(capture("sweep", repos={
        "livelane": root, "livehd": root / "thirdparty" / "livehd",
        "lhdsuite": suite, "yosys": root / "thirdparty" / "yosys"},
        pdk_version=os.environ.get("CIEL_PDK_VERSION"),
        liberty_sha256=os.environ.get("LIVELANE_LIBERTY_SHA256"),
        seat_kind=seat_kind, thinking_budget=cfg.thinking_budget).to_json())

    judged_runs: list[str] = []
    ran, failed, stopped = 0, 0, None
    for i, c in enumerate(todo):
        try:
            budget.check(f"before {c.design}/{c.arm}/seed{c.seed}")
        except BudgetExhausted as e:
            stopped = str(e)
            if verbose:
                print(f"\n!!! STOPPING: {e}")
            break

        top, srcdir, flist, clk_port, clk_period = DESIGNS[c.design]
        if verbose:
            print(f"\n[{i + 1}/{len(todo)}] {c.design} {c.arm} {c.model} seed={c.seed}"
                  f"   (spent ${budget.spent_usd:.4f} / ${budget.max_usd:.2f})")

        if c.lane == "L":
            if lhd_bin is None:
                print("    lhd not built; skipping lane L cell")
                continue
            evaluator = LaneLEvaluator(binary=lhd_bin, liberty=liberty,
                                       reader="slang", filelist=None)
        else:
            evaluator = LaneSEvaluator(
                yosys=yosys, sta=sta, liberty=liberty,
                script=SCRIPTS["baseline-2026-09-02"], read_cmd="read_slang",
                clock_port=clk_port, clock_period_ns=clk_period)

        delay = DelayNode(seconds=c.delay_s, mode=DelayMode.ADDITIVE, verbose=False)
        lec = None if cfg.skip_lec else LecGateNode(
            eqy=eqy, workdir=root / "var" / "workdirs" / "lec",
            read_cmd="read_slang", strategies=DEFAULT_STRATEGIES, timeout_s=1800.0)
        agent = _make_agent(c.model, seat_kind, cfg.thinking_budget)

        loop = LiveLaneLoop(
            cfg=LoopConfig(design=c.design, top=top, source_dir=srcdir,
                           filelist_name=flist, seed=c.seed, model=c.model,
                           budget_wall_s=cfg.budget_wall_s,
                           budget_iters=cfg.budget_iters,
                           skip_lec=cfg.skip_lec),
            evaluator=evaluator, delay=delay, store=store, agent=agent, lec=lec,
            workroot=root / "var" / "workdirs", verbose=verbose)
        try:
            loop.run_loop(provenance=prov)
            ran += 1
            if loop.run is not None:
                judged_runs.append(loop.run.run_id)
        except BudgetExhausted:
            raise
        except Exception as e:
            failed += 1
            print(f"    CELL FAILED: {type(e).__name__}: {str(e)[:200]}")
        finally:
            # Charge this cell's tokens to the sweep budget even if it crashed,
            # a failed run still spent money.
            usage = getattr(agent, "usage_summary", lambda: {})()
            budget.add_usage(usage)
            if verbose and usage:
                cost = usage.get("cost_usd")
                cost_s = "unpriced" if cost is None else f"${cost:.4f}"
                print(f"    seat: {usage.get('turns')} turns, {cost_s}, "
                      f"{usage.get('malformed_replies', 0)} malformed"
                      f"   | budget: {budget.summary()}")
                warn = budget.warn_if_dollar_cap_inert()
                if warn and (i == 0):
                    print(f"    !! {warn}")

    # --- neutral judge, AFTER every arm has finished -------------------------
    # Deliberately not interleaved: judging is a full synthesis per candidate and
    # would contend for CPU with the arms still running, contaminating the very
    # wall-clock this experiment measures.
    judge_report: dict[str, Any] = {}
    if judge_finals and judged_runs:
        try:
            from livelane.judge import NeutralJudge, judge_run, proxy_vs_judge
            # The judge MUST apply the same clock constraint the arms ran under.
            # Without it the judge reports the unconstrained combinational I/O
            # path while the evaluator reported the register-to-register critical
            # path, 0.1959 vs 12.7612 on picorv32, and the H4 correlation
            # silently compares two different physical quantities.
            j_clk = j_per = None
            for rid in judged_runs:
                dname = store.query("SELECT design FROM runs WHERE run_id=?",
                                    (rid,))[0]["design"]
                j_clk, j_per = DESIGNS[dname][3], DESIGNS[dname][4]
                break
            nj = NeutralJudge(yosys=yosys, sta=sta, liberty=liberty,
                              clock_port=j_clk, clock_period_ns=j_per,
                              verbose=False)
            if verbose:
                print(f"\n=== neutral judge: re-scoring {len(judged_runs)} runs "
                      f"through ONE pinned flow ===")
            for rid in judged_runs:
                design = store.query("SELECT design FROM runs WHERE run_id=?",
                                     (rid,))[0]["design"]
                top = DESIGNS[design][0]
                rep = judge_run(store, rid, judge=nj, top=top,
                                workroot=root / "var" / "workdirs",
                                filelist_name=DESIGNS[design][2], verbose=False)
                rho = proxy_vs_judge(store, rid)
                judge_report[rid] = {"judged": rep.n_judged, "valid": rep.n_valid,
                                     "complete": rep.complete, "spearman_rho": rho}
                if verbose:
                    print(f"  {rid[:44]:<44} judged={rep.n_judged} "
                          f"complete={rep.complete} rho={rho}")
        except Exception as e:
            print(f"  judge step failed: {type(e).__name__}: {str(e)[:160]}")

    summary = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "cells_total": len(cells), "cells_todo": len(todo),
        "cells_ran": ran, "cells_failed": failed,
        "stopped_early": stopped,
        "judge": judge_report,
        "spent_usd": budget.spent_usd, "spent_tokens": budget.spent_tokens,
        "dollar_cap_inert": budget.dollar_cap_is_inert,
        "priced_turns": budget.priced_turns, "unpriced_turns": budget.unpriced_turns,
        "elapsed_s": budget.elapsed_s(),
        "db": str(db),
    }
    if verbose:
        print(f"\n=== sweep done: {ran} ran, {failed} failed, "
              f"${budget.spent_usd:.4f} spent, {budget.elapsed_s() / 60:.1f} min ===")
        if stopped:
            print(f"    stopped early: {stopped}")
    store.close()
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=Path(__file__).resolve().parents[2])
    ap.add_argument("--designs", nargs="*", default=["Alu", "DecodeUnit"])
    ap.add_argument("--delays", nargs="*", type=float, default=list(LADDER_S))
    ap.add_argument("--models", nargs="*", default=["gemini-3.8-flash"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--seat", default="vertex", choices=["vertex", "claude", "recorded"])
    ap.add_argument("--include-lane-l", action="store_true")
    ap.add_argument("--iters", type=int, default=20, help="iterations per cell")
    ap.add_argument("--cell-wall-s", type=float, default=1800.0)
    ap.add_argument("--max-usd", type=float, default=25.0,
                    help="HARD dollar stop for the whole sweep")
    ap.add_argument("--max-hours", type=float, default=6.0)
    ap.add_argument("--max-tokens", type=int, default=20_000_000,
                    help="HARD token stop; the backstop when a seat reports no cost")
    ap.add_argument("--db", default=None)
    ap.add_argument("--thinking-budget", type=int, default=0,
                    help="0 minimises extended thinking (widest ladder); "
                         "-1 leaves the model default")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the neutral judge pass (H4 cannot be computed)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = SweepConfig(designs=a.designs, delays_s=a.delays, models=a.models,
                      seeds=a.seeds, include_lane_l=a.include_lane_l,
                      budget_wall_s=a.cell_wall_s, budget_iters=a.iters,
                      thinking_budget=(None if a.thinking_budget < 0
                                       else a.thinking_budget))
    budget = SweepBudget(max_usd=a.max_usd, max_wall_s=a.max_hours * 3600.0,
                         max_tokens=a.max_tokens)
    res = run_sweep(Path(a.root), cfg, budget, seat_kind=a.seat,
                    db_path=Path(a.db) if a.db else None, dry_run=a.dry_run,
                    judge_finals=not a.no_judge)
    out = Path(a.root) / "var" / "runs" / "sweep_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
