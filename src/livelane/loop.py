"""The LiveLane loop, one pipeline, one arm per run.

    seed -> [ sample parent -> agent edit -> DELAY -> functional gate ->
              equivalence gate -> QoR score -> persist ] * N -> summary

Solid edges are programmatic Python control flow the agent cannot influence.
Only the "agent edit" step is agentic.  Scoring, gating and persistence are never
in the agent's hands: that separation is what makes the numbers believable, and
it is enforced structurally here rather than by convention.

The arm is entirely determined by two objects handed to the loop: an
:class:`~livelane.evaluators.Evaluator` (lane S or lane L) and a
:class:`~chia_livelane.delay_node.DelayNode`.  ``I(0)`` is lane S with a zero
delay, the same code path, so the control arm cannot drift from the treatment
arms.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from chia_livelane.delay_node import DelayNode
from chia_livelane.lec_gate import LecGateNode, PROVEN
from livelane.agent.base import Agent, AgentContext, Proposal
from livelane.db.store import RunHandle, VariantStore
from livelane.evaluators import DesignWorkspace, Evaluator
from livelane.state.reports import QorReport


@dataclass
class LoopConfig:
    design: str
    top: str
    source_dir: Path
    filelist_name: str | None = None
    #: Fixed budgets, set in advance and never extended after seeing results.
    budget_wall_s: float = 7200.0
    budget_iters: int = 50
    #: Parent selection: sample from the best K variants so far.
    top_k: int = 3
    seed: int = 0
    model: str = "scripted"
    #: Area guardrail on the primary (delay) objective.
    area_epsilon: float = 0.02
    #: Skip the equivalence gate (for harness smoke tests ONLY, never for a
    #: reported run; every reported arm gates every accepted candidate).
    skip_lec: bool = False
    #: Consecutive agent refusals tolerated before a run ends. A single
    #: stochastic "{"stop": true}" on the first turn killed 3 of 4 arms of a real
    #: sweep while 1800s of budget remained, turning the comparison into "did the
    #: model happen to give up". The same allowance applies to every arm, so it
    #: biases none of them; the count is recorded per run.
    max_consecutive_stops: int = 3


@dataclass
class IterationResult:
    index: int
    variant_id: int | None
    accepted: bool
    stage: str
    note: str = ""
    qor: QorReport | None = None
    lec_verdict: str | None = None
    observed_latency_s: float = 0.0
    eval_wall_s: float = 0.0
    gate_wall_s: float = 0.0
    injected_delay_s: float = 0.0


@dataclass
class LiveLaneLoop:
    cfg: LoopConfig
    evaluator: Evaluator
    delay: DelayNode
    store: VariantStore
    agent: Agent
    lec: LecGateNode | None = None
    workroot: Path = Path("var/workdirs")
    verbose: bool = True

    run: RunHandle | None = field(default=None, init=False)
    iterations: list[IterationResult] = field(default_factory=list, init=False)
    _best: tuple[int, QorReport] | None = field(default=None, init=False)
    #: variant_id -> that variant's RTL tree, so the gate compares a candidate
    #: against the exact parent it came from rather than against the seed.
    _rtl: dict[int, Path] = field(default_factory=dict, init=False)
    #: variant_id -> its TimingReport. The agent is optimising the critical path,
    #: so it must be able to SEE the critical path: the plan's tool surface
    #: promises read_timing() returning "slack, endpoint pin, and the RTL
    #: file:line that drives it". Passing only the scalar delay makes the agent
    #: guess which logic matters, and a measured run confirmed it: a formally
    #: proven edit that moved delay the WRONG way, 12.76ns -> 13.57ns.
    _timing: dict[int, Any] = field(default_factory=dict, init=False)
    _stops: int = field(default=0, init=False)

    #, helpers -------------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _arm_label(self) -> str:
        if self.evaluator.lane == "L":
            return "L"
        return f"I({int(self.delay.seconds)})"

    def _turn_usage(self) -> dict[str, Any]:
        """Four-class token accounting for the turn the agent just took.

        Without this the iterations table records zeros and the cost table,
        a headline deliverable, is empty. The agent accumulates one Usage per
        turn; take the LAST one, because add_iteration is called once per turn.
        """
        usages = getattr(self.agent, "usages", None)
        if not usages:
            return {}
        u = usages[-1]
        out: dict[str, Any] = {
            "tokens_in": u.tokens_in,
            "tokens_out": u.tokens_out + getattr(u, "thoughts", 0),
            "tokens_cache_read": u.cache_read,
            "tokens_cache_write": u.cache_write,
            "cost_source": u.cost_source,
        }
        if u.cost_usd is not None:
            out["cost_usd"] = u.cost_usd
        return out

    def _persist_tool_runs(self, variant_id: int | None) -> None:
        """Persist every instrumented tool invocation this iteration made.

        The plan requires that any published wall-clock be traceable back to a
        specific argv. Evaluators accumulate ToolRun records in `.tool_runs`;
        drain them so each is attributed to the variant it produced and cannot be
        double-counted by the next iteration.
        """
        assert self.run is not None
        runs = getattr(self.evaluator, "tool_runs", None)
        if not runs:
            return
        for tr in runs:
            try:
                self.store.add_tool_run(self.run.run_id, tr, variant_id=variant_id)
            except Exception as e:      # provenance must never kill a run
                self._log(f"         [warn] could not persist tool run: {e}")
        runs.clear()

    def _lec_sources(self, rtl_dir: Path) -> list[str]:
        """Sources handed to the equivalence gate.

        A filelist design is passed as its listed files, in order, so the gate
        elaborates the same module set the evaluator did.
        """
        if self.cfg.filelist_name:
            fl = rtl_dir / self.cfg.filelist_name
            names = [l.strip() for l in fl.read_text().splitlines()
                     if l.strip() and not l.strip().startswith(("#", "//"))]
            # Order matters: the gate reads these in ONE command, and the
            # filelist order is the order the evaluator used.
            return [str(rtl_dir / n) for n in names if (rtl_dir / n).exists()]
        return [str(p) for p in sorted(rtl_dir.rglob("*"))
                if p.suffix in (".v", ".sv")]

    def _workspace(self, tag: str,
                   parent_rtl: Path | None = None) -> DesignWorkspace:
        """A fresh working copy, seeded FROM THE PARENT when there is one.

        Materialising from ``cfg.source_dir`` every iteration silently discards
        the parent's edit, so the loop stops being a tree search and becomes a
        series of independent single edits against the pristine design. Observed
        directly: the agent proposed the same edit on two consecutive
        iterations, because each time it was handed the same untouched RTL, and
        the run then stalled with no accumulated progress.
        """
        wd = self.workroot / (self.run.run_id if self.run else "seed") / tag
        src = Path(parent_rtl) if parent_rtl is not None else self.cfg.source_dir
        return DesignWorkspace(source_dir=src, workdir=wd,
                               filelist_name=self.cfg.filelist_name).materialize()

    #, the loop ------------------------------------------------------------
    def start(self, provenance: dict[str, Any] | None = None) -> RunHandle:
        run_id = f"{self.cfg.design}-{self._arm_label()}-{self.cfg.model}-s{self.cfg.seed}-{uuid.uuid4().hex[:8]}"
        self.run = self.store.start_run(
            run_id, design=self.cfg.design, lane=self.evaluator.lane,
            arm=self._arm_label(), model=self.cfg.model, seed=self.cfg.seed,
            arm_delay_s=self.delay.seconds, arm_delay_mode=self.delay.mode.value,
            arm_delay_virtual=self.delay.virtual,
            budget_wall_s=self.cfg.budget_wall_s, budget_iters=self.cfg.budget_iters,
            liberty_path=getattr(self.evaluator, "liberty", None),
            provenance=provenance,
            # The evaluator's own configuration, so a later comparison can check
            # it was measuring the same quantity as the judge.
            note=json.dumps(self.evaluator.config())
            if hasattr(self.evaluator, "config") else None)
        if provenance:
            self.store.record_provenance(run_id, provenance)
        return self.run

    def seed(self) -> int:
        """Evaluate the unmodified design. Iteration 0, the baseline."""
        assert self.run is not None
        self._log(f"\n[iter 0] seed: evaluating the unmodified design")
        ws = self._workspace("seed")
        t0 = time.monotonic()
        qor, timing = self.evaluator.evaluate(
            [str(f) for f in ws.files()], self.cfg.top, ws.workdir,
            filelist=ws.filelist)
        eval_wall = time.monotonic() - t0
        rec = self.delay.apply(eval_wall, succeeded=qor.valid)

        vid = self.store.add_variant(
            self.run, iteration_index=0, parent_id=None,
            edit_note="seed", verilog_sha256=ws.sha256(),
            accepted=1 if qor.valid else 0, functional_pass=None,
            lec_verdict="skipped", qor_source=qor.source,
            qor_cells=qor.cells, qor_area_um2=qor.area_um2,
            qor_max_delay_ns=qor.max_delay_ns, qor_slack_ns=qor.slack_ns,
            qor_json=qor.raw_json,
            timing_json=(timing.raw_json if timing else None),
            status="ok" if qor.valid else "eval-failed")
        self.store.add_iteration(
            self.run, iteration_index=0, variant_id=vid,
            t_tool_ms=eval_wall * 1000.0, t_delay_ms=rec.slept_s * 1000.0,
            injected_delay_s=self.delay.seconds, delay_mode=self.delay.mode.value,
            delay_interrupted=int(rec.interrupted),
            evaluator_cpu_s=self.evaluator.cpu_s)
        self._persist_tool_runs(vid)
        self._rtl[vid] = ws.rtl_dir
        if timing is not None:
            self._timing[vid] = timing
        if qor.valid:
            self._best = (vid, qor)
        self._log(f"         baseline: cells={qor.cells} area={qor.area_um2} "
                  f"delay={qor.max_delay_ns} ({eval_wall:.2f}s + {rec.slept_s:.1f}s injected)")
        self.iterations.append(IterationResult(
            0, vid, qor.valid, "seed", "seed", qor, "skipped",
            rec.observed_latency_s, eval_wall, 0.0, rec.slept_s))
        return vid

    def _sample_parent(self) -> tuple[int, QorReport]:
        """Top-K by the primary objective; ties broken toward the newest."""
        assert self.run is not None
        rows = self.store.query(
            """SELECT id, qor_cells, qor_area_um2, qor_max_delay_ns, qor_slack_ns
               FROM variants
               WHERE run_id=? AND accepted=1 AND qor_area_um2 IS NOT NULL
               ORDER BY (CASE WHEN qor_max_delay_ns IS NULL THEN 1 ELSE 0 END),
                        qor_max_delay_ns ASC, qor_area_um2 ASC, id DESC
               LIMIT ?""",
            (self.run.run_id, self.cfg.top_k))
        if not rows:
            assert self._best is not None, "no accepted variant to build on"
            return self._best
        r = rows[0]
        return int(r["id"]), QorReport(
            top=self.cfg.top, cells=r["qor_cells"], area_um2=r["qor_area_um2"],
            max_delay_ns=r["qor_max_delay_ns"], slack_ns=r["qor_slack_ns"])

    def step(self, index: int) -> IterationResult:
        assert self.run is not None
        parent_id, parent_qor = self._sample_parent()
        # Build ON the parent, not on the original design.
        ws = self._workspace(f"iter{index}", parent_rtl=self._rtl.get(parent_id))

        ctx = AgentContext(
            design=self.cfg.design, top=self.cfg.top, iteration=index,
            files=[str(p.relative_to(ws.rtl_dir)) for p in ws.files()],
            qor=parent_qor.to_agent_dict(),
            timing=(self._timing[parent_id].to_agent_dict()
                    if parent_id in self._timing else {}),
            history=self.store.history(self.run.run_id, limit=10),
            read_file=ws.read, primary_file=ws.primary_file(self.cfg.top))

        t_llm0 = time.monotonic()
        proposal = self.agent.propose(ctx)
        t_llm = time.monotonic() - t_llm0

        if proposal is None:
            self._stops += 1
            if self._stops >= self.cfg.max_consecutive_stops:
                self._log(f"[iter {index}] agent declined "
                          f"{self._stops}x consecutively -- stopping")
                return IterationResult(index, None, False, "agent-stop",
                                       f"declined {self._stops} times")
            self._log(f"[iter {index}] agent declined "
                      f"({self._stops}/{self.cfg.max_consecutive_stops}) -- retrying")
            return IterationResult(index, None, False, "agent-declined",
                                   "declined, retrying")
        self._stops = 0
        if proposal.is_noop():
            self._log(f"[iter {index}] no-op edit rejected before evaluation")
            return IterationResult(index, None, False, "noop", "edit was a no-op")

        applied, occurrences, msg = ws.apply_edit(proposal.path, proposal.old,
                                                  proposal.new)
        self._log(f"[iter {index}] edit {proposal.path}: {msg}"
                  f"{'' if applied else ' -- REJECTED'}  ({proposal.note[:60]})")
        if not applied:
            vid = self.store.add_variant(
                self.run, iteration_index=index, parent_id=parent_id,
                edit_note=proposal.note, accepted=0, status="edit-failed",
                note=msg)
            return IterationResult(index, vid, False, "edit", msg)

        # --- evaluate, then inject the arm's latency -------------------------
        t0 = time.monotonic()
        qor, timing = self.evaluator.evaluate(
            [str(f) for f in ws.files()], self.cfg.top, ws.workdir,
            filelist=ws.filelist)
        eval_wall = time.monotonic() - t0
        rec = self.delay.apply(eval_wall, succeeded=qor.valid)

        # --- equivalence gate: runs in EVERY arm, charged to this arm's clock -
        # gold = the parent this candidate was derived from; gate = the candidate.
        lec_verdict, gate_wall, lec_backend = "skipped", 0.0, None
        if not self.cfg.skip_lec and self.lec is not None:
            parent_rtl = self._rtl.get(parent_id)
            if parent_rtl is None or not Path(parent_rtl).exists():
                lec_verdict = "error"
                self._log("         LEC: parent RTL unavailable -- failing closed")
            else:
                gold = self._lec_sources(Path(parent_rtl))
                gate = self._lec_sources(ws.rtl_dir)
                g0 = time.monotonic()
                r = self.lec.check(gold, gate, self.cfg.top, name=f"lec{index}")
                gate_wall = time.monotonic() - g0
                lec_verdict, lec_backend = r.verdict, r.backend
                self._log(f"         LEC: {r.evidence_strength} "
                          f"({gate_wall:.1f}s, backend={r.backend})")

        improved = qor.improves_on(parent_qor, self.cfg.area_epsilon) if qor.valid else False
        gate_ok = (lec_verdict in (PROVEN, "skipped"))
        accepted = bool(qor.valid and gate_ok)

        vid = self.store.add_variant(
            self.run, iteration_index=index, parent_id=parent_id,
            edit_note=proposal.note, verilog_sha256=ws.sha256(),
            accepted=1 if accepted else 0, functional_pass=None,
            lec_verdict=lec_verdict, lec_backend=lec_backend,
            lec_wall_s=gate_wall,
            qor_source=qor.source, qor_cells=qor.cells,
            qor_area_um2=qor.area_um2, qor_max_delay_ns=qor.max_delay_ns,
            qor_slack_ns=qor.slack_ns, qor_json=qor.raw_json,
            timing_json=(timing.raw_json if timing else None),
            status="ok" if qor.valid else "eval-failed",
            note=qor.message)
        self.store.add_iteration(
            self.run, iteration_index=index, variant_id=vid,
            t_llm_ms=t_llm * 1000.0, t_tool_ms=eval_wall * 1000.0,
            t_gate_ms=gate_wall * 1000.0, t_delay_ms=rec.slept_s * 1000.0,
            injected_delay_s=self.delay.seconds, delay_mode=self.delay.mode.value,
            delay_interrupted=int(rec.interrupted),
            evaluator_cpu_s=self.evaluator.cpu_s,
            **self._turn_usage())

        self._persist_tool_runs(vid)
        self._rtl[vid] = ws.rtl_dir
        if timing is not None:
            self._timing[vid] = timing
        if accepted and improved and qor.valid:
            self._best = (vid, qor)
        self._log(f"         -> {'ACCEPTED' if accepted else 'rejected'}"
                  f"{' (improved)' if improved else ''}: cells={qor.cells} "
                  f"area={qor.area_um2} delay={qor.max_delay_ns} "
                  f"[{eval_wall:.2f}s eval + {rec.slept_s:.1f}s injected]")

        return IterationResult(index, vid, accepted, "done", proposal.note, qor,
                               lec_verdict, rec.observed_latency_s, eval_wall,
                               gate_wall, rec.slept_s)

    def run_loop(self, provenance: dict[str, Any] | None = None) -> dict[str, Any]:
        self.start(provenance)
        assert self.run is not None
        self._log(f"=== LiveLane run {self.run.run_id} ===")
        self._log(f"    arm={self._arm_label()} lane={self.evaluator.lane} "
                  f"delay={self.delay.seconds}s design={self.cfg.design}")
        self.seed()
        i = 1
        while i <= self.cfg.budget_iters:
            if self.run.elapsed() > self.cfg.budget_wall_s:
                self._log(f"[budget] wall-clock budget "
                          f"{self.cfg.budget_wall_s}s exhausted at iteration {i}")
                break
            res = self.step(i)
            self.iterations.append(res)
            if res.stage == "agent-stop":
                break
            if res.stage == "agent-declined":
                i += 1
                continue
            i += 1
        self.store.finish_run(self.run.run_id)
        summary = self.store.run_summary(self.run.run_id)
        self._log(f"\n=== run complete: {summary['variants']} variants, "
                  f"{summary['accepted']} accepted, "
                  f"{summary['delay_s']:.0f}s injected delay ===")
        return summary


__all__ = ["LiveLaneLoop", "LoopConfig", "IterationResult"]
