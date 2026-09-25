"""Each iteration must build on its PARENT, not on the pristine design.

Materialising the workspace from the original source every iteration silently
turns the loop from a tree search into a series of independent single edits.
It is invisible in the schema, rows still have parent_id set, best-so-far still
looks monotone, and it was only caught by noticing the agent propose the *same*
edit twice in a live run, because each time it was handed the same untouched RTL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from chia_livelane.delay_node import DelayNode
from livelane.agent.base import Proposal, ScriptedAgent
from livelane.db.store import VariantStore
from livelane.loop import LiveLaneLoop, LoopConfig
from livelane.state.reports import QorReport, TimingReport

RTL = "module w(input a, output b);\n  assign b = a;  // MARK0\nendmodule\n"


@dataclass
class CountingEvaluator:
    """Records the RTL it was asked to score. No tools involved."""

    lane: str = "S"
    seen: list[str] = field(default_factory=list)
    tool_runs: list = field(default_factory=list)

    def evaluate(self, sources, top, workdir, filelist=None):
        text = (Path(workdir) / "rtl" / "w.v").read_text()
        self.seen.append(text)
        # Score improves with each accumulated MARK, so a working tree search
        # shows a falling delay and a broken one plateaus.
        marks = text.count("MARK")
        return (QorReport(top=top, cells=10, area_um2=100.0,
                          max_delay_ns=10.0 - marks, valid=True, source="stub"),
                TimingReport(top=top, slack_ns=None, max_delay_ns=10.0 - marks,
                             source="stub"))

    @property
    def cpu_s(self) -> float:
        return 0.0


def _design(tmp_path: Path) -> Path:
    d = tmp_path / "design"
    d.mkdir()
    (d / "w.v").write_text(RTL)
    (d / "filelist.f").write_text("w.v\n")
    return d


def test_second_edit_sees_the_first(tmp_path):
    src = _design(tmp_path)
    store = VariantStore(tmp_path / "db.sqlite", verbose=False)
    ev = CountingEvaluator()
    agent = ScriptedAgent(edits=[
        Proposal("w.v", "// MARK0", "// MARK0\n  // MARK1", "first"),
        Proposal("w.v", "// MARK1", "// MARK1\n  // MARK2", "second, needs the first"),
    ])
    loop = LiveLaneLoop(
        cfg=LoopConfig(design="w", top="w", source_dir=src,
                       filelist_name="filelist.f", budget_iters=2,
                       budget_wall_s=60, skip_lec=True),
        evaluator=ev, delay=DelayNode(0.0, verbose=False), store=store,
        agent=agent, lec=None, workroot=tmp_path / "work", verbose=False)
    loop.run_loop()

    # The SECOND edit only applies if the first one is present in the tree it was
    # given. If the workspace were re-seeded from source, `// MARK1` would not
    # exist and the edit would be rejected as non-matching.
    rows = store.query(
        "SELECT iteration_index, accepted, status FROM variants "
        "WHERE run_id=? ORDER BY iteration_index", (loop.run.run_id,))
    assert len(rows) == 3, [dict(r) for r in rows]
    assert all(r["accepted"] == 1 for r in rows), [dict(r) for r in rows]
    assert rows[2]["status"] != "edit-failed"

    # And the evaluator must have seen accumulating content.
    assert ev.seen[0].count("MARK") == 1
    assert ev.seen[1].count("MARK") == 2
    assert ev.seen[2].count("MARK") == 3, "third evaluation lost the parent's edits"
    store.close()


def test_best_so_far_actually_improves_when_edits_accumulate(tmp_path):
    src = _design(tmp_path)
    store = VariantStore(tmp_path / "db.sqlite", verbose=False)
    agent = ScriptedAgent(edits=[
        Proposal("w.v", "// MARK0", "// MARK0\n  // MARK1", "first"),
        Proposal("w.v", "// MARK1", "// MARK1\n  // MARK2", "second"),
    ])
    loop = LiveLaneLoop(
        cfg=LoopConfig(design="w", top="w", source_dir=src,
                       filelist_name="filelist.f", budget_iters=2,
                       budget_wall_s=60, skip_lec=True),
        evaluator=CountingEvaluator(), delay=DelayNode(0.0, verbose=False),
        store=store, agent=agent, lec=None, workroot=tmp_path / "w", verbose=False)
    loop.run_loop()
    curve = store.best_so_far(loop.run.run_id)
    assert [v for _, v in curve] == [9.0, 8.0, 7.0], curve
    store.close()


def test_one_refusal_does_not_end_a_run_with_budget_left(tmp_path):
    """A single stochastic {"stop": true} killed 3 of 4 arms of a real sweep
    while 1800s of budget remained, turning the comparison into "did the model
    happen to give up". Up to `max_consecutive_stops` refusals are tolerated."""
    from livelane.agent.base import AgentContext, Proposal

    class FlakyAgent:
        """Declines, then proposes a real edit."""
        name = "flaky"

        def __init__(self):
            self.calls = 0

        def propose(self, ctx: AgentContext):
            self.calls += 1
            if self.calls == 1:
                return None                     # a first-turn refusal
            if self.calls == 2:
                return Proposal("w.v", "// MARK0", "// MARK0\n  // MARK1", "real")
            return None

    src = _design(tmp_path)
    store = VariantStore(tmp_path / "db.sqlite", verbose=False)
    agent = FlakyAgent()
    loop = LiveLaneLoop(
        cfg=LoopConfig(design="w", top="w", source_dir=src,
                       filelist_name="filelist.f", budget_iters=6,
                       budget_wall_s=60, skip_lec=True),
        evaluator=CountingEvaluator(), delay=DelayNode(0.0, verbose=False),
        store=store, agent=agent, lec=None, workroot=tmp_path / "w", verbose=False)
    loop.run_loop()
    rows = store.query("SELECT COUNT(*) c FROM variants WHERE run_id=?",
                       (loop.run.run_id,))
    # seed + the edit that came after the refusal
    assert rows[0]["c"] == 2, "the run died on the first refusal"
    store.close()


def test_repeated_refusals_do_end_the_run(tmp_path):
    class AlwaysDeclines:
        name = "nope"

        def propose(self, ctx):
            return None

    src = _design(tmp_path)
    store = VariantStore(tmp_path / "db.sqlite", verbose=False)
    loop = LiveLaneLoop(
        cfg=LoopConfig(design="w", top="w", source_dir=src,
                       filelist_name="filelist.f", budget_iters=50,
                       budget_wall_s=60, skip_lec=True,
                       max_consecutive_stops=3),
        evaluator=CountingEvaluator(), delay=DelayNode(0.0, verbose=False),
        store=store, agent=AlwaysDeclines(), lec=None,
        workroot=tmp_path / "w", verbose=False)
    loop.run_loop()
    assert loop._stops == 3, loop._stops
    store.close()
