"""Sweep driver: matrix shape, arm interleaving, resume, and the hard spend cap."""

import pytest

from livelane.db.store import VariantStore
from livelane.sweep import (LADDER_S, BudgetExhausted, SweepBudget, SweepConfig,
                            completed_cells)


def test_matrix_size():
    cfg = SweepConfig(designs=("Alu", "DecodeUnit"), models=("m",), seeds=(0, 1, 2))
    assert len(cfg.cells()) == 2 * 4 * 1 * 3


def test_arms_are_interleaved_not_blocked():
    """Blocking by arm would confound the treatment with session drift.

    If all of I(0) ran first and all of I(600) last, anything that changes over
    the session, thermal state, a background job, a model-side deployment,
    would land entirely on one arm and look like a latency effect.
    """
    cfg = SweepConfig(designs=("Alu",), models=("m",), seeds=(0, 1, 2))
    arms = [c.arm for c in cfg.cells()]
    # The first four cells must cover all four arms.
    assert len(set(arms[:4])) == 4, arms[:4]


def test_lane_l_is_an_extra_arm_not_a_replacement():
    base = SweepConfig(designs=("Alu",), models=("m",), seeds=(0,))
    withl = SweepConfig(designs=("Alu",), models=("m",), seeds=(0,),
                        include_lane_l=True)
    assert len(withl.cells()) == len(base.cells()) + 1
    assert any(c.lane == "L" for c in withl.cells())


def test_ladder_matches_the_fixed_design():
    assert LADDER_S == (0.0, 30.0, 120.0, 600.0)
    arms = {c.arm for c in SweepConfig(designs=("Alu",), models=("m",), seeds=(0,)).cells()}
    assert arms == {"I(0)", "I(30)", "I(120)", "I(600)"}


def test_hard_dollar_cap_raises_not_warns():
    """The GCP budget only alerts; this is the only thing that stops spending."""
    b = SweepBudget(max_usd=1.0)
    b.add(0.5)
    b.check()                      # under cap: fine
    b.add(0.6)
    with pytest.raises(BudgetExhausted):
        b.check()


def test_wall_clock_cap_also_stops():
    b = SweepBudget(max_usd=1e9, max_wall_s=0.0)
    with pytest.raises(BudgetExhausted):
        b.check()


def test_budget_ignores_negative_and_none_costs():
    b = SweepBudget(max_usd=1.0)
    b.add(None)      # a seat that reported no cost
    b.add(-5.0)      # nonsense must not create credit
    assert b.spent_usd == 0.0


def test_resume_is_by_identity(tmp_path):
    s = VariantStore(tmp_path / "t.db", verbose=False)
    assert completed_cells(s) == set()
    run = s.start_run("r1", design="Alu", lane="S", arm="I(30)", model="m", seed=2)
    # An unfinished run must NOT count as done.
    assert completed_cells(s) == set()
    s.finish_run("r1")
    assert completed_cells(s) == {("Alu", "I(30)", "m", 2)}
    s.close()


def test_completed_cells_matches_cell_key(tmp_path):
    s = VariantStore(tmp_path / "t.db", verbose=False)
    s.start_run("r", design="Alu", lane="S", arm="I(0)", model="gemini", seed=1)
    s.finish_run("r")
    cfg = SweepConfig(designs=("Alu",), models=("gemini",), seeds=(1,))
    done = completed_cells(s)
    todo = [c for c in cfg.cells() if c.key not in done]
    assert len(todo) == 3          # I(0) already done, three arms remain
    assert "I(0)" not in {c.arm for c in todo}
    s.close()


# --- the cap that survives an unpriced seat ----------------------------------

class _UnpricedSeat:
    """A seat that reports tokens but no billed cost, i.e. Vertex."""
    name = "unpriced"

    def ask(self, system, user, timeout_s=0.0):
        from livelane.agent.llm import Usage
        return '{"stop": true}', Usage(tokens_in=18, tokens_out=2, thoughts=92,
                                       cost_usd=None, cost_source="unknown")


def test_token_cap_stops_a_seat_that_reports_no_cost():
    """Vertex returns no billed cost, so a $-only cap would sit at $0.00 forever
    while the sweep spent real money. The token cap does not depend on anyone's
    price table."""
    b = SweepBudget(max_usd=25.0, max_tokens=200)
    b.add_usage({"turns": 1, "cost_usd": None, "tokens_in": 100,
                 "tokens_out": 50, "thoughts": 60})
    with pytest.raises(BudgetExhausted, match="token cap"):
        b.check()


def test_inert_dollar_cap_is_flagged_loudly():
    b = SweepBudget(max_usd=25.0)
    b.add_usage({"turns": 3, "cost_usd": None, "tokens_in": 10})
    assert b.dollar_cap_is_inert
    assert "INERT" in (b.warn_if_dollar_cap_inert() or "")
    assert "$ unpriced" in b.summary()


def test_a_priced_seat_is_not_flagged_inert():
    b = SweepBudget(max_usd=25.0)
    b.add_usage({"turns": 1, "cost_usd": 0.01, "tokens_in": 10})
    assert not b.dollar_cap_is_inert
    assert b.warn_if_dollar_cap_inert() is None


def test_thinking_tokens_count_toward_the_cap():
    """On gemini-3.x a measured PONG was 2 output tokens and 92 thinking tokens.
    Omitting thoughts undercounts billed output by ~45x."""
    b = SweepBudget(max_tokens=10_000)
    b.add_usage({"turns": 1, "cost_usd": None, "tokens_in": 18,
                 "tokens_out": 2, "thoughts": 92})
    assert b.spent_tokens == 112


def test_agent_with_no_priceable_turn_reports_none_not_zero():
    from livelane.agent.base import AgentContext
    from livelane.agent.llm import LLMAgent
    ctx = AgentContext(design="d", top="t", iteration=0, files=[], qor={},
                       timing={}, history=[])
    a = LLMAgent(seat=_UnpricedSeat())
    a.propose(ctx)
    assert a.total_cost_usd is None, "0.0 would read as 'this was free'"
    assert a.usage_summary()["thoughts"] == 92


def test_recorded_replay_really_is_free():
    from livelane.agent.base import AgentContext
    from livelane.agent.llm import LLMAgent, RecordedSeat
    ctx = AgentContext(design="d", top="t", iteration=0, files=[], qor={},
                       timing={}, history=[])
    a = LLMAgent(seat=RecordedSeat(transcript=['{"stop": true}']))
    a.propose(ctx)
    assert a.total_cost_usd == 0.0
