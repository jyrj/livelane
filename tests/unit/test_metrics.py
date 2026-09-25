"""Analysis statistics. These produce every number in the paper, so the
properties that could silently corrupt a figure are tested explicitly."""

import math

import pytest

from livelane.analysis.metrics import (ArmSummary, auc, best_so_far, bootstrap,
                                       bootstrap_difference, is_monotone_decreasing,
                                       kendall_tau_ordered, predicted_speedup,
                                       saturation_point, spearman,
                                       time_to_first_improvement, value_at_time)

CURVE = [(0.0, 10.0), (10.0, 9.0), (20.0, 9.0), (30.0, 8.5)]


def test_best_so_far_is_monotone_and_ignores_regressions():
    pts = [(0.0, 10.0), (5.0, 11.0), (10.0, 9.0), (15.0, 9.5)]
    c = best_so_far(pts)
    assert [v for _, v in c] == [10.0, 10.0, 9.0, 9.0]


def test_best_so_far_sorts_by_time():
    c = best_so_far([(10.0, 9.0), (0.0, 10.0)])
    assert c[0][0] == 0.0


def test_auc_uses_a_common_horizon_so_a_short_arm_is_not_rewarded():
    # A slow arm whose curve ends early must still be integrated to the horizon.
    short = [(0.0, 10.0), (5.0, 9.0)]
    long = [(0.0, 10.0), (5.0, 9.0), (100.0, 9.0)]
    assert auc(short, horizon_s=100.0) == pytest.approx(auc(long, horizon_s=100.0))


def test_auc_is_zero_when_nothing_improves():
    assert auc([(0.0, 10.0), (50.0, 10.0)], horizon_s=100.0) == 0.0


def test_time_to_first_improvement():
    assert time_to_first_improvement(CURVE) == 10.0
    assert time_to_first_improvement([(0.0, 10.0), (9.0, 10.0)]) is None


def test_iso_time_comparison():
    assert value_at_time(CURVE, 25.0) == 9.0
    assert value_at_time(CURVE, 0.0) == 10.0
    assert value_at_time(CURVE, -1.0) is None


def test_saturation_point_is_where_the_search_stops_paying():
    t, v = saturation_point(CURVE)
    assert (t, v) == (30.0, 8.5)
    flat = [(0.0, 5.0), (10.0, 5.0), (20.0, 5.0)]
    assert saturation_point(flat) == (0.0, 5.0)


def test_bootstrap_interval_brackets_the_point_estimate():
    ci = bootstrap([1.0, 2.0, 3.0, 4.0, 5.0], resamples=2000, seed=1)
    assert ci.lo <= ci.point <= ci.hi and ci.n == 5


def test_bootstrap_is_reproducible():
    a = bootstrap([1.0, 5.0, 2.0, 8.0], resamples=1000, seed=7)
    b = bootstrap([1.0, 5.0, 2.0, 8.0], resamples=1000, seed=7)
    assert (a.lo, a.hi) == (b.lo, b.hi)


def test_bootstrap_single_value_is_degenerate_not_a_crash():
    ci = bootstrap([3.0])
    assert ci.point == 3.0 and ci.n == 1


def test_effect_size_ci_detects_a_real_difference():
    d = bootstrap_difference([10.0] * 8, [1.0] * 8, resamples=2000, seed=2)
    assert d.point == pytest.approx(9.0) and d.excludes_zero


def test_effect_size_ci_does_not_claim_a_difference_that_is_not_there():
    d = bootstrap_difference([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], resamples=2000, seed=3)
    assert not d.excludes_zero


def test_spearman_perfect_and_inverse():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_needs_enough_points():
    assert spearman([1, 2], [1, 2]) is None


def test_h1_is_tested_over_the_whole_ladder_not_endpoints():
    # Monotone decreasing across the ordered ladder.
    assert is_monotone_decreasing([100.0, 50.0, 20.0, 5.0])
    assert kendall_tau_ordered([100.0, 50.0, 20.0, 5.0]) == pytest.approx(1.0)
    # A non-monotone ladder must not read as a clean trend even if the
    # endpoints alone would suggest one.
    assert not is_monotone_decreasing([100.0, 20.0, 50.0, 5.0])
    assert kendall_tau_ordered([100.0, 20.0, 50.0, 5.0]) < 1.0


def test_gate_rejection_rate_excludes_undecided():
    """H5 is about refutations. An undecided partition is not a rejection."""
    s = ArmSummary(arm="I(0)", design="d", model="m", delay_s=0.0,
                   iterations=10, lec_refuted=2, lec_undecided=5)
    assert s.gate_rejection_rate == pytest.approx(2 / 5)


def test_improvements_per_hour():
    s = ArmSummary(arm="I(0)", design="d", model="m", delay_s=0.0,
                   verified_improvements=3, wall_s=1800.0)
    assert s.improvements_per_hour == pytest.approx(6.0)


def test_time_split_sums_to_one():
    s = ArmSummary(arm="I(600)", design="d", model="m", delay_s=600.0,
                   llm_s=10.0, tool_s=5.0, gate_s=5.0, injected_s=600.0)
    assert sum(s.time_split.values()) == pytest.approx(1.0)
    assert s.time_split["injected"] > 0.9


def test_predicted_speedup_bounds_the_claim():
    """If the model's turn dominates, no evaluator speedup can move much."""
    assert predicted_speedup(t_llm_s=90.0, t_tool_fast_s=1.0,
                             t_tool_slow_s=46.0) == pytest.approx(136 / 91)
    # A fast seat exposes tool time and the achievable speedup grows.
    fast_seat = predicted_speedup(2.0, 1.0, 46.0)
    slow_seat = predicted_speedup(90.0, 1.0, 46.0)
    assert fast_seat > slow_seat


def test_proxy_vs_judge_refuses_a_timing_basis_mismatch(tmp_path):
    """A clock-constrained arm and an unconstrained judge measure different
    physical quantities. Correlating them produced rho from -0.64 to +0.92 across
    arms of one real sweep, numbers that looked like a result and meant nothing.
    """
    import json

    import pytest

    from livelane.db.store import VariantStore
    from livelane.judge import _same_timing_basis, proxy_vs_judge

    s = VariantStore(tmp_path / "t.db", verbose=False)
    run = s.start_run("r", design="d", lane="S", arm="I(0)", model="m", seed=0,
                      note=json.dumps({"clock_port": "clk"}))
    for i, (q, j) in enumerate([(10.0, 1.0), (9.0, 2.0), (8.0, 3.0)]):
        s.add_variant(run, iteration_index=i, accepted=1, qor_max_delay_ns=q,
                      judge_max_delay_ns=j,
                      judge_json=json.dumps({"config": {"clock_port": None}}))
    assert _same_timing_basis(s, "r") is False
    with pytest.raises(ValueError, match="different timing bases"):
        proxy_vs_judge(s, "r")
    s.close()


def test_proxy_vs_judge_allows_a_matching_basis(tmp_path):
    import json

    from livelane.db.store import VariantStore
    from livelane.judge import _same_timing_basis, proxy_vs_judge

    s = VariantStore(tmp_path / "t.db", verbose=False)
    run = s.start_run("r", design="d", lane="S", arm="I(0)", model="m", seed=0,
                      note=json.dumps({"clock_port": "clk"}))
    for i, (q, j) in enumerate([(10.0, 10.0), (9.0, 9.0), (8.0, 8.0)]):
        s.add_variant(run, iteration_index=i, accepted=1, qor_max_delay_ns=q,
                      judge_max_delay_ns=j,
                      judge_json=json.dumps({"config": {"clock_port": "clk"}}))
    assert _same_timing_basis(s, "r") is True
    assert proxy_vs_judge(s, "r") == pytest.approx(1.0)
    s.close()


def test_proxy_vs_judge_refuses_a_front_end_mismatch(tmp_path):
    """A different RTL front end is a different measurement, not a nuance.

    On picorv32, same recipe, same Liberty, same clock, read_slang gives
    6563 cells / 12.7612 ns and read_verilog -sv gives 6691 / 14.7771 ns. That
    is 15.8% on the critical path, larger than most improvements the arms
    report, so correlating a judge that used one front end against arms that
    used the other compares two different quantities.
    """
    import json

    from livelane.db.store import VariantStore
    from livelane.judge import _same_timing_basis, proxy_vs_judge

    s = VariantStore(tmp_path / "t.db", verbose=False)
    run = s.start_run("r", design="d", lane="S", arm="I(0)", model="m", seed=0,
                      note=json.dumps({"clock_port": "clk",
                                       "read_cmd": "read_slang",
                                       "script": "baseline-2026-09-02"}))
    for i, (q, j) in enumerate([(10.0, 10.0), (9.0, 9.0), (8.0, 8.0)]):
        s.add_variant(run, iteration_index=i, accepted=1, qor_max_delay_ns=q,
                      judge_max_delay_ns=j,
                      judge_json=json.dumps({"config": {
                          "clock_port": "clk",
                          "read_cmd": "read_verilog -sv",
                          "script": "baseline-2026-09-02"}}))
    assert _same_timing_basis(s, "r") is False
    with pytest.raises(ValueError, match="different timing bases"):
        proxy_vs_judge(s, "r")
    s.close()


def test_same_timing_basis_tolerates_an_unrecorded_front_end(tmp_path):
    """Missing must not become 'unknown': None does not block, only False does.

    Runs recorded before the evaluator config was persisted carry a clock but no
    front end. Downgrading those to None would weaken the guard on exactly the
    runs it exists to protect, so a recorded clock match still returns True.
    """
    import json

    from livelane.db.store import VariantStore
    from livelane.judge import _same_timing_basis

    s = VariantStore(tmp_path / "t.db", verbose=False)
    run = s.start_run("r", design="d", lane="S", arm="I(0)", model="m", seed=0,
                      note=json.dumps({"clock_port": "clk"}))
    s.add_variant(run, iteration_index=0, accepted=1, qor_max_delay_ns=10.0,
                  judge_max_delay_ns=10.0,
                  judge_json=json.dumps({"config": {"clock_port": "clk",
                                                    "read_cmd": "read_slang"}}))
    assert _same_timing_basis(s, "r") is True
    s.close()
