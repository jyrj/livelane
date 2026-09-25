"""The plotting layer, tested for the properties that silently corrupt a figure.

``figures.py`` had no coverage here at all, which is how a module-level ``NameError``
on every path through Figure 1 survived long enough to ship alongside three stale
PNGs that looked like proof it worked.  File existence is not evidence, and neither
is a rendered picture: the numbers underneath it can be wrong while the axes look
perfect.  So these tests assert the *refusals*, the cases where the honest answer
is "no number", rather than re-checking arithmetic that lives in ``metrics``.
"""

from __future__ import annotations

import warnings

import pytest

# matplotlib is in the `analysis` extra, not a core dependency. Without it
# a plain `pip install -e .` made the WHOLE suite fail at collection;
# skipping this module with a reason keeps the rest of the suite honest.
matplotlib = pytest.importorskip("matplotlib")

matplotlib.use("Agg")

from livelane.analysis import figures as F
from livelane.db.store import VariantStore


@pytest.fixture
def store():
    s = VariantStore(":memory:", verbose=False)
    yield s
    s.close()


def _run(store: VariantStore, run_id: str, **kw):
    kw.setdefault("design", "Alu")
    kw.setdefault("lane", "S")
    kw.setdefault("arm", "I(0)")
    kw.setdefault("model", "m")
    kw.setdefault("seed", 1)
    return store.start_run(run_id, **kw)


# --- the module must import at all (the regression that started this) --------


def test_module_exposes_the_fixed_ladder():
    # DELAY_LADDER is referenced by _delay_colours; when it was missing, every
    # path through Figure 1 raised NameError and nothing in the suite noticed.
    assert F.DELAY_LADDER == (0.0, 30.0, 120.0, 600.0)
    assert F._delay_colours([0.0]) == {0.0: "#0072B2"}


# --- colour ------------------------------------------------------------------


def test_ladder_colours_do_not_shift_when_a_figure_omits_an_arm():
    full = F._delay_colours([0.0, 30.0, 120.0, 600.0])
    assert F._delay_colours([0.0, 600.0]) == {0.0: full[0.0], 600.0: full[600.0]}


def test_every_ladder_arm_gets_a_distinct_colour():
    full = F._delay_colours(F.DELAY_LADDER)
    assert len(set(full.values())) == len(F.DELAY_LADDER)


def test_an_off_ladder_delay_never_borrows_a_ladder_colour():
    # Two arms sharing a colour on one set of axes is a misread, not a blemish.
    c = F._delay_colours([0.0, 30.0, 120.0, 600.0, 45.0])
    assert len(set(c.values())) == len(c)
    assert c[45.0] not in F._delay_colours(F.DELAY_LADDER).values()


def test_exhausting_the_palette_warns_rather_than_colliding_silently(caplog):
    F._delay_colours([0.0, 45.0, 90.0, 180.0])
    assert "REUSES a colour" in caplog.text


# --- the baseline: the number that must not be invented ----------------------


def test_seed_baseline_is_iteration_zero_and_nothing_else(store):
    """loop.seed() writes iteration 0 with a NULL QoR when the evaluation could
    not be parsed. Falling through to the next scored variant would re-base the
    run onto an *agent edit* and report the rest as improvements over the seed."""
    r = _run(store, "nullseed")
    store.add_variant(r, iteration_index=0, wall_offset_s=0.0, accepted=0,
                      functional_pass=None, lec_verdict="skipped",
                      qor_max_delay_ns=None, qor_area_um2=None,
                      status="eval-failed")
    for k, ns in ((1, 9.0), (2, 8.0)):
        store.add_variant(r, iteration_index=k, wall_offset_s=10.0 * k, accepted=1,
                          functional_pass=1, lec_verdict="proven",
                          qor_max_delay_ns=ns, qor_area_um2=1000.0)

    assert F._seed_baseline(store, "nullseed") is None
    c = F.verified_improvements(store, "nullseed")
    assert c.no_baseline is True and c.verified == 0


def test_a_scored_seed_still_counts_its_improvements(store):
    r = _run(store, "ok")
    store.add_variant(r, iteration_index=0, wall_offset_s=0.0, accepted=1,
                      functional_pass=1, lec_verdict="skipped",
                      qor_max_delay_ns=10.0, qor_area_um2=1000.0)
    for k, ns in ((1, 9.0), (2, 8.0)):
        store.add_variant(r, iteration_index=k, wall_offset_s=10.0 * k, accepted=1,
                          functional_pass=1, lec_verdict="proven",
                          qor_max_delay_ns=ns, qor_area_um2=1000.0)
    assert F.verified_improvements(store, "ok").verified == 2


# --- the gate must fail closed -----------------------------------------------


@pytest.mark.parametrize("bad,why", [
    ({"functional_pass": None, "lec_verdict": "proven"}, "no oracle record"),
    ({"functional_pass": 0, "lec_verdict": "proven"}, "oracle failed"),
    ({"functional_pass": 1, "lec_verdict": None}, "no verdict"),
    ({"functional_pass": 1, "lec_verdict": "error"}, "gate errored"),
    ({"functional_pass": 1, "lec_verdict": "refuted"}, "not equivalent"),
])
def test_a_better_qor_is_refused_without_both_gates(store, bad, why):
    r = _run(store, "gate")
    store.add_variant(r, iteration_index=0, wall_offset_s=0.0, accepted=1,
                      functional_pass=1, lec_verdict="skipped",
                      qor_max_delay_ns=10.0, qor_area_um2=1000.0)
    store.add_variant(r, iteration_index=1, wall_offset_s=10.0, accepted=1,
                      qor_max_delay_ns=5.0, qor_area_um2=1000.0, **bad)
    assert F.verified_improvements(store, "gate").verified == 0, why


def test_the_area_guardrail_is_enforced(store):
    r = _run(store, "area")
    store.add_variant(r, iteration_index=0, wall_offset_s=0.0, accepted=1,
                      functional_pass=1, lec_verdict="skipped",
                      qor_max_delay_ns=10.0, qor_area_um2=1000.0)
    store.add_variant(r, iteration_index=1, wall_offset_s=10.0, accepted=1,
                      functional_pass=1, lec_verdict="proven",
                      qor_max_delay_ns=9.0, qor_area_um2=1500.0)
    assert F.verified_improvements(store, "area").verified == 0


# --- exclusions --------------------------------------------------------------


def test_virtual_delay_runs_never_reach_a_latency_figure(store):
    _run(store, "replay", arm_delay_s=600.0, arm_delay_virtual=True)
    assert F.load_runs(store, None) == []
    assert [r.run_id for r in F.load_runs(store, None, include_virtual=True)] == ["replay"]


def test_interrupted_delays_are_counted_and_excluded_never_averaged(store):
    r = _run(store, "intr", arm_delay_s=30.0)
    store.add_iteration(r, iteration_index=1, wall_offset_s=10.0,
                        t_llm_ms=2000.0, t_delay_ms=30000.0, injected_delay_s=30.0)
    store.add_iteration(r, iteration_index=2, wall_offset_s=20.0,
                        t_llm_ms=9999.0, t_delay_ms=100.0, injected_delay_s=30.0,
                        delay_interrupted=1)
    t = F.time_split(store, F.load_runs(store, None)[0])
    assert (t.iterations, t.interrupted, t.llm_s) == (1, 1, 2.0)


def test_a_run_that_persisted_nothing_has_no_wall_clock(store):
    _run(store, "crashed")
    assert F._run_wall_s(store, "crashed") is None
    assert F.load_runs(store, None)[0].wall_s is None


def test_an_unknown_run_id_is_skipped_not_raised(store, caplog):
    assert F.load_runs(store, ["nope"]) == []
    assert "not in" in caplog.text


# --- metric allow-list -------------------------------------------------------


@pytest.mark.parametrize("metric", ["qor_bogus", "area; DROP TABLE variants"])
def test_an_unknown_metric_never_reaches_the_database(store, metric):
    with pytest.raises(ValueError):
        F.accepted_curve(store, "any", metric)


def test_only_accepted_variants_enter_the_curve(store):
    r = _run(store, "acc")
    store.add_variant(r, iteration_index=0, wall_offset_s=0.0, accepted=1,
                      qor_max_delay_ns=10.0)
    store.add_variant(r, iteration_index=1, wall_offset_s=10.0, accepted=0,
                      lec_verdict="refuted", qor_max_delay_ns=1.0)
    assert F.accepted_curve(store, "acc") == [(0.0, 10.0)]


def test_slack_is_plotted_as_higher_is_better(store):
    r = _run(store, "slack")
    for k, sl in enumerate((-1.0, -0.5, -0.8), start=1):
        store.add_variant(r, iteration_index=k, wall_offset_s=10.0 * k, accepted=1,
                          qor_slack_ns=sl)
    assert [v for _, v in F.accepted_curve(store, "slack", "qor_slack_ns")] == \
        [-1.0, -0.5, -0.5]


# --- rendering ---------------------------------------------------------------


def test_render_all_on_the_synthetic_store(tmp_path):
    s = F.build_demo_store(delays=(0.0, 600.0), models=("m",), seeds=(1,),
                           budget_iters=6)
    try:
        paths = F.render_all(s, None, tmp_path, tag="t")
    finally:
        s.close()
    assert len(paths) == 3 and all(p.stat().st_size > 20_000 for p in paths)


def test_an_empty_database_draws_empty_frames_not_invented_data(store, tmp_path,
                                                                caplog):
    paths = F.render_all(store, None, tmp_path)
    assert all(p.exists() for p in paths)
    assert "no run had plottable data" in caplog.text


def test_an_arm_with_no_measured_clock_renders_without_a_singular_axis(store, tmp_path):
    # An all-zero-clock arm used to hand matplotlib xlim(0, 0); it rescaled the
    # axis itself and warned, so the bar was drawn against an invented range.
    r = _run(store, "zero")
    v = store.add_variant(r, iteration_index=1, wall_offset_s=5.0, accepted=1,
                          qor_max_delay_ns=9.0)
    store.add_iteration(r, iteration_index=1, variant_id=v, wall_offset_s=5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert F.fig2_time_split(store, None, tmp_path / "z.png").exists()
