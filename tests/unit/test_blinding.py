"""The agent must not be able to tell which lane it is in.

This is the experiment's central control. If it fails, every arm comparison is
confounded, so it is tested rather than assumed.
"""

from livelane.db.store import VariantStore
from livelane.state.reports import (LANE_REVEALING_FIELDS, CandidateVerdict,
                                    FunctionalResult, LecVerdict, QorReport,
                                    TimingReport)


def _lane_s_qor() -> QorReport:
    return QorReport(top="t", cells=6691, area_um2=75663.8, max_delay_ns=9.4,
                     source="yosys+abc[baseline-2026-09-02]", wall_s=46.02,
                     cpu_s=52.1, peak_rss_kb=3_874_000, raw_json='{"yosys":1}')


def _lane_l_qor() -> QorReport:
    return QorReport(top="t", cells=6691, area_um2=75663.8, max_delay_ns=9.4,
                     source="lhd", wall_s=39.59, cpu_s=44.4,
                     peak_rss_kb=2_699_000, raw_json='{"lhd":1}')


def test_lanes_emit_identical_agent_views():
    assert _lane_s_qor().to_agent_dict() == _lane_l_qor().to_agent_dict()


def test_no_lane_revealing_field_reaches_the_agent():
    for r in (_lane_s_qor(), _lane_l_qor()):
        leaked = set(r.to_agent_dict()) & LANE_REVEALING_FIELDS
        assert not leaked, f"leaked {leaked}"


def test_ledger_keeps_what_the_agent_view_drops():
    r = _lane_s_qor()
    assert r.to_record()["source"] == "yosys+abc[baseline-2026-09-02]"
    assert r.to_record()["wall_s"] == 46.02


def test_nested_reports_are_scrubbed_too():
    cv = CandidateVerdict(
        accepted=True, stage="done",
        functional=FunctionalResult(passed=True, wall_s=3.0),
        lec=LecVerdict("proven", backend="eqy", wall_s=11.5),
        qor=_lane_s_qor(), variant_id=7,
        observed_latency_s=632.8, injected_delay_s=600.0)
    d = cv.to_agent_dict()
    for k in ("variant_id", "injected_delay_s", "observed_latency_s"):
        assert k not in d
    assert "backend" not in d["lec"] and "source" not in d["qor"]
    assert d["lec"]["verdict"] == "proven"


def test_history_view_leaks_nothing(tmp_path):
    s = VariantStore(tmp_path / "t.db", verbose=False)
    run = s.start_run("r", design="d", lane="L", arm="L", model="m", seed=0)
    s.add_variant(run, iteration_index=0, wall_offset_s=0.0, accepted=1,
                  qor_cells=1, qor_area_um2=2.0, qor_max_delay_ns=3.0,
                  qor_source="lhd", lec_backend="eqy")
    h = s.history("r", limit=5)
    assert h, "history returned nothing"
    forbidden = {"wall_offset_s", "qor_source", "lec_backend", "injected_delay_s"}
    assert not (set(h[0]) & forbidden)
    s.close()


def test_timing_report_blinding():
    a = TimingReport(top="t", slack_ns=0.5, max_delay_ns=9.4, source="opensta",
                     wall_s=0.26, raw_json="{}")
    b = TimingReport(top="t", slack_ns=0.5, max_delay_ns=9.4, source="lhd",
                     wall_s=0.01, raw_json="{}")
    assert a.to_agent_dict() == b.to_agent_dict()


def test_lane_configs_are_explicit_not_defaulted():
    """A silent config asymmetry between lanes invalidates every comparison.

    Lane L's `pass.abc.flatten` defaulted to `auto` in LiveHD, which partitioned
    DecodeUnit into 51 regions and produced a 16x worse critical path than lane
    S's flattened mapping, an artifact that looked exactly like a tool
    difference. Both lanes must now declare their configuration.
    """
    from livelane.evaluators import LaneLEvaluator, LaneSEvaluator
    from livelane.nodes.yosys_sta import BASELINE

    s = LaneSEvaluator(yosys="y", sta="s", liberty="l", script=BASELINE)
    l = LaneLEvaluator(binary="b", liberty="l")
    assert s.config()["lane"] == "S" and l.config()["lane"] == "L"
    # Lane L must default to the SAME mapping regime as lane S, not to LiveHD's
    # partitioned default.
    assert l.config()["pass.abc.flatten"] == "true"
    assert l.config()["pass.abc.delay_ps"] == 10000
