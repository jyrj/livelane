"""Tool-output parsers, tested against REAL captured output.

Every fixture here was copied from an actual run on this host. A parser that
silently returns None turns into a missing data point in a plot, so the
"unparseable input yields None, never a fabricated number" property is tested
explicitly.
"""

from livelane.nodes.lhd import LhdEnvelope
from livelane.nodes.yosys_sta import parse_sta_report, parse_yosys_stat

# `stat -liberty` uses a DIFFERENT format from plain `stat`. Lane S always passes
# -liberty, so this is the format that actually matters.
LIBERTY_STAT = """
=== Alu ===
     7841        - wires
     7692 4.32E+04 cells
       43  376.611   sky130_fd_sc_hd__a2111oi_0
   Chip area for module '\\Alu': 43193.926400
"""
PLAIN_STAT = """
   Number of cells:               6691
   Chip area for module '\\picorv32': 75663.817600
"""


def test_liberty_stat_format():
    assert parse_yosys_stat(LIBERTY_STAT) == (7692, 43193.9264)


def test_plain_stat_format():
    cells, area = parse_yosys_stat(PLAIN_STAT)
    assert cells == 6691 and abs(area - 75663.8176) < 1e-6


def test_unparseable_stat_returns_none_not_a_guess():
    assert parse_yosys_stat("nothing useful here") == (None, None)


STA_OUT = """
Startpoint: cpuregs_reg_1_ (rising edge-triggered flip-flop clocked by clk)
Endpoint: mem_addr_reg_7_ (rising edge-triggered flip-flop clocked by clk)
             9.4210   data arrival time
             0.5790   slack (MET)
"""


def test_sta_report():
    p = parse_sta_report(STA_OUT)
    assert abs(p["slack_ns"] - 0.579) < 1e-9
    assert abs(p["max_delay_ns"] - 9.421) < 1e-9
    assert p["endpoint"] == "mem_addr_reg_7_" and p["met"] is True


def test_sta_worst_of_n_slack_wins():
    multi = STA_OUT + STA_OUT.replace("0.5790   slack (MET)",
                                      "-3.0000   slack (VIOLATED)")
    p = parse_sta_report(multi)
    assert abs(p["slack_ns"] + 3.0) < 1e-9 and p["met"] is False


def test_lhd_envelope_prefers_whole_design_sta_over_abc_estimate():
    env = LhdEnvelope(raw={
        "status": "pass",
        "qor": {"kind": "synth",
                "abc": {"total": {"gates": 100, "area": 1000.0, "max_delay": 9.1}},
                "sta": {"designs": [{"module": "top", "max_delay": 9.42,
                                     "critical_pin": "u/Z"}]}}})
    q = env.to_qor_report("top")
    assert q.max_delay_ns == 9.42


def test_lhd_unknown_lec_verdict_never_reads_as_a_pass():
    for raw, want in (("proven", "proven"), ("refuted", "refuted"),
                      ("unknown", "error"), ("nonsense", "error")):
        env = LhdEnvelope(raw={"status": "pass", "lec": {"verdict": raw}})
        assert env.to_lec_verdict().verdict == want


def test_systemverilog_has_no_compile_tier():
    """LiveHD's compile cache is Pyrope-only (lhd_kernel_compile.cpp:1736)."""
    env = LhdEnvelope(raw={"status": "pass",
                           "incremental": {"abc": {"enabled": True, "hits": 3}}})
    assert env.tier("compile") == {}
    assert env.compile_cache_engaged is False


def test_yosys_lane_workdir_is_absolute(tmp_path, monkeypatch):
    """Tools run with cwd=workdir, so a relative workdir makes the generated
    script path relative to itself and yosys cannot open it."""
    import os
    from livelane.nodes.yosys_sta import YosysStaLane
    lib = tmp_path / "x.lib"
    lib.write_text("/* liberty */")
    monkeypatch.chdir(tmp_path)
    lane = YosysStaLane(yosys="yosys", sta="sta", liberty=str(lib),
                        workdir="relative/dir")
    assert lane.workdir.is_absolute(), lane.workdir
