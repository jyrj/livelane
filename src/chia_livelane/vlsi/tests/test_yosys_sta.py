"""Unit tests for the open-source QoR node's parsers and guards.

Every number this node reports is read out of tool text, so the parsers are the
correctness surface: a silently-wrong area or delay corrupts every comparison
downstream and nothing else in the flow would notice.
"""

import json

import pytest

from chia_livelane.vlsi.yosys_sta import (BASELINE, SCRIPTS, TUNED, QorReport,
                                          YosysStaNode,
                                          annotate_path_with_sources,
                                          load_cell_sources, parse_sta_report,
                                          parse_yosys_stat, yosys_sta_qor)

# Real captured `stat -liberty` output, 2026-09-03, XiangShan Alu on sky130.
# This is the format the node actually produces; the plain "Number of cells:"
# line does not appear in it at all.
LIBERTY_STAT = """
7. Printing statistics.

=== Alu ===

        +----------Local Count, excluding submodules.
        |        +-Local Area, excluding submodules.
     7841        - wires
    23642        - wire bits
       12        - ports
     7692 4.32E+04 cells
       43  376.611   sky130_fd_sc_hd__a2111oi_0
      113  989.699   sky130_fd_sc_hd__xor2_1

   Chip area for module '\\Alu': 43193.926400
     of which used for sequential elements: 0.000000 (0.00%)
"""

PLAIN_STAT = """
=== picorv32 ===
   Number of wires:               3000
   Number of cells:               9174
=== design hierarchy ===
   Number of cells:               6691
   Chip area for module '\\picorv32': 75664.115200
"""

STA_REPORT = """
Startpoint: cpuregs_reg_1_ (rising edge-triggered flip-flop clocked by clk)
Endpoint: mem_addr_reg_7_ (rising edge-triggered flip-flop clocked by clk)
Path Group: clk
Path Type: max

   0.0000    0.0000 ^ cpuregs_reg_1_/CLK (sky130_fd_sc_hd__dfxtp_1)
   4.2100    9.4210 ^ mem_addr_reg_7_/D (sky130_fd_sc_hd__dfxtp_1)
             9.4210   data arrival time

            10.0000   data required time
            -9.4210   data arrival time
  ---------------------------------------
             0.5790   slack (MET)
"""


class TestStatParsing:
    def test_liberty_format(self):
        cells, area = parse_yosys_stat(LIBERTY_STAT)
        assert cells == 7692
        assert area == pytest.approx(43193.9264)

    def test_plain_format_takes_the_last_block(self):
        # With hierarchy preserved, stat prints one block per module and the
        # top-level summary comes last.
        cells, area = parse_yosys_stat(PLAIN_STAT)
        assert cells == 6691
        assert area == pytest.approx(75664.1152)

    def test_unparseable_never_fabricates_a_number(self):
        assert parse_yosys_stat("no numbers here") == (None, None)

    def test_area_is_full_precision_not_the_rounded_column(self):
        # The inline column says 4.32E+04; the Chip-area line says 43193.9264.
        _, area = parse_yosys_stat(LIBERTY_STAT)
        assert area != pytest.approx(43200.0)


class TestStaParsing:
    def test_slack_and_arrival(self):
        p = parse_sta_report(STA_REPORT)
        assert p["slack_ns"] == pytest.approx(0.579)
        assert p["max_delay_ns"] == pytest.approx(9.421)
        assert p["met"] is True

    def test_endpoints(self):
        p = parse_sta_report(STA_REPORT)
        assert p["startpoint"] == "cpuregs_reg_1_"
        assert p["endpoint"] == "mem_addr_reg_7_"

    def test_negative_slack(self):
        viol = STA_REPORT.replace("0.5790   slack (MET)",
                                  "-1.2500   slack (VIOLATED)")
        p = parse_sta_report(viol)
        assert p["slack_ns"] == pytest.approx(-1.25)
        assert p["met"] is False

    def test_worst_of_n_paths_wins(self):
        multi = STA_REPORT + STA_REPORT.replace("0.5790   slack (MET)",
                                                "-3.0000   slack (VIOLATED)")
        assert parse_sta_report(multi)["slack_ns"] == pytest.approx(-3.0)

    def test_path_summary_is_carried(self):
        # Scalars alone are unactionable: post-synthesis endpoint names are
        # mangled, so the node carries the path block that names the cells.
        p = parse_sta_report(STA_REPORT)
        assert "Startpoint:" in str(p["path_summary"])

    def test_empty_report_yields_nothing(self):
        assert parse_sta_report("") == {}


class TestCellSourceRecovery:
    def test_maps_cell_to_rtl_line(self, tmp_path):
        j = tmp_path / "cells.json"
        j.write_text(json.dumps({"modules": {"top": {"cells": {
            "_05679_": {"attributes": {"src": "picorv32.v:1122|other.v:3"}}}}}}))
        assert load_cell_sources(j) == {"_05679_": "picorv32.v:1122"}

    def test_annotates_a_path_line(self):
        line = "   4.2100    9.4210 ^ _05679_/D (sky130_fd_sc_hd__dfxtp_1)"
        out = annotate_path_with_sources(line, {"_05679_": "picorv32.v:1122"})
        assert out.endswith("<- picorv32.v:1122")

    def test_unknown_cell_left_alone(self):
        line = "   4.2100    9.4210 ^ _99999_/D (sky130_fd_sc_hd__dfxtp_1)"
        assert annotate_path_with_sources(line, {"_05679_": "x.v:1"}) == line

    def test_missing_map_degrades_rather_than_raising(self, tmp_path):
        # A missing map costs RTL line numbers, nothing else. It must never
        # take down an evaluation.
        assert load_cell_sources(tmp_path / "nope.json") == {}

    def test_corrupt_map_degrades_rather_than_raising(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        assert load_cell_sources(bad) == {}


class TestSynthScripts:
    def test_both_recipes_registered(self):
        assert set(SCRIPTS) == {"baseline-flat", "tuned-hier"}

    def test_baseline_renders_its_abc_target_and_top(self):
        r = BASELINE.render(top="picorv32", lib="/pdk/sky130.lib",
                            netlist="/tmp/n.v", stat="/tmp/s.txt")
        assert "abc -D 10000 -liberty /pdk/sky130.lib" in r
        assert "-top picorv32" in r

    def test_stat_is_teed_to_a_file(self):
        # `yosys -q` prints no log at all, so parsing stdout returns nothing.
        r = BASELINE.render(top="t", lib="L", netlist="N", stat="/tmp/s.txt")
        assert "tee -o /tmp/s.txt stat" in r

    def test_tuned_preserves_hierarchy_and_uses_fast_abc(self):
        r = TUNED.render(top="T", lib="L", netlist="N", stat="S")
        assert "abc -fast" in r
        assert "synth -top" not in r          # no full flatten

    def test_every_recipe_names_its_rationale(self):
        for s in SCRIPTS.values():
            assert s.rationale.strip()


@pytest.fixture
def liberty(tmp_path):
    lib = tmp_path / "fake.lib"
    lib.write_text("library(fake){}")
    return lib


class TestNodeGuards:
    def test_missing_liberty_is_fatal_at_construction(self, tmp_path):
        # Two results are only comparable if they used the same Liberty, so a
        # missing one is fatal rather than a warning.
        with pytest.raises(FileNotFoundError):
            YosysStaNode(yosys="yosys", sta="sta", liberty="/does/not/exist.lib",
                         workdir=tmp_path)

    def test_workdir_is_absolutised(self, tmp_path, liberty, monkeypatch):
        # The tools run with cwd=workdir, so a relative workdir makes the
        # generated script path relative to itself and yosys cannot open it.
        monkeypatch.chdir(tmp_path)
        n = YosysStaNode(yosys="yosys", sta="sta", liberty=str(liberty),
                         workdir="rel", verbose=False)
        assert n.workdir.is_absolute()

    def test_empty_source_list_rejected(self, tmp_path, liberty):
        # Synthesising nothing reports cells=0 in milliseconds and looks like a
        # fast success.
        n = YosysStaNode(yosys="yosys", sta="sta", liberty=str(liberty),
                         workdir=tmp_path, verbose=False)
        with pytest.raises(ValueError):
            n.synthesize([], "top")

    def test_unknown_recipe_rejected(self, tmp_path, liberty):
        with pytest.raises(ValueError):
            yosys_sta_qor(["a.v"], "t", str(liberty), workdir=str(tmp_path),
                          script="does-not-exist")


class TestClockConstraint:
    def test_unconstrained_by_default(self, tmp_path, liberty):
        n = YosysStaNode(yosys="yosys", sta="sta", liberty=str(liberty),
                         workdir=tmp_path, verbose=False)
        assert n.constrained is False
        assert "set_max_delay" in n._sta_script(tmp_path / "n.v", "t")
        assert "create_clock" not in n._sta_script(tmp_path / "n.v", "t")

    def test_both_period_and_port_are_required(self, tmp_path, liberty):
        half = YosysStaNode(yosys="yosys", sta="sta", liberty=str(liberty),
                            workdir=tmp_path, clock_period_ns=10.0, verbose=False)
        assert half.constrained is False

    def test_constrained_emits_create_clock(self, tmp_path, liberty):
        n = YosysStaNode(yosys="yosys", sta="sta", liberty=str(liberty),
                         workdir=tmp_path, clock_period_ns=10.0, clock_port="clk",
                         verbose=False)
        assert n.constrained is True
        assert ("create_clock -name clk -period 10.0 [get_ports clk]"
                in n._sta_script(tmp_path / "n.v", "t"))

    def test_report_carries_the_constrained_flag(self):
        # Measured on picorv32/sky130: 12.7612 ns constrained at a 10 ns clock
        # vs 0.1959 ns unconstrained. Without this flag the two are
        # indistinguishable in a results table.
        assert QorReport(top="t", success=True).constrained is False
        assert QorReport(top="t", success=True, constrained=True).constrained


class TestReport:
    def test_as_dict_is_json_serialisable(self):
        d = QorReport(top="picorv32", success=True, cells=6691,
                      area_um2=75663.8176, max_delay_ns=12.7612).as_dict()
        assert json.loads(json.dumps(d))["cells"] == 6691

    def test_failure_carries_no_claim(self):
        r = QorReport(top="t", success=False, message="yosys rc=1")
        assert r.success is False and r.cells is None and r.area_um2 is None
