"""End-to-end tests against real yosys/ABC/OpenSTA.

Named ``*_live`` to match ``chia/vlsi/tests/test_hammer_live.py``: these need
tools, and skip cleanly when they are absent instead of failing.

The Liberty and design in ``smoke/`` are small fixtures, so the test runs in
seconds wherever yosys and OpenSTA are installed.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from chia_livelane.vlsi.yosys_sta import YosysStaNode, yosys_sta_qor

SMOKE = Path(__file__).parent / "smoke"
LIB = SMOKE / "smoke.lib"
SRC = SMOKE / "smoke.v"
TOP = "smoke_top"

pytestmark = pytest.mark.skipif(
    shutil.which("yosys") is None or shutil.which("sta") is None,
    reason="yosys and/or sta (OpenSTA) not on PATH")


@pytest.fixture
def node(tmp_path):
    return YosysStaNode(yosys="yosys", sta="sta", liberty=str(LIB),
                        workdir=tmp_path, clock_period_ns=10.0,
                        clock_port="clk", verbose=False)


class TestRealFlow:
    def test_constrained_evaluation_reports_every_number(self, node):
        r = node.evaluate([str(SRC)], TOP)
        assert r.success is True, r.message
        assert r.cells == 5, r.cells
        assert r.area_um2 == pytest.approx(13.0)
        assert r.max_delay_ns == pytest.approx(0.16, abs=1e-3)
        assert r.slack_ns == pytest.approx(9.82, abs=1e-3)
        assert r.constrained is True

    def test_cost_is_measured_not_guessed(self, node):
        r = node.evaluate([str(SRC)], TOP)
        assert r.wall_s > 0.0
        assert r.cpu_s > 0.0
        assert r.peak_rss_kb > 0

    def test_critical_path_names_real_cells(self, node):
        r = node.evaluate([str(SRC)], TOP)
        # Scalars alone are unactionable; the path block is what says WHICH
        # logic to change.
        assert "Startpoint:" in r.path_summary
        assert "DFF" in r.path_summary

    def test_unconstrained_run_says_so_instead_of_reporting_a_small_number(
            self, tmp_path):
        # Measured on picorv32/sky130: 12.7612 ns constrained at a 10 ns clock
        # vs 0.1959 ns unconstrained. Anything optimising the second number is
        # optimising noise, so an unconstrained report has to announce itself.
        n = YosysStaNode(yosys="yosys", sta="sta", liberty=str(LIB),
                         workdir=tmp_path, verbose=False)
        r = n.evaluate([str(SRC)], TOP)
        assert r.constrained is False
        assert "UNCONSTRAINED" in r.message

    def test_both_recipes_run_and_are_recorded(self, tmp_path):
        from chia_livelane.vlsi.yosys_sta import SCRIPTS
        for name in SCRIPTS:
            n = YosysStaNode(yosys="yosys", sta="sta", liberty=str(LIB),
                             workdir=tmp_path / name, script=SCRIPTS[name],
                             clock_period_ns=10.0, clock_port="clk",
                             verbose=False)
            r = n.evaluate([str(SRC)], TOP)
            assert r.success is True, (name, r.message)
            # Every measurement records the recipe that produced it, so two
            # numbers are never compared across recipes by accident.
            assert r.script == name

    def test_relative_paths_work(self, tmp_path, monkeypatch):
        # yosys and sta run with cwd=workdir, so relative source and Liberty
        # paths used to resolve against the wrong directory: yosys exited 1
        # with an empty stat file and the node reported cells=None, which reads
        # as "this design has no cells" rather than "you gave me a bad path".
        shutil.copy(SRC, tmp_path / "smoke.v")
        shutil.copy(LIB, tmp_path / "smoke.lib")
        monkeypatch.chdir(tmp_path)
        n = YosysStaNode(yosys="yosys", sta="sta", liberty="smoke.lib",
                         workdir="wd", clock_period_ns=10.0, clock_port="clk",
                         verbose=False)
        r = n.evaluate(["smoke.v"], TOP)
        assert r.success is True, r.message
        assert r.cells == 5

    def test_node_function_returns_a_json_safe_dict(self, tmp_path):
        import json
        fn = getattr(yosys_sta_qor, "_chia_original", yosys_sta_qor)
        d = fn([str(SRC)], TOP, str(LIB), workdir=str(tmp_path),
               clock_period_ns=10.0, clock_port="clk")
        assert d["success"] is True
        assert json.loads(json.dumps(d))["cells"] == 5

    def test_missing_source_is_named(self, node, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            node.evaluate([str(tmp_path / "absent.v")], TOP)
