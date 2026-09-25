"""Helpers the multi-file loop relies on.

The loop shows the agent ONE file per iteration and applies its edit to ONE
file. Both choices are derived from an STA location string. Get the derivation
wrong and the loop keeps working, it just optimises the wrong module, or
edits a file nobody asked it to, so these are pinned rather than left to the
one end-to-end run that happened to exercise them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "chia"))

from loop import DESIGNS, _file_of  # noqa: E402


class TestFileOf:
    @pytest.mark.parametrize("loc,want", [
        ("../parent.v:1402.2", "parent.v"),
        ("../src/ibex_counter.sv:76.5", "ibex_counter.sv"),
        ("/abs/path/x.sv:12.3", "x.sv"),
        ("a/b/c.sv:1.2-3.4", "c.sv"),          # a line RANGE, not a point
        ("no-colon.sv", "no-colon.sv"),        # already bare
    ])
    def test_extracts_the_basename(self, loc, want):
        assert _file_of(loc) == want

    @pytest.mark.parametrize("loc", [None, "", 0])
    def test_absent_location_is_none(self, loc):
        # STA reports no RTL location when the endpoint is a cell with no `src`
        # attribute. That must select no focus file, so the caller falls back,
        # never raise, and never return a truthy nonsense name.
        assert _file_of(loc) is None

    def test_deep_relative_prefix_is_stripped(self):
        # yosys emits locations relative to the eqy workdir, which can be many
        # levels below the source tree.
        assert _file_of("../../../../../a/b/ibex_counter.sv:76.5") \
            == "ibex_counter.sv"


class TestDesignTable:
    def test_every_design_declares_a_top_and_clock(self):
        for name, d in DESIGNS.items():
            assert d.get("top"), f"{name} has no top module"
            assert d.get("clk"), f"{name} has no clock port"
            assert d.get("period"), f"{name} has no clock period"

    def test_each_design_is_single_file_or_multi_file_not_both(self):
        # `rtl` means "one editable file, fixed read command"; `tree` means
        # "resolve the source list from a checkout". Declaring both would make
        # which branch runs depend on statement order in the setup code.
        for name, d in DESIGNS.items():
            single, multi = "rtl" in d, "tree" in d
            assert single != multi, (
                f"{name} declares {'both' if single and multi else 'neither'} "
                f"`rtl` and `tree`")

    def test_single_file_designs_fix_their_front_end(self):
        for name, d in DESIGNS.items():
            if "rtl" in d:
                assert d.get("read_cmd"), f"{name} has no read_cmd"

    def test_multi_file_designs_restrict_what_the_agent_may_edit(self):
        # Without `editable_prefix` the agent could rewrite a vendored
        # primitive that is context, not a target, and the edit would be
        # applied to a file outside the design's own RTL.
        for name, d in DESIGNS.items():
            if "tree" in d:
                assert d.get("editable_prefix"), \
                    f"{name} does not bound the editable set"
                assert d.get("filelist") or d.get("rtl_glob"), \
                    f"{name} has no way to seed its source list"

    def test_multi_file_trees_are_relative_paths(self):
        # Absolute paths here would make a run depend on one machine's layout.
        for name, d in DESIGNS.items():
            if "tree" in d:
                assert not Path(d["tree"]).is_absolute(), \
                    f"{name} pins an absolute checkout path"
