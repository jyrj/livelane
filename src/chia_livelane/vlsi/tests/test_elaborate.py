"""What the elaboration resolver must and must not do.

The resolver exists under a soundness claim: it decides which sources, include
paths and defines a miter is built from. A mistake here does not look like a
crash, it looks like a proof. These tests pin the parts that would fail
silently.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from chia_livelane.vlsi.elaborate import (ASSERT_GUARDS, Elaboration, _declares,
                                          _pick, resolve, seed_sources)


class TestSeedSources:
    def test_filelist_wins_over_glob(self, tmp_path: Path) -> None:
        rtl = tmp_path / "rtl"
        rtl.mkdir()
        for n in ("a.sv", "b.sv", "poison.sv"):
            (rtl / n).write_text("// x\n")
        (rtl / "top.f").write_text("a.sv\nb.sv\n")
        got = seed_sources(tmp_path, "rtl/top.f", "rtl/*.sv")
        assert [p.name for p in got] == ["a.sv", "b.sv"]

    def test_filelist_comments_are_not_paths(self, tmp_path: Path) -> None:
        # A filelist line like `a.sv  // the core` names one file, not two.
        # Treating the comment as a path is how CVA6's list first broke.
        rtl = tmp_path / "rtl"
        rtl.mkdir()
        (rtl / "a.sv").write_text("// x\n")
        (rtl / "top.f").write_text("// header comment\n\na.sv // the core\n")
        assert [p.name for p in seed_sources(tmp_path, "rtl/top.f", "rtl/*.sv")] \
            == ["a.sv"]

    def test_falls_back_to_glob_when_filelist_absent(self, tmp_path: Path) -> None:
        rtl = tmp_path / "rtl"
        rtl.mkdir()
        (rtl / "a.sv").write_text("// x\n")
        assert [p.name for p in seed_sources(tmp_path, "rtl/nope.f", "rtl/*.sv")] \
            == ["a.sv"]

    def test_filelist_entries_that_do_not_exist_are_dropped(
            self, tmp_path: Path) -> None:
        # Stale filelists outlive the files they name; a missing entry must not
        # be handed to the front end as a path.
        rtl = tmp_path / "rtl"
        rtl.mkdir()
        (rtl / "a.sv").write_text("// x\n")
        (rtl / "top.f").write_text("a.sv\ngone.sv\n")
        assert [p.name for p in seed_sources(tmp_path, "rtl/top.f", None)] \
            == ["a.sv"]


class TestDeclarationSearch:
    def test_finds_module_and_package(self, tmp_path: Path) -> None:
        (tmp_path / "m.sv").write_text("module widget (input a);\nendmodule\n")
        (tmp_path / "p.sv").write_text("package pkg_x;\nendpackage\n")
        assert _declares(tmp_path, "module", "widget", set())[0].name == "m.sv"
        assert _declares(tmp_path, "package", "pkg_x", set())[0].name == "p.sv"

    def test_prefix_is_not_a_match(self, tmp_path: Path) -> None:
        # `module widget_extra` must not satisfy a search for `widget`, or the
        # source list silently acquires the wrong implementation.
        (tmp_path / "m.sv").write_text("module widget_extra (input a);\nendmodule\n")
        assert _declares(tmp_path, "module", "widget", set()) == []

    def test_instantiation_is_not_a_declaration(self, tmp_path: Path) -> None:
        (tmp_path / "u.sv").write_text(
            "module top;\n  widget u_widget (.a(a));\nendmodule\n")
        assert _declares(tmp_path, "module", "widget", set()) == []

    def test_excluded_files_are_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "m.sv"
        f.write_text("module widget;\nendmodule\n")
        assert _declares(tmp_path, "module", "widget", {f}) == []


class TestPickIsDeterministic:
    def test_same_input_same_choice(self, tmp_path: Path) -> None:
        hits = [tmp_path / "b" / "x.sv", tmp_path / "a" / "x.sv"]
        assert _pick(hits) == _pick(list(reversed(hits)))

    def test_simulation_model_preferred(self, tmp_path: Path) -> None:
        deep_sim = tmp_path / "examples" / "sim" / "rtl" / "cg.sv"
        shallow = tmp_path / "fpga" / "cg.sv"
        assert _pick([shallow, deep_sim]) == deep_sim

    def test_shallower_path_breaks_ties(self, tmp_path: Path) -> None:
        near = tmp_path / "a" / "cg.sv"
        far = tmp_path / "a" / "b" / "c" / "cg.sv"
        assert _pick([far, near]) == near


class TestElaborationRecord:
    def test_not_ok_without_sources(self) -> None:
        assert not Elaboration(error="boom").ok
        assert not Elaboration(sources=[]).ok

    def test_read_cmd_carries_includes_and_defines(self) -> None:
        e = Elaboration(sources=[Path("a.sv")], includes=[Path("/i")],
                        defines=["SYNTHESIS"])
        cmd = e.read_cmd()
        assert cmd.startswith("read_slang ")
        assert "-I /i" in cmd and "-D SYNTHESIS" in cmd

    def test_read_cmd_is_stable(self) -> None:
        # Both sides of a miter are built from this string. If it varied
        # between calls, gold and gate could be elaborated differently and the
        # comparison would be meaningless.
        e = Elaboration(sources=[Path("a.sv")], includes=[Path("/i"), Path("/j")],
                        defines=["SYNTHESIS", "VERILATOR"])
        assert e.read_cmd() == e.read_cmd()


class TestFailClosed:
    def test_unresolvable_error_is_reported_not_swallowed(
            self, tmp_path: Path) -> None:
        # A front end that fails for a reason we cannot act on must surface the
        # failure. Returning a partial source list would elaborate *something*
        # and prove it.
        src = tmp_path / "a.sv"
        src.write_text("module a; endmodule\n")
        fake = tmp_path / "fake-yosys"
        fake.write_text("#!/bin/sh\necho \"a.sv:1:1: error: syntax error\" >&2\n"
                        "exit 1\n")
        fake.chmod(0o755)
        got = resolve(tmp_path, [src], "a", str(fake))
        assert not got.ok
        assert "syntax error" in (got.error or "")

    def test_missing_module_that_exists_nowhere_fails(self, tmp_path: Path) -> None:
        src = tmp_path / "a.sv"
        src.write_text("module a; endmodule\n")
        fake = tmp_path / "fake-yosys"
        fake.write_text("#!/bin/sh\necho \"a.sv:1:1: error: unknown module "
                        "'nowhere'\" >&2\nexit 1\n")
        fake.chmod(0o755)
        got = resolve(tmp_path, [src], "a", str(fake))
        assert not got.ok, "must not claim success with an unresolved module"

    def test_non_convergence_is_an_error(self, tmp_path: Path) -> None:
        # A front end that reports a genuinely new missing include every round
        #, each in its own directory, so every round really does make
        # progress, would loop forever. The round cap must end the loop as a
        # failure, never as success.
        src = tmp_path / "a.sv"
        src.write_text("module a; endmodule\n")
        counter = tmp_path / "n"
        counter.write_text("0")
        fake = tmp_path / "fake-yosys"
        fake.write_text(
            "#!/bin/sh\n"
            f"n=$(cat {counter}); n=$((n+1)); echo $n > {counter}\n"
            f"mkdir -p {tmp_path}/d$n\n"
            f"touch {tmp_path}/d$n/h$n.svh\n"
            f"echo \"a.sv:1:1: error: 'h$n.svh': No such file or directory\" >&2\n"
            "exit 1\n")
        fake.chmod(0o755)
        got = resolve(tmp_path, [src], "a", str(fake), max_rounds=3)
        assert not got.ok, "the round cap must not be reported as success"
        assert "converge" in (got.error or "")

    def test_stalled_resolution_stops_immediately(self, tmp_path: Path) -> None:
        # The other termination path: the front end keeps complaining but
        # nothing new can be resolved. Stopping at once and reporting the
        # front end's own diagnostic beats burning the full round budget.
        src = tmp_path / "a.sv"
        src.write_text("module a; endmodule\n")
        (tmp_path / "h.svh").touch()
        fake = tmp_path / "fake-yosys"
        fake.write_text(
            "#!/bin/sh\n"
            f"echo \"a.sv:1:1: error: 'h.svh': No such file or directory\" >&2\n"
            "exit 1\n")
        fake.chmod(0o755)
        got = resolve(tmp_path, [src], "a", str(fake))
        assert not got.ok
        assert "h.svh" in (got.error or ""), \
            "the front end's own diagnostic is what the user needs to see"


class TestAssertionGuards:
    def test_guards_escalate_one_at_a_time(self, tmp_path: Path) -> None:
        # SYNTHESIS alone does not silence ibex's SVA, it is guarded by
        # `ifndef VERILATOR. The resolver must keep going rather than give up,
        # and must not define everything at once.
        src = tmp_path / "a.sv"
        src.write_text("module a; endmodule\n")
        fake = tmp_path / "fake-yosys"
        fake.write_text(
            "#!/bin/sh\n"
            'case "$*" in\n'
            '  *"-D VERILATOR"*) exit 0 ;;\n'
            '  *) echo "a.sv:1:1: error: encountered unsupported SVA feature" >&2\n'
            '     exit 1 ;;\n'
            'esac\n')
        fake.chmod(0o755)
        got = resolve(tmp_path, [src], "a", str(fake))
        assert got.ok
        assert got.defines == ["SYNTHESIS", "VERILATOR"], \
            "guards must escalate in order, not all at once"

    def test_no_guards_defined_when_nothing_blocks(self, tmp_path: Path) -> None:
        # The abstraction must never be applied speculatively.
        src = tmp_path / "a.sv"
        src.write_text("module a; endmodule\n")
        fake = tmp_path / "fake-yosys"
        fake.write_text("#!/bin/sh\nexit 0\n")
        fake.chmod(0o755)
        got = resolve(tmp_path, [src], "a", str(fake))
        assert got.ok and got.defines == [] and got.notes == []

    def test_guard_use_is_recorded(self, tmp_path: Path) -> None:
        src = tmp_path / "a.sv"
        src.write_text("module a; endmodule\n")
        fake = tmp_path / "fake-yosys"
        fake.write_text(
            "#!/bin/sh\n"
            'case "$*" in\n'
            '  *"-D SYNTHESIS"*) exit 0 ;;\n'
            '  *) echo "a.sv:1:1: error: encountered unsupported SVA feature" >&2\n'
            '     exit 1 ;;\n'
            'esac\n')
        fake.chmod(0o755)
        got = resolve(tmp_path, [src], "a", str(fake))
        assert any("SYNTHESIS" in n for n in got.notes), \
            "an applied abstraction must appear in the audit trail"

    def test_guard_list_is_exhaustible(self) -> None:
        assert len(ASSERT_GUARDS) == len(set(ASSERT_GUARDS))


class TestFallbackDefinition:
    def test_fallback_copies_into_stash_and_records_it(
            self, tmp_path: Path) -> None:
        tree = tmp_path / "tree"
        (tree).mkdir()
        src = tree / "a.sv"
        src.write_text("module a; endmodule\n")
        head = tmp_path / "head"
        (head / "syn" / "rtl").mkdir(parents=True)
        (head / "syn" / "rtl" / "cg.v").write_text(
            "module prim_clock_gating; endmodule\n")
        stash = tmp_path / "stash"
        fake = tmp_path / "fake-yosys"
        fake.write_text(
            "#!/bin/sh\n"
            'case "$*" in\n'
            '  *cg.v*) exit 0 ;;\n'
            "  *) echo \"a.sv:1:1: error: unknown module 'prim_clock_gating'\" >&2\n"
            '     exit 1 ;;\n'
            'esac\n')
        fake.chmod(0o755)
        got = resolve(tree, [src], "a", str(fake), fallback=head, stash=stash)
        assert got.ok
        assert (stash / "cg.v").exists(), "must be self-contained, not a reference"
        assert any("@fallback" in n for n in got.notes), \
            "a definition taken from outside the checkout must be declared"

    def test_no_fallback_means_failure_not_invention(self, tmp_path: Path) -> None:
        tree = tmp_path / "tree"
        tree.mkdir()
        src = tree / "a.sv"
        src.write_text("module a; endmodule\n")
        fake = tmp_path / "fake-yosys"
        fake.write_text(
            "#!/bin/sh\n"
            "echo \"a.sv:1:1: error: unknown module 'prim_clock_gating'\" >&2\n"
            "exit 1\n")
        fake.chmod(0o755)
        got = resolve(tree, [src], "a", str(fake))
        assert not got.ok, "without a fallback the honest outcome is failure"


class TestQualifiedIncludes:
    def test_subdirectory_qualified_include_adds_the_right_root(
            self, tmp_path: Path) -> None:
        # `include "common_cells/registers.svh"` needs the directory CONTAINING
        # common_cells on the search path. Adding the header's own parent would
        # leave the qualified path just as unresolvable, and the resolver would
        # loop until the round cap.
        tree = tmp_path / "tree"
        (tree / "vendor" / "common_cells").mkdir(parents=True)
        (tree / "vendor" / "common_cells" / "registers.svh").touch()
        src = tree / "a.sv"
        src.write_text("module a; endmodule\n")
        fake = tmp_path / "fake-yosys"
        fake.write_text(
            "#!/bin/sh\n"
            'case "$*" in\n'
            '  *"-I "*) exit 0 ;;\n'
            "  *) echo \"a.sv:1:1: error: 'common_cells/registers.svh': "
            "No such file or directory\" >&2\n"
            '     exit 1 ;;\n'
            'esac\n')
        fake.chmod(0o755)
        got = resolve(tree, [src], "a", str(fake))
        assert got.ok
        assert got.includes == [tree / "vendor"], \
            "the include root is the path minus the qualified suffix"

    def test_plain_include_still_adds_the_parent(self, tmp_path: Path) -> None:
        tree = tmp_path / "tree"
        (tree / "inc").mkdir(parents=True)
        (tree / "inc" / "defs.svh").touch()
        src = tree / "a.sv"
        src.write_text("module a; endmodule\n")
        fake = tmp_path / "fake-yosys"
        fake.write_text(
            "#!/bin/sh\n"
            'case "$*" in\n'
            '  *"-I "*) exit 0 ;;\n'
            "  *) echo \"a.sv:1:1: error: 'defs.svh': No such file or "
            "directory\" >&2\n"
            '     exit 1 ;;\n'
            'esac\n')
        fake.chmod(0o755)
        got = resolve(tree, [src], "a", str(fake))
        assert got.ok and got.includes == [tree / "inc"]


class TestFilelistVariables:
    def test_variable_expands_to_checkout_root(self, tmp_path: Path) -> None:
        # CVA6's filelist is 188 lines of ${CVA6_REPO_DIR}/... . An unknown
        # variable falls back to the checkout root, which is what that spelling
        # means in every project that uses it.
        (tmp_path / "core").mkdir()
        (tmp_path / "core" / "a.sv").touch()
        (tmp_path / "f.f").write_text("${CVA6_REPO_DIR}/core/a.sv\n")
        assert [p.name for p in seed_sources(tmp_path, "f.f")] == ["a.sv"]

    def test_explicit_variable_wins(self, tmp_path: Path) -> None:
        other = tmp_path / "elsewhere"
        other.mkdir()
        (other / "b.sv").touch()
        (tmp_path / "f.f").write_text("${OTHER_DIR}/b.sv\n")
        got = seed_sources(tmp_path, "f.f", variables={"OTHER_DIR": str(other)})
        assert [p.name for p in got] == ["b.sv"]

    def test_unresolvable_variable_drops_only_that_entry(
            self, tmp_path: Path) -> None:
        # One unset variable must cost one line, not the whole filelist,
        # CVA6's list names ${HPDCACHE_DIR}, which the build sets and we do not.
        (tmp_path / "a.sv").touch()
        (tmp_path / "f.f").write_text("a.sv\n${NOPE}/deep/missing.sv\n")
        assert [p.name for p in seed_sources(tmp_path, "f.f")] == ["a.sv"]

    def test_incdir_lines_are_not_sources(self, tmp_path: Path) -> None:
        (tmp_path / "a.sv").touch()
        (tmp_path / "inc").mkdir()
        (tmp_path / "f.f").write_text("+incdir+${X}/inc/\na.sv\n")
        assert [p.name for p in seed_sources(tmp_path, "f.f")] == ["a.sv"]

    def test_directory_entry_is_not_a_source(self, tmp_path: Path) -> None:
        (tmp_path / "adir").mkdir()
        (tmp_path / "a.sv").touch()
        (tmp_path / "f.f").write_text("adir\na.sv\n")
        assert [p.name for p in seed_sources(tmp_path, "f.f")] == ["a.sv"]
