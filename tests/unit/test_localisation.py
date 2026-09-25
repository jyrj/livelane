"""Three parser hazards, each of which produced a plausible wrong answer.

`localisation.py` classifies every reported failing partition as in the changed
module, on the same net as one of its ports, or elsewhere. All three failure
modes below returned a NUMBER rather than an error, which is the only reason
they are worth tests:

  1. a stray `)` inside an ibex comment closed the port-list paren counter 22
     lines early, so ibex_decoder had 15 ports instead of 67;
  2. `module foo import pkg::*; #(params) (ports);`, stopping at the first
     `(` grabs the parameters and stopping at the first `;` stops on the
     import, both giving an empty port set;
  3. the net graph was mined from the WORKING TREE while every instance's
     partition names come from its own `base.sha`. That alone moved five
     partitions from "elsewhere" to "same net" once corrected, a 24.3% ->
     10.8% swing in the headline bucket.

None of the three raised. Each just moved the answer.
"""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "localisation", ROOT / "scripts" / "chia" / "localisation.py")
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)


class TestDecomment:
    def test_paren_in_a_line_comment_is_blanked(self) -> None:
        """ibex_decoder.sv:44 really does end a comment with `fan-out)`."""
        src = "a ( // replicated to ease fan-out)\n b )\n"
        out = L.decomment(src)
        assert out.count("(") == 1
        assert out.count(")") == 1        # only the real one survives

    def test_offsets_are_preserved(self) -> None:
        """Callers index back into the original text."""
        src = "module x; // )))\nendmodule\n"
        assert len(L.decomment(src)) == len(src)
        assert L.decomment(src).count("\n") == src.count("\n")

    def test_block_comments_and_strings(self) -> None:
        out = L.decomment('a /* ) ) */ b $display("(((") c\n')
        # the comment's parens and the STRING's parens go; $display's own
        # parens are code and stay
        assert out.count("(") == 1 and out.count(")") == 1
        assert "a " in out and " c" in out
        assert "$display" in out


class TestPortList:
    def _rtl(self, tmp_path: Path, body: str) -> Path:
        d = tmp_path / "rtl"
        d.mkdir(exist_ok=True)
        (d / "m.sv").write_text(body)
        return d

    def test_import_clause_then_parameters(self, tmp_path: Path) -> None:
        d = self._rtl(tmp_path, "module m import p::*; #(\n"
                               "  parameter int W = 1\n"
                               ") (\n"
                               "  input  logic clk_i,\n"
                               "  output logic q_o\n"
                               ");\nendmodule\n")
        assert L.ports_of(d, "m") == {"clk_i", "q_o"}

    def test_parameters_are_not_mistaken_for_ports(self, tmp_path: Path) -> None:
        d = self._rtl(tmp_path, "module m #(\n  parameter int Wide = 4\n) (\n"
                               "  input logic a_i\n);\nendmodule\n")
        assert L.ports_of(d, "m") == {"a_i"}

    def test_stray_paren_in_a_comment_does_not_truncate(
            self, tmp_path: Path) -> None:
        d = self._rtl(tmp_path, "module m (\n"
                               "  input logic a_i,   // fan-out)\n"
                               "  input logic b_i,\n"
                               "  output logic c_o\n"
                               ");\nendmodule\n")
        assert L.ports_of(d, "m") == {"a_i", "b_i", "c_o"}


class TestNetMerge:
    def _rtl(self, tmp_path: Path) -> Path:
        d = tmp_path / "rtl"
        d.mkdir()
        (d / "child.sv").write_text(
            "module child (\n  input logic a_i,\n  output logic y_o\n);\n"
            "endmodule\n")
        (d / "sink.sv").write_text(
            "module sink (\n  input logic s_i\n);\nendmodule\n")
        (d / "top.sv").write_text(
            "module top (\n  output logic out_o\n);\n"
            "  logic w;\n"
            "  child child_i (\n    .a_i (in_a),\n    .y_o (w)\n  );\n"
            "  sink sink_i (\n    .s_i (w)\n  );\n"
            "endmodule\n")
        return d

    def test_signal_in_the_changed_module_is_in_module(
            self, tmp_path: Path) -> None:
        d = self._rtl(tmp_path)
        nets = L.build(d)
        assert L.classify(nets, d, "top.child_i.y_o", "child") == "in-module"

    def test_same_net_seen_from_a_consumer(self, tmp_path: Path) -> None:
        """The #48/#157 shape: the same wire, named at the far end."""
        d = self._rtl(tmp_path)
        nets = L.build(d)
        assert L.classify(nets, d, "top.sink_i.s_i",
                          "child") == "same net as a changed-module port"

    def test_unconnected_signal_is_elsewhere(self, tmp_path: Path) -> None:
        d = self._rtl(tmp_path)
        nets = L.build(d)
        assert L.classify(nets, d, "top.out_o", "child") == "elsewhere"


class TestReadsTheRightCommit:
    """The hazard that moved the headline without raising anything."""

    def _repo(self, tmp_path: Path) -> tuple[Path, str]:
        r = tmp_path / "repo"
        (r / "rtl").mkdir(parents=True)
        def git(*a):
            p = subprocess.run(["git", *a], cwd=r, capture_output=True,
                               text=True)
            assert p.returncode == 0, f"git {a}: {p.stderr}"
        git("init", "-q")
        git("config", "user.email", "t@t")
        git("config", "user.name", "t")
        # The user's GLOBAL config has commit.gpgsign=true, which a throwaway
        # repo inherits. gpg-agent then prompts, times out, and git exits 128
        #, which looks exactly like a flake, because whether it fires depends
        # on the agent's passphrase cache. It failed once here, "passed three
        # times in a row" when the cache was warm, and was written off as load.
        git("config", "commit.gpgsign", "false")
        git("config", "tag.gpgsign", "false")
        (r / "rtl" / "child.sv").write_text(
            "module child (\n  output logic y_o\n);\nendmodule\n")
        (r / "rtl" / "top.sv").write_text(
            "module top;\n  child child_i (\n    .y_o (w)\n  );\n"
            "  sink sink_i (\n    .s_i (w)\n  );\nendmodule\n")
        (r / "rtl" / "sink.sv").write_text(
            "module sink (\n  input logic s_i\n);\nendmodule\n")
        git("add", "-A")
        git("commit", "-qm", "old")
        old = subprocess.run(["git", "rev-parse", "HEAD"], cwd=r,
                             capture_output=True, text=True).stdout.strip()
        # the port is renamed later, exactly the ibex_cs_registers.pc_set_i
        # situation, where the signal simply does not exist any more
        (r / "rtl" / "sink.sv").write_text(
            "module sink (\n  input logic renamed_i\n);\nendmodule\n")
        (r / "rtl" / "top.sv").write_text(
            "module top;\n  child child_i (\n    .y_o (w)\n  );\n"
            "  sink sink_i (\n    .renamed_i (w)\n  );\nendmodule\n")
        git("add", "-A")
        git("commit", "-qm", "new")
        return r, old

    def test_the_old_sha_classifies_the_old_name(self, tmp_path: Path) -> None:
        r, old = self._repo(tmp_path)
        tree = L.Tree(r, old, r / "rtl")
        nets = L.build(tree)
        assert L.classify(nets, tree, "top.sink_i.s_i",
                          "child") == "same net as a changed-module port"

    def test_the_working_tree_gets_it_wrong(self, tmp_path: Path) -> None:
        """Same partition name, current tree: 'elsewhere'. A finding that isn't."""
        r, _ = self._repo(tmp_path)
        nets = L.build(r / "rtl")
        assert L.classify(nets, r / "rtl", "top.sink_i.s_i",
                          "child") == "elsewhere"


class TestScannerHandlesWhatTheRegexDropped:
    """Each of these vanished from the net graph with no diagnostic."""

    def _rtl(self, tmp_path: Path, top: str, extra: dict | None = None) -> Path:
        d = tmp_path / "rtl"
        d.mkdir(exist_ok=True)
        (d / "child.sv").write_text(
            "module child (\n  input logic a_i,\n  output logic y_o\n);\n"
            "endmodule\n")
        (d / "top.sv").write_text(top)
        for name, body in (extra or {}).items():
            (d / f"{name}.sv").write_text(body)
        return d

    def test_parameter_override_nesting_two_levels(self, tmp_path: Path) -> None:
        """'#(.W($bits(t)))' nests twice; the old regex allowed one."""
        d = self._rtl(tmp_path,
                      "module top;\n"
                      "  child #(\n    .W($bits(some_t))\n  ) child_i (\n"
                      "    .a_i (in_a),\n    .y_o (w)\n  );\n"
                      "endmodule\n")
        nets = L.build(d)
        assert nets.inst_module.get("child_i") == "child"
        assert nets.same(("child", "y_o"), ("top", "w"))

    def test_connection_expression_containing_parens(self, tmp_path: Path) -> None:
        """'.a_i (foo | bar(x))' matched nothing AND was not counted."""
        d = self._rtl(tmp_path,
                      "module top;\n  child child_i (\n"
                      "    .a_i (sel ? f(x) : g(y)),\n    .y_o (w)\n  );\n"
                      "endmodule\n")
        nets = L.build(d)
        assert nets.same(("child", "y_o"), ("top", "w"))
        ports = [p for _i, p, _e in nets.unresolved]
        assert "a_i" in ports, "a non-bare expression must be COUNTED, not lost"

    def test_unpacked_array_port_is_not_dropped(self, tmp_path: Path) -> None:
        """ibex_alu.sv:23 really is 'input logic [31:0] imd_val_q_i[2],'."""
        d = self._rtl(tmp_path, "module top;\nendmodule\n", extra={"arr":
            "module arr (\n"
            "  input  logic [31:0] imd_val_q_i[2],\n"
            "  output logic        done_o\n"
            ");\nendmodule\n"})
        assert L.ports_of(d, "arr") == {"imd_val_q_i", "done_o"}

    def test_instantiation_of_a_module_declared_elsewhere_still_merges(
            self, tmp_path: Path) -> None:
        """A vendored prim_* has no source under rtl/; the old 'declared'
        filter dropped it and the merge stopped at that boundary."""
        d = self._rtl(tmp_path,
                      "module top;\n  prim_buf u_buf (\n"
                      "    .i (a),\n    .o (b)\n  );\nendmodule\n")
        nets = L.build(d)
        assert nets.inst_module.get("u_buf") == "prim_buf"
        assert nets.same(("prim_buf", "o"), ("top", "b"))

    def test_keywords_are_not_mistaken_for_instantiations(
            self, tmp_path: Path) -> None:
        d = self._rtl(tmp_path,
                      "module top;\n"
                      "  always_ff @(posedge clk) begin\n    q <= d;\n  end\n"
                      "  if (cond) begin : gen_x\n  end\n"
                      "endmodule\n")
        nets = L.build(d)
        assert "posedge" not in nets.inst_module
        assert not any(k in nets.inst_module for k in ("begin", "cond", "gen_x"))


class TestCollisionsAreReportedNotResolved:
    def test_same_instance_name_two_modules(self, tmp_path: Path) -> None:
        """ibex really does this: multdiv_i is both _fast and _slow."""
        d = tmp_path / "rtl"
        d.mkdir()
        (d / "a.sv").write_text("module a (\n  output logic y_o\n);\nendmodule\n")
        (d / "b.sv").write_text("module b (\n  output logic y_o\n);\nendmodule\n")
        (d / "top.sv").write_text(
            "module top;\n"
            "  a u_thing (\n    .y_o (w1)\n  );\n"
            "  b u_thing (\n    .y_o (w2)\n  );\n"
            "endmodule\n")
        nets = L.build(d)
        assert "u_thing" in nets.collisions
        assert nets.collisions["u_thing"] == {"a", "b"}
