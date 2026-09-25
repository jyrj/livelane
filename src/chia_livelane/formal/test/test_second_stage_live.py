"""The second stage, against known answers, with the real tools.

Three fixtures, each for a way this stage could plausibly be wrong:

* an edit correct at the parameters the design uses but WRONG at the module's
  defaults, a standalone check at default parameters would call it broken,
  and did exactly that on ibex_counter;
* a genuinely broken edit, which must produce a counterexample;
* an edit that is correct only because both copies share register state,
  which the state-agnostic proof must still accept.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from chia_livelane.formal.second_stage import (CONFIRMED, PROVEN, UNDECIDED,
                                                prove_any_state, settle,
                                                settle_explicit)

TOOLS = os.environ.get("LIVELANE_TOOLS")
pytestmark = pytest.mark.skipif(
    not TOOLS or not (Path(TOOLS) / "bin" / "sby").exists()
    or shutil.which("yosys-abc") is None and not (Path(TOOLS) / "bin" / "yosys-abc").exists(),
    reason="needs yosys (with slang), sby and yosys-abc; source env.sh")

TOP = """module top(input clk, input rst_n, input [7:0] a, output [7:0] q);
  sub #(.W(4)) u (.clk(clk), .rst_n(rst_n), .a(a), .q(q));
endmodule
"""

SUB = """module sub #(parameter int W = 8) (input clk, input rst_n,
                                       input [7:0] a, output logic [7:0] q);
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) q <= '0; else q <= a & MASK_EXPR;
endmodule
"""


def _design(tmp: Path, mask: str) -> list[str]:
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "top.sv").write_text(TOP)
    (tmp / "sub.sv").write_text(SUB.replace("MASK_EXPR", mask))
    return [str(tmp / "top.sv"), str(tmp / "sub.sv")]


def _parent(tmp: Path) -> list[str]:
    return _design(tmp / "gold", "8'((1 << W) - 1)")


class TestParametersAreTheDesigns:
    def test_correct_at_the_instance_width_is_proven(self, tmp_path: Path) -> None:
        """`a & 8'h0f` equals `a & ((1<<W)-1)` at W=4, the width `top` uses,
        and differs at the default W=8. Instance scope must see W=4."""
        gold = _parent(tmp_path)
        gate = _design(tmp_path / "gate", "8'h0f")
        r = settle(gold, gate, "top", "sub", [], [], str(tmp_path / "wd"))
        assert r["verdict"] == PROVEN, r

    def test_broken_edit_yields_a_counterexample(self, tmp_path: Path) -> None:
        gold = _parent(tmp_path)
        gate = _design(tmp_path / "gate", "8'h07")
        r = settle(gold, gate, "top", "sub", [], [], str(tmp_path / "wd"))
        assert r["verdict"] == CONFIRMED, r

    def test_uninstantiated_module_is_undecided_not_proven(self, tmp_path: Path) -> None:
        """No instance, no proof: a vacuous pass must never admit an edit."""
        gold = _parent(tmp_path)
        gate = _design(tmp_path / "gate", "8'h0f")
        r = settle(gold, gate, "top", "nosuch", [], [], str(tmp_path / "wd"))
        assert r["verdict"] not in (PROVEN, CONFIRMED)


class TestAnyState:
    def test_equivalent_rewrite_holds_from_any_shared_state(self, tmp_path: Path) -> None:
        gold = _parent(tmp_path)
        gate = _design(tmp_path / "gate", "8'h0f")
        r = prove_any_state(gold, gate, "top", "sub", [], [],
                            str(tmp_path / "wd"))
        assert r["verdict"] == "ANY-STATE", r

    def test_broken_rewrite_does_not(self, tmp_path: Path) -> None:
        gold = _parent(tmp_path)
        gate = _design(tmp_path / "gate", "8'h07")
        r = prove_any_state(gold, gate, "top", "sub", [], [],
                            str(tmp_path / "wd"))
        assert r["verdict"] != "ANY-STATE", r


# --- initial state stated explicitly ---------------------------------------

FSM_TOP = """module top(input clk, input rst_ni, input go, output busy);
  fsm u (.clk(clk), .rst_ni(rst_ni), .go(go), .busy(busy));
endmodule
"""

FSM = """module fsm(input clk, input rst_ni, input go, output logic busy);
  typedef enum logic [ENC_W] {IDLE ENC_IDLE, RUN ENC_RUN, DONE ENC_DONE} st_e;
  st_e s;
  always_ff @(posedge clk or negedge rst_ni)
    if (!rst_ni) s <= IDLE;
    else case (s)
      IDLE: if (go) s <= RUN;
      RUN:  s <= DONE;
      default: s <= IDLE;
    endcase
  assign busy = (s == RUN);
endmodule
"""


def _fsm(tmp: Path, one_hot: bool) -> list[str]:
    tmp.mkdir(parents=True, exist_ok=True)
    enc = (("2:0", " = 3'b001", " = 3'b010", " = 3'b100") if one_hot
           else ("1:0", "", "", ""))
    body = FSM.replace("ENC_W", enc[0]).replace("ENC_IDLE", enc[1]) \
              .replace("ENC_RUN", enc[2]).replace("ENC_DONE", enc[3])
    (tmp / "top.sv").write_text(FSM_TOP)
    (tmp / "fsm.sv").write_text(body)
    return [str(tmp / "top.sv"), str(tmp / "fsm.sv")]


NR_TOP = """module top(input clk, input rst_ni, input we, input [7:0] d,
           output [7:0] q);
  buf8 u (.clk(clk), .rst_ni(rst_ni), .we(we), .d(d), .q(q));
endmodule
"""

NR_GOLD = """module buf8(input clk, input rst_ni, input we, input [7:0] d,
            output [7:0] q);
  logic [7:0] data;                         // no reset
  always_ff @(posedge clk) if (we) data <= d;
  assign q = data;
endmodule
"""

# "Equivalent" only if the un-reset register powers up to zero.
NR_GATE = """module buf8(input clk, input rst_ni, input we, input [7:0] d,
            output [7:0] q);
  logic [7:0] data;
  logic       seen;                         // no reset either
  always_ff @(posedge clk) if (we) data <= d;
  always_ff @(posedge clk) if (we) seen <= 1'b1;
  assign q = seen ? data : 8'h00;
endmodule
"""


CNT_TOP = """module top(input clk, input rst_ni, output o, output lsb);
  cnt3 u (.clk(clk), .rst_ni(rst_ni), .o(o), .lsb(lsb));
endmodule
"""

CNT = """module cnt3(input clk, input rst_ni, output o, output lsb);
  logic [1:0] cnt;
  always_ff @(posedge clk or negedge rst_ni)
    if (!rst_ni) cnt <= '0; else cnt <= (cnt == 2'd2) ? 2'd0 : cnt + 2'd1;
  assign o = OUT;
  assign lsb = cnt[0];              // keeps the counter alive in both copies
endmodule
"""


def _files(tmp: Path, top: str, sub_name: str, sub: str) -> list[str]:
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "top.sv").write_text(top)
    (tmp / f"{sub_name}.sv").write_text(sub)
    return [str(tmp / "top.sv"), str(tmp / f"{sub_name}.sv")]


class TestExplicitInitialState:
    def test_one_hot_reencoding_is_proven_from_reset(self, tmp_path: Path) -> None:
        """RESET = 3'b001: the all-zero start is an illegal state, and a
        zero-init check reports a counterexample no reset reaches."""
        gold, gate = _fsm(tmp_path / "gold", False), _fsm(tmp_path / "gate", True)
        z = settle(gold, gate, "top", "fsm", [], [], str(tmp_path / "zero"))
        assert z["verdict"] == CONFIRMED, z            # the artefact
        r = settle_explicit(gold, gate, "top", "fsm", [], [], str(tmp_path / "wd"))
        assert r["from_reset"] == PROVEN, r
        # the state register exists in both copies at different widths, so
        # nothing ties the copies' state together without reset
        assert r["any_state"] == "REACHABLE-ONLY", r

    def test_broken_edit_is_confirmed_from_reset(self, tmp_path: Path) -> None:
        gold = _parent(tmp_path)
        gate = _design(tmp_path / "gate", "8'h07")
        r = settle_explicit(gold, gate, "top", "sub", [], [], str(tmp_path / "wd"))
        assert r["from_reset"] == CONFIRMED, r

    def test_equivalent_rewrite_holds_from_any_shared_state(self, tmp_path: Path) -> None:
        gold = _parent(tmp_path)
        gate = _design(tmp_path / "gate", "8'h0f")
        r = settle_explicit(gold, gate, "top", "sub", [], [], str(tmp_path / "wd"))
        assert (r["from_reset"], r["any_state"]) == (PROVEN, "ANY-STATE"), r
        assert r["instances"][0]["register_cut"] == "PASS", r   # the fast path

    def test_correct_only_on_reachable_states(self, tmp_path: Path) -> None:
        """cnt counts 0,1,2,0,...: `cnt == 3` never holds after reset, so
        replacing it by 0 is correct from reset and wrong from cnt = 3. The
        register cut must fail and hand over to the sequential proofs."""
        gold = _files(tmp_path / "gold", CNT_TOP, "cnt3", CNT.replace("OUT", "cnt == 2'd3"))
        gate = _files(tmp_path / "gate", CNT_TOP, "cnt3", CNT.replace("OUT", "1'b0"))
        r = settle_explicit(gold, gate, "top", "cnt3", [], [], str(tmp_path / "wd"))
        assert r["instances"][0]["register_cut"] == "FAIL", r
        assert (r["from_reset"], r["any_state"]) == (PROVEN, "REACHABLE-ONLY"), r

    def test_edit_assuming_unreset_flops_power_up_zero_is_refuted(
            self, tmp_path: Path) -> None:
        """Zero-init accepts it; silicon does not power up to zero."""
        gold = _files(tmp_path / "gold", NR_TOP, "buf8", NR_GOLD)
        gate = _files(tmp_path / "gate", NR_TOP, "buf8", NR_GATE)
        z = settle(gold, gate, "top", "buf8", [], [], str(tmp_path / "zero"))
        assert z["verdict"] == PROVEN, z               # the artefact
        r = settle_explicit(gold, gate, "top", "buf8", [], [], str(tmp_path / "wd"))
        assert r["from_reset"] == CONFIRMED, r

    def test_uninstantiated_module_is_undecided(self, tmp_path: Path) -> None:
        gold = _parent(tmp_path)
        r = settle_explicit(gold, gold, "top", "nosuch", [], [], str(tmp_path / "wd"))
        assert (r["from_reset"], r["any_state"]) == (UNDECIDED, UNDECIDED), r


PKG = """package p_pkg;
  typedef enum logic [ENC_W] {A ENC_A, B ENC_B, C ENC_C} sel_e;
endpackage
"""

PKG_TOP = """module top(input clk, input rst_ni, input [1:0] k, output [7:0] y);
  import p_pkg::*;
  sel_e s;
  pick u_pick (.k(k), .s(s));
  sink u_sink (.s(s), .y(y));
endmodule
"""

PKG_PICK = """module pick import p_pkg::*; (input [1:0] k, output sel_e s);
  always_comb case (k) 2'd0: s = A; 2'd1: s = B; default: s = C; endcase
endmodule
"""

PKG_USE = """module sink import p_pkg::*; (input sel_e s, output logic [7:0] y);
  always_comb case (s) A: y = 8'h11; B: y = 8'h22; default: y = 8'h33; endcase
endmodule
"""


def _pkg(tmp: Path, one_hot: bool) -> list[str]:
    tmp.mkdir(parents=True, exist_ok=True)
    enc = (("2:0", " = 3'b001", " = 3'b010", " = 3'b100") if one_hot
           else ("1:0", "", "", ""))
    body = PKG.replace("ENC_W", enc[0]).replace("ENC_A", enc[1]) \
              .replace("ENC_B", enc[2]).replace("ENC_C", enc[3])
    for name, text in (("p_pkg.sv", body), ("top.sv", PKG_TOP),
                       ("pick.sv", PKG_PICK), ("sink.sv", PKG_USE)):
        (tmp / name).write_text(text)
    return [str(tmp / n) for n in ("p_pkg.sv", "pick.sv", "sink.sv", "top.sv")]


class TestPackageEdits:
    def test_a_type_crossing_a_port_is_proven_at_the_top(self, tmp_path: Path) -> None:
        """Re-encoding sel_e changes both modules' ports: neither module is
        equivalent alone, the design is."""
        from chia_livelane.formal.second_stage import second_stage, edit_scope
        gold, gate = _pkg(tmp_path / "gold", False), _pkg(tmp_path / "gate", True)
        scope, changed = edit_scope(Path(gold[0]), Path(gate[0]),
                                    [Path(x) for x in gold], "top")
        assert (scope, changed) == ("top", ["sel_e"])
        fn = getattr(second_stage, "__wrapped__", second_stage)
        r = fn(gold, gate, "top", "p_pkg", [], [], str(tmp_path / "wd"),
               init="reset")
        assert (r["verdict"], r["scope"]) == (PROVEN, "top"), r


IBEX = Path(__file__).resolve().parents[4] / "thirdparty" / "ibex"


@pytest.mark.skipif(not (IBEX / "rtl" / "ibex_pkg.sv").exists(),
                    reason="needs thirdparty/ibex (scripts/setup)")
class TestWholeCorePackageEdits:
    """The stronger agent's enum re-encodings, at ibex_core scale."""

    @pytest.fixture(scope="class")
    def core(self, tmp_path_factory):
        from chia_livelane.vlsi.elaborate import resolve, seed_sources
        wd = tmp_path_factory.mktemp("core")
        e = resolve(IBEX, seed_sources(IBEX, "rtl/ibex_core.f", "rtl/*.sv"),
                    "ibex_core", str(Path(TOOLS) / "bin" / "yosys"),
                    stash=wd / "stubs")
        srcs = [str(s) for s in e.sources]
        return wd, srcs, [str(i) for i in e.includes] + [str(IBEX / "rtl")], \
            list(e.defines)

    def _edit(self, core, name, old, new):
        wd, srcs, incs, defs = core
        text = (IBEX / "rtl" / "ibex_pkg.sv").read_text()
        assert old in text
        (wd / name).mkdir()
        cand = wd / name / "ibex_pkg.sv"
        cand.write_text(text.replace(old, new))
        gate = [str(cand) if Path(s).name == "ibex_pkg.sv" else s for s in srcs]
        from chia_livelane.formal.second_stage import second_stage
        fn = getattr(second_stage, "__wrapped__", second_stage)
        return fn(srcs, gate, "ibex_core", "ibex_pkg", incs, defs,
                  str(wd / name / "wd"), init="reset")

    def test_one_hot_pc_select_is_proven_at_the_top(self, core) -> None:
        """pc_sel_e crosses controller -> id_stage -> core -> if_stage."""
        r = self._edit(core, "onehot", """typedef enum logic [2:0] {
    PC_BOOT,
    PC_JUMP,
    PC_EXC,
    PC_ERET,
    PC_DRET,
    PC_BP
  } pc_sel_e;""", """typedef enum logic [5:0] {
    PC_BOOT = 6'b000001,
    PC_JUMP = 6'b000010,
    PC_EXC  = 6'b000100,
    PC_ERET = 6'b001000,
    PC_DRET = 6'b010000,
    PC_BP   = 6'b100000
  } pc_sel_e;""")
        assert (r["verdict"], r["scope"]) == (PROVEN, "ibex_core"), r
        assert r["instances"][0].get("register_cut") == "PASS", r

    def test_a_moved_csr_is_not(self, core) -> None:
        """The same whole-core cut must fail on a real change (non-vacuity).
        Only the cut is run: the sequential fallback at this scale is slow."""
        from chia_livelane.formal.second_stage import (_miter_il, _prove_comb,
                                                        cut_registers, edit_scope)
        wd, srcs, incs, defs = core
        text = (IBEX / "rtl" / "ibex_pkg.sv").read_text()
        (wd / "csr").mkdir()
        cand = wd / "csr" / "ibex_pkg.sv"
        cand.write_text(text.replace("CSR_MSCRATCH  = 12'h340",
                                     "CSR_MSCRATCH  = 12'h5A5"))
        scope, changed = edit_scope(IBEX / "rtl" / "ibex_pkg.sv", cand,
                                    [Path(s) for s in srcs], "ibex_core")
        assert scope == "ibex_core" and "csr_num_e" in changed, (scope, changed)
        gate = [str(cand) if Path(s).name == "ibex_pkg.sv" else s for s in srcs]
        il = _miter_il(srcs, gate, "ibex_core", "ibex_core", incs, defs,
                       wd / "csr" / "m")
        cut, info = cut_registers(il)
        assert cut is not None, info
        assert _prove_comb(cut, wd / "csr" / "comb", 600) == "FAIL"


class TestEditScope:
    """Where a package edit is proven. No tools needed."""

    def _scope(self, tmp: Path, gold: str, gate: str, mods: dict[str, str]):
        from chia_livelane.formal.second_stage import edit_scope
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "g").mkdir(); (tmp / "c").mkdir()
        (tmp / "g" / "p_pkg.sv").write_text(gold)
        (tmp / "c" / "p_pkg.sv").write_text(gate)
        srcs = [tmp / "g" / "p_pkg.sv"]
        for name, body in mods.items():
            (tmp / f"{name}.sv").write_text(body)
            srcs.append(tmp / f"{name}.sv")
        return edit_scope(tmp / "g" / "p_pkg.sv", tmp / "c" / "p_pkg.sv", srcs, "top")

    GOLD = """package p_pkg;
  typedef enum logic [1:0] {A, B, C} e_t;
  typedef struct packed { e_t sel; logic v; } s_t;
  parameter int W = 4;
endpackage
"""

    def test_one_user(self, tmp_path: Path) -> None:
        gate = self.GOLD.replace("{A, B, C} e_t", "{A = 2'd2, B = 2'd1, C = 2'd0} e_t")
        mods = {"m1": "module m1; e_t x; endmodule", "m2": "module m2; endmodule"}
        # s_t holds e_t, but nobody names s_t here
        assert self._scope(tmp_path, self.GOLD, gate, mods) == ("m1", ["e_t", "s_t"])

    def test_a_struct_holding_the_enum_reaches_its_users(self, tmp_path: Path) -> None:
        """m2 names only s_t, which holds the re-encoded e_t."""
        gate = self.GOLD.replace("{A, B, C} e_t", "{A = 2'd2, B = 2'd1, C = 2'd0} e_t")
        mods = {"m1": "module m1; e_t x; endmodule",
                "m2": "module m2; s_t y; endmodule"}
        assert self._scope(tmp_path, self.GOLD, gate, mods)[0] == "top"

    def test_enum_members_count_as_uses(self, tmp_path: Path) -> None:
        gate = self.GOLD.replace("{A, B, C} e_t", "{A = 2'd2, B = 2'd1, C = 2'd0} e_t")
        mods = {"m1": "module m1; e_t x; endmodule",
                "m2": "module m2; wire z = (q == B); endmodule"}
        assert self._scope(tmp_path, self.GOLD, gate, mods)[0] == "top"

    def test_a_change_outside_named_items_goes_to_the_top(self, tmp_path: Path) -> None:
        gate = self.GOLD.replace("endpackage", "  import q_pkg::*;\nendpackage")
        mods = {"m1": "module m1; e_t x; endmodule"}
        assert self._scope(tmp_path, self.GOLD, gate, mods)[0] == "top"

    def test_comments_do_not_count(self, tmp_path: Path) -> None:
        gate = self.GOLD.replace("parameter int W = 4;", "parameter int W = 4; // wide")
        assert self._scope(tmp_path, self.GOLD, gate, {"m1": "module m1; endmodule"}) \
            == ("top", [])


# --- regressions: edits an adversarial review showed the proofs got wrong ---

def _pair(tmp: Path, gold: dict[str, str], gate: dict[str, str]):
    out = []
    for side, files in (("gold", gold), ("gate", gate)):
        d = tmp / side
        d.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (d / name).write_text(text)
        out.append([str(d / n) for n in files])
    return out


def _explicit(tmp, gold, gate, module, top="top"):
    g, c = _pair(tmp, gold, gate)
    return settle_explicit(g, c, top, module, [], [], str(tmp / "wd"))


SUB_TOP = """module top(input clk, input rst_ni, input [3:0] d, output [3:0] o);
  sub u (.clk(clk), .rst_ni(rst_ni), .d(d), .o(o));
endmodule
"""


class TestAdversarialRegressions:
    def test_a_declaration_initialiser_does_not_pin_the_power_up(self, tmp_path):
        """`logic q = 0` is ignored by ASIC synthesis. Kept, it fixed the
        register cut's "arbitrary" state and proved a counter that counts
        0,1,0 equal to one that counts 0,1,2."""
        gold = {"top.sv": SUB_TOP, "sub.sv": """module sub(input clk, input rst_ni, input [3:0] d, output [3:0] o);
  logic [1:0] q;
  always_ff @(posedge clk or negedge rst_ni)
    if (!rst_ni) q <= 2'd0; else q <= q + 2'd1;
  assign o = {2'b00, q};
endmodule
"""}
        gate = {"top.sv": SUB_TOP, "sub.sv": """module sub(input clk, input rst_ni, input [3:0] d, output [3:0] o);
  logic [1:0] q = 2'd0;
  always_ff @(posedge clk or negedge rst_ni)
    if (!rst_ni) q <= 2'd0; else q <= (q == 2'd0) ? 2'd1 : 2'd0;
  assign o = {2'b00, q};
endmodule
"""}
        r = _explicit(tmp_path, gold, gate, "sub")
        assert r["from_reset"] == CONFIRMED, r

    def test_a_register_moved_to_the_other_edge_is_not_proven(self, tmp_path):
        body = """module sub(input clk, input rst_ni, input [3:0] d, output [3:0] o);
  logic [3:0] q1, q2;
  always_ff @(EDGE clk or negedge rst_ni) if (!rst_ni) q1 <= '0; else q1 <= d;
  always_ff @(posedge clk) q2 <= q1;
  assign o = q2;
endmodule
"""
        r = _explicit(tmp_path, {"top.sv": SUB_TOP, "sub.sv": body.replace("EDGE", "posedge")},
                      {"top.sv": SUB_TOP, "sub.sv": body.replace("EDGE", "negedge")}, "sub")
        assert r["from_reset"] != PROVEN, r

    def test_an_assume_in_the_edited_rtl_constrains_nothing(self, tmp_path):
        gold = {"top.sv": SUB_TOP, "sub.sv": """module sub(input clk, input rst_ni, input [3:0] d, output [3:0] o);
  logic [3:0] q;
  always_ff @(posedge clk or negedge rst_ni) if (!rst_ni) q <= '0; else q <= d;
  assign o = q;
endmodule
"""}
        gate = {"top.sv": SUB_TOP, "sub.sv": """module sub(input clk, input rst_ni, input [3:0] d, output [3:0] o);
  logic [3:0] q;
  always_ff @(posedge clk or negedge rst_ni) if (!rst_ni) q <= '0; else q <= d & 4'h7;
  assign o = q;
  always_comb assume (!d[3]);
endmodule
"""}
        r = _explicit(tmp_path, gold, gate, "sub")
        assert r["from_reset"] == CONFIRMED, r
        g, c = _pair(tmp_path / "zero", gold, gate)
        assert settle(g, c, "top", "sub", [], [], str(tmp_path / "z"))["verdict"] == CONFIRMED

    def test_a_tied_off_reset_is_not_assumed(self, tmp_path):
        """The parent ties rst_ni high: the instance is never reset, so a
        state that reset would clear can be where the silicon starts."""
        top = """module top(input clk, input rst_ni, output o);
  c3 u (.clk(clk), .rst_ni(1'b1), .o(o));
endmodule
"""
        body = """module c3(input clk, input rst_ni, output o);
  logic [1:0] c;
  always_ff @(posedge clk or negedge rst_ni)
    if (!rst_ni) c <= '0; else c <= NEXT;
  assign o = (c == 2'd3);
endmodule
"""
        r = _explicit(tmp_path,
                      {"top.sv": top, "c3.sv": body.replace("NEXT", "(c == 2'd2) ? 2'd0 : c + 2'd1")},
                      {"top.sv": top, "c3.sv": body.replace(
                          "NEXT", "(c == 2'd2) ? 2'd0 : (c == 2'd3) ? 2'd3 : c + 2'd1")}, "c3")
        assert r["instances"][0]["reset_assumed"] is False, r
        assert r["from_reset"] != PROVEN, r

    def test_a_driven_reset_is_assumed(self, tmp_path):
        """...and the FSM re-encoding still needs, and gets, its reset."""
        gold, gate = _fsm(tmp_path / "gold", False), _fsm(tmp_path / "gate", True)
        r = settle_explicit(gold, gate, "top", "fsm", [], [], str(tmp_path / "wd"))
        assert r["instances"][0]["reset_assumed"] is True and r["from_reset"] == PROVEN, r


class TestEditScopeReach:
    def _scope(self, tmp, gold, gate, files, pkg):
        from chia_livelane.formal.second_stage import edit_scope
        (tmp / "g").mkdir(parents=True); (tmp / "c").mkdir()
        (tmp / "g" / pkg).write_text(gold)
        (tmp / "c" / pkg).write_text(gate)
        srcs = [tmp / "g" / pkg]
        for name, text in files.items():
            (tmp / name).write_text(text)
            srcs.append(tmp / name)
        return edit_scope(tmp / "g" / pkg, tmp / "c" / pkg, srcs, "top")[0]

    def test_a_package_that_uses_the_package_sends_the_proof_to_the_top(self, tmp_path):
        files = {"b_pkg.sv": "package b_pkg;\n  typedef logic [a_pkg::WIDTH-1:0] word_t;\nendpackage\n",
                 "m.sv": "module m(input [7:0] a, output [7:0] y);\n  b_pkg::word_t t;\n  assign t = a;\n  assign y = 8'(t);\nendmodule\n",
                 "n.sv": "module n(input [7:0] a, output [7:0] z);\n  localparam int Unused = a_pkg::WIDTH;\n  assign z = a;\nendmodule\n"}
        assert self._scope(tmp_path, "package a_pkg;\n  parameter int WIDTH = 4;\nendpackage\n",
                           "package a_pkg;\n  parameter int WIDTH = 3;\nendpackage\n",
                           files, "a_pkg.sv") == "top"

    def test_a_multi_name_parameter_sends_the_proof_to_the_top(self, tmp_path):
        files = {"ma.sv": "module ma(input [7:0] a, output [7:0] y);\n  assign y = a & p_pkg::MaskA;\nendmodule\n",
                 "mb.sv": "module mb(input [7:0] a, output [7:0] z);\n  assign z = a & p_pkg::MaskB;\nendmodule\n"}
        assert self._scope(tmp_path,
                           "package p_pkg;\n  localparam logic [7:0] MaskA = 8'hff, MaskB = 8'h0f;\nendpackage\n",
                           "package p_pkg;\n  localparam logic [7:0] MaskA = 8'hff, MaskB = 8'h07;\nendpackage\n",
                           files, "p_pkg.sv") == "top"

    def test_an_attributed_module_header_is_a_user(self, tmp_path):
        files = {"ma.sv": "(* keep_hierarchy *) module ma(input [7:0] a, output [7:0] y);\n  assign y = a << p_pkg::V;\nendmodule\n",
                 "mb.sv": "module mb(input [7:0] a, output [7:0] z);\n  localparam int X = p_pkg::V;\n  assign z = a;\nendmodule\n"}
        assert self._scope(tmp_path,
                           "package p_pkg;\n  parameter int W = 4;\n  parameter int V = 2;\nendpackage\n",
                           "package p_pkg;\n  parameter int W = 4;\n  parameter int V = 3;\nendpackage\n",
                           files, "p_pkg.sv") == "top"


    def test_an_unrelated_package_sharing_a_name_is_not_a_user(self, tmp_path):
        """`RESET` in some other package is that package's own identifier."""
        files = {"ma.sv": "module ma(input [7:0] a, output [7:0] y);\n  import p_pkg::*;\n  assign y = (a == RESET) ? 8'd1 : 8'd0;\nendmodule\n",
                 "other_pkg.sv": "package other_pkg;\n  typedef enum logic {RESET, RUN} o_e;\nendpackage\n"}
        gold = "package p_pkg;\n  typedef enum logic [1:0] {RESET, GO} s_e;\nendpackage\n"
        gate = "package p_pkg;\n  typedef enum logic [1:0] {RESET = 2'd2, GO = 2'd1} s_e;\nendpackage\n"
        assert self._scope(tmp_path, gold, gate, files, "p_pkg.sv") == "ma"


    def test_literal_digits_are_not_names(self, tmp_path):
        """A one-hot encoding is full of `10'h001`; `h001` is not a name that
        every file with a `32'h001` literal suddenly uses."""
        files = {"ma.sv": "module ma(input [1:0] a, output y);\n  import p_pkg::*;\n  assign y = (a == GO);\nendmodule\n",
                 "mb.sv": "module mb(output [31:0] z);\n  assign z = 32'h002;\nendmodule\n"}
        gold = "package p_pkg;\n  typedef enum logic [1:0] {IDLE, GO} s_e;\nendpackage\n"
        gate = "package p_pkg;\n  typedef enum logic [1:0] {IDLE = 2'h001, GO = 2'h002} s_e;\nendpackage\n"
        assert self._scope(tmp_path, gold, gate, files, "p_pkg.sv") == "ma"
