"""Unit tests for the equivalence gate: fail closed, and never guess.

The two things that matter here are (1) that nothing except a positive proof
admits an edit, and (2) that "I found a counterexample" and "I could not decide"
stay separate answers. ``eqy`` reports them with the SAME summary line and the
SAME exit code, so getting this wrong is easy and silent.
"""

import os
import tempfile
import time
from pathlib import Path

import pytest

from chia_livelane.formal.lec_gate import (DEFAULT_STRATEGIES, ERROR, PROVEN,
                                           REFUTED, SKIPPED, TIMEOUT,
                                           UNDECIDED, UNDEF_INIT_VALUES,
                                           VALID_VERDICTS, LecGateNode,
                                           LecResult, auto_jobs, cross_check,
                                           parse_eqy_log)

# Verbatim from real eqy runs. Note the identical summary line and rc.
UNDECIDED_LOG = (
    "EQY run: Could not prove equivalence of partition 'picorv32.count_cycle' "
    "using strategy 'sby': equivalence unknown\n"
    "EQY Warning: Failed to prove equivalence for 2/473 partitions:\n"
    "EQY Failed to prove equivalence of partition picorv32.count_cycle\n"
    "EQY DONE (FAIL, rc=2)")
REFUTED_LOG = (
    "EQY run: Could not prove equivalence of partition 'nerv.next_rd' "
    "using strategy 'sby': partitions not equivalent\n"
    "EQY Warning: Failed to prove equivalence for 1/44 partitions:\n"
    "EQY Failed to prove equivalence of partition nerv.next_rd\n"
    "EQY DONE (FAIL, rc=2)")
PROVEN_LOG = ("EQY Successfully proved designs equivalent\n"
              "EQY DONE (PASS, rc=0)")


class TestLogParsing:
    def test_undecided_is_not_a_refutation(self):
        p = parse_eqy_log(UNDECIDED_LOG)
        assert p["unknown_partitions"] == ["picorv32.count_cycle"]
        assert not p.get("not_equivalent_partitions")

    def test_counterexample_is_a_refutation(self):
        p = parse_eqy_log(REFUTED_LOG)
        assert p["not_equivalent_partitions"] == ["nerv.next_rd"]
        assert not p.get("unknown_partitions")

    def test_the_two_logs_are_otherwise_identical(self):
        # This is why the per-partition reason has to be parsed: the summary
        # line and the exit code carry no information distinguishing them.
        a, b = parse_eqy_log(UNDECIDED_LOG), parse_eqy_log(REFUTED_LOG)
        assert a["done_rc"] == b["done_rc"] == 2
        assert a["done_status"] == b["done_status"] == "FAIL"
        assert a["proved"] is b["proved"] is False

    def test_proved_log(self):
        assert parse_eqy_log(PROVEN_LOG)["proved"] is True

    def test_partition_counts(self):
        p = parse_eqy_log(UNDECIDED_LOG)
        assert p["partitions_failed"] == 2
        assert p["partitions_total"] == 473

    def test_unparsable_log_yields_no_facts(self):
        # Absent keys, not None values: a caller can tell "not stated" from
        # "stated as zero".
        assert parse_eqy_log("total garbage") == {}


class TestVerdictSemantics:
    @pytest.mark.parametrize("verdict",
                             [REFUTED, UNDECIDED, ERROR, TIMEOUT, SKIPPED])
    def test_gate_fails_closed(self, verdict):
        assert LecResult(verdict, "eqy", 1.0).admits_edit is False

    def test_only_proven_admits(self):
        assert LecResult(PROVEN, "eqy", 1.0).admits_edit is True

    @pytest.mark.parametrize("verdict", [UNDECIDED, ERROR, TIMEOUT, SKIPPED])
    def test_only_a_counterexample_is_a_refutation(self, verdict):
        # Counting an undecided or errored check as a refutation would inflate
        # any rejection rate computed from these results.
        assert LecResult(verdict, "eqy", 1.0).is_refutation is False

    def test_refuted_is_a_refutation(self):
        assert LecResult(REFUTED, "eqy", 1.0).is_refutation is True

    def test_bounded_proof_admits_but_is_labelled(self):
        b = LecResult(PROVEN, "eqy", 1.0, bounded=True, depth=2)
        assert b.admits_edit is True
        assert b.is_unbounded_proof is False
        assert b.evidence_strength == "bounded-proof(depth=2)"

    def test_unbounded_proof_is_the_stronger_claim(self):
        u = LecResult(PROVEN, "eqy", 1.0)
        assert u.is_unbounded_proof is True
        assert u.evidence_strength == "proof"

    def test_unknown_verdict_string_rejected(self):
        with pytest.raises(ValueError):
            LecResult("probably_fine", "eqy", 1.0)

    def test_vocabulary_is_closed(self):
        assert VALID_VERDICTS == {PROVEN, REFUTED, UNDECIDED, ERROR, TIMEOUT,
                                  SKIPPED}


class TestCrossCheck:
    def test_disagreement_is_reported_not_averaged(self):
        d = cross_check(LecResult(PROVEN, "eqy", 1.0),
                        LecResult(REFUTED, "circt-lec", 0.5))
        assert d["agree"] is False
        assert "DISAGREEMENT" in str(d["note"])

    def test_agreement_carries_no_note(self):
        d = cross_check(LecResult(PROVEN, "eqy", 1.0),
                        LecResult(PROVEN, "circt-lec", 0.5))
        assert d["agree"] is True and d["note"] == ""


@pytest.fixture
def srcs(tmp_path):
    """Four real source files; the node now requires sources to exist."""
    out = []
    for name in ("a.v", "b.v", "c.v", "d.v"):
        f = tmp_path / name
        f.write_text("// placeholder\n")
        out.append(str(f))
    return out


class TestConfigGeneration:
    def test_all_sources_share_one_read_command(self, tmp_path, srcs):
        # One read line per side, never per file: read_slang elaborates each
        # invocation independently, so per-file reads make every module
        # invisible to the others and the whole check errors out.
        n = LecGateNode(workdir=tmp_path, verbose=False)
        body = n.write_config(srcs[:3], srcs[:1] + srcs[3:], "top",
                              tmp_path / "t.eqy").read_text()
        reads = [ln for ln in body.splitlines() if ln.startswith("read_verilog")]
        assert len(reads) == 2, reads
        assert reads[0].count(".v") == 3, reads[0]

    def test_sources_are_absolutised(self, tmp_path, monkeypatch):
        # eqy runs with cwd=workdir, so a relative source path resolves against
        # the workdir instead of the caller's cwd and yosys never finds it.
        # That surfaced as rc=2 with an unparsable log, i.e. an ERROR verdict
        # that looks like a cautious gate rather than a misconfigured one.
        (tmp_path / "a.v").write_text("// x\n")
        (tmp_path / "b.v").write_text("// x\n")
        monkeypatch.chdir(tmp_path)
        n = LecGateNode(workdir=tmp_path, verbose=False)
        body = n.write_config(["a.v"], ["b.v"], "t", tmp_path / "t.eqy").read_text()
        for line in body.splitlines():
            if line.startswith("read_verilog"):
                assert line.split()[-1].startswith("/"), line

    def test_read_slang_gets_an_explicit_top(self, tmp_path, srcs):
        n = LecGateNode(workdir=tmp_path, read_cmd="read_slang", verbose=False)
        body = n.write_config(srcs[:1], srcs[1:2], "Alu",
                              tmp_path / "s.eqy").read_text()
        assert "read_slang --top Alu " in body

    def test_per_side_read_override(self, tmp_path, srcs):
        n = LecGateNode(workdir=tmp_path, gate_read_cmd="read_verilog -sv -DCAND",
                        verbose=False)
        body = n.write_config(srcs[:1], srcs[:1], "t",
                              tmp_path / "t.eqy").read_text()
        assert "read_verilog -sv /" in body
        assert "read_verilog -sv -DCAND /" in body

    def test_strategy_ladder_is_emitted_in_order(self, tmp_path, srcs):
        n = LecGateNode(workdir=tmp_path, strategies=DEFAULT_STRATEGIES,
                        verbose=False)
        body = n.write_config(srcs[:1], srcs[1:2], "t",
                              tmp_path / "t.eqy").read_text()
        # `sat` leads, which is sound ONLY with initial values resolved on both
        # sides: without `setundef -zero -init`, `sat` returns PASS on a
        # sequential partition that is not equivalent (smoke/refuted_const.v).
        # So the order and the setundef are pinned together, not separately.
        assert body.index("[strategy simple]") < body.index("[strategy smt]")
        assert "use sat" in body and "use sby" in body
        assert "setundef -zero -init" in body

    def test_sat_first_without_undef_init_is_not_the_default(self, tmp_path, srcs):
        """Opting out of undef_init must be a deliberate, visible choice."""
        n = LecGateNode(workdir=tmp_path, strategies=DEFAULT_STRATEGIES,
                        verbose=False)
        assert n.undef_init == "zero"

    def test_single_strategy_fallback_carries_depth_and_engine(self, tmp_path, srcs):
        n = LecGateNode(workdir=tmp_path, depth=7, engine="smtbmc z3",
                        verbose=False)
        body = n.write_config(srcs[:1], srcs[1:2], "t",
                              tmp_path / "t.eqy").read_text()
        assert "depth 7" in body and "engine smtbmc z3" in body


class TestRelativePaths:
    def test_relative_workdir_is_absolutised(self, tmp_path, monkeypatch):
        # eqy runs with cwd=workdir, so a relative workdir made the generated
        # config path relative to itself; eqy then exited 2, the same code it
        # uses for a failed proof, with an unparsable log, and the gate
        # returned ERROR. A misconfiguration wearing the costume of a cautious
        # gate is the exact failure this module exists to prevent.
        monkeypatch.chdir(tmp_path)
        (tmp_path / "wd").mkdir()
        n = LecGateNode(workdir=Path("wd"), verbose=False)
        assert n.workdir.is_absolute()

    def test_unparsable_log_carries_its_tail(self, tmp_path, srcs):
        # Without the tail, "the solver crashed" and "you pointed me at a file
        # that does not exist" are the same ERROR verdict with an empty message.
        stub = tmp_path / "shouty"
        stub.write_text("#!/bin/sh\necho 'eqy: error: cannot open config' >&2\n"
                        "exit 2\n")
        stub.chmod(0o755)
        r = LecGateNode(eqy=str(stub), workdir=tmp_path,
                        verbose=False).check(srcs[:1], srcs[1:2], "t", name="s")
        assert r.verdict == ERROR
        assert "cannot open config" in r.message


class TestOperatorErrorsAreNamed:
    def test_missing_source_is_named_not_blamed_on_the_solver(self, tmp_path, srcs):
        # eqy exits 2 with an unparsable log when a source is missing, which is
        # indistinguishable from a solver crash. Catch it first and say so.
        r = LecGateNode(workdir=tmp_path, verbose=False).check(
            srcs[:1], [str(tmp_path / "gone.v")], "t", name="gone")
        assert r.verdict == ERROR and "not found" in r.message
        assert "gone.v" in r.message

    def test_empty_side_proves_nothing(self, tmp_path, srcs):
        r = LecGateNode(workdir=tmp_path, verbose=False).check(
            srcs[:1], [], "t", name="empty")
        assert r.verdict == ERROR and r.admits_edit is False


class TestToolFailuresFailClosed:
    def test_missing_binary_is_an_error_not_an_exception(self, tmp_path, srcs):
        n = LecGateNode(eqy="definitely-not-eqy-xyz", workdir=tmp_path,
                        verbose=False)
        r = n.check(srcs[:1], srcs[1:2], "top", name="missing")
        assert r.verdict == ERROR
        assert r.admits_edit is False and r.is_refutation is False
        assert "not found" in r.message

    def test_a_binary_that_cannot_be_executed_is_an_error_not_an_exception(
            self, tmp_path, srcs):
        # REGRESSION, same trap as the kepler backend: an eqy path that EXISTS
        # but is not executable raises PermissionError, and `except
        # FileNotFoundError` alone let it propagate out of check().
        broken = tmp_path / "eqy-not-executable"
        broken.write_text("#!/bin/sh\necho hi\n")
        broken.chmod(0o000)
        r = LecGateNode(eqy=str(broken), workdir=tmp_path,
                        verbose=False).check(srcs[:1], srcs[1:2], "t",
                                             name="noexec")
        assert r.verdict == ERROR and r.admits_edit is False
        assert "could not be executed" in r.message

    def test_rc_zero_with_an_unparsable_log_is_not_a_pass(self, tmp_path, srcs):
        liar = tmp_path / "liar"
        liar.write_text("#!/bin/sh\necho 'everything is fine'\nexit 0\n")
        liar.chmod(0o755)
        r = LecGateNode(eqy=str(liar), workdir=tmp_path,
                        verbose=False).check(srcs[:1], srcs[1:2], "t", name="liar")
        assert r.returncode == 0
        assert r.verdict == ERROR and r.admits_edit is False

    def test_timeout_kills_a_sigterm_ignoring_tool_tree(self, tmp_path, srcs):
        # subprocess.run(timeout=) only kills the direct child; eqy spawns
        # yosys/sby/a solver, and a surviving grandchild holding the stdout
        # pipe makes the timeout never fire at all.
        stub = tmp_path / "hang"
        stub.write_text("#!/bin/sh\ntrap '' TERM\nsleep 60 &\nwait\n")
        stub.chmod(0o755)
        n = LecGateNode(eqy=str(stub), workdir=tmp_path, timeout_s=1.0,
                        verbose=False)
        t0 = time.monotonic()
        r = n.check(srcs[:1], srcs[1:2], "t", name="slow")
        assert time.monotonic() - t0 < 20.0
        assert r.verdict == TIMEOUT and r.admits_edit is False

    def test_log_is_always_written(self, tmp_path, srcs):
        stub = tmp_path / "noisy"
        stub.write_text("#!/bin/sh\necho hello-from-the-checker\nexit 3\n")
        stub.chmod(0o755)
        r = LecGateNode(eqy=str(stub), workdir=tmp_path,
                        verbose=False).check(srcs[:1], srcs[1:2], "t", name="n")
        assert r.log_path and Path(r.log_path).exists()
        assert "hello-from-the-checker" in Path(r.log_path).read_text()


# --- undef_init: the false-refutation defence ---------------------------------
# picorv32's count_cycle/count_instr are free-running 64-bit registers with no
# initial value (picorv32.v:1433). Under eqy's safe-replacement semantics the
# gold side is read 3-valued and the gate side 2-valued with each x replaced by
# an ARBITRARY unconstrained value, so the two sides' undefined state are
# independent variables and an identical pair is refutable. Measured: the gate
# returned REFUTED on exactly those two partitions, with a counterexample whose
# only difference was bit 63. These tests pin the defence, and pin that the
# proof it buys is never reported as an unqualified one.

def _cfg(**kw):
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        node = LecGateNode(workdir=d, verbose=False, **kw)
        return node.write_config(["/g/a.v"], ["/g/b.v"], "top", d / "c.eqy").read_text()


class TestUndefInit:
    def test_resolution_is_emitted_into_both_sides(self):
        # An asymmetric resolution would itself manufacture a counterexample,
        # so this counts occurrences rather than merely asserting presence.
        assert _cfg().count("setundef -zero -init") == 2

    @pytest.mark.parametrize("block", ["gold", "gate"])
    def test_resolution_lands_after_prep_and_before_memory_map(self, block):
        # The flip-flops must exist before their init values can be set. This
        # is the order validated against picorv32.
        body = _cfg()
        section = body.split(f"[{block}]")[1].split("[")[0]
        assert section.index("prep -top") < section.index("setundef -zero -init")
        assert section.index("setundef -zero -init") < section.index("memory_map")

    def test_opting_out_emits_nothing(self):
        assert "setundef" not in _cfg(undef_init=None)

    @pytest.mark.parametrize("value", sorted(UNDEF_INIT_VALUES))
    def test_every_accepted_value_renders(self, value):
        assert f"setundef -{value} -init" in _cfg(undef_init=value)

    def test_a_bad_value_fails_at_construction(self):
        # Not an hour later inside a solver, where a bad yosys command surfaces
        # as rc=2 with an unparsable log, an ERROR verdict indistinguishable
        # from a cautious gate.
        with pytest.raises(ValueError):
            LecGateNode(undef_init="anyseq")

    def test_a_proof_under_a_resolved_init_is_labelled_not_bare(self):
        r = LecResult(PROVEN, "eqy", 1.0, abstraction="undef-init=zero")
        assert r.evidence_strength == "proof[undef-init=zero]"
        assert r.admits_edit is True

    def test_an_unqualified_proof_stays_unqualified(self):
        assert LecResult(PROVEN, "eqy", 1.0).evidence_strength == "proof"


# --- jobs: partitions are independent, so prove them concurrently -------------
# eqy writes one make target chain per partition and a single `all:` depending
# on all of them (eqy.py:1138), then runs `make{kopt}{jopt} -f strategies.mk`
# (eqy.py:1156). Without -j that is one proof at a time, 590 of them on
# picorv32, however many cores the worker holds.

class TestJobs:
    def test_jobs_none_keeps_the_plain_command(self):
        n = LecGateNode(verbose=False)
        assert n._argv("eqy", Path("/w/c.eqy")) == ["eqy", "-f", "/w/c.eqy"]

    def test_jobs_is_passed_through_to_eqy(self):
        n = LecGateNode(verbose=False, jobs=12)
        assert n._argv("eqy", Path("/w/c.eqy")) == ["eqy", "-j", "12", "-f", "/w/c.eqy"]

    def test_a_nonsense_job_count_fails_at_construction(self):
        with pytest.raises(ValueError):
            LecGateNode(jobs=0)

    def test_auto_jobs_is_derived_from_the_runtime_not_hardcoded(self):
        n = auto_jobs()
        assert 1 <= n <= (os.cpu_count() or 1)
