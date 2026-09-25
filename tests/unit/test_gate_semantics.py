"""The equivalence gate: fail closed, and never call 'undecided' a refutation."""

import pytest

from chia_livelane.lec_gate import (ERROR, PROVEN, REFUTED, TIMEOUT, LecResult,
                                    cross_check, parse_eqy_log)

# eqy emits the SAME summary line and the SAME exit code for a real
# counterexample and for a partition it could not decide. Only the per-partition
# reason separates them. Both strings below are verbatim from real runs.
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


def test_undecided_is_not_a_refutation():
    p = parse_eqy_log(UNDECIDED_LOG)
    assert p["unknown_partitions"] == ["picorv32.count_cycle"]
    assert not p.get("not_equivalent_partitions")


def test_counterexample_is_a_refutation():
    p = parse_eqy_log(REFUTED_LOG)
    assert p["not_equivalent_partitions"] == ["nerv.next_rd"]
    assert not p.get("unknown_partitions")


def test_proved_log():
    assert parse_eqy_log(PROVEN_LOG)["proved"] is True


@pytest.mark.parametrize("verdict", [REFUTED, ERROR, TIMEOUT, "skipped"])
def test_gate_fails_closed(verdict):
    assert LecResult(verdict, "eqy", 1.0).admits_edit is False


def test_only_proven_admits():
    assert LecResult(PROVEN, "eqy", 1.0).admits_edit is True


def test_bounded_proof_admits_but_is_labelled():
    b = LecResult(PROVEN, "eqy", 1.0, bounded=True, depth=2)
    assert b.admits_edit is True
    assert b.is_unbounded_proof is False
    assert b.evidence_strength == "bounded-proof(depth=2)"


def test_unknown_verdict_string_rejected():
    with pytest.raises(ValueError):
        LecResult("probably_fine", "eqy", 1.0)


def test_backend_disagreement_is_reported_not_averaged():
    d = cross_check(LecResult(PROVEN, "eqy", 1.0),
                    LecResult(REFUTED, "lhd-lec", 0.5))
    assert d["agree"] is False and "DISAGREEMENT" in d["note"]
