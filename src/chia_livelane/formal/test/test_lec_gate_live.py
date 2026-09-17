"""End-to-end tests against a real ``eqy``, on a pair whose answer is known.

Named ``*_live`` to match ``chia/vlsi/tests/test_hammer_live.py`` and
``chia/simulators/tests/test_champsim_live.py``: these need a tool, and skip
cleanly when it is absent instead of failing.

The fixture in ``smoke/`` is the same one ``dockerfiles/EqyDockerfile`` runs at
image-build time, so a green unit test and a green image build are making the
same claim about the same files.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from chia_livelane.formal.lec_gate import (DEFAULT_STRATEGIES, PROVEN, REFUTED,
                                           LecGateNode, lec_gate)

SMOKE = Path(__file__).parent / "smoke"

pytestmark = pytest.mark.skipif(
    shutil.which("eqy") is None,
    reason="eqy not on PATH; run inside ghcr.io/ucb-bar/chia-eqy")


def _check(tmp_path, candidate, name):
    node = LecGateNode(workdir=tmp_path, strategies=DEFAULT_STRATEGIES,
                       timeout_s=300.0, verbose=False)
    return node.check([str(SMOKE / "gold.v")], [str(SMOKE / candidate)],
                      "tiny", name=name)


class TestRealEqy:
    def test_a_harmless_refactor_is_proved(self, tmp_path):
        # Commuted operands, sum hoisted into a wire. A gate that cannot prove
        # this would reject every harmless refactor an agent proposes.
        r = _check(tmp_path, "equivalent.v", "live_ok")
        assert r.verdict == PROVEN, r
        assert r.admits_edit is True
        assert r.is_refutation is False

    def test_a_functional_bug_is_refuted(self, tmp_path):
        # Subtraction where the design adds: it elaborates, synthesises and
        # simulates cleanly for every a >= b, so nothing short of an
        # equivalence check catches it.
        r = _check(tmp_path, "refuted.v", "live_bad")
        assert r.verdict == REFUTED, r
        assert r.is_refutation is True
        assert r.admits_edit is False

    def test_the_fast_strategy_cannot_admit_a_stuck_output(self, tmp_path):
        # REGRESSION, found by running the ladder rather than reading it.
        # `y <= 4'b0` against `y <= a + b` is not equivalent by any reading, but
        # eqy's `use sat` strategy reports "Induction step proven: SUCCESS!" on
        # it and eqy turns that into PASS. With `induct` listed first the gate
        # returned verdict=proven / admits_edit=True -- a false accept, the one
        # outcome this module exists to make impossible.
        #
        # This asserts the OUTCOME (not admitted), so it keeps holding whatever
        # the ladder is reordered to, and fails loudly if `induct` is ever put
        # back in front.
        r = _check(tmp_path, "refuted_const.v", "live_const")
        assert r.admits_edit is False, (
            f"FALSE ACCEPT: a stuck-at-constant output was admitted: {r}")
        assert r.verdict == REFUTED, r
        assert r.is_refutation is True

    def test_refutation_names_the_failing_partition(self, tmp_path):
        r = _check(tmp_path, "refuted.v", "live_named")
        assert "tiny.y" in r.message, r.message

    def test_a_counterexample_outranks_an_undecided_strategy(self, tmp_path):
        # The precedence rule in check(): when ONE log carries both reasons for
        # the SAME partition, REFUTED must win. Returning UNDECIDED there would
        # throw away a real refutation and admit a bug.
        #
        # Producing that log needs `induct` to run and answer "unknown" BEFORE
        # `smt` refutes, so this test pins the ladder explicitly instead of
        # using DEFAULT_STRATEGIES. The default now leads with `smt` (see
        # DEFAULT_STRATEGIES: `induct` first is a false-accept risk), and `smt`
        # refutes this fixture outright, so the default log carries only one
        # reason and could not exercise the precedence rule at all.
        # undef_init=None is load-bearing here, not incidental. The precedence
        # rule can only be exercised by a partition some strategy calls
        # "unknown", and `induct` only says that while the fixture's registers
        # still carry undefined initial state. Resolving it (the default) lets
        # `induct` decide the partition outright, the log then carries a single
        # reason, and this test would pass vacuously.
        node = LecGateNode(
            workdir=tmp_path, timeout_s=300.0, verbose=False, undef_init=None,
            strategies=(("induct", ("use sat", "depth 10")),
                        ("smt", ("use sby", "engine smtbmc yices", "depth 10"))))
        r = node.check([str(SMOKE / "gold.v")], [str(SMOKE / "refuted.v")],
                       "tiny", name="live_ladder")
        log = Path(r.log_path).read_text()
        assert "equivalence unknown" in log
        assert "partitions not equivalent" in log
        assert r.verdict == REFUTED

    def test_log_is_kept_for_audit(self, tmp_path):
        r = _check(tmp_path, "equivalent.v", "live_log")
        assert r.log_path and Path(r.log_path).exists()
        assert "eqy" in Path(r.log_path).read_text().lower()

    def test_node_function_returns_real_bools(self, tmp_path):
        # The CHIA convention: gate functions return bools callers can branch
        # on, not strings they have to parse.
        fn = getattr(lec_gate, "_chia_original", lec_gate)
        d = fn([str(SMOKE / "gold.v")], [str(SMOKE / "refuted.v")], "tiny",
               workdir=str(tmp_path),
               strategies=[("induct", ["use sat", "depth 10"]),
                           ("smt", ["use sby", "engine smtbmc yices",
                                    "depth 10"])])
        assert d["equivalent"] is False
        assert d["refuted"] is True
        assert d["verdict"] == REFUTED
        assert isinstance(d["wall_s"], float)
