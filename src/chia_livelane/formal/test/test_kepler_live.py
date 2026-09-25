"""End-to-end tests against a real ``kepler-formal``, on pairs whose answer is known.

Named ``*_live`` to match ``test_lec_gate_live.py``: these need the tool and
skip cleanly when it is absent instead of failing.

``smoke/sec_gold.v`` vs ``smoke/sec_retimed.v`` is the pair that justifies
carrying a second backend at all. It is equivalent, 2000 cycles of Verilator
simulation with random stimulus report 0 mismatches, and on this host
``eqy`` REFUTES it ("partitions not equivalent: rt.r, rt.q") while
``kepler-formal`` proves it. Two formal tools disagreeing is a finding, and
:func:`cross_check` is what reports it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from chia_livelane.formal.lec_gate import (DEFAULT_STRATEGIES, PROVEN, REFUTED,
                                           UNDECIDED, KeplerBackend,
                                           LecGateNode, cross_check)

SMOKE = Path(__file__).parent / "smoke"

pytestmark = pytest.mark.skipif(
    shutil.which("kepler-formal") is None,
    reason="kepler-formal not on PATH; build it with scripts/build/kepler.sh")


def _kepler(tmp_path, gold, gate, top, name, **kw):
    b = KeplerBackend(workdir=tmp_path, timeout_s=600.0, verbose=False, **kw)
    return b.check([str(SMOKE / gold)], [str(SMOKE / gate)], top, name=name)


class TestRealKepler:
    def test_a_harmless_refactor_is_proved(self, tmp_path):
        r = _kepler(tmp_path, "gold.v", "equivalent.v", "tiny", "ok")
        assert r.verdict == PROVEN, r
        assert r.admits_edit is True and r.is_refutation is False
        # A proof that covered only some outputs is not a proof of the design.
        assert r.coverage_pct == 100.0, r

    def test_a_functional_bug_is_refuted(self, tmp_path):
        r = _kepler(tmp_path, "gold.v", "refuted.v", "tiny", "bad")
        assert r.verdict == REFUTED, r
        assert r.is_refutation is True and r.admits_edit is False

    def test_the_proof_names_its_abstraction(self, tmp_path):
        # kepler's SEC proof holds on cycles where both outputs are
        # binary-defined. That caveat must reach the caller, not stop at the log.
        r = _kepler(tmp_path, "gold.v", "equivalent.v", "tiny", "abstr")
        assert r.abstraction, r
        assert r.abstraction in r.evidence_strength

    def test_a_retimed_design_is_proved(self, tmp_path):
        # THE reason this backend exists: the sequential boundary has moved.
        r = _kepler(tmp_path, "sec_gold.v", "sec_retimed.v", "rt", "retimed")
        assert r.verdict == PROVEN, r
        assert r.coverage_pct == 100.0, r


@pytest.mark.skipif(shutil.which("eqy") is None, reason="eqy not on PATH")
class TestAgainstEqy:
    def _both(self, tmp_path, gold, gate, top):
        k = _kepler(tmp_path, gold, gate, top, "k")
        e = LecGateNode(workdir=tmp_path, strategies=DEFAULT_STRATEGIES,
                        timeout_s=600.0, verbose=False).check(
            [str(SMOKE / gold)], [str(SMOKE / gate)], top, name="e")
        return e, k

    def test_the_backends_agree_on_the_combinational_fixtures(self, tmp_path):
        for gate in ("equivalent.v", "refuted.v"):
            e, k = self._both(tmp_path, "gold.v", gate, "tiny")
            assert cross_check(e, k)["agree"] is True, (gate, e, k)

    def test_the_backends_disagree_on_the_retimed_pair(self, tmp_path):
        # Not a flaky test: it pins a real, reproducible capability gap. If eqy
        # ever learns to prove this, THAT is the news, and this test says so.
        e, k = self._both(tmp_path, "sec_gold.v", "sec_retimed.v", "rt")
        d = cross_check(e, k)
        assert k.verdict == PROVEN, k
        assert e.verdict == REFUTED, e
        assert d["agree"] is False and "DISAGREEMENT" in d["note"]


class TestFailsClosed:
    def test_a_design_it_cannot_ingest_is_not_a_pass(self, tmp_path):
        # naja's SNL lowering rejects plain-Verilog constructs the SystemVerilog
        # front end parses happily; the result must be an error, never a pass.
        bad = tmp_path / "bad.sv"
        bad.write_text("module rt; initial $display(\"no ports\"); endmodule\n")
        b = KeplerBackend(workdir=tmp_path, timeout_s=120.0, verbose=False)
        r = b.check([str(SMOKE / "sec_gold.v")], [str(bad)], "rt", name="ingest")
        assert r.admits_edit is False, r
        assert r.verdict != PROVEN and r.verdict != REFUTED, r

    def test_an_impossible_budget_times_out_rather_than_passing(self, tmp_path):
        b = KeplerBackend(workdir=tmp_path, timeout_s=0.001, verbose=False)
        r = b.check([str(SMOKE / "gold.v")], [str(SMOKE / "equivalent.v")],
                    "tiny", name="tmo")
        assert r.admits_edit is False, r
