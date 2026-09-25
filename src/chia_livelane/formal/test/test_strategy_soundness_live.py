"""Why the strategy ladder is ordered the way it is, re-measured, not assumed.

Needs a real `eqy` and yosys, so it is skipped where they are absent.

The history this pins down:

1. `eqy`'s `use sat` strategy (yosys's `sat -tempinduct`) was measured to
   report PASS on a sequential partition that is NOT equivalent, so the ladder
   was ordered `smt` first and `sat`/`induct` only as a fallback.
2. Separately, `setundef -zero -init` was added to both sides of the miter for
   an unrelated reason: without it a design was refuted against *itself*,
   because uninitialised flops became independent free variables.
3. Fix (2) also closed hazard (1). The false accept came from the miter's
   `in_gold === 1'bx` escape making the induction assertion vacuous while gold
   flops sat at X. Once they are initialised, they never sit at X.

Nobody re-tested the ordering after (2), so the ladder kept paying for a hazard
that no longer existed, roughly 9x on proving.

These tests pin both halves, because the cheap ordering is only safe *while*
`setundef` is there. If someone removes `setundef -zero -init`, the second test
fails and says exactly what it costs.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SMOKE = Path(__file__).parent / "smoke"
FAST_FIRST = ("[strategy simple]\nuse sat\ndepth 3\n\n"
              "[strategy smt]\nuse sby\nengine smtbmc yices\ndepth 10\n")
SLOW_FIRST = ("[strategy smt]\nuse sby\nengine smtbmc yices\ndepth 10\n\n"
              "[strategy induct]\nuse sat\ndepth 10\n")

pytestmark = pytest.mark.skipif(
    shutil.which("eqy") is None or shutil.which("yosys") is None,
    reason="needs a real eqy and yosys on PATH")


def _check(tmp_path: Path, gate_name: str, ladder: str, undef: bool):
    from chia_livelane.formal.proof_cache import (PartitionProofCache,
                                                  incremental_check)

    def blk(f: Path) -> str:
        setundef = "setundef -zero -init\n" if undef else ""
        return f"read_verilog -sv {f}\nprep -top tiny\n{setundef}memory_map"

    cfg = tmp_path / "c.eqy"
    cfg.write_text(f"[gold]\n{blk(SMOKE / 'gold.v')}\n\n"
                   f"[gate]\n{blk(SMOKE / gate_name)}\n\n[collect *]\n\n{ladder}")
    wd = tmp_path / "wd"
    shutil.rmtree(wd, ignore_errors=True)
    return incremental_check(cfg, wd, PartitionProofCache(tmp_path / "cache"),
                             jobs=4)


@pytest.mark.parametrize("gate", ["refuted_const.v", "refuted.v"])
@pytest.mark.parametrize("ladder,label", [(FAST_FIRST, "fast-first"),
                                          (SLOW_FIRST, "slow-first")])
def test_both_ladders_refute_with_setundef(tmp_path, gate, ladder, label):
    """With `setundef -zero -init`, ladder order does not change the answer.

    `refuted_const.v` is the fixture that historically exposed the false
    accept; `refuted.v` is the sibling that did not, and is kept so a
    regression cannot hide behind the easy case.
    """
    got = _check(tmp_path, gate, ladder, undef=True)
    assert got.verdict == "refuted", (
        f"{label} admitted a non-equivalent design ({gate}): "
        f"verdict={got.verdict}")


def test_setundef_is_load_bearing_for_the_fast_ladder(tmp_path):
    """Drop `setundef` and the cheap strategy admits a design that is wrong.

    This is the test that justifies the coupling. It is not describing a bug to
    be fixed, it is pinning WHY `setundef -zero -init` may not be removed
    while the ladder leads with `sat`.
    """
    got = _check(tmp_path, "refuted_const.v", FAST_FIRST, undef=False)
    assert got.verdict == "proven", (
        "expected the documented false accept without setundef; if this now "
        "refutes, the hazard is gone by some other route and the ladder "
        "comment needs re-deriving rather than trusting")


def test_slow_ladder_survives_without_setundef(tmp_path):
    """Ordering is an independent mitigation, not a redundant one.

    Without `setundef`, `smt`-first still refutes: `sby` decides the partition
    before `induct` is ever reached, so the vacuous induction never gets to
    answer. That is exactly what the original ladder comment claimed, and it
    is why leading with `smt` was the right call at the time.

    So there are TWO independent ways to close the hazard:
      - lead with a strategy that does not rely on induction (`smt`), or
      - initialise the flops so the induction is not vacuous (`setundef`).

    Today the config has both. That is what makes leading with the cheap
    strategy safe, and it is also why removing `setundef` while leading with
    `sat` is the one combination that breaks, pinned by the test above.
    """
    got = _check(tmp_path, "refuted_const.v", SLOW_FIRST, undef=False)
    assert got.verdict == "refuted", (
        "smt-first should refute even without setundef; if it does not, the "
        "ladder has only one mitigation left and the cheap ordering must go")
