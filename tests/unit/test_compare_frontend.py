"""The front-end adjudicator must gate on the verdict and nothing else.

A strategy-ladder comparison would have been the obvious tool for the flat-vs-
hierarchical run, and it would have reported a soundness failure that does not
exist: it requires reuse to be identical and failing-partition NAMES to match,
and `read_slang --keep-hierarchy` changes both by construction. Its output
would have been "reuse differs" and "DIFFERENT failing partitions" on every
instance, exit 2, indistinguishable from a real refusal.

So the asymmetry here is the same as the ladder comparator's, but the gate is
narrower: a verdict disagreement disqualifies the flag; everything else is
reported. These tests hold that line from both sides, it must not pass a
verdict flip, and it must not fail a rename.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "chia" / "compare_frontend.py"


def _row(number: int, verdict: str, failed: list[str], *, file: str,
         parts: int = 6000, reuse: float = 96.0, wall: float = 20.0,
         setup: float = 12.0, status: str = "ok") -> dict:
    return {
        "number": number, "status": status, "file": file,
        "warm": {"verdict": "proven", "prove_s": 200.0, "wall_s": 220.0,
                 "setup_s": 8.0, "partitions": parts},
        "measure": {"verdict": verdict, "failed": failed, "reuse_pct": reuse,
                    "prove_s": 8.0, "wall_s": wall, "setup_s": setup,
                    "partitions": parts},
    }


def _run(tmp_path: Path, a_rows, b_rows, rtl: Path | None = None):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"rows": a_rows}))
    b.write_text(json.dumps({"rows": b_rows}))
    cmd = [sys.executable, str(SCRIPT), str(a), str(b),
           "--name-a", "flat", "--name-b", "hier"]
    if rtl is not None:
        cmd += ["--rtl", str(rtl)]
    return subprocess.run(cmd, capture_output=True, text=True)


class TestTheGate:
    def test_verdict_flip_is_disqualifying(self, tmp_path: Path) -> None:
        """The direction that matters: one arm sees a bug, the other does not."""
        a = [_row(1, "refuted", ["core.x.sig"], file="rtl/ibex_decoder.sv")]
        b = [_row(1, "proven", [], file="rtl/ibex_decoder.sv")]
        r = _run(tmp_path, a, b)
        assert r.returncode == 2
        assert "VERDICT DISAGREEMENT" in r.stdout
        assert "does not ship" in r.stdout

    def test_renamed_partitions_are_not_a_failure(self, tmp_path: Path) -> None:
        """Module-scoped names differ from flat paths. That is the flag working."""
        a = [_row(1, "refuted", ["ibex_core.if_stage_i.illegal_c_insn"],
                  file="rtl/ibex_compressed_decoder.sv", parts=5261)]
        b = [_row(1, "refuted",
                  ["ibex_core.if_stage_i.compressed_decoder_i.illegal_instr_o"],
                  file="rtl/ibex_compressed_decoder.sv", parts=2613)]
        r = _run(tmp_path, a, b)
        assert r.returncode == 0
        assert "every instance agrees on the verdict" in r.stdout

    def test_different_reuse_is_not_a_failure(self, tmp_path: Path) -> None:
        """The two arms do not share partitions, so reuse cannot match."""
        a = [_row(1, "refuted", ["c.x"], file="rtl/ibex_decoder.sv", reuse=98.3)]
        b = [_row(1, "refuted", ["c.x"], file="rtl/ibex_decoder.sv", reuse=94.1)]
        r = _run(tmp_path, a, b)
        assert r.returncode == 0
        assert "expected to differ" in r.stdout

    def test_unmeasured_side_is_skipped_not_counted(self, tmp_path: Path) -> None:
        a = [_row(1, "refuted", ["c.x"], file="rtl/ibex_decoder.sv")]
        b = [_row(1, "refuted", [], file="rtl/ibex_decoder.sv",
                  status="patch-failed")]
        r = _run(tmp_path, a, b)
        # The wording tightened: one-arm-only coverage is now named as a hole
        # rather than reported as a neutral skip.
        assert "ASYMMETRIC" in r.stdout
        assert r.returncode == 0


class TestLocalisation:
    def _rtl(self, tmp_path: Path) -> Path:
        d = tmp_path / "rtl"
        d.mkdir()
        (d / "ibex_if_stage.sv").write_text(
            "module ibex_if_stage;\n"
            "  ibex_compressed_decoder compressed_decoder_i (\n"
            "    .clk_i(clk)\n"
            "  );\n"
            "endmodule\n")
        (d / "ibex_compressed_decoder.sv").write_text(
            "module ibex_compressed_decoder;\nendmodule\n")
        return d

    def test_counts_the_arm_that_names_the_changed_module(
            self, tmp_path: Path) -> None:
        """The real #48: flat names the parent's wire, hier the changed module."""
        a = [_row(1, "refuted", ["ibex_core.if_stage_i.illegal_c_insn"],
                  file="rtl/ibex_compressed_decoder.sv")]
        b = [_row(1, "refuted",
                  ["ibex_core.if_stage_i.compressed_decoder_i.illegal_instr_o"],
                  file="rtl/ibex_compressed_decoder.sv")]
        r = _run(tmp_path, a, b, rtl=self._rtl(tmp_path))
        assert "flat: 0/1 localised into the changed module" in r.stdout
        assert "hier: 1/1 localised into the changed module" in r.stdout

    def test_unmappable_module_is_excluded_not_scored_zero(
            self, tmp_path: Path) -> None:
        """A module we cannot find an instance for is unknown, not a miss."""
        a = [_row(1, "refuted", ["c.x"], file="rtl/ibex_icache.sv")]
        b = [_row(1, "refuted", ["c.x"], file="rtl/ibex_icache.sv")]
        r = _run(tmp_path, a, b, rtl=self._rtl(tmp_path))
        assert "could not be mapped" in r.stdout
        assert "localised into the changed module" not in r.stdout


class TestSharesTheLoadGuard:
    """The comparator divides two wall-clocks, so it must disqualify the same
    instances the harness does. A guard present at one call site and absent at
    the other is how a contaminated number ships, which is why the rule lives
    in loadguard.py and both import it rather than each keeping a copy."""

    def _rows(self, dirty_peak: float | None):
        rows = []
        for i, n in enumerate((1, 2, 3, 4, 5)):
            pk = 12.0
            if dirty_peak is not None and n == 5:
                pk = dirty_peak
            r = _row(n, "refuted", ["c.x"], file="rtl/ibex_decoder.sv",
                     wall=100.0 if pk > 12.0 else 20.0)
            r["load_at_start"] = {"load1": 12.0, "cpus": 24}
            r["load_at_end"] = {"load1": pk, "cpus": 24}
            rows.append(r)
        return rows

    def test_contended_instance_leaves_the_timing_medians(
            self, tmp_path: Path) -> None:
        a = self._rows(None)
        b = self._rows(30.0)          # #5 ran at 30 against a baseline of 12
        r = _run(tmp_path, a, b)
        assert "load drift, timings not counted" in r.stdout
        assert "dropped for load drift: [5]" in r.stdout
        assert r.returncode == 0      # drift is not a soundness failure

    def test_partition_counts_still_count_every_instance(
            self, tmp_path: Path) -> None:
        """Counts are not times. Dropping them would throw away good evidence."""
        r = _run(tmp_path, self._rows(None), self._rows(30.0))
        assert "(5 instances; counts, so load cannot move them)" in r.stdout

    def test_clean_run_drops_nothing(self, tmp_path: Path) -> None:
        r = _run(tmp_path, self._rows(None), self._rows(None))
        assert "load drift" not in r.stdout
        assert "(5 instances)" in r.stdout


class TestTheGateRefusesToOverclaim:
    """Each of these used to be absorbed into 'every instance agrees'."""

    def test_asymmetric_coverage_is_named(self, tmp_path: Path) -> None:
        """One arm measured it, the other did not. That is a hole, not a pass."""
        a = [_row(1, "refuted", ["c.x"], file="rtl/ibex_decoder.sv"),
             _row(2, "refuted", ["c.x"], file="rtl/ibex_decoder.sv")]
        b = [_row(1, "refuted", ["c.x"], file="rtl/ibex_decoder.sv"),
             _row(2, "refuted", [], file="rtl/ibex_decoder.sv",
                  status="patch-failed")]
        r = _run(tmp_path, a, b)
        assert "ASYMMETRIC" in r.stdout
        assert "measured on ONE arm only" in r.stdout
        assert "(1 compared)" in r.stdout, "the hole must leave the count"

    def test_both_sides_skipped_is_not_asymmetric(self, tmp_path: Path) -> None:
        a = [_row(1, "refuted", [], file="rtl/ibex_decoder.sv",
                  status="file-outside-cone")]
        b = [_row(1, "refuted", [], file="rtl/ibex_decoder.sv",
                  status="file-outside-cone")]
        r = _run(tmp_path, a, b)
        assert "neither arm measured it" in r.stdout
        assert "ASYMMETRIC" not in r.stdout

    def test_two_timeouts_are_not_agreement(self, tmp_path: Path) -> None:
        """'timeout' == 'timeout' is two non-answers."""
        a = [_row(1, "timeout", [], file="rtl/ibex_decoder.sv"),
             _row(2, "refuted", ["c.x"], file="rtl/ibex_decoder.sv")]
        b = [_row(1, "timeout", [], file="rtl/ibex_decoder.sv"),
             _row(2, "refuted", ["c.x"], file="rtl/ibex_decoder.sv")]
        r = _run(tmp_path, a, b)
        assert "did not settle" in r.stdout
        assert "(1 compared)" in r.stdout

    def test_error_on_one_side_is_not_a_verdict_disagreement(
            self, tmp_path: Path) -> None:
        """It is a non-answer, which is a different report from 'unsound'."""
        a = [_row(1, "refuted", ["c.x"], file="rtl/ibex_decoder.sv")]
        b = [_row(1, "error", [], file="rtl/ibex_decoder.sv")]
        r = _run(tmp_path, a, b)
        assert "did not settle" in r.stdout
        assert "VERDICT DISAGREEMENT" not in r.stdout
        assert r.returncode == 0

    def test_unvetted_timings_are_declared(self, tmp_path: Path) -> None:
        """Below MIN_BASELINE the drift guard disables itself silently."""
        a = [_row(n, "refuted", ["c.x"], file="rtl/ibex_decoder.sv")
             for n in (1, 2)]
        b = [_row(n, "refuted", ["c.x"], file="rtl/ibex_decoder.sv")
             for n in (1, 2)]
        for rows in (a, b):
            for r_ in rows:
                r_["load_at_start"] = {"load1": 12.0, "cpus": 24}
                r_["load_at_end"] = {"load1": 12.0, "cpus": 24}
        out = _run(tmp_path, a, b).stdout
        assert "UNVETTED" in out
        assert "not clean" in out
