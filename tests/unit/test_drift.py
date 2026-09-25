"""The load-drift guard must disqualify contended timings, and only those.

This guard exists because two corpus runs were reported with contaminated
numbers before anyone noticed, one of them contaminated by a recursive grep
started on the same box while the run was in flight.

Its first version used a within-instance ratio and got BOTH directions wrong on
the same pair of runs:

  * it missed #167, which rose only 1.85x but rose to 22.2 against a run
    baseline of 13.3, and came out 2x slower than its neighbours;
  * it flagged #48, which "rose" 9.8x purely because it was the FIRST instance
    and sampled an idle box before the harness's own 12 jobs started.

A guard that only fires correctly in one direction is worse than no guard,
because it looks like it is working. So both directions are tested here, with
the real numbers from those runs.
"""
from __future__ import annotations

import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _hwebench():
    spec = importlib.util.spec_from_file_location(
        "hwebench", ROOT / "scripts" / "chia" / "hwebench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(number: int, start: float, end: float, *, wall: float = 20.0,
         warm_wall: float = 200.0, prove: float = 8.0,
         warm_prove: float = 200.0, status: str = "ok") -> dict:
    return {
        "number": number, "status": status, "file": "rtl/x.sv",
        "scope": "whole-design",
        "load_at_start": {"load1": start, "cpus": 24},
        "load_at_end": {"load1": end, "cpus": 24},
        "warm": {"verdict": "proven", "prove_s": warm_prove,
                 "wall_s": warm_wall, "setup_s": 8.0, "partitions": 6000},
        "measure": {"verdict": "refuted", "failed": ["a.b"], "reuse_pct": 96.0,
                    "prove_s": prove, "wall_s": wall, "setup_s": 12.0,
                    "partitions": 6000},
    }


def _summarise(rows: list[dict]) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        _hwebench().summarise(rows)
    return buf.getvalue()


class TestCatchesContention:
    def test_flags_the_instance_that_ran_hot(self) -> None:
        """#167's real numbers: peak 22.2 against a 13.3 baseline."""
        rows = [_row(48, 9.79, 13.21), _row(54, 13.21, 13.30),
                _row(157, 13.30, 12.09), _row(166, 12.09, 12.03),
                _row(167, 12.03, 22.22, wall=45.0)]
        out = _summarise(rows)
        assert "#167" in out
        assert "load drift on 1 instance" in out
        assert "EXCLUDED" in out

    def test_drifted_timings_leave_the_median(self) -> None:
        """Flagging without excluding is a warning nobody acts on."""
        rows = [_row(n, 12.0, 12.0, wall=20.0, warm_wall=200.0)
                for n in (1, 2, 3, 4)]
        rows.append(_row(5, 12.0, 30.0, wall=100.0, warm_wall=200.0))
        out = _summarise(rows)
        # the clean four are all 10.00x; the drifted one is 2.00x and must not
        # drag the median down
        assert "10.00x" in out
        assert "4 of 5" in out

    def test_contended_for_its_whole_duration_is_still_caught(self) -> None:
        """Start == end, so a within-instance ratio sees nothing."""
        rows = [_row(n, 12.0, 12.0) for n in (1, 2, 3, 4)]
        rows.append(_row(5, 30.0, 30.0))
        assert "#5" in _summarise(rows)


class TestDoesNotCryWolf:
    def test_first_instance_spin_up_is_not_drift(self) -> None:
        """#48's real numbers: 1.3 -> 12.7 is the harness's OWN jobs starting.

        12.7 is this run's steady state. Calling that drift throws away a
        perfectly good measurement, and #48's ratio is if anything conservative:
        its no-cache baseline ran on the idle box and its cached run did not.
        """
        rows = [_row(48, 1.33, 12.67), _row(54, 12.67, 12.9),
                _row(157, 12.9, 13.1), _row(166, 13.1, 12.8),
                _row(167, 12.8, 13.0)]
        out = _summarise(rows)
        assert "load drift" not in out
        assert "5 of 5" in out

    def test_steady_run_flags_nothing(self) -> None:
        rows = [_row(n, 12.0, 12.5) for n in range(1, 7)]
        assert "load drift" not in _summarise(rows)


class TestRefusesToVetWithoutABaseline:
    def test_three_instances_is_not_a_baseline(self) -> None:
        """With n<4 the median peak is noise. Say so; do not report clean."""
        rows = [_row(1, 12.0, 12.0), _row(2, 12.0, 12.0), _row(3, 1.0, 40.0)]
        out = _summarise(rows)
        assert "too few to establish a baseline" in out
        assert "unvetted" in out
        assert "load drift on" not in out


class TestPeakIsThisInstanceOnly:
    """peak() took max(start, end). Instances run sequentially, so an
    instance's load_at_start IS its predecessor's load_at_end, the same
    reading. One spike then disqualified two instances, and did."""

    def _chain(self, loads: list[float]) -> list[dict]:
        """Rows whose samples chain the way a sequential run's really do."""
        rows = []
        for i, end in enumerate(loads):
            start = loads[i - 1] if i else 1.0
            rows.append(_row(i + 1, start, end))
        return rows

    def test_the_successor_of_a_spike_is_not_flagged(self) -> None:
        """The real hierarchical arm: #167 ran hot, #176 merely followed it."""
        import importlib.util
        from pathlib import Path as P
        spec = importlib.util.spec_from_file_location(
            "loadguard", ROOT / "scripts" / "chia" / "loadguard.py")
        lg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lg)
        rows = self._chain([13.21, 13.30, 12.09, 12.03, 22.22, 15.63,
                            12.37, 11.64, 12.25, 13.23, 10.81, 12.31])
        drift, _ = lg.drifted(rows)
        nums = sorted(drift)
        assert 5 in nums, "the instance that actually ran hot must be flagged"
        assert 6 not in nums, "its successor inherited the sample, not the load"

    def test_first_instance_start_sample_is_ignored(self) -> None:
        """#48 sampled an idle box before the harness's own jobs existed."""
        rows = self._chain([12.67, 12.9, 13.1, 12.8, 13.0])
        rows[0]["load_at_start"] = {"load1": 1.33, "cpus": 24}
        assert "load drift" not in _summarise(rows)


class TestBaselineIsATrueMedian:
    def test_even_n_averages_the_middle_pair(self) -> None:
        """peaks[len//2] is the UPPER middle, which biases the guard permissive
       , the mistake hwebench's own mid() carries a warning about."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "loadguard", ROOT / "scripts" / "chia" / "loadguard.py")
        lg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lg)
        rows = [_row(i + 1, 1.0, v) for i, v in enumerate([10.0, 10.0, 20.0, 20.0])]
        assert lg.baseline(rows) == 15.0          # not 20.0, the upper middle

        # And it changes a verdict. The candidate has to be IN the sample,
        # appending it shifts the median, which is what made the first version
        # of this test wrong. peaks = [10,10,10,20,20,23]: true median 15, so
        # 23 is 1.53x and drifts; upper-middle 20 makes it 1.15x and clean.
        rows = [_row(i + 1, 1.0, v) for i, v in
                enumerate([10.0, 10.0, 10.0, 20.0, 20.0, 23.0])]
        assert lg.baseline(rows) == 15.0
        assert 6 in lg.drifted(rows)[0]


class TestTheBaselineChecksItself:
    """The median assumes contention is a MINORITY. It need not be."""

    def _lg(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "loadguard", ROOT / "scripts" / "chia" / "loadguard.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def _rows(self, ends):
        return [{"number": i + 1, "load_at_end": {"load1": v, "cpus": 24}}
                for i, v in enumerate(ends)]

    def test_half_contended_run_is_declared_unvetted(self) -> None:
        """[12,12,12,22,22,22]: median 17, nothing over 1.5x, run looks clean."""
        lg = self._lg()
        rows = self._rows([12, 12, 12, 22, 22, 22])
        assert not lg.drifted(rows)[0], "nothing exceeds the threshold"
        why = lg.baseline_is_suspect(rows)
        assert why and "unvetted" in why

    def test_a_single_spike_is_drift_not_a_suspect_baseline(self) -> None:
        """The guard working normally must not also cry 'unvetted'."""
        lg = self._lg()
        rows = self._rows([12, 12, 12, 12, 12, 22])
        assert 6 in lg.drifted(rows)[0]
        assert lg.baseline_is_suspect(rows) is None

    def test_a_steady_run_is_neither(self) -> None:
        lg = self._lg()
        rows = self._rows([12, 12.2, 11.9, 12.1, 12.3, 12.0])
        assert not lg.drifted(rows)[0]
        assert lg.baseline_is_suspect(rows) is None

    def test_the_real_runs(self) -> None:
        """flat2 was clean; hier had one real spike and caught it."""
        import json
        lg = self._lg()
        for name, expect_drift in (("flat2", False), ("hier", True)):
            f = ROOT / "var" / "final" / f"ibex-{name}.json"
            if not f.exists():
                continue
            rows = json.loads(f.read_text())["rows"]
            assert bool(lg.drifted(rows)[0]) is expect_drift
            assert lg.baseline_is_suspect(rows) is None
