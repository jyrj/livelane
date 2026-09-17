"""Unit tests for the instrumented runner.

Everything this module reports is a measurement, so these tests check that the
measurements are real (a burned CPU second shows up as a CPU second) and that
the failure paths -- non-zero exit, missing binary, a tool that ignores SIGTERM
-- are all distinguishable rather than collapsing into one "it didn't work".
"""

import json
import time

import pytest

from chia_livelane.vlsi.tool_run import (DEFAULT_SALIENT_ENV, ToolNotFound,
                                         ToolRun, append_jsonl, run_tool)


class TestMeasurement:
    def test_wall_clock_is_real(self):
        r = run_tool(["/bin/sh", "-c", "sleep 0.2"], verbose=False)
        assert r.ok
        assert 0.15 < r.wall_s < 5.0

    def test_cpu_and_rss_are_captured(self):
        burn = ("python3 -c \"import time;b=bytearray(200*1024*1024);"
                "t=time.time();x=0\nwhile time.time()-t<0.5: x+=1\nprint(len(b),x)\"")
        r = run_tool(["/bin/sh", "-c", burn], verbose=False)
        assert r.cpu_s > 0.3, f"cpu time not captured: {r.cpu_s}"
        assert r.peak_rss_mb > 150, f"peak RSS not captured: {r.peak_rss_mb}"

    def test_cpu_s_is_user_plus_sys(self):
        r = ToolRun(argv=["x"], cwd="/", returncode=0, wall_s=1.0,
                    cpu_user_s=2.0, cpu_sys_s=0.5, peak_rss_kb=2048)
        assert r.cpu_s == 2.5
        assert r.peak_rss_mb == 2.0

    def test_salient_env_is_recorded(self):
        r = run_tool(["/bin/sh", "-c", "true"], verbose=False,
                     env={"PATH": "/bin:/usr/bin", "OMP_NUM_THREADS": "4"})
        assert r.env_salient["OMP_NUM_THREADS"] == "4"
        assert set(r.env_salient) == set(DEFAULT_SALIENT_ENV)


class TestFailureModes:
    def test_non_zero_exit_preserved(self):
        r = run_tool(["/bin/sh", "-c", "exit 7"], verbose=False)
        assert r.returncode == 7
        assert r.ok is False and r.timed_out is False

    def test_missing_binary_raises_tool_not_found(self):
        # Distinct from a non-zero exit on purpose: a missing tool means the
        # worker is running the wrong image, not a result about the design.
        with pytest.raises(ToolNotFound):
            run_tool(["definitely-not-a-real-binary-xyz"], verbose=False)

    def test_timeout_kills_a_sigterm_ignoring_tree(self):
        t0 = time.monotonic()
        r = run_tool(["/bin/sh", "-c", "trap '' TERM; sleep 30 & wait"],
                     timeout_s=1.0, verbose=False)
        assert time.monotonic() - t0 < 10.0
        assert r.timed_out is True and r.ok is False

    def test_a_timed_out_run_is_still_a_record(self):
        # A failed evaluation still costs wall-clock, and that cost belongs in
        # the ledger rather than being dropped.
        r = run_tool(["/bin/sh", "-c", "sleep 30"], timeout_s=0.5, verbose=False)
        assert isinstance(r, ToolRun) and r.wall_s > 0.4


class TestLogging:
    def test_streams_are_captured_to_disk(self, tmp_path):
        r = run_tool(["/bin/sh", "-c", "echo out; echo err >&2"],
                     log_dir=tmp_path, label="x", verbose=False)
        assert (tmp_path / "x.stdout.log").read_text().strip() == "out"
        assert (tmp_path / "x.stderr.log").read_text().strip() == "err"
        assert r.stdout_path and r.stderr_path

    def test_no_log_dir_means_no_paths(self):
        r = run_tool(["/bin/sh", "-c", "true"], verbose=False)
        assert r.stdout_path is None and r.stderr_path is None

    def test_input_text_reaches_stdin(self, tmp_path):
        r = run_tool(["/bin/sh", "-c", "cat"], input_text="hello\n",
                     log_dir=tmp_path, label="cat", verbose=False)
        assert r.ok
        assert (tmp_path / "cat.stdout.log").read_text() == "hello\n"


class TestLedger:
    def test_as_dict_is_json_serialisable(self):
        r = run_tool(["/bin/sh", "-c", "true"], label="t", verbose=False)
        d = r.as_dict()
        assert json.loads(json.dumps(d))["ok"] is True
        assert "cpu_s" in d and "peak_rss_mb" in d

    def test_append_jsonl_creates_parents_and_appends(self, tmp_path):
        p = tmp_path / "deep" / "ledger.jsonl"
        append_jsonl(p, {"b": 2, "a": 1})
        append_jsonl(p, {"a": 3})
        lines = p.read_text().strip().splitlines()
        assert len(lines) == 2
        # Sorted keys so two runs of the same flow diff cleanly.
        assert lines[0] == '{"a": 1, "b": 2}'
