"""Instrumented external-tool execution for EDA nodes.

Why this exists
---------------
A CHIA node that shells out to an EDA tool is usually measured on what it
*produced*, and the cost of producing it is thrown away.  For any flow that
reasons about evaluator cost -- scheduling, cluster sizing, or an experiment
whose result is a claim about time -- that cost IS the measurement, so the thing
that measures it has to be trustworthy first.

:func:`run_tool` records, for every invocation:

* wall-clock from :func:`time.monotonic`, which is immune to NTP steps (unlike
  :func:`time.time`);
* CPU time (user + system) of the child *and its reaped descendants*, which is
  what "evaluator CPU-hours" is actually made of;
* peak RSS of that same process tree, because synthesis memory is a real
  cluster constraint and should be measured rather than guessed;
* the exact argv, cwd and the environment variables that change results.

The rusage numbers come from :func:`os.wait4`, which attributes them to *this*
child. ``resource.getrusage(RUSAGE_CHILDREN)`` is cumulative over every child
the process has ever reaped, so it would be wrong the moment two evaluations run
concurrently in one worker -- which is the normal case under Ray.

The child is started in its own session so a timeout can kill the whole tool
tree: ``yosys`` spawns ``abc``, ``eqy`` spawns ``sby`` and an SMT solver.
Killing only the direct child leaves a grandchild holding the stdout pipe, and
the parent then blocks forever on a timeout that never fires.
"""

from __future__ import annotations

import json
import os
import resource
import shlex
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

#: Environment variables recorded with every run because they change the answer.
#: Extend per call site with ``salient_env=``; the defaults are the ones that
#: change an open-source ASIC flow's numbers.
DEFAULT_SALIENT_ENV: tuple[str, ...] = ("PATH", "OMP_NUM_THREADS", "LIBERTY")


@dataclass
class ToolRun:
    """One instrumented external-tool invocation.

    Attributes:
        argv (list[str]): The exact command executed.
        cwd (str): Working directory it ran in.
        returncode (int | None): Exit status, negative for a fatal signal,
            ``None`` if it was never reaped.
        wall_s (float): Monotonic wall-clock seconds.
        cpu_user_s (float): User CPU seconds of this child and its descendants.
        cpu_sys_s (float): System CPU seconds of the same.
        peak_rss_kb (int): Peak resident set size in kilobytes (Linux units).
        timed_out (bool): The process group was killed on the deadline.
        stdout_path (str | None): Captured stdout on disk, when ``log_dir`` was
            given.
        stderr_path (str | None): Captured stderr on disk, likewise.
        env_salient (dict): Values of the recorded environment variables.
        label (str | None): Caller's name for this step, also the log basename.
    """

    argv: list[str]
    cwd: str
    returncode: int | None
    # --- the measurement ---
    wall_s: float
    cpu_user_s: float
    cpu_sys_s: float
    peak_rss_kb: int
    # --- context ---
    timed_out: bool = False
    stdout_path: str | None = None
    stderr_path: str | None = None
    env_salient: dict[str, str | None] = field(default_factory=dict)
    label: str | None = None

    @property
    def cpu_s(self) -> float:
        """Total CPU seconds, user plus system."""
        return self.cpu_user_s + self.cpu_sys_s

    @property
    def peak_rss_mb(self) -> float:
        """Peak resident set size in megabytes."""
        return self.peak_rss_kb / 1024.0

    @property
    def ok(self) -> bool:
        """True only on a clean exit that did not hit the deadline."""
        return self.returncode == 0 and not self.timed_out

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe record including the derived fields.

        Returns:
            dict: Every field plus ``cpu_s``, ``peak_rss_mb`` and ``ok``.
        """
        d = asdict(self)
        d["cpu_s"] = self.cpu_s
        d["peak_rss_mb"] = self.peak_rss_mb
        d["ok"] = self.ok
        return d

    def summary(self) -> str:
        """One-line human summary: status, wall, CPU, peak RSS, command."""
        cmd = " ".join(shlex.quote(a) for a in self.argv)
        if len(cmd) > 110:
            cmd = cmd[:107] + "..."
        status = "OK " if self.ok else (
            "TIMEOUT" if self.timed_out else f"rc={self.returncode}")
        return (
            f"[{status}] {self.wall_s:8.2f}s wall  {self.cpu_s:8.2f}s cpu  "
            f"{self.peak_rss_mb:8.1f} MB peak  | {cmd}"
        )


class ToolNotFound(RuntimeError):
    """Raised when the executable is not on PATH.

    Deliberately distinct from a non-zero exit: a missing tool means the worker
    is running the wrong container image, which is an operator error, not a
    result about the design.
    """


def run_tool(
    argv: Sequence[str],
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    timeout_s: float | None = None,
    log_dir: str | os.PathLike[str] | None = None,
    label: str | None = None,
    input_text: str | None = None,
    salient_env: Sequence[str] = DEFAULT_SALIENT_ENV,
    verbose: bool = True,
) -> ToolRun:
    """Run one external tool, fully instrumented.

    Args:
        argv (Sequence[str]): Command and arguments.
        cwd (str | PathLike | None): Working directory; defaults to the caller's.
        env (Mapping | None): Full environment; ``None`` inherits ``os.environ``.
        timeout_s (float | None): Deadline in seconds. On expiry the tool's
            whole process group is SIGKILLed and ``timed_out`` is set.
        log_dir (str | PathLike | None): Directory for ``<label>.stdout.log``
            and ``<label>.stderr.log``. ``None`` discards both streams.
        label (str | None): Name for this step and the log basename.
        input_text (str | None): Written to the tool's stdin, which is then
            closed. ``None`` gives the tool an empty stdin.
        salient_env (Sequence[str]): Environment variable names to record.
        verbose (bool): Print the command and the result line.

    Returns:
        ToolRun: Always, even when the tool fails or times out -- a failed
        evaluation still costs wall-clock and that cost belongs in the ledger.

    Raises:
        ToolNotFound: If ``argv[0]`` is not executable or not on PATH.
    """
    argv = [str(a) for a in argv]
    cwd = str(cwd) if cwd is not None else os.getcwd()
    run_env = dict(os.environ if env is None else env)

    stdout_path = stderr_path = None
    out_f = err_f = None
    if log_dir is not None:
        d = Path(log_dir)
        d.mkdir(parents=True, exist_ok=True)
        stem = label or Path(argv[0]).name
        stdout_path = str(d / f"{stem}.stdout.log")
        stderr_path = str(d / f"{stem}.stderr.log")
        out_f = open(stdout_path, "w")
        err_f = open(stderr_path, "w")

    if verbose:
        print(f"    $ {' '.join(shlex.quote(a) for a in argv)}", flush=True)
        print(f"      cwd={cwd}" + (f"  logs={log_dir}" if log_dir else ""), flush=True)

    timed_out = False
    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=run_env,
            stdout=out_f or subprocess.DEVNULL,
            stderr=err_f or subprocess.DEVNULL,
            stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
            # Own process group, so a timeout kills the whole tool tree
            # (yosys spawns abc; eqy spawns sby and a solver).
            start_new_session=True,
        )
    except (FileNotFoundError, PermissionError) as exc:
        for f in (out_f, err_f):
            if f:
                f.close()
        raise ToolNotFound(f"{argv[0]} not found or not executable") from exc

    if input_text is not None and proc.stdin is not None:
        try:
            # Popen was opened WITHOUT text=True, so stdin is a binary pipe and
            # writing a str raises TypeError. Encode explicitly rather than
            # switching the whole call to text mode: stdout/stderr go straight
            # to file objects we opened ourselves, and text mode there would
            # re-encode tool output that may not be valid UTF-8.
            proc.stdin.write(input_text.encode())
        finally:
            proc.stdin.close()

    rc: int | None = None
    ru: resource.struct_rusage | None = None
    deadline = None if timeout_s is None else t0 + timeout_s

    while True:
        # os.wait4 attributes rusage to THIS child, unlike RUSAGE_CHILDREN.
        pid, status, ru = os.wait4(proc.pid, os.WNOHANG)
        if pid == proc.pid:
            rc = -os.WTERMSIG(status) if os.WIFSIGNALED(status) else os.WEXITSTATUS(status)
            break
        if deadline is not None and time.monotonic() > deadline:
            timed_out = True
            if verbose:
                print(f"      !! timeout after {timeout_s}s -- killing process group",
                      flush=True)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            pid, status, ru = os.wait4(proc.pid, 0)
            rc = -os.WTERMSIG(status) if os.WIFSIGNALED(status) else os.WEXITSTATUS(status)
            break
        time.sleep(0.005)

    wall = time.monotonic() - t0
    proc.returncode = rc
    for f in (out_f, err_f):
        if f:
            f.close()

    run = ToolRun(
        argv=argv,
        cwd=cwd,
        returncode=rc,
        wall_s=wall,
        cpu_user_s=ru.ru_utime if ru else 0.0,
        cpu_sys_s=ru.ru_stime if ru else 0.0,
        # Linux reports ru_maxrss in kilobytes.
        peak_rss_kb=int(ru.ru_maxrss) if ru else 0,
        timed_out=timed_out,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        env_salient={k: run_env.get(k) for k in salient_env},
        label=label,
    )
    if verbose:
        print("      " + run.summary(), flush=True)
    return run


def append_jsonl(path: str | os.PathLike[str], record: Mapping[str, Any]) -> None:
    """Append one record to a JSONL ledger, creating parent directories.

    Args:
        path (str | PathLike): The ledger file.
        record (Mapping): A JSON-serialisable record, written with sorted keys
            so two runs of the same flow diff cleanly.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


__all__ = ["ToolRun", "ToolNotFound", "run_tool", "append_jsonl",
           "DEFAULT_SALIENT_ENV"]


if __name__ == "__main__":
    print("=== chia.vlsi.tool_run self-test ===")

    r = run_tool(["/bin/sh", "-c", "echo hi; sleep 0.2"], label="trivial",
                 verbose=False)
    assert r.ok, r
    assert 0.15 < r.wall_s < 2.0, f"wall {r.wall_s}"
    print(f"    trivial run: ok={r.ok} wall={r.wall_s:.3f}s")

    # Burn measurable CPU and allocate ~200 MB so cpu_s and peak_rss are real.
    burn = (
        "python3 -c \"import time;"
        "b=bytearray(200*1024*1024);"
        "t=time.time();x=0\n"
        "while time.time()-t<0.5: x+=1\n"
        "print(len(b),x)\""
    )
    r2 = run_tool(["/bin/sh", "-c", burn], label="burn", verbose=False)
    assert r2.cpu_s > 0.3, f"cpu time not captured: {r2.cpu_s}"
    assert r2.peak_rss_mb > 150, f"peak RSS not captured: {r2.peak_rss_mb}"
    print(f"    rusage captured: cpu={r2.cpu_s:.2f}s peak={r2.peak_rss_mb:.1f}MB")

    r3 = run_tool(["/bin/sh", "-c", "exit 7"], label="failing", verbose=False)
    assert r3.returncode == 7 and not r3.ok, r3
    print(f"    non-zero exit preserved: rc={r3.returncode} ok={r3.ok}")

    # A tool that ignores SIGTERM must still be killed, and its child with it.
    t0 = time.monotonic()
    r4 = run_tool(["/bin/sh", "-c", "trap '' TERM; sleep 30 & wait"],
                  timeout_s=1.0, label="timeout", verbose=False)
    took = time.monotonic() - t0
    assert r4.timed_out and not r4.ok, r4
    assert took < 10.0, f"timeout did not fire: {took:.1f}s"
    print(f"    SIGTERM-ignoring tool tree killed in {took:.2f}s, timed_out=True")

    try:
        run_tool(["definitely-not-a-real-binary-xyz"], label="missing",
                 verbose=False)
        raise AssertionError("expected ToolNotFound")
    except ToolNotFound as e:
        print(f"    ToolNotFound raised, not a silent failure: {e}")

    d = r2.as_dict()
    assert d["ok"] is True and d["cpu_s"] == r2.cpu_s and "peak_rss_mb" in d
    assert json.dumps(d), "as_dict must be JSON-serialisable for the ledger"
    print("    as_dict() is JSON-serialisable and carries the derived fields")

    print("=== all chia.vlsi.tool_run self-tests passed ===")
