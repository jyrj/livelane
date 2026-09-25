"""Instrumented subprocess execution.

LiveLane's entire result is a claim about *time*, so the thing that measures
time has to be trustworthy before anything else is.  Every EDA invocation in
either lane goes through :func:`run_tool`, which records:

* wall-clock, from ``time.monotonic()`` (immune to NTP steps, unlike ``time.time``);
* CPU time (user + system) of the child *and its reaped descendants*, which is
  what "evaluator CPU-hours" in the cost table is actually made of;
* peak RSS of that same process tree, because the medium XiangShan block peaks
  near 4 GB in lane S and reportedly 23-31 GB in lane L, a number we must
  measure rather than repeat;
* the exact argv, cwd, and the environment variables that change results.

The rusage numbers come from :func:`os.wait4`, which attributes them to *this*
child rather than to the whole process (``RUSAGE_CHILDREN`` is cumulative and
would be wrong the moment two evaluations run concurrently).
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

# Environment variables recorded with every run because they change the answer.
_SALIENT_ENV = ("HAGENT_TECH_DIR", "LIVELANE_LIBERTY", "OMP_NUM_THREADS", "PATH")


@dataclass
class ToolRun:
    """One instrumented external-tool invocation."""

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
    # Latency deliberately injected on top of the real evaluation (arm I(d)).
    injected_delay_s: float = 0.0
    label: str | None = None

    @property
    def cpu_s(self) -> float:
        return self.cpu_user_s + self.cpu_sys_s

    @property
    def peak_rss_mb(self) -> float:
        return self.peak_rss_kb / 1024.0

    @property
    def observed_latency_s(self) -> float:
        """What the *agent* waited for: real work plus any injected delay."""
        return self.wall_s + self.injected_delay_s

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cpu_s"] = self.cpu_s
        d["peak_rss_mb"] = self.peak_rss_mb
        d["observed_latency_s"] = self.observed_latency_s
        d["ok"] = self.ok
        return d

    def summary(self) -> str:
        cmd = " ".join(shlex.quote(a) for a in self.argv)
        if len(cmd) > 110:
            cmd = cmd[:107] + "..."
        status = "OK " if self.ok else ("TIMEOUT" if self.timed_out else f"rc={self.returncode}")
        return (
            f"[{status}] {self.wall_s:8.2f}s wall  {self.cpu_s:8.2f}s cpu  "
            f"{self.peak_rss_mb:8.1f} MB peak  | {cmd}"
        )


class ToolNotFound(RuntimeError):
    pass


def run_tool(
    argv: Sequence[str],
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    timeout_s: float | None = None,
    log_dir: str | os.PathLike[str] | None = None,
    label: str | None = None,
    input_text: str | None = None,
    verbose: bool = True,
) -> ToolRun:
    """Run one external tool, fully instrumented.

    Returns a :class:`ToolRun` even when the tool fails or times out, a failed
    evaluation still costs wall-clock, and that cost belongs in the arm's ledger.
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
            # (yosys spawns abc; lhd spawns its own helpers).
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        for f in (out_f, err_f):
            if f:
                f.close()
        raise ToolNotFound(f"{argv[0]} not found on PATH") from exc

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
                print(f"      !! timeout after {timeout_s}s -- killing process group", flush=True)
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
        env_salient={k: run_env.get(k) for k in _SALIENT_ENV},
        label=label,
    )
    if verbose:
        print("      " + run.summary(), flush=True)
    return run


def append_jsonl(path: str | os.PathLike[str], record: Mapping[str, Any]) -> None:
    """Append one record to a JSONL ledger, creating parents as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    # Self-test against real processes with known behaviour.
    print("=== run_tool self-test ===")

    r = run_tool(["/bin/sh", "-c", "echo hi; sleep 0.2"], label="trivial")
    assert r.ok, r
    assert 0.15 < r.wall_s < 2.0, f"wall {r.wall_s}"

    # Burn measurable CPU and allocate ~200 MB so cpu_s and peak_rss are non-trivial.
    burn = (
        "python3 -c \"import time;"
        "b=bytearray(200*1024*1024);"
        "t=time.time();x=0\n"
        "while time.time()-t<0.5: x+=1\n"
        "print(len(b),x)\""
    )
    r2 = run_tool(["/bin/sh", "-c", burn], label="burn")
    print(f"    cpu={r2.cpu_s:.2f}s peak={r2.peak_rss_mb:.1f}MB")
    assert r2.cpu_s > 0.3, f"cpu time not captured: {r2.cpu_s}"
    assert r2.peak_rss_mb > 150, f"peak RSS not captured: {r2.peak_rss_mb}"

    r3 = run_tool(["/bin/sh", "-c", "exit 7"], label="failing")
    assert r3.returncode == 7 and not r3.ok, r3

    # A tool that ignores SIGTERM must still be killed, and its child with it.
    r4 = run_tool(["/bin/sh", "-c", "trap '' TERM; sleep 30"], timeout_s=1.0, label="timeout")
    assert r4.timed_out and not r4.ok, r4
    assert r4.wall_s < 5.0, f"timeout took too long: {r4.wall_s}"

    try:
        run_tool(["definitely-not-a-real-binary-xyz"], label="missing")
        raise AssertionError("expected ToolNotFound")
    except ToolNotFound as e:
        print(f"    ToolNotFound raised correctly: {e}")

    print("=== all run_tool self-tests passed ===")
