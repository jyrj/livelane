"""Provenance capture.

Every measurement LiveLane publishes must carry enough context to be reproduced
or invalidated.  LiveHD re-salts its content-hash caches on rebuild, so a "warm"
number is meaningless unless we also record *which* run after the rebuild it was
(the first warm run after a rebuild misses; always take run #2).  Yosys/ABC
heuristics move between versions, so a QoR number without a tool version is
noise.

Nothing here guesses: every field is read out of the running system, and a probe
that cannot answer records ``None`` rather than a plausible-looking default.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import sys
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _run(
    cmd: list[str], timeout: float = 30.0, *, require_success: bool = False
) -> str | None:
    """Run a probe command; return its output or None. Never raises.

    ``require_success`` matters more than it looks.  Version probes are allowed
    to exit non-zero (``yosys-abc -h`` does, and several tools print their
    version to stderr), so by default we keep whatever they printed.  But a
    query like ``git rev-parse HEAD`` prints the *literal string* ``HEAD`` on
    stderr when the branch is unborn and exits 128, which, merged blindly into
    stdout, becomes a commit sha of "HEAD".  Any probe whose output is a fact
    rather than a banner must pass ``require_success=True``.
    """
    exe = shutil.which(cmd[0])
    if exe is None:
        return None
    try:
        proc = subprocess.run(
            [exe, *cmd[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if require_success and proc.returncode != 0:
        return None
    out = (proc.stdout or "") + (proc.stderr or "")
    return out.strip() or None


def _first_line(text: str | None) -> str | None:
    return text.splitlines()[0].strip() if text else None


def git_sha(repo: str | os.PathLike[str]) -> str | None:
    """Exact HEAD of a checkout, or None if it is not a git repo."""
    repo = Path(repo)
    if not (repo / ".git").exists():
        return None
    out = _run(["git", "-C", str(repo), "rev-parse", "HEAD"], require_success=True)
    if not out:
        return None  # unborn branch: no commit exists yet
    sha = out.split()[0]
    return sha if re.fullmatch(r"[0-9a-f]{40}", sha) else None


def git_dirty(repo: str | os.PathLike[str]) -> bool | None:
    """True if the checkout has uncommitted changes (invalidates a pin claim)."""
    repo = Path(repo)
    if not (repo / ".git").exists():
        return None
    out = _run(["git", "-C", str(repo), "status", "--porcelain"], require_success=True)
    return bool(out)


# --- tool version probes -----------------------------------------------------
# Each returns the raw version string as the tool prints it, plus a parsed
# short form where we can extract one confidently.

_VERSION_PROBES: dict[str, list[str]] = {
    "yosys": ["yosys", "-V"],
    "verilator": ["verilator", "--version"],
    "sta": ["sta", "-version"],
    "eqy": ["eqy", "--version"],
    "sby": ["sby", "--version"],
    "sv2v": ["sv2v", "--version"],
    "iverilog": ["iverilog", "-V"],
    "abc": ["yosys-abc", "-h"],
    "gcc": ["gcc", "-dumpfullversion"],
}


def tool_version(name: str) -> str | None:
    cmd = _VERSION_PROBES.get(name)
    if cmd is None:
        return None
    return _first_line(_run(cmd))


def yosys_git_sha() -> str | None:
    """Yosys prints its own git sha in -V; pull it out so QoR is attributable."""
    raw = tool_version("yosys")
    if not raw:
        return None
    m = re.search(r"git sha1 ([0-9a-f]{7,40})", raw)
    return m.group(1) if m else None


def lhd_version(lhd_bin: str | os.PathLike[str] | None = None) -> str | None:
    """LiveHD has no --version; the pinned git sha of its checkout is the truth."""
    if lhd_bin is None:
        return None
    p = Path(lhd_bin)
    if not p.exists():
        return None
    # Resolve through the bazel-bin symlink to the real build output.
    return str(p.resolve())


@dataclass(frozen=True)
class HostInfo:
    hostname: str
    platform: str
    kernel: str
    cpu_model: str | None
    cpu_count: int
    mem_total_kb: int | None

    @staticmethod
    def probe() -> "HostInfo":
        cpu_model = None
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
        mem_kb = None
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    mem_kb = int(line.split()[1])
                    break
        except OSError:
            pass
        return HostInfo(
            hostname=socket.gethostname(),
            platform=platform.platform(),
            kernel=platform.release(),
            cpu_model=cpu_model,
            cpu_count=os.cpu_count() or 0,
            mem_total_kb=mem_kb,
        )


@dataclass
class Provenance:
    """Everything needed to reproduce, or to refuse to trust, one measurement."""

    run_id: str
    host: HostInfo
    tools: dict[str, str | None]
    repos: dict[str, dict[str, Any]]
    env: dict[str, str | None]
    # LiveHD re-salts caches on rebuild: which warm run after the rebuild is this?
    warm_run_index: int | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(asdict(self), indent=indent, sort_keys=True)

    def write(self, path: str | os.PathLike[str]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())
        return p


# Environment variables that materially change a result.
_TRACKED_ENV = (
    "HAGENT_TECH_DIR",
    "LIVELANE_ROOT",
    "LIVELANE_JOBS",
    "LIVELANE_LIBERTY",
    "CCACHE_DISABLE",
    "OMP_NUM_THREADS",
)


def capture(
    run_id: str,
    repos: dict[str, str | os.PathLike[str]] | None = None,
    warm_run_index: int | None = None,
    **notes: Any,
) -> Provenance:
    """Probe the live system. Absent facts are None, never invented."""
    repo_info: dict[str, dict[str, Any]] = {}
    for name, path in (repos or {}).items():
        repo_info[name] = {
            "path": str(path),
            "sha": git_sha(path),
            "dirty": git_dirty(path),
        }

    tools = {name: tool_version(name) for name in _VERSION_PROBES}
    tools["yosys_git_sha"] = yosys_git_sha()
    # The interpreter running this code, not whatever python3 is on PATH.
    tools["python"] = sys.version.split()[0]
    tools["python_executable"] = sys.executable

    return Provenance(
        run_id=run_id,
        host=HostInfo.probe(),
        tools=tools,
        repos=repo_info,
        env={k: os.environ.get(k) for k in _TRACKED_ENV},
        warm_run_index=warm_run_index,
        notes=notes,
    )


if __name__ == "__main__":  # cheap self-test: python -m livelane.harness.provenance
    root = Path(__file__).resolve().parents[3]
    p = capture(
        run_id="provenance-selftest",
        repos={
            "livelane": root,
            "livehd": root / "thirdparty" / "livehd",
            "chia": root / "thirdparty" / "chia",
            "lhdsuite": root / "thirdparty" / "lhdsuite",
        },
    )
    print(p.to_json())
