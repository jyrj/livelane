"""Run a Verilog design against its own testbench under Verilator.

What this node is for, and what it is NOT for
---------------------------------------------
Simulation is the CHEAP stage of an edit cascade: it costs about a second where
an equivalence proof costs tens, so it is worth running first to reject edits
that are obviously broken. It is *evidence*, never proof, and this matters
concretely rather than philosophically.

Measured on picorv32 with the design's own ``testbench_ez.v``: inverting the
``BGE`` comparison (``alu_out_0 = !alu_lts`` -> ``alu_out_0 = alu_lts``) produces
a simulation trace that is **byte-identical** to the correct design's,
identical sha256 over all 274 design-visible lines. The reason is structural,
not statistical: that testbench's program is six instructions

    li x1,1020 / sw x0,0(x1) / loop: lw x2,0(x1) / addi x2,x2,1 / sw x2,0(x1) / j loop

and contains no conditional branch at all, so an inverted branch comparison is
never exercised. The same edit is scored *better* than the correct design by
synthesis (75,074.50 um2 against 75,663.82, 235 fewer cells) and is refuted only
by the formal gate.

So: a passing simulation bounds nothing. A cascade that accepts on it admits
functional bugs, and the coverage of the cheap stage is a property of its
stimulus, not of the simulator. Use :attr:`SimResult.passed` to REJECT early,
never to ACCEPT.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from chia_livelane.vlsi.tool_run import ToolNotFound, run_tool

try:
    from chia.base.ChiaFunction import ChiaFunction
except Exception:  # pragma: no cover - degrades to a plain decorator
    def ChiaFunction(**_kwargs):  # type: ignore[misc]
        def deco(fn):
            return fn
        return deco

#: Resource token. A cluster node type declaring ``verilator_run`` binds this
#: call to a worker whose image actually carries the tool.
VERILATOR_RESOURCE = {"verilator_run": 1}

#: Verilator's own report lines. They carry host walltime and simulation speed,
#: which differ run to run on identical designs, so they are excluded before any
#: trace comparison, otherwise two identical designs never compare equal.
_REPORT_RE = re.compile(r"^- (Verilator|S i m u l a t i o n|V e r i l a t i o n)")


@dataclass
class SimResult:
    """One simulation: did it build, did it run, and what did it emit."""

    status: str                      # pass | fail | error
    built: bool
    ran: bool
    passed: bool
    build_s: float | None = None
    run_s: float | None = None
    wall_s: float = 0.0
    returncode: int | None = None
    trace_sha256: str | None = None
    trace_lines: int = 0
    finished: bool = False           # the testbench reached $finish
    assertion_failures: list[str] = field(default_factory=list)
    message: str = ""
    log_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean_trace(text: str) -> list[str]:
    """Design-visible output only, with the simulator's own report stripped."""
    return [ln for ln in text.splitlines() if not _REPORT_RE.match(ln)]


@dataclass
class VerilatorSimNode:
    """Build and run one Verilog testbench, and report what it showed.

    Attributes:
        verilator: Executable, by name or absolute path.
        workdir: Where ``obj_dir`` and logs are written. Named, never a temp
            dir, so a warm rebuild is possible and measurable.
        top: Top module name, the TESTBENCH's module, which frequently differs
            from the file name (picorv32's ``testbench_ez.v`` declares
            ``testbench``). A mismatch is a hard Verilator error, not a warning.
        timing: Pass ``--timing``. Required for any testbench that uses delay
            control (``always #5 clk = ~clk``); without it the build fails.
        jobs: ``-j`` for the C++ build.
        extra_args: Appended verbatim.
        timeout_s: Budget for build and for run, each.
    """

    verilator: str = "verilator"
    workdir: Path = Path(".")
    top: str = "testbench"
    timing: bool = True
    jobs: int = 0
    extra_args: Sequence[str] = field(default_factory=tuple)
    timeout_s: float = 1800.0
    verbose: bool = False

    def __post_init__(self) -> None:
        self.workdir = Path(self.workdir).resolve()
        if self.jobs < 0:
            raise ValueError(f"jobs must be >= 0; got {self.jobs}")

    def _build_argv(self, sources: Sequence[str], objdir: Path) -> list[str]:
        argv = [self.verilator, "--cc", "--exe", "--build", "--main",
                "--top-module", self.top, "-Wno-fatal",
                "--Mdir", str(objdir)]
        if self.timing:
            argv.append("--timing")
        if self.jobs:
            argv += ["-j", str(self.jobs)]
        argv += list(self.extra_args)
        argv += [str(Path(s).resolve()) for s in sources]
        return argv

    def run(self, sources: Sequence[str], *, name: str = "sim",
            plusargs: Sequence[str] = ()) -> SimResult:
        """Verilate, build, and run. Never raises; returns a verdict.

        Args:
            sources: Verilog sources. The testbench must be among them.
            name: Label for the workdir and logs.
            plusargs: ``+key=value`` arguments passed to the built binary.

        Returns:
            SimResult: ``passed`` is True only if the build succeeded, the
            binary ran, exited 0, and emitted no assertion failure.
        """
        wd = self.workdir / name
        wd.mkdir(parents=True, exist_ok=True)
        objdir = wd / "obj_dir"

        try:
            build = run_tool(self._build_argv(sources, objdir),
                             timeout_s=self.timeout_s, log_dir=wd,
                             label=f"{name}-build", verbose=self.verbose)
        except ToolNotFound as e:
            return SimResult(status="error", built=False, ran=False, passed=False,
                             message=str(e))
        if build.returncode != 0:
            return SimResult(status="error", built=False, ran=False, passed=False,
                             build_s=build.wall_s, wall_s=build.wall_s,
                             returncode=build.returncode,
                             message=f"verilate/build failed rc={build.returncode}",
                             log_path=str(build.stdout_path or ""))

        exe = objdir / f"V{self.top}"
        if not exe.exists():
            return SimResult(status="error", built=True, ran=False, passed=False,
                             build_s=build.wall_s, wall_s=build.wall_s,
                             message=f"build reported success but {exe} is absent")

        run = run_tool([str(exe), *plusargs], timeout_s=self.timeout_s,
                       log_dir=wd, label=f"{name}-run", verbose=self.verbose)
        text = ""
        if run.stdout_path and Path(run.stdout_path).exists():
            text = Path(run.stdout_path).read_text(errors="replace")

        lines = _clean_trace(text)
        sha = hashlib.sha256("\n".join(lines).encode()).hexdigest()
        # Verilog's own failure vocabulary. $finish is normal termination;
        # $stop, an assertion, or a fatal is not.
        fails = [ln for ln in lines
                 if re.search(r"(Assertion failed|%Error|\$stop|FAILED|ERROR:)", ln)]
        finished = "$finish" in text
        ok = run.returncode == 0 and not fails and not run.timed_out

        return SimResult(
            status="pass" if ok else "fail",
            built=True, ran=True, passed=ok,
            build_s=build.wall_s, run_s=run.wall_s,
            wall_s=(build.wall_s or 0.0) + (run.wall_s or 0.0),
            returncode=run.returncode, trace_sha256=sha, trace_lines=len(lines),
            finished=finished, assertion_failures=fails[:5],
            message="" if ok else (fails[0] if fails else
                                   f"rc={run.returncode} timed_out={run.timed_out}"),
            log_path=str(run.stdout_path or ""),
        )


@ChiaFunction(resources=VERILATOR_RESOURCE)
def verilator_sim(sources: list[str], workdir: str, top: str = "testbench",
                  verilator: str = "verilator", timing: bool = True,
                  jobs: int = 0, name: str = "sim",
                  timeout_s: float = 1800.0) -> dict:
    """Simulate a Verilog testbench and report what it showed.

    Use this as the CHEAP stage of a cascade: reject on ``passed == False``,
    but never accept on ``passed == True``. A passing simulation is bounded by
    its stimulus, picorv32's own ``testbench_ez`` produces a byte-identical
    trace for a design with an inverted ``BGE``, because its six-instruction
    program has no conditional branch to exercise.

    Args:
        sources: Verilog sources including the testbench.
        workdir: Named work directory (``obj_dir`` and logs land under it).
        top: The TESTBENCH module name, which often differs from its filename.
        verilator: Executable, by name or path.
        timing: Pass ``--timing``; required for delay-control testbenches.
        jobs: ``-j`` for the C++ build; 0 leaves it to Verilator.
        name: Label for this run's subdirectory.
        timeout_s: Budget for the build and for the run, each.

    Returns:
        dict: :class:`SimResult`. ``passed`` is the decision bit;
        ``trace_sha256`` lets two runs be compared exactly, with the
        simulator's own timing report excluded so identical designs compare equal.
    """
    node = VerilatorSimNode(verilator=verilator, workdir=Path(workdir), top=top,
                            timing=timing, jobs=jobs, timeout_s=timeout_s)
    return node.run(sources, name=name).as_dict()


__all__ = ["SimResult", "VerilatorSimNode", "verilator_sim", "VERILATOR_RESOURCE"]
