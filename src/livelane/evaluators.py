"""The two evaluator lanes, behind one identical interface.

The experiment only means anything if the agent cannot tell the lanes apart, so
both implement the same :class:`Evaluator` protocol and return the same report
types with the same fields.  Everything that differs, tool names, wall-clock,
peak RSS, the ``source`` string, lives on the internal side of the report and
is stripped by :meth:`~livelane.state.reports._AgentVisible.to_agent_dict`.

``LaneS`` is Yosys + ABC + OpenSTA; ``LaneL`` is ``lhd synth``.  ``I(d)`` is not
a third implementation: it is ``LaneS`` wrapped in a
:class:`~chia_livelane.delay_node.DelayNode`, which is the whole point, the
tools are literally the same objects.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from livelane.harness.run import ToolRun
from livelane.nodes.lhd import Lhd
from livelane.nodes.yosys_sta import BASELINE, SynthScript, YosysStaLane
from livelane.state.reports import QorReport, TimingReport


@runtime_checkable
class Evaluator(Protocol):
    """What every arm must provide. Identical signatures in every lane."""

    lane: str

    def evaluate(self, sources: Sequence[str], top: str, workdir: Path,
                 filelist: str | None = None) -> tuple[QorReport, TimingReport | None]:
        """Score the design described by ``sources`` (or ``filelist``).

        ``sources``/``filelist`` describe the CANDIDATE's working copy and change
        every iteration; nothing about the design may be captured at
        construction time.
        """
        ...

    @property
    def cpu_s(self) -> float:
        ...


@dataclass
class LaneSEvaluator:
    """Lane S: Yosys + ABC technology mapping on sky130, then OpenSTA."""

    yosys: str
    sta: str
    liberty: str
    script: SynthScript = BASELINE
    #: `read_slang` reads XiangShan SystemVerilog directly, so lane S and lane L
    #: consume the SAME files. `read_verilog -sv` is for plain-Verilog designs.
    read_cmd: str = "read_slang"
    filelist: str | None = None
    #: Clock constraint. A SEQUENTIAL design MUST have one, or OpenSTA reports
    #: the longest unconstrained combinational I/O path instead of the
    #: register-to-register critical path. On picorv32 that is 0.196 ns against a
    #: real 12.76 ns critical path, a 65x error, and an objective the agent
    #: cannot meaningfully improve. A purely combinational design (XiangShan's
    #: Alu, DecodeUnit: 0% sequential elements) correctly has none.
    clock_port: str | None = None
    clock_period_ns: float | None = None
    timeout_s: float = 7200.0
    lane: str = "S"
    tool_runs: list[ToolRun] = field(default_factory=list)

    def evaluate(self, sources: Sequence[str], top: str, workdir: Path,
                 filelist: str | None = None) -> tuple[QorReport, TimingReport | None]:
        lane = YosysStaLane(yosys=self.yosys, sta=self.sta, liberty=self.liberty,
                            workdir=workdir, script=self.script, read_cmd="",
                            clock_port=self.clock_port,
                            clock_period_ns=self.clock_period_ns)
        fl = filelist if filelist is not None else self.filelist
        if fl and self.read_cmd.startswith("read_slang"):
            read = f"read_slang --top {top} -F {fl}"
        elif fl:
            # read_verilog cannot consume a filelist; expand it in listed order.
            base = Path(fl).parent
            names = [l.strip() for l in Path(fl).read_text().splitlines()
                     if l.strip() and not l.strip().startswith(("#", "//"))]
            read = "\n".join(f"{self.read_cmd} {(base / n).as_posix()}" for n in names)
        else:
            read = "\n".join(f"{self.read_cmd} {Path(s).as_posix()}" for s in sources)
        if not read.strip():
            raise ValueError("evaluator was given no sources and no filelist; "
                             "it would synthesise an empty design")
        lane.script = SynthScript(name=self.script.name,
                                  rationale=self.script.rationale,
                                  body=read + "\n" + self.script.body)
        qor, timing = lane.evaluate([], top, timeout_s=self.timeout_s)
        self.tool_runs.extend(lane.tool_runs)
        return qor, timing

    @property
    def cpu_s(self) -> float:
        return sum(r.cpu_s for r in self.tool_runs)

    def config(self) -> dict[str, object]:
        """The settings that must appear alongside any number this lane produces."""
        return {"lane": "S", "script": self.script.name,
                "read_cmd": self.read_cmd,
                "clock_port": self.clock_port,
                "clock_period_ns": self.clock_period_ns,
                "timing": ("register-to-register critical path"
                           if self.clock_port else
                           "unconstrained combinational I/O path")}


@dataclass
class LaneLEvaluator:
    """Lane L: ``lhd synth``, fused compile + color + abc + opentimer."""

    binary: str
    liberty: str
    reader: str = "slang"
    filelist: str | None = None
    incremental: bool = True
    #: ABC timing budget in PICOSECONDS. lhd's `pass.abc.delay` defaults to
    #: EMPTY (no budget at all) while lane S passes `abc -D 10000`. Leaving it
    #: unset is a silent asymmetry, so it is required here and recorded per run.
    abc_delay_ps: int | None = 10000
    #: Whole-design flatten. This is the single most consequential lane-L knob:
    #: `auto` (LiveHD's default) partitioned DecodeUnit into 51 independently
    #: mapped regions, which is exactly what makes the ABC cache work, and
    #: costs ~16x on the critical path (115.6 ns vs 7.0 ns flattened).
    #: You can have the reuse or the quality, not both. Whichever is chosen, it
    #: MUST be identical across every arm and recorded with the measurement.
    abc_flatten: str = "true"
    #: Persistent workdir. Every incremental tier is gated on a USER-NAMED
    #: workdir; letting lhd mint a scratch one silently disables all reuse.
    persistent_workdir: Path | None = None
    timeout_s: float = 7200.0
    lane: str = "L"
    tool_runs: list[ToolRun] = field(default_factory=list)

    def evaluate(self, sources: Sequence[str], top: str, workdir: Path,
                 filelist: str | None = None) -> tuple[QorReport, TimingReport | None]:
        fl = filelist if filelist is not None else self.filelist
        wd = Path(self.persistent_workdir or workdir)
        sets: dict[str, str] = {"pass.abc.flatten": self.abc_flatten}
        if self.abc_delay_ps is not None:
            sets["pass.abc.delay"] = str(self.abc_delay_ps)
        lhd = Lhd(binary=self.binary, workdir=wd, liberty=self.liberty,
                  reader=self.reader, incremental=self.incremental,
                  extra_sets=sets, log_dir=workdir)
        from livelane.harness.run import run_tool
        rj = wd / "synth.result.json"
        wd.mkdir(parents=True, exist_ok=True)
        if rj.exists():
            rj.unlink()
        args = ["synth"] + ([] if fl else [str(s) for s in sources]) \
               + ["--top", top, "--stats"]
        trailing = ["--", "-F", fl] if fl else []
        run = run_tool([self.binary, *args, *lhd._common(rj), *trailing],
                       timeout_s=self.timeout_s, log_dir=workdir,
                       label="lhd-synth", verbose=False)
        self.tool_runs.append(run)
        if not rj.exists():
            return (QorReport(top=top, cells=None, area_um2=None,
                              max_delay_ns=None, valid=False,
                              message=f"lhd produced no envelope (rc={run.returncode})",
                              source="lhd", wall_s=run.wall_s), None)
        import json
        from livelane.nodes.lhd import LhdEnvelope
        env = LhdEnvelope(raw=json.loads(rj.read_text()), path=rj)
        qor = env.to_qor_report(top)
        qor.wall_s, qor.cpu_s, qor.peak_rss_kb = run.wall_s, run.cpu_s, run.peak_rss_kb
        return qor, env.to_timing_report(top)

    @property
    def cpu_s(self) -> float:
        return sum(r.cpu_s for r in self.tool_runs)

    def config(self) -> dict[str, object]:
        """The settings that must appear alongside any number this lane produces."""
        return {"lane": "L", "reader": self.reader,
                "incremental": self.incremental,
                "pass.abc.flatten": self.abc_flatten,
                "pass.abc.delay_ps": self.abc_delay_ps}


@dataclass
class DesignWorkspace:
    """A private, writable copy of one design's RTL that an edit can mutate.

    Every variant gets its own workspace so a rejected edit can never leak into
    the next iteration, and so the parent can be reproduced exactly.
    """

    source_dir: Path
    workdir: Path
    filelist_name: str | None = None
    _files: list[Path] = field(default_factory=list)

    def materialize(self) -> "DesignWorkspace":
        self.workdir.mkdir(parents=True, exist_ok=True)
        dst = self.workdir / "rtl"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(self.source_dir, dst, symlinks=False)
        self._files = sorted(p for p in dst.rglob("*")
                             if p.suffix in (".v", ".sv", ".svh", ".vh"))
        return self

    @property
    def rtl_dir(self) -> Path:
        return self.workdir / "rtl"

    @property
    def filelist(self) -> str | None:
        if self.filelist_name is None:
            return None
        return str(self.rtl_dir / self.filelist_name)

    def files(self) -> list[Path]:
        return list(self._files)

    def primary_file(self, top: str) -> str | None:
        """The file that DECLARES the top module, as a workspace-relative path.

        Without this the agent is handed ``files[0]``, alphabetically first,
        which on XiangShan's 1,088-file corpus is ``AddModule.sv`` while the task
        says "optimise Alu". The model would be reading one design and editing
        another, and every iteration would be wasted.

        Matching is on a real declaration (``module <top>`` at a word boundary),
        not a substring, so ``Alu`` does not match ``AluDataModule``.
        """
        import re
        pat = re.compile(rf"^\s*module\s+{re.escape(top)}\s*(?:#|\(|;|$)", re.M)
        for f in self._files:
            if f.suffix not in (".v", ".sv"):
                continue
            try:
                if pat.search(f.read_text(errors="ignore")):
                    return str(f.relative_to(self.rtl_dir))
            except OSError:
                continue
        return None

    def read(self, relpath: str) -> str:
        return (self.rtl_dir / relpath).read_text()

    def apply_edit(self, relpath: str, old: str, new: str) -> tuple[bool, int, str]:
        """Exact-match replacement. Returns (applied, occurrences, message)."""
        p = self.rtl_dir / relpath
        if not p.exists():
            return False, 0, f"no such file in the working design: {relpath}"
        text = p.read_text()
        n = text.count(old)
        if n == 0:
            return False, 0, "old text not found; the edit must match exactly"
        if n > 1:
            return False, n, (f"old text appears {n} times; make it unique so the "
                              f"edit is unambiguous")
        p.write_text(text.replace(old, new, 1))
        return True, 1, "applied"

    def sha256(self) -> str:
        """Content hash of the whole RTL tree, the variant's identity."""
        import hashlib
        h = hashlib.sha256()
        for f in sorted(self._files):
            h.update(f.relative_to(self.rtl_dir).as_posix().encode())
            h.update(f.read_bytes())
        return h.hexdigest()


__all__ = ["Evaluator", "LaneSEvaluator", "LaneLEvaluator", "DesignWorkspace"]
