"""Open-source ASIC QoR: Yosys + ABC technology mapping, then OpenSTA timing.

Why this exists
---------------
CHIA's only logic-synthesis-to-netlist path is Cadence Genus, via
``chia/vlsi/hammer.py`` and ``vlsi.core.synthesis_tool: "hammer.synthesis.genus"``.
Vivado FPGA synthesis exists (``chia/esp``, ``chia/firesim``), but there is no
open-source ASIC synthesis anywhere in the tree, so every cell-count/area/delay
number a CHIA flow can produce today requires a commercial licence.  That is a
hard floor on who can reproduce a CHIA result.

This node closes that gap with Yosys + ABC for mapping and OpenSTA for timing,
against any Liberty file.

Two things shape the code
-------------------------
* **The synthesis script is data, not a hardcoded string.**  A reviewer will
  reasonably ask whether a slow or bad baseline was left unoptimised -- and they
  would be right to: we measured 52% of one block's 24-minute synthesis going to
  ``opt_dff``, not to ABC.  So the recipe is a named, recordable object
  (:class:`SynthScript`), reported with every measurement, and more than one is
  published.  See :data:`SCRIPTS`.
* **Nothing is estimated.**  Cell count and area come from ``stat -liberty``;
  the critical path comes from OpenSTA over the mapped netlist.  If a parse
  fails the report is marked ``success=False`` rather than silently reporting
  ``None`` as if it were a good number.

The clock constraint is not optional
------------------------------------
``report_checks`` on an *unconstrained* design reports the longest path it can
find with no clock to relate it to, and that is a different quantity.  Measured
on picorv32 with sky130: constrained at a 10 ns clock the critical path is
**12.7612 ns**; unconstrained the same netlist reports **0.1959 ns**.  Anything
optimising the second number is optimising noise.  So this node records
:attr:`QorReport.constrained` on every report, and an unconstrained run says so
in ``message`` instead of quietly returning the small number.

Container
---------
Needs ``yosys`` (with ABC) and OpenSTA's ``sta`` on PATH plus a Liberty file;
see ``dockerfiles/YosysStaDockerfile`` and bind it with a cluster node type
whose ``resources: {"yosys_sta": 1}`` matches :func:`yosys_sta_qor`'s token.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

try:  # CHIA is optional: the parsers are testable without a cluster.
    from chia.base.ChiaFunction import ChiaFunction
except Exception:  # pragma: no cover - exercised only outside a CHIA install
    def ChiaFunction(**_kwargs):  # type: ignore[misc]
        def deco(fn):
            return fn
        return deco

try:  # Installed form: a sibling module in chia.vlsi.
    from chia.vlsi.tool_run import ToolNotFound, ToolRun, run_tool
except Exception:  # pragma: no cover - staging form, before upstreaming
    from chia_livelane.vlsi.tool_run import ToolNotFound, ToolRun, run_tool


@dataclass(frozen=True)
class SynthScript:
    """A named, recordable Yosys recipe.

    Attributes:
        name (str): Identifier recorded with every measurement this recipe
            produced, so two numbers are never compared across recipes by
            accident.
        rationale (str): Why this recipe exists and what it trades away.
        body (str): Yosys script text. ``{top}``, ``{lib}``, ``{netlist}``,
            ``{stat}`` and ``{celljson}`` are substituted by :meth:`render`.
    """

    name: str
    rationale: str
    body: str

    def render(self, *, top: str, lib: str, netlist: str, stat: str,
               celljson: str = "/dev/null") -> str:
        """Substitute the placeholders and return runnable Yosys script text.

        Args:
            top (str): Top module name.
            lib (str): Path to the Liberty file.
            netlist (str): Where to write the mapped Verilog netlist.
            stat (str): Where to tee the ``stat -liberty`` report.
            celljson (str): Where to write the JSON netlist used to map
                post-synthesis cells back to RTL lines.

        Returns:
            str: The rendered script.
        """
        return self.body.format(top=top, lib=lib, netlist=netlist, stat=stat,
                                celljson=celljson)


#: Straightforward full-flatten recipe: ``synth``, then ABC with a fixed target.
#: Reported as the *naive* baseline, never as a tuned result.
BASELINE = SynthScript(
    name="baseline-flat",
    rationale=(
        "Full flatten via `synth`, ABC with a fixed -D 10000. The obvious "
        "recipe, and the one to quote when claiming a baseline, because it is "
        "the one a reader would have written."
    ),
    body=(
        "hierarchy -top {top}\n"
        "synth -top {top}\n"
        "dfflibmap -liberty {lib}\n"
        "abc -D 10000 -liberty {lib}\n"
        "opt_clean\n"
        "tee -o {stat} stat -liberty {lib}\n"
        "write_verilog -noattr {netlist}\n"
        "write_json {celljson}\n"
    ),
)

#: Hierarchy-preserving recipe. Measured motivation: on one medium block, 52% of
#: a 24-minute run was ``opt_dff`` working on a fully flattened, front-end-
#: inflated netlist. Keeping hierarchy and using ABC's fast script targets that
#: pathology directly.
TUNED = SynthScript(
    name="tuned-hier",
    rationale=(
        "Preserves hierarchy (no full flatten) and uses `abc -fast`, targeting "
        "the measured pathology that 52% of one block's synthesis time was "
        "opt_dff on a flattened netlist."
    ),
    body=(
        "hierarchy -check -top {top}\n"
        "proc\n"
        "opt -fast\n"
        "fsm\n"
        "opt -fast\n"
        "memory_map\n"
        "techmap\n"
        "opt -fast\n"
        "dfflibmap -liberty {lib}\n"
        "abc -fast -D 10000 -liberty {lib}\n"
        "opt_clean\n"
        "tee -o {stat} stat -liberty {lib}\n"
        "write_verilog -noattr {netlist}\n"
        "write_json {celljson}\n"
    ),
)

#: Recipes by name. Pass ``script="tuned-hier"`` to :func:`yosys_sta_qor`.
SCRIPTS: dict[str, SynthScript] = {s.name: s for s in (BASELINE, TUNED)}


# --- parsers -----------------------------------------------------------------
# Written against real output and unit-tested against captured text, because a
# silently-wrong area number corrupts every downstream comparison.

# `stat` has TWO output formats and the difference is not cosmetic:
#   without -liberty:  "   Number of cells:               6691"
#   with    -liberty:  "     7692 4.32E+04 cells"
# This node always passes -liberty (it must, to get area), so the second form is
# the one that matters. Matching only the first -- which an earlier version of
# this parser did -- silently returns cells=None on every real run.
_CELLS_PLAIN_RE = re.compile(r"^\s*Number of cells:\s+(\d+)\s*$", re.M)
_CELLS_LIB_RE = re.compile(r"^\s*(\d+)\s+([0-9.eE+-]+)\s+cells\s*$", re.M)
_AREA_RE = re.compile(r"^\s*Chip area for (?:top )?module '?\\?([^':]+)'?:\s*"
                      r"([0-9.eE+-]+)\s*$", re.M)


def parse_yosys_stat(text: str) -> tuple[int | None, float | None]:
    """Extract (cell count, chip area) from ``stat`` output, either format.

    Takes the LAST match of each: with hierarchy preserved ``stat`` prints one
    block per module and the top-level summary comes last.

    Chip area is preferred over the inline per-block area column because it is
    the whole-design number and is printed at full precision, whereas the inline
    column is rounded ("4.32E+04").

    Args:
        text (str): The ``stat`` report, as tee'd to a file or captured.

    Returns:
        tuple: ``(cells, area_um2)``, either of which is ``None`` when the
        report did not state it. Never a fabricated number.
    """
    cells = None
    for m in _CELLS_LIB_RE.finditer(text):
        cells = int(m.group(1))
    if cells is None:
        for m in _CELLS_PLAIN_RE.finditer(text):
            cells = int(m.group(1))
    area = None
    for m in _AREA_RE.finditer(text):
        try:
            area = float(m.group(2))
        except ValueError:
            pass
    return cells, area


_SLACK_RE = re.compile(r"^\s*(-?[0-9.]+)\s+slack\s+\((MET|VIOLATED)\)", re.M)
_ARRIVAL_RE = re.compile(r"^\s*(-?[0-9.]+)\s+data arrival time\s*$", re.M)
_STARTPOINT_RE = re.compile(r"^Startpoint:\s*(\S+)", re.M)
_ENDPOINT_RE = re.compile(r"^Endpoint:\s*(\S+)", re.M)


def parse_sta_report(text: str) -> dict[str, object]:
    """Extract worst slack, arrival and endpoints from ``report_checks`` output.

    Args:
        text (str): OpenSTA's captured output.

    Returns:
        dict: Any of ``slack_ns`` (float, the worst i.e. most negative across
        all reported paths), ``met`` (bool), ``max_delay_ns`` (float),
        ``startpoint`` (str), ``endpoint`` (str) and ``path_summary`` (str,
        the full worst-path block). Keys are absent when the report did not
        state them.
    """
    out: dict[str, object] = {}
    sl = _SLACK_RE.findall(text)
    if sl:
        # Worst (most negative) slack across reported paths.
        vals = [float(v) for v, _ in sl]
        out["slack_ns"] = min(vals)
        out["met"] = all(k == "MET" for _, k in sl)
    ar = _ARRIVAL_RE.findall(text)
    if ar:
        out["max_delay_ns"] = max(float(v) for v in ar)
    sp = _STARTPOINT_RE.search(text)
    ep = _ENDPOINT_RE.search(text)
    if sp:
        out["startpoint"] = sp.group(1)
    if ep:
        out["endpoint"] = ep.group(1)
    # The scalars alone are not actionable. Post-synthesis endpoint names are
    # mangled ("_10883_"), so anything told only "slack is -2.92ns at _10883_"
    # cannot know WHICH logic to change and edits blind. Carry the actual
    # report_checks path block: it names the cells and nets on the critical
    # path, which is the most that survives synthesis without source tracking.
    blk = _worst_path_block(text)
    if blk:
        out["path_summary"] = blk
    return out


def _worst_path_block(text: str, max_lines: int = 40) -> str:
    """Return the ``report_checks`` block for the worst path, trimmed."""
    starts = [m.start() for m in re.finditer(r"^Startpoint:", text, re.M)]
    if not starts:
        return ""
    worst, worst_slack = None, None
    for i, st in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        block = text[st:end]
        m = _SLACK_RE.search(block)
        if not m:
            continue
        sl = float(m.group(1))
        if worst_slack is None or sl < worst_slack:
            worst, worst_slack = block, sl
    block = worst or text[starts[0]:]
    lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
    if len(lines) > max_lines:
        head, tail = lines[: max_lines - 8], lines[-8:]
        lines = head + [f"    ... {len(lines) - max_lines + 8} lines elided ..."] + tail
    return "\n".join(lines)


def load_cell_sources(json_path: str | os.PathLike[str]) -> dict[str, str]:
    """Map post-synthesis cell name -> originating RTL ``file:line``.

    OpenSTA names critical-path nodes by their synthesised cell (``_05679_``),
    which is unactionable for anything editing RTL. Yosys keeps the originating
    location in each cell's ``src`` attribute, but ``write_verilog -noattr``
    discards it, so the mapping is recovered from a JSON netlist instead.

    Args:
        json_path (str | PathLike): Path written by ``write_json``.

    Returns:
        dict: ``{cell_name: "file:line"}``. Returns ``{}`` rather than raising:
        a missing or malformed map degrades the report to bare cell names and
        must never take down an evaluation.
    """
    p = Path(json_path)
    if not p.exists():
        return {}
    try:
        doc = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    out: dict[str, str] = {}
    for mod in (doc.get("modules") or {}).values():
        for cell_name, cell in (mod.get("cells") or {}).items():
            src = (cell.get("attributes") or {}).get("src")
            if isinstance(src, str) and src:
                # yosys may record several locations, pipe-separated.
                out[cell_name] = src.split("|")[0]
    return out


_CELL_REF_RE = re.compile(r"^(\s*[\d.]+\s+[\d.]+\s+[v^]\s+)(\S+?)/(\S+)(\s+\(.*)$")


def annotate_path_with_sources(path_block: str,
                               cell_src: dict[str, str]) -> str:
    """Append ``file:line`` to each critical-path line whose cell can be placed.

    Args:
        path_block (str): A ``report_checks`` path block.
        cell_src (dict): Mapping from :func:`load_cell_sources`.

    Returns:
        str: The block with ``<- file:line`` appended where known, unchanged
        where not.
    """
    if not path_block or not cell_src:
        return path_block
    out = []
    for line in path_block.splitlines():
        m = _CELL_REF_RE.match(line)
        if m:
            src = cell_src.get(m.group(2))
            if src:
                line = f"{line}   <- {src}"
        out.append(line)
    return "\n".join(out)


# --- the report --------------------------------------------------------------

@dataclass
class QorReport:
    """Cell count, area and critical-path timing for one design.

    Attributes:
        top (str): Top module the numbers describe.
        success (bool): Every stage completed and the numbers were parsed. A
            report with ``success=False`` carries no usable number, whatever
            the other fields say.
        cells (int | None): Mapped cell count from ``stat -liberty``.
        area_um2 (float | None): Chip area from ``stat -liberty``.
        max_delay_ns (float | None): Worst data-arrival time from OpenSTA.
        slack_ns (float | None): Worst slack. Negative means the clock is
            violated.
        constrained (bool): A clock was created before timing. **False means
            ``max_delay_ns`` is NOT the clock-constrained critical path** and
            must not be compared against a constrained number.
        clock_period_ns (float | None): The clock period, when constrained.
        startpoint (str | None): Worst path's start.
        endpoint (str | None): Worst path's end.
        path_summary (str): The worst path itself, annotated with RTL
            ``file:line`` wherever a cell could be placed back in the source.
        script (str): Which :class:`SynthScript` produced this.
        message (str): Why the report failed, or a warning about it.
        wall_s (float): Total wall-clock across every tool invocation.
        cpu_s (float): Total CPU seconds across the same.
        peak_rss_kb (int): Largest peak RSS of any single invocation.
        netlist_path (str | None): Mapped netlist, when one was written.
    """

    top: str
    success: bool
    cells: int | None = None
    area_um2: float | None = None
    max_delay_ns: float | None = None
    slack_ns: float | None = None
    constrained: bool = False
    clock_period_ns: float | None = None
    startpoint: str | None = None
    endpoint: str | None = None
    path_summary: str = ""
    script: str = ""
    message: str = ""
    wall_s: float = 0.0
    cpu_s: float = 0.0
    peak_rss_kb: int = 0
    netlist_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe record of every field, for a ledger or a Ray return."""
        return asdict(self)


# --- the node ----------------------------------------------------------------

@dataclass
class YosysStaNode:
    """Synthesise with Yosys/ABC, then time the mapped netlist with OpenSTA.

    Attributes:
        yosys (str): ``yosys`` executable, by name or absolute path.
        sta (str): OpenSTA ``sta`` executable, likewise.
        liberty (str): Liberty file used by ``dfflibmap``, ``abc``, ``stat``
            and ``read_liberty``. Two results are only comparable if they used
            the same one.
        workdir (Path): Where scripts, netlists and reports are written.
            Resolved to an absolute path in ``__post_init__``.
        script (SynthScript): Recipe to run. Recorded in every report.
        read_cmd (str): Yosys front-end command for the sources.
        clock_period_ns (float | None): Clock period for ``create_clock``.
            Together with ``clock_port`` this decides whether the timing result
            is the clock-constrained critical path or an unconstrained number.
        clock_port (str | None): Port to attach the clock to.
        log_dir (Path | None): Where tool logs go; defaults to ``workdir``.
        env (dict | None): Environment for the tools; ``None`` inherits.
        verbose (bool): Print each tool invocation.
    """

    yosys: str
    sta: str
    liberty: str
    workdir: Path
    script: SynthScript = BASELINE
    read_cmd: str = "read_verilog -sv"
    clock_period_ns: float | None = None
    clock_port: str | None = None
    log_dir: Path | None = None
    env: dict[str, str] | None = None
    verbose: bool = True
    tool_runs: list[ToolRun] = field(default_factory=list)
    _celljson: Path | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # ABSOLUTE, always. The tools run with cwd=workdir, so a relative
        # workdir makes the generated script path relative to itself and yosys
        # dies with "Can't open script file".
        self.workdir = Path(self.workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        if not Path(self.liberty).exists():
            raise FileNotFoundError(
                f"Liberty not found: {self.liberty}. Area and delay from two "
                f"different Liberty files are not comparable, so this is fatal "
                f"rather than a warning.")
        # ABSOLUTE for the same reason as workdir: the Liberty path is written
        # into a script that yosys and sta run with cwd=workdir, so a relative
        # one resolves against the wrong directory. yosys then exits 1 with an
        # empty stat file and the node reports cells=None -- which reads as
        # "this design has no cells", not as "you gave me a bad path".
        self.liberty = str(Path(self.liberty).resolve())

    @property
    def constrained(self) -> bool:
        """True when a clock will actually be created before timing."""
        return bool(self.clock_period_ns) and bool(self.clock_port)

    # -- synthesis ------------------------------------------------------------
    def synthesize(self, sources: Sequence[str | os.PathLike[str]], top: str, *,
                   timeout_s: float | None = 3600.0,
                   label: str = "yosys-synth") -> tuple[QorReport, Path | None]:
        """Map ``sources`` to the Liberty's cells and report cells and area.

        Args:
            sources (Sequence): RTL source files.
            top (str): Top module name.
            timeout_s (float | None): Wall-clock budget for Yosys.
            label (str): Log basename for this invocation.

        Returns:
            tuple: ``(QorReport, netlist path or None)``. The report carries no
            timing yet; :meth:`timing` fills that in.
        """
        if not sources:
            raise ValueError(
                "no sources given; synthesising an empty design would report "
                "cells=0 in milliseconds and look like a fast success")
        netlist = self.workdir / f"{top}.mapped.v"
        statfile = self.workdir / f"{top}.{self.script.name}.stat.txt"
        self._celljson = self.workdir / f"{top}.cells.json"
        missing = [str(x) for x in sources if not Path(x).exists()]
        if missing:
            raise FileNotFoundError(
                f"source file(s) not found: {missing[:4]}. Reported here rather "
                f"than as a synthesis failure, because yosys would exit 1 with "
                f"an empty stat file and the node would report cells=None.")
        # ABSOLUTE: the script is run with cwd=workdir (see __post_init__).
        reads = "\n".join(f"{self.read_cmd} {Path(s).resolve().as_posix()}"
                           for s in sources)
        script = reads + "\n" + self.script.render(
            top=top, lib=Path(self.liberty).as_posix(),
            netlist=netlist.as_posix(), stat=statfile.as_posix(),
            celljson=self._celljson.as_posix())
        script_path = self.workdir / f"{top}.{self.script.name}.ys"
        script_path.write_text(script)

        run = run_tool([self.yosys, "-q", "-s", str(script_path)],
                       cwd=self.workdir, timeout_s=timeout_s,
                       log_dir=self.log_dir or self.workdir,
                       label=label, env=self.env, verbose=self.verbose)
        self.tool_runs.append(run)

        # Parse the tee'd stat file, not stdout: `yosys -q` prints no log at all.
        text = statfile.read_text() if statfile.exists() else ""
        if not text:
            for p_ in (run.stdout_path, run.stderr_path):
                if p_ and Path(p_).exists():
                    text += Path(p_).read_text()

        cells, area = parse_yosys_stat(text)
        ok = run.ok and netlist.exists() and cells is not None
        return (
            QorReport(
                top=top, success=ok, cells=cells, area_um2=area,
                constrained=self.constrained,
                clock_period_ns=self.clock_period_ns,
                script=self.script.name,
                message="" if ok else (
                    f"yosys rc={run.returncode} timed_out={run.timed_out} "
                    f"netlist={netlist.exists()} cells={cells}"),
                wall_s=run.wall_s, cpu_s=run.cpu_s, peak_rss_kb=run.peak_rss_kb,
                netlist_path=str(netlist) if netlist.exists() else None,
            ),
            netlist if netlist.exists() else None,
        )

    # -- timing ---------------------------------------------------------------
    def _sta_script(self, netlist: Path, top: str) -> str:
        lines = [
            f"read_liberty {Path(self.liberty).as_posix()}",
            f"read_verilog {netlist.as_posix()}",
            f"link_design {top}",
        ]
        if self.constrained:
            lines.append(
                f"create_clock -name clk -period {self.clock_period_ns} "
                f"[get_ports {self.clock_port}]")
        else:
            # No clock: report the longest combinational path so a number still
            # exists, but the caller is told it is NOT the constrained critical
            # path. Measured on picorv32/sky130: 12.7612 ns constrained at a
            # 10 ns clock vs 0.1959 ns unconstrained -- different quantities.
            lines.append("set_max_delay 0 -from [all_inputs] -to [all_outputs]")
        lines += [
            "report_checks -path_delay max -format full_clock_expanded -digits 4",
            "exit",
        ]
        return "\n".join(lines) + "\n"

    def timing(self, netlist: Path, top: str, *, timeout_s: float | None = 1800.0,
               label: str = "opensta") -> dict[str, object]:
        """Time a mapped netlist with OpenSTA.

        Args:
            netlist (Path): Mapped Verilog netlist from :meth:`synthesize`.
            top (str): Top module name.
            timeout_s (float | None): Wall-clock budget for OpenSTA.
            label (str): Log basename for this invocation.

        Returns:
            dict: The parsed report from :func:`parse_sta_report`, with
            ``path_summary`` annotated with RTL locations where possible, plus
            ``success`` (bool) and ``constrained`` (bool).
        """
        sp = self.workdir / f"{top}.sta.tcl"
        sp.write_text(self._sta_script(netlist, top))
        run = run_tool([self.sta, "-no_splash", "-exit", str(sp)],
                       cwd=self.workdir, timeout_s=timeout_s,
                       log_dir=self.log_dir or self.workdir,
                       label=label, env=self.env, verbose=self.verbose)
        self.tool_runs.append(run)
        text = ""
        for p in (run.stdout_path, run.stderr_path):
            if p and Path(p).exists():
                text += Path(p).read_text()
        parsed = parse_sta_report(text)
        # Place each critical-path cell back in the RTL, when we can.
        cell_src = load_cell_sources(self._celljson) if self._celljson else {}
        if cell_src and parsed.get("path_summary"):
            parsed["path_summary"] = annotate_path_with_sources(
                str(parsed["path_summary"]), cell_src)
        parsed["success"] = run.ok and bool(parsed.get("max_delay_ns") is not None)
        parsed["constrained"] = self.constrained
        return parsed

    def evaluate(self, sources: Sequence[str | os.PathLike[str]], top: str,
                 **kw) -> QorReport:
        """Full evaluation: synthesise, then time the mapped netlist.

        Args:
            sources (Sequence): RTL source files.
            top (str): Top module name.
            **kw: Forwarded to :meth:`synthesize`.

        Returns:
            QorReport: One report carrying cells, area, timing and cost. Its
            ``success`` is True only if BOTH stages succeeded.
        """
        qor, netlist = self.synthesize(sources, top, **kw)
        if netlist is None:
            return qor
        t = self.timing(netlist, top)
        qor.max_delay_ns = t.get("max_delay_ns")          # type: ignore[assignment]
        qor.slack_ns = t.get("slack_ns")                  # type: ignore[assignment]
        qor.startpoint = t.get("startpoint")              # type: ignore[assignment]
        qor.endpoint = t.get("endpoint")                  # type: ignore[assignment]
        qor.path_summary = str(t.get("path_summary", ""))
        qor.success = bool(qor.success and t.get("success"))
        qor.wall_s = self.total_wall_s
        qor.cpu_s = self.total_cpu_s
        qor.peak_rss_kb = self.peak_rss_kb
        if not qor.constrained:
            qor.message = (qor.message + " UNCONSTRAINED: max_delay_ns is the "
                           "longest combinational path, NOT the clock-constrained "
                           "critical path; do not compare it with a constrained "
                           "number.").strip()
        elif not qor.success:
            qor.message = (qor.message + f" sta_success={t.get('success')}").strip()
        return qor

    @property
    def total_cpu_s(self) -> float:
        """CPU seconds summed over every tool this node has run."""
        return sum(r.cpu_s for r in self.tool_runs)

    @property
    def total_wall_s(self) -> float:
        """Wall-clock seconds summed over every tool this node has run."""
        return sum(r.wall_s for r in self.tool_runs)

    @property
    def peak_rss_kb(self) -> int:
        """Largest peak RSS of any single tool this node has run."""
        return max((r.peak_rss_kb for r in self.tool_runs), default=0)


# The resource token is the ONLY binding between this function and the worker
# image carrying yosys/abc/OpenSTA: a cluster node type declaring
# `resources: {"yosys_sta": 1}` with
# `docker.image: ghcr.io/ucb-bar/chia-yosys-sta:latest` is what puts this call
# in a container that can run the tools.
@ChiaFunction(resources={"yosys_sta": 1})
def yosys_sta_qor(sources: list[str], top: str, liberty: str,
                  workdir: str = ".", script: str = "baseline-flat",
                  clock_period_ns: float | None = None,
                  clock_port: str | None = None,
                  yosys: str = "yosys", sta: str = "sta",
                  read_cmd: str = "read_verilog -sv",
                  timeout_s: float = 3600.0) -> dict:
    """Synthesise RTL with open-source tools and report area, cells and timing.

    Maps ``sources`` onto the cells of ``liberty`` with Yosys and ABC, then
    times the mapped netlist with OpenSTA. Needs no commercial licence, so a
    result from this node is reproducible by anyone.

    Pass BOTH ``clock_period_ns`` and ``clock_port`` whenever you care about the
    critical path. Without them the reported ``max_delay_ns`` is the longest
    combinational path with no clock to relate it to, which is a different
    quantity: on picorv32 with sky130 the same netlist reads 12.7612 ns
    constrained at a 10 ns clock and 0.1959 ns unconstrained. The returned
    ``constrained`` flag says which one you got.

    Args:
        sources (list[str]): Paths to the RTL source files to synthesise.
        top (str): Name of the top module.
        liberty (str): Path to the Liberty (.lib) file. Two results are only
            comparable if they used the same one.
        workdir (str): Directory for generated scripts, the netlist and logs.
        script (str): Synthesis recipe name: ``"baseline-flat"`` (full flatten,
            the obvious recipe) or ``"tuned-hier"`` (preserves hierarchy, uses
            ``abc -fast``).
        clock_period_ns (float | None): Clock period in ns. Required, with
            ``clock_port``, for a clock-constrained critical path.
        clock_port (str | None): Name of the clock port, e.g. ``"clk"``.
        yosys (str): ``yosys`` executable, by name or absolute path.
        sta (str): OpenSTA ``sta`` executable, likewise.
        read_cmd (str): Yosys front-end command, e.g. ``"read_verilog -sv"``.
        timeout_s (float): Wall-clock budget for synthesis. On expiry the whole
            tool tree is killed and ``success`` is false.

    Returns:
        dict: ``success`` (bool -- false means no field below is usable),
        ``cells`` (int | None), ``area_um2`` (float | None),
        ``max_delay_ns`` (float | None), ``slack_ns`` (float | None),
        ``constrained`` (bool), ``clock_period_ns`` (float | None),
        ``startpoint`` / ``endpoint`` (str | None), ``path_summary`` (str, the
        worst path annotated with RTL file:line where recoverable), ``script``
        (str), ``message`` (str), ``wall_s`` / ``cpu_s`` (float),
        ``peak_rss_kb`` (int) and ``netlist_path`` (str | None).

    Raises:
        ValueError: If ``script`` is not a known recipe, or ``sources`` is empty.
        FileNotFoundError: If ``liberty`` does not exist.
        ToolNotFound: If ``yosys`` or ``sta`` is missing -- which means the
            worker is not running the chia-yosys-sta image.
    """
    if script not in SCRIPTS:
        raise ValueError(f"unknown synthesis script {script!r}; "
                         f"known: {sorted(SCRIPTS)}")
    node = YosysStaNode(
        yosys=yosys, sta=sta, liberty=liberty, workdir=Path(workdir),
        script=SCRIPTS[script], read_cmd=read_cmd,
        clock_period_ns=clock_period_ns, clock_port=clock_port,
    )
    return node.evaluate(sources, top, timeout_s=timeout_s).as_dict()


__all__ = ["YosysStaNode", "QorReport", "SynthScript", "SCRIPTS",
           "BASELINE", "TUNED", "yosys_sta_qor", "ToolNotFound",
           "parse_yosys_stat", "parse_sta_report", "load_cell_sources",
           "annotate_path_with_sources"]


if __name__ == "__main__":
    import tempfile

    print("=== chia.vlsi.yosys_sta self-test ===")

    # REAL captured `stat -liberty` output. This is the format the node actually
    # produces; the plain "Number of cells:" line does not appear at all.
    lib_stat = """
7. Printing statistics.

=== Alu ===

        +----------Local Count, excluding submodules.
        |        +-Local Area, excluding submodules.
     7841        - wires
    23642        - wire bits
       12        - ports
     7692 4.32E+04 cells
       43  376.611   sky130_fd_sc_hd__a2111oi_0
      113  989.699   sky130_fd_sc_hd__xor2_1

   Chip area for module '\\Alu': 43193.926400
     of which used for sequential elements: 0.000000 (0.00%)
"""
    c2, a2 = parse_yosys_stat(lib_stat)
    assert c2 == 7692, f"liberty-format cell count not parsed: {c2}"
    assert abs(a2 - 43193.9264) < 1e-6, a2
    print(f"    stat(-liberty) -> cells={c2} area={a2} (real captured output)")

    plain = """
=== picorv32 ===
   Number of cells:               9174
=== design hierarchy ===
   Number of cells:               6691
   Chip area for module '\\picorv32': 75664.115200
"""
    cells, area = parse_yosys_stat(plain)
    assert cells == 6691 and abs(area - 75664.1152) < 1e-6, (cells, area)
    print(f"    stat(plain) -> cells={cells} area={area} (last block wins)")

    assert parse_yosys_stat("no numbers here") == (None, None)
    print("    unparseable stat -> (None, None), never a fabricated number")

    sta_out = """
Startpoint: cpuregs_reg_1_ (rising edge-triggered flip-flop clocked by clk)
Endpoint: mem_addr_reg_7_ (rising edge-triggered flip-flop clocked by clk)
Path Group: clk
Path Type: max

   0.0000    0.0000 ^ cpuregs_reg_1_/CLK (sky130_fd_sc_hd__dfxtp_1)
   4.2100    9.4210 ^ mem_addr_reg_7_/D (sky130_fd_sc_hd__dfxtp_1)
             9.4210   data arrival time

            10.0000   data required time
            -9.4210   data arrival time
  ---------------------------------------
             0.5790   slack (MET)
"""
    p = parse_sta_report(sta_out)
    assert abs(p["slack_ns"] - 0.579) < 1e-9, p
    assert abs(p["max_delay_ns"] - 9.421) < 1e-9, p
    assert p["endpoint"] == "mem_addr_reg_7_" and p["startpoint"] == "cpuregs_reg_1_"
    assert p["met"] is True
    print(f"    sta -> slack={p['slack_ns']} arrival={p['max_delay_ns']} "
          f"endpoint={p['endpoint']}")

    multi = sta_out + sta_out.replace("0.5790   slack (MET)",
                                      "-3.0000   slack (VIOLATED)")
    pm = parse_sta_report(multi)
    assert abs(pm["slack_ns"] + 3.0) < 1e-9 and pm["met"] is False
    print(f"    worst-of-N selected across paths: slack={pm['slack_ns']} met={pm['met']}")

    # Cell -> RTL line recovery, and its graceful degradation.
    with tempfile.TemporaryDirectory() as td:
        j = Path(td) / "cells.json"
        j.write_text(json.dumps({"modules": {"top": {"cells": {
            "mem_addr_reg_7_": {"attributes": {"src": "picorv32.v:1122|x.v:3"}}}}}}))
        m = load_cell_sources(j)
        assert m == {"mem_addr_reg_7_": "picorv32.v:1122"}, m
        ann = annotate_path_with_sources(
            "   4.2100    9.4210 ^ mem_addr_reg_7_/D (sky130_fd_sc_hd__dfxtp_1)", m)
        assert ann.endswith("<- picorv32.v:1122"), ann
        print(f"    cell placed back in RTL: ...{ann[-40:]}")
        assert load_cell_sources(Path(td) / "nope.json") == {}
        (Path(td) / "bad.json").write_text("{not json")
        assert load_cell_sources(Path(td) / "bad.json") == {}
        print("    missing/corrupt cell map degrades to {}, never raises")

        assert set(SCRIPTS) == {"baseline-flat", "tuned-hier"}
        r = BASELINE.render(top="picorv32", lib="/pdk/sky130.lib",
                            netlist="/tmp/n.v", stat="/tmp/s.txt")
        assert "abc -D 10000 -liberty /pdk/sky130.lib" in r and "-top picorv32" in r
        assert "tee -o /tmp/s.txt stat" in r, \
            "stat must be tee'd to a file (yosys -q hides stdout)"
        assert "abc -fast" in TUNED.render(top="T", lib="L", netlist="N", stat="S")
        print(f"    recipes registered and rendered: {sorted(SCRIPTS)}")

        # A missing Liberty is fatal at construction, not a silent bad number.
        try:
            YosysStaNode(yosys="yosys", sta="sta", liberty="/does/not/exist.lib",
                         workdir=Path(td))
            raise AssertionError("missing Liberty accepted")
        except FileNotFoundError as e:
            print(f"    Liberty presence enforced: {str(e)[:60]}...")

        lib = Path(td) / "fake.lib"
        lib.write_text("library(x){}")
        n = YosysStaNode(yosys="yosys", sta="sta", liberty=str(lib),
                         workdir=Path(td), verbose=False)
        assert n.constrained is False
        assert "set_max_delay" in n._sta_script(Path("n.v"), "t")
        n2 = YosysStaNode(yosys="yosys", sta="sta", liberty=str(lib),
                          workdir=Path(td), clock_period_ns=10.0, clock_port="clk",
                          verbose=False)
        assert n2.constrained is True
        assert "create_clock -name clk -period 10.0 [get_ports clk]" \
            in n2._sta_script(Path("n.v"), "t")
        print("    constrained flag drives create_clock vs set_max_delay")

        try:
            n.synthesize([], "t")
            raise AssertionError("empty source list accepted")
        except ValueError as e:
            print(f"    empty source list rejected: {str(e)[:56]}...")

        try:
            yosys_sta_qor(["a.v"], "t", str(lib), workdir=td, script="nope")
            raise AssertionError("unknown script accepted")
        except ValueError as e:
            print(f"    unknown recipe rejected: {str(e)[:52]}...")

    print("=== all chia.vlsi.yosys_sta self-tests passed ===")
