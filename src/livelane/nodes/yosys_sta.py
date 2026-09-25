"""Lane S: Yosys + ABC technology mapping, then OpenSTA timing.

This is the arm the whole experiment is calibrated against, and the arm the
latency-injection arms *are* (``I(d)`` is exactly this stack plus ``d`` seconds).
Two consequences shape the code:

* The synthesis script is **data**, not a hardcoded string.  A reviewer will
  argue the slow baseline was left unoptimised, 52% of the medium block's 24
  minutes was ``opt_dff``, not ABC, so the script must be swappable, named, and
  recorded with every measurement, and the tuned variant must be published
  alongside the naive one.  See :data:`SCRIPTS`.
* Nothing is estimated.  Cell count and area come from ``stat -liberty`` and the
  critical path from OpenSTA over the mapped netlist; if a parse fails the report
  is marked invalid rather than silently reporting ``None`` as a good number.

CHIA today ships only a Cadence Genus synthesis path; this node is the
open-source alternative.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from livelane.harness.run import ToolRun, run_tool
from livelane.state.reports import QorReport, TimingReport


@dataclass(frozen=True)
class SynthScript:
    """A named, recordable Yosys recipe."""

    name: str
    rationale: str
    #: ``{top}``, ``{lib}``, ``{netlist}``, ``{stat}`` and ``{celljson}`` are
    #: substituted.
    body: str

    def render(self, *, top: str, lib: str, netlist: str, stat: str,
               celljson: str = "/dev/null") -> str:
        return self.body.format(top=top, lib=lib, netlist=netlist, stat=stat,
                                celljson=celljson)


#: The script the 2026-09-02 baseline used. Kept verbatim so its numbers stay
#: reproducible, and reported as the *naive* baseline, never as the tuned one.
BASELINE = SynthScript(
    name="baseline-2026-09-02",
    rationale=(
        "The recipe behind the published lane-S numbers (Alu 2.81 s, DivUnit "
        "3.53 s, DecodeUnit 5.66 s, RenameTableWrapper 1447.66 s). Full flatten "
        "via `synth`, ABC with a fixed -D 10000."
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
        "write_json {celljson}\n"
    ),
)

#: Good-faith tuning, per the plan's confound table: keep hierarchy so `opt_dff`
#: is not handed an 807k-cell flattened design, and let ABC use its fast script.
TUNED = SynthScript(
    name="tuned-hier",
    rationale=(
        "Preserves hierarchy (-flatten omitted, `synth` run without full flatten) "
        "and uses `abc -fast`, targeting the measured pathology that 52% of the "
        "medium block's runtime was opt_dff on an sv2v-inflated netlist."
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
    ),
)

SCRIPTS: dict[str, SynthScript] = {s.name: s for s in (BASELINE, TUNED)}


# --- parsers -----------------------------------------------------------------
# Both are written against the real output format and are unit-tested against
# captured text, because a silently-wrong area number would corrupt every plot.

# `stat` has TWO output formats and the difference is not cosmetic:
#   without -liberty:  "   Number of cells:               6691"
#   with    -liberty:  "     7692 4.32E+04 cells"
# Lane S always passes -liberty (it must, to get area), so the second form is the
# one that matters. Matching only the first, which an earlier version of this
# parser did, silently returns cells=None on every real run.
_CELLS_PLAIN_RE = re.compile(r"^\s*Number of cells:\s+(\d+)\s*$", re.M)
_CELLS_LIB_RE = re.compile(r"^\s*(\d+)\s+([0-9.eE+-]+)\s+cells\s*$", re.M)
_AREA_RE = re.compile(r"^\s*Chip area for (?:top )?module '?\\?([^':]+)'?:\s*"
                      r"([0-9.eE+-]+)\s*$", re.M)


def parse_yosys_stat(text: str) -> tuple[int | None, float | None]:
    """Extract (cells, chip area) from ``stat`` output, either format.

    Takes the LAST match of each: with hierarchy preserved ``stat`` prints one
    block per module and the top-level summary comes last.

    Chip area is preferred over the per-block area column because it is the
    whole-design number and is printed at full precision, whereas the inline
    column is rounded ("4.32E+04").
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
    """Extract worst slack / arrival / endpoints from ``report_checks`` output."""
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
    # mangled ("_10883_"), so an agent told only "slack is -2.92ns at _10883_"
    # cannot know WHICH logic to change and edits blind, observed once as a
    # formally-proven edit that moved delay the wrong way. Carry the actual
    # report_checks path block: it names the cells and nets on the critical path,
    # which is the most that survives synthesis without source tracking.
    blk = _worst_path_block(text)
    if blk:
        out["path_summary"] = blk
    return out


def _worst_path_block(text: str, max_lines: int = 40) -> str:
    """The report_checks block for the worst path, trimmed."""
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
    lines = [l.rstrip() for l in block.splitlines() if l.strip()]
    if len(lines) > max_lines:
        head, tail = lines[: max_lines - 8], lines[-8:]
        lines = head + [f"    ... {len(lines) - max_lines + 8} lines elided ..."] + tail
    return "\n".join(lines)


def load_cell_sources(json_path: str | os.PathLike[str]) -> dict[str, str]:
    """Map post-synthesis cell name -> originating RTL ``file:line``.

    OpenSTA names critical-path nodes by their synthesised cell (``_05679_``),
    which is unactionable for something editing RTL. Yosys keeps the originating
    location in each cell's ``src`` attribute, but ``write_verilog -noattr``
    discards it, so the mapping is recovered from a JSON netlist instead.

    Returns {} rather than raising: a missing or malformed map degrades the
    report to cell names, which is what it was before, and must never take down
    an evaluation.
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
    """Append ``file:line`` to each critical-path line whose cell we can place."""
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


# --- the node ----------------------------------------------------------------

@dataclass
class YosysStaLane:
    """Lane S evaluator: synthesise with Yosys/ABC, then time with OpenSTA."""

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
    tool_runs: list[ToolRun] = field(default_factory=list)
    _celljson: Path | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # ABSOLUTE, always. The tools run with cwd=workdir, so a relative workdir
        # makes the generated script path relative to itself and yosys dies with
        # "Can't open script file". It cost the entire neutral-judge pass of a
        # 178-minute sweep, which reported rho=None rather than failing loudly.
        self.workdir = Path(self.workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        if not Path(self.liberty).exists():
            raise FileNotFoundError(
                f"Liberty not found: {self.liberty}. Both lanes MUST use the same "
                f"Liberty or area and delay are not comparable."
            )

    #, synthesis ------------------------------------------------------------
    def synthesize(self, sources: Sequence[str | os.PathLike[str]], top: str, *,
                   timeout_s: float | None = 3600.0,
                   label: str = "yosys-synth") -> tuple[QorReport, Path | None]:
        netlist = self.workdir / f"{top}.mapped.v"
        statfile = self.workdir / f"{top}.{self.script.name}.stat.txt"
        self._celljson = self.workdir / f"{top}.cells.json"
        reads = "\n".join(f"{self.read_cmd} {Path(s).as_posix()}" for s in sources)
        script = reads + "\n" + self.script.render(
            top=top, lib=Path(self.liberty).as_posix(),
            netlist=netlist.as_posix(), stat=statfile.as_posix(),
            celljson=self._celljson.as_posix())
        script_path = self.workdir / f"{top}.{self.script.name}.ys"
        script_path.write_text(script)

        run = run_tool([self.yosys, "-q", "-s", str(script_path)],
                       cwd=self.workdir, timeout_s=timeout_s,
                       log_dir=self.log_dir or self.workdir,
                       label=label, env=self.env)
        self.tool_runs.append(run)

        # Parse the tee'd stat file, not stdout: `yosys -q` prints no log at all.
        text = statfile.read_text() if statfile.exists() else ""
        if not text:
            for p_ in (run.stdout_path, run.stderr_path):
                if p_ and Path(p_).exists():
                    text += Path(p_).read_text()

        cells, area = parse_yosys_stat(text)
        ok = run.ok and netlist.exists()
        return (
            QorReport(
                top=top,
                cells=cells,
                area_um2=area,
                max_delay_ns=None,   # filled in by timing()
                valid=ok and cells is not None,
                message="" if ok else (
                    f"yosys rc={run.returncode} timed_out={run.timed_out}"),
                source=f"yosys+abc[{self.script.name}]",
                wall_s=run.wall_s, cpu_s=run.cpu_s, peak_rss_kb=run.peak_rss_kb,
            ),
            netlist if netlist.exists() else None,
        )

    #, timing ---------------------------------------------------------------
    def _sta_script(self, netlist: Path, top: str) -> str:
        lines = [
            f"read_liberty {Path(self.liberty).as_posix()}",
            f"read_verilog {netlist.as_posix()}",
            f"link_design {top}",
        ]
        if self.clock_period_ns and self.clock_port:
            lines.append(
                f"create_clock -name clk -period {self.clock_period_ns} "
                f"[get_ports {self.clock_port}]")
        else:
            # No clock constraint: report the longest combinational path so a
            # delay number still exists. Recorded as unconstrained.
            lines.append("set_max_delay 0 -from [all_inputs] -to [all_outputs]")
        lines += [
            "report_checks -path_delay max -format full_clock_expanded -digits 4",
            "exit",
        ]
        return "\n".join(lines) + "\n"

    def timing(self, netlist: Path, top: str, *, timeout_s: float | None = 1800.0,
               label: str = "opensta") -> TimingReport:
        sp = self.workdir / f"{top}.sta.tcl"
        sp.write_text(self._sta_script(netlist, top))
        run = run_tool([self.sta, "-no_splash", "-exit", str(sp)],
                       cwd=self.workdir, timeout_s=timeout_s,
                       log_dir=self.log_dir or self.workdir,
                       label=label, env=self.env)
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
        return TimingReport(
            top=top,
            slack_ns=parsed.get("slack_ns"),          # type: ignore[arg-type]
            max_delay_ns=parsed.get("max_delay_ns"),  # type: ignore[arg-type]
            clock_period_ns=self.clock_period_ns,
            endpoint=parsed.get("endpoint"),          # type: ignore[arg-type]
            startpoint=parsed.get("startpoint"),      # type: ignore[arg-type]
            path_summary=str(parsed.get("path_summary", "")),
            valid=run.ok and bool(parsed),
            message="" if run.ok else f"sta rc={run.returncode}",
            source="opensta",
            wall_s=run.wall_s,
        )

    def evaluate(self, sources: Sequence[str | os.PathLike[str]], top: str,
                 **kw) -> tuple[QorReport, TimingReport | None]:
        """Full lane-S evaluation: synthesise, then time the mapped netlist."""
        qor, netlist = self.synthesize(sources, top, **kw)
        if netlist is None:
            return qor, None
        t = self.timing(netlist, top)
        # Fold the timing result into the QoR the agent sees, so lane S and
        # lane L return the same shape.
        qor.max_delay_ns = t.max_delay_ns
        qor.slack_ns = t.slack_ns
        return qor, t

    @property
    def total_cpu_s(self) -> float:
        return sum(r.cpu_s for r in self.tool_runs)


if __name__ == "__main__":
    print("=== yosys/sta parser self-test ===")

    stat = """
=== picorv32 ===

   Number of wires:               3000
   Number of cells:               9174
     $_AND_                       100

=== design hierarchy ===

   Number of cells:               6691
     sky130_fd_sc_hd__a2111o_1      12
     sky130_fd_sc_hd__nand2_1     900

   Chip area for module '\\picorv32': 75664.115200
"""
    cells, area = parse_yosys_stat(stat)
    # Must take the LAST block (the mapped summary), not the first generic one.
    assert cells == 6691, cells
    assert abs(area - 75664.1152) < 1e-6, area
    print(f"    stat(plain) -> cells={cells} area={area} (last block wins)")

    # REAL captured output from `stat -liberty` on XiangShan Alu, 2026-09-03.
    # This is the format lane S actually produces; the plain "Number of cells:"
    # line does not appear at all.
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

    viol = sta_out.replace("0.5790   slack (MET)", "-1.2500   slack (VIOLATED)")
    pv = parse_sta_report(viol)
    assert abs(pv["slack_ns"] + 1.25) < 1e-9 and pv["met"] is False
    print(f"    negative slack parsed: {pv['slack_ns']} met={pv['met']}")

    # Worst (most negative) slack must win across multiple reported paths.
    multi = sta_out + sta_out.replace("0.5790   slack (MET)", "-3.0000   slack (VIOLATED)")
    assert abs(parse_sta_report(multi)["slack_ns"] + 3.0) < 1e-9
    print("    worst-of-N slack selected across multiple paths")

    assert set(SCRIPTS) == {"baseline-2026-09-02", "tuned-hier"}
    r = BASELINE.render(top="picorv32", lib="/pdk/sky130.lib", netlist="/tmp/n.v",
                        stat="/tmp/s.txt")
    assert "abc -D 10000 -liberty /pdk/sky130.lib" in r and "-top picorv32" in r
    assert "tee -o /tmp/s.txt stat" in r, "stat must be tee'd to a file (yosys -q hides stdout)"
    assert "abc -fast" in TUNED.render(top="T", lib="L", netlist="N", stat="S")
    print(f"    scripts registered: {sorted(SCRIPTS)}")

    try:
        YosysStaLane(yosys="yosys", sta="sta", liberty="/does/not/exist.lib",
                     workdir=Path("/tmp/x"))
        raise AssertionError("missing Liberty accepted")
    except FileNotFoundError as e:
        print(f"    Liberty presence enforced: {str(e)[:80]}...")
    print("=== all yosys/sta parser self-tests passed ===")
