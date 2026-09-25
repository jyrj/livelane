"""The neutral final judge: ONE pinned flow re-scores every arm's survivors.

Lane S and lane L do not produce comparable QoR.  On identical ``DecodeUnit``
source against the identical Liberty they differ by 3.84x in cells and 5.09x in
area, and, before the partitioning confound was found, by 18.6x in max
delay.  Those gaps are
properties of the *evaluator*, not of the design the agent produced.  So a
cross-arm sentence of the form "arm X found a better design than arm Y" is
meaningless while each arm is scored by its own tool: the winner would be
decided by which mapper was in the loop.

This module removes that degree of freedom.  Every surviving candidate from
every arm is re-synthesised here, by the same tools, the same recipe, the same
Liberty, and the resulting number is the only one a cross-arm quality claim may
cite.  Two consequences shape the code:

* **The configuration is not a parameter of scoring.**  ``NeutralJudge`` is a
  frozen dataclass and the recipe (:data:`JUDGE_SCRIPT`) and front end
  (:data:`JUDGE_READ_CMD`) are module constants, not fields.  There is
  deliberately no per-candidate knob: a judge that could be tuned per candidate
  is not a judge.  What configuration there is (tool paths, Liberty, an optional
  pinned clock) is recorded in :attr:`JudgeResult.config` on every single result,
  so a table of judge numbers carries its own provenance.
* **Nothing is estimated.**  The flow is Yosys ``stat -liberty`` plus OpenSTA
  over the mapped netlist, and a parse failure yields ``valid=False`` with
  ``None`` numbers rather than a plausible-looking one.

The judge is also the instrument for **H4** (:func:`proxy_vs_judge`): does the
in-lane proxy *rank* candidates the way the neutral judge does?  Since lane L's
fast configuration is 16x off in absolute delay, ranking fidelity
(Spearman rho >= :data:`H4_RHO_THRESHOLD` over >= :data:`H4_MIN_N` candidates)
is the only thing that could make it usable as an inner loop.

TIMING CAVEAT.  With no clock constrained, the default, matching the published
lane-S baseline, OpenSTA is asked for the longest *combinational* input-to-
output path, which on a register-heavy design is not the critical path at all
(picorv32: 0.1959 ns unconstrained vs 12.7612 ns with a 10 ns clock on ``clk``).
Cells and area are unaffected.  Pin ``clock_port``/``clock_period_ns`` on the
judge when a delay number has to mean something; it is recorded in the config
either way, and ``timing_constrained`` says which was used.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from livelane.analysis.metrics import spearman
from livelane.db.store import VariantStore
from livelane.harness.provenance import tool_version, yosys_git_sha
from livelane.nodes.yosys_sta import SCRIPTS, SynthScript, YosysStaLane

#: Bumped whenever a change here could move a judge number. Recorded per result
#: so two judge numbers can be checked for comparability instead of assumed to be.
JUDGE_ID = "neutral-judge-1"

#: The pinned recipe. Indexing SCRIPTS (rather than importing BASELINE) makes a
#: rename of the published baseline a loud KeyError at import, not a silent
#: change of judge.
JUDGE_SCRIPT_NAME = "baseline-2026-09-02"
JUDGE_SCRIPT: SynthScript = SCRIPTS[JUDGE_SCRIPT_NAME]

#: `read_slang` so the judge consumes the SAME files both lanes do, including
#: XiangShan SystemVerilog. All sources go into ONE invocation: a per-file
#: `read_slang` elaborates each file in isolation and cannot resolve a
#: multi-file hierarchy.
JUDGE_READ_CMD = "read_slang"

#: Files handed to the front end when a directory is judged. Headers (.svh/.vh)
#: are includes, not compilation units, and are deliberately excluded.
RTL_SUFFIXES: tuple[str, ...] = (".v", ".sv")

#: Metrics that exist as both a `qor_*` (proxy) and a `judge_*` (neutral) column.
#: Used as an allow-list, these names are interpolated into SQL.
H4_METRICS: frozenset[str] = frozenset({"cells", "area_um2", "max_delay_ns", "slack_ns"})

#: The threshold for a usable proxy ranking (H4).
H4_RHO_THRESHOLD = 0.8

#: The other half of the same criterion: "Spearman rho >= 0.8 over >= 30
#: candidates". A rho over three candidates is not a small H4 result, it is no
#: H4 result, so the count is a precondition of the decision, not a caveat on it.
H4_MIN_N = 30


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _sha256_file(path: str | os.PathLike[str]) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- the result --------------------------------------------------------------


@dataclass
class JudgeResult:
    """One candidate's score under the neutral flow, with the flow attached.

    ``config`` travels with the number on purpose: a judge number quoted without
    its configuration is not checkable, and the partitioning result is precisely
    the case where an unrecorded default silently changed the answer.
    """

    top: str
    cells: int | None
    area_um2: float | None
    max_delay_ns: float | None
    slack_ns: float | None
    valid: bool
    message: str
    wall_s: float | None
    cpu_s: float | None
    config: dict[str, Any]
    judged_at: str
    #: Timing is reported separately from ``valid``: cells and area can be
    #: trustworthy on a design whose STA run produced nothing parseable.
    timing_valid: bool = False
    netlist_sha256: str | None = None
    log_paths: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        """JSON-native dict for the ``judge_json`` column."""
        return asdict(self)

    @property
    def columns(self) -> dict[str, Any]:
        """The typed `judge_*` columns, exactly as the schema names them.

        Fail closed, because this is the surface analysis reads: the schema has
        no ``judge_valid``, and both :meth:`VariantStore.best_so_far` (which
        plots ``judge_max_delay_ns`` and ``judge_area_um2``) and
        :func:`proxy_vs_judge` take a non-NULL column as a measurement. An
        invalid result therefore contributes NULL, which also clears a stale
        number left by an earlier judging, and keeps its detail in
        ``judge_json``, where the reason it failed is auditable.
        """
        ok = self.valid
        timed = ok and self.timing_valid
        return {
            "judge_cells": self.cells if ok else None,
            "judge_area_um2": self.area_um2 if ok else None,
            "judge_max_delay_ns": self.max_delay_ns if timed else None,
            "judge_slack_ns": self.slack_ns if timed else None,
        }

    def summary(self) -> str:
        status = "OK " if self.valid else "INVALID"
        d = "None" if self.max_delay_ns is None else f"{self.max_delay_ns:.4f}"
        a = "None" if self.area_um2 is None else f"{self.area_um2:.1f}"
        return (f"[{status}] {self.top}: cells={self.cells} area={a} um2 "
                f"delay={d} ns  ({self.wall_s or 0.0:.2f}s wall)"
                + (f"  {self.message}" if self.message else ""))


# --- the judge ---------------------------------------------------------------


@dataclass(frozen=True)
class NeutralJudge:
    """Yosys + ABC + OpenSTA, pinned, identical for every candidate in every arm.

    Frozen by design.  The recipe and front end are not fields at all, so the
    only way to change what the judge does is to edit this module, which moves
    :data:`JUDGE_ID` and is visible in every recorded config.
    """

    yosys: str = "yosys"
    sta: str = "sta"
    #: Defaults to ``$LIVELANE_LIBERTY``. Both lanes and the judge MUST use the
    #: same file or area and delay are not comparable.
    liberty: str = ""
    timeout_s: float = 3600.0
    sta_timeout_s: float = 1800.0
    #: Optional, judge-wide (never per-candidate) clock constraint. Without it
    #: OpenSTA reports only combinational I/O paths, see the module docstring.
    clock_port: str | None = None
    clock_period_ns: float | None = None
    verbose: bool = True
    #: Filled once in __post_init__; contents are provenance, not settings.
    _pinned: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        lib = self.liberty or os.environ.get("LIVELANE_LIBERTY", "")
        if not lib:
            raise ValueError(
                "NeutralJudge has no Liberty: pass liberty=... or set "
                "$LIVELANE_LIBERTY. Refusing to score against an unknown library."
            )
        if not Path(lib).is_file():
            raise FileNotFoundError(
                f"Liberty not found: {lib}. The judge must use the SAME Liberty "
                f"as both lanes or its numbers are not comparable to theirs."
            )
        object.__setattr__(self, "liberty", str(Path(lib)))
        if (self.clock_port is None) != (self.clock_period_ns is None):
            raise ValueError(
                "clock_port and clock_period_ns must be set together; a half-"
                "specified clock would silently fall back to an unconstrained "
                "timing run and report a combinational path as the critical one."
            )
        self._pinned.update(self._probe())

    #, configuration --------------------------------------------------------
    def _probe(self) -> dict[str, Any]:
        """Versions and hashes of what this judge is actually made of.

        A version is recorded ONLY when the configured binary is the one
        ``tool_version`` would probe (it probes by name, off PATH). Reporting
        PATH's yosys version for a differently-configured binary would be a
        fabricated provenance field, so it is left None instead.
        """
        out: dict[str, Any] = {
            "liberty_sha256": _sha256_file(self.liberty),
            "script_sha256": hashlib.sha256(JUDGE_SCRIPT.body.encode()).hexdigest(),
            "yosys_version": None,
            "yosys_git_sha": None,
            "opensta_version": None,
        }
        for name, configured, keys in (
            ("yosys", self.yosys, ("yosys_version",)),
            ("sta", self.sta, ("opensta_version",)),
        ):
            on_path = shutil.which(name)
            # Matching on the BASENAME would call /opt/other-build/yosys "the
            # same as" PATH's yosys and record PATH's version for it, exactly
            # the fabricated provenance field this method exists to avoid. Only
            # the bare command name (what `tool_version` itself resolves off
            # PATH) or a path resolving to the same file counts as the same tool.
            same = on_path is not None and (
                str(configured) == name
                or (Path(configured).exists()
                    and Path(configured).resolve() == Path(on_path).resolve())
            )
            if same:
                v = tool_version(name)
                for k in keys:
                    out[k] = v
        if out["yosys_version"]:
            out["yosys_git_sha"] = yosys_git_sha()
        return out

    def config(self) -> dict[str, Any]:
        """The full pinned configuration, recorded with every number produced.

        A fresh dict each call: a caller mutating what it got back must not be
        able to change what the next candidate is judged under.
        """
        return {
            "judge": JUDGE_ID,
            "flow": "yosys+abc+opensta",
            "script": JUDGE_SCRIPT_NAME,
            "read_cmd": JUDGE_READ_CMD,
            "liberty": self.liberty,
            "yosys": self.yosys,
            "sta": self.sta,
            "clock_port": self.clock_port,
            "clock_period_ns": self.clock_period_ns,
            "timing_constrained": self.clock_port is not None,
            "timeout_s": self.timeout_s,
            "sta_timeout_s": self.sta_timeout_s,
            **dict(self._pinned),
        }

    #, scoring --------------------------------------------------------------
    def _invalid(self, top: str, message: str) -> JudgeResult:
        """Fail closed: no number at all rather than a wrong one."""
        return JudgeResult(
            top=top, cells=None, area_um2=None, max_delay_ns=None, slack_ns=None,
            valid=False, message=message, wall_s=None, cpu_s=None,
            config=self.config(), judged_at=_utcnow(),
        )

    def _read_stanza(self, top: str,
                     sources: Sequence[str | os.PathLike[str]] | None,
                     filelist: str | os.PathLike[str] | None) -> str:
        """The front-end lines prepended to the pinned recipe.

        Every path is made absolute: Yosys runs with ``cwd=workdir``, so a
        relative filelist resolves against the scratch directory and the run
        dies with rc=1 after 0.01s, a failure that looks like a broken design.
        """
        if filelist is not None:
            fl = Path(filelist).resolve()
            if not fl.is_file():
                raise ValueError(f"filelist not found: {fl}")
            return f"{JUDGE_READ_CMD} --top {top} -F {fl.as_posix()}"
        files = [Path(s).resolve() for s in (sources or [])]
        missing = [f for f in files if not f.exists()]
        if missing:
            raise ValueError(f"source(s) not found: {[str(m) for m in missing[:3]]}")
        files = [f for f in files if f.suffix in RTL_SUFFIXES]
        if not files:
            raise ValueError(
                "no filelist and no .v/.sv sources; the judge will not score an "
                "empty design")
        return (f"{JUDGE_READ_CMD} --top {top} "
                + " ".join(f.as_posix() for f in files))

    def _purge_stale(self, wd: Path, top: str) -> None:
        """Delete the outputs a previous judging left in this workdir.

        :class:`YosysStaLane` reads back whatever it finds at those paths and
        treats it as *this* run's output, so a workdir judged twice is a
        fabrication path: if the second yosys dies, ``stat`` is parsed from the
        first run's file and OpenSTA times the first run's netlist, producing a
        full set of real-looking numbers for a design this run never built.
        The names mirror ``YosysStaLane.synthesize``; a rename there must be
        mirrored here, which is why the recipe name is not free-form either.
        """
        for p in (wd / f"{top}.mapped.v",
                  wd / f"{top}.{JUDGE_SCRIPT.name}.stat.txt"):
            p.unlink(missing_ok=True)

    def judge(self, top: str, *,
              sources: Sequence[str | os.PathLike[str]] | None = None,
              filelist: str | os.PathLike[str] | None = None,
              workdir: str | os.PathLike[str]) -> JudgeResult:
        """Re-score one candidate under the pinned flow.

        Note the signature: ``top`` and the sources, and nowhere to pass a
        script, a front end, a Liberty or a clock. That absence is the contract.
        """
        wd = Path(workdir)
        wd.mkdir(parents=True, exist_ok=True)
        try:
            read = self._read_stanza(top, sources, filelist)
        except ValueError as e:
            return self._invalid(top, str(e))
        self._purge_stale(wd, top)

        # YosysStaLane renders `script.body`; the read stanza is folded into the
        # body (read_cmd="") so the pinned recipe text itself is never edited.
        lane = YosysStaLane(
            yosys=self.yosys, sta=self.sta, liberty=self.liberty, workdir=wd,
            read_cmd="",
            script=SynthScript(name=JUDGE_SCRIPT.name,
                               rationale=JUDGE_SCRIPT.rationale,
                               body=read + "\n" + JUDGE_SCRIPT.body),
            clock_period_ns=self.clock_period_ns, clock_port=self.clock_port,
        )
        qor, netlist = lane.synthesize([], top, timeout_s=self.timeout_s,
                                       label="judge-yosys")
        timing = None
        if netlist is not None:
            timing = lane.timing(netlist, top, timeout_s=self.sta_timeout_s,
                                 label="judge-opensta")

        # A number printed by a failed STA run is not a measurement. OpenSTA can
        # exit non-zero (or be killed on timeout) having already printed a
        # partial report, and that report parses, so without this gate a delay
        # from a crashed run would reach `judge_max_delay_ns`, which best_so_far
        # plots and proxy_vs_judge correlates. No number instead.
        timing_ok = bool(timing is not None and timing.valid)

        notes: list[str] = []
        if qor.message:
            notes.append(qor.message)
        if netlist is None:
            notes.append("no mapped netlist produced")
        if qor.cells is None or qor.area_um2 is None:
            notes.append("stat -liberty produced no parseable cells/area")
        if timing is not None and not timing_ok:
            notes.append((timing.message or "STA produced no parseable path")
                         + "; delay/slack discarded")
        if timing_ok and self.clock_port is None:
            notes.append("UNCONSTRAINED timing: longest combinational I/O path, "
                         "NOT the critical path")

        # Cells and area are the judge's load-bearing outputs; a design whose
        # timing failed still yields a usable area comparison, so validity is
        # scoped to what was actually measured.
        valid = bool(qor.valid and qor.cells is not None and qor.area_um2 is not None)

        logs: list[str] = []
        for r in lane.tool_runs:
            logs += [p for p in (r.stdout_path, r.stderr_path) if p]

        res = JudgeResult(
            top=top,
            cells=qor.cells,
            area_um2=qor.area_um2,
            max_delay_ns=timing.max_delay_ns if timing_ok else None,
            slack_ns=timing.slack_ns if timing_ok else None,
            valid=valid,
            message="; ".join(notes),
            wall_s=sum(r.wall_s for r in lane.tool_runs),
            cpu_s=sum(r.cpu_s for r in lane.tool_runs),
            config=self.config(),
            judged_at=_utcnow(),
            timing_valid=timing_ok,
            netlist_sha256=_sha256_file(netlist) if netlist is not None else None,
            log_paths=logs,
        )
        if self.verbose:
            print(f"  [judge] {res.summary()}", flush=True)
        return res

    def judge_dir(self, rtl_dir: str | os.PathLike[str], top: str, *,
                  workdir: str | os.PathLike[str] | None = None,
                  filelist_name: str | None = None) -> JudgeResult:
        """Score a variant from its stored RTL tree.

        The loop keeps each variant's tree at ``var/workdirs/<run_id>/<tag>/rtl``
        (:class:`livelane.evaluators.DesignWorkspace`), so this is the entry
        point used for re-scoring after a run. The path is always passed in,
        see :func:`variant_rtl_dir` for why it is never inferred silently.
        """
        d = Path(rtl_dir)
        if not d.is_dir():
            return self._invalid(top, f"RTL directory does not exist: {d}")
        fl: Path | None = None
        if filelist_name:
            fl = d / filelist_name
            if not fl.is_file():
                return self._invalid(
                    top, f"filelist {filelist_name!r} not found under {d}")
        srcs = None if fl else sorted(p for p in d.rglob("*")
                                      if p.suffix in RTL_SUFFIXES)
        wd = Path(workdir) if workdir is not None else d.parent / "judge"
        return self.judge(top, sources=srcs, filelist=fl, workdir=wd)


# --- locating a variant's RTL ------------------------------------------------


def variant_rtl_dir(workroot: str | os.PathLike[str], run_id: str,
                    iteration_index: int) -> Path | None:
    """The loop's conventional RTL path for one iteration, or ``None``.

    ``LiveLaneLoop._workspace`` tags iteration 0 ``seed`` and iteration N
    ``iterN``, both under ``<workroot>/<run_id>/<tag>/rtl``. Returns ``None``
    rather than a plausible path when nothing is there: judging a directory that
    does not hold the variant's RTL would produce a real number for the wrong
    design, which is worse than producing none.
    """
    base = Path(workroot) / run_id
    tags = [f"iter{iteration_index}"]
    if iteration_index == 0:
        tags.append("seed")
    for tag in tags:
        d = base / tag / "rtl"
        if d.is_dir():
            return d
    return None


# --- re-scoring a whole run --------------------------------------------------


@dataclass
class JudgeRecord:
    variant_id: int
    iteration_index: int
    rtl_dir: str | None
    result: JudgeResult | None


@dataclass
class JudgeRunReport:
    run_id: str
    top: str
    config: dict[str, Any]
    judged_at: str
    n_accepted: int
    n_judged: int
    n_valid: int
    n_missing_rtl: int
    n_timing_valid: int = 0
    records: list[JudgeRecord] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Every accepted variant produced a valid neutral score.

        Fail closed: a partially-judged run must not back a cross-arm claim.
        Scoped to what ``valid`` covers, cells and area. Timing can fail on
        its own (see :attr:`JudgeResult.timing_valid`), so a cross-arm *delay*
        claim must check :attr:`timing_complete` instead of this.
        """
        return self.n_accepted > 0 and self.n_valid == self.n_accepted

    @property
    def timing_complete(self) -> bool:
        """Every accepted variant also produced a usable delay."""
        return self.n_accepted > 0 and self.n_timing_valid == self.n_accepted


def judge_run(store: VariantStore, run_id: str, *, judge: NeutralJudge, top: str,
              workroot: str | os.PathLike[str] = Path("var/workdirs"),
              filelist_name: str | None = None,
              rtl_dirs: Mapping[int, str | os.PathLike[str]] | None = None,
              judge_workroot: str | os.PathLike[str] | None = None,
              write: bool = True, verbose: bool = True) -> JudgeRunReport:
    """Re-score every ACCEPTED variant of ``run_id`` and write the results back.

    Only accepted variants are judged: a candidate the functional or equivalence
    gate rejected is not a design, and re-scoring it would put a number next to
    a netlist that is not equivalent to the seed.

    ``rtl_dirs`` overrides the location per variant id, for runs whose workdirs
    were moved or archived. ``top`` is required rather than derived from the
    ``runs`` row, which stores the design name and not the top module.
    """
    if not store.query("SELECT run_id FROM runs WHERE run_id=?", (run_id,)):
        raise KeyError(f"no such run: {run_id}")
    rows = store.query(
        """SELECT id, iteration_index FROM variants
           WHERE run_id=? AND accepted=1 ORDER BY iteration_index""",
        (run_id,))
    overrides = {int(k): Path(v) for k, v in (rtl_dirs or {}).items()}
    jroot = Path(judge_workroot) if judge_workroot is not None \
        else Path("var/judge") / run_id

    records: list[JudgeRecord] = []
    n_valid = n_timing = n_missing = 0
    for row in rows:
        vid, idx = int(row["id"]), int(row["iteration_index"])
        d = overrides.get(vid) or variant_rtl_dir(workroot, run_id, idx)
        if d is None or not Path(d).is_dir():
            n_missing += 1
            records.append(JudgeRecord(vid, idx, None if d is None else str(d), None))
            if verbose:
                print(f"  [judge] variant {vid} (iter {idx}): NO RTL -- not judged",
                      flush=True)
            continue
        if verbose:
            print(f"  [judge] variant {vid} (iter {idx}) <- {d}", flush=True)
        res = judge.judge_dir(d, top, workdir=jroot / f"v{vid}",
                              filelist_name=filelist_name)
        n_valid += int(res.valid)
        n_timing += int(res.valid and res.timing_valid)
        records.append(JudgeRecord(vid, idx, str(d), res))
        if write:
            # All five fields move together: leaving a stale judge_* number next
            # to a fresh judge_json would make the row self-contradictory.
            store.update_variant(vid, **res.columns, judge_json=res.to_record())

    report = JudgeRunReport(
        run_id=run_id, top=top, config=judge.config(), judged_at=_utcnow(),
        n_accepted=len(rows), n_judged=len(rows) - n_missing, n_valid=n_valid,
        n_missing_rtl=n_missing, n_timing_valid=n_timing, records=records)
    if verbose:
        print(f"  [judge] run {run_id}: {report.n_valid}/{report.n_accepted} "
              f"accepted variants scored by {JUDGE_ID}"
              + (f" ({n_missing} missing RTL)" if n_missing else "")
              + ("" if report.timing_complete else
                 f"; only {n_timing}/{report.n_accepted} have a usable delay"),
              flush=True)
    return report


# --- H4: does the in-lane proxy rank the way the judge does? -----------------


def _pairs(store: VariantStore, run_id: str, metric: str) -> list[tuple[float, float]]:
    """(proxy, judge) for every accepted variant scored by both, in run order."""
    if metric not in H4_METRICS:
        raise ValueError(
            f"refusing to correlate unknown metric {metric!r}; "
            f"expected one of {sorted(H4_METRICS)}")
    rows = store.query(
        f"""SELECT qor_{metric} AS proxy, judge_{metric} AS neutral
            FROM variants
            WHERE run_id=? AND accepted=1
              AND qor_{metric} IS NOT NULL AND judge_{metric} IS NOT NULL
            ORDER BY iteration_index""",
        (run_id,))
    return [(r["proxy"], r["neutral"]) for r in rows]


def judged_pairs(store: VariantStore, run_id: str,
                 metric: str = "max_delay_ns") -> int:
    """How many candidates the H4 rho for ``metric`` would rest on.

    Reported next to every rho: the criterion asks for >= 30 candidates
    (:data:`H4_MIN_N`), and a rho quoted without its n cannot be checked against
    that half of the criterion.
    """
    return len(_pairs(store, run_id, metric))


def _same_timing_basis(store, run_id: str) -> bool | None:
    """Were the proxy and the judge measuring the same physical quantity?

    A clock-constrained arm reports the register-to-register critical path; an
    unconstrained judge reports the longest combinational I/O path. On picorv32
    that is 12.7612 against 0.1959. Correlating them yields a plausible-looking
    number for two different quantities, observed live as rho swinging from
    -0.64 to +0.92 across arms of one sweep.

    The clock is not the only thing that has to match. The RTL FRONT END changes
    the answer too: on picorv32, same recipe, same Liberty, same clock,

        read_slang        6563 cells   73992.2 um2   12.7612 ns
        read_verilog -sv  6691 cells   75663.8 um2   14.7771 ns

   , 15.8% on the critical path, which is larger than most of the improvements
    the arms report. The recipe matters for the same reason. So the basis is the
    triple (clock constrained?, front end, recipe), and all three are compared.

    Returns None when either side did not record its configuration, so "unknown"
    is never silently reported as "fine". Runs recorded before the evaluator
    config was persisted therefore return None rather than a false pass.
    """
    jr = store.query(
        "SELECT judge_json FROM variants WHERE run_id=? AND judge_json IS NOT NULL"
        " LIMIT 1", (run_id,))
    rr = store.query("SELECT note FROM runs WHERE run_id=?", (run_id,))
    if not jr or not rr or not rr[0]["note"]:
        return None
    try:
        jcfg = json.loads(jr[0]["judge_json"]).get("config") or {}
        acfg = json.loads(rr[0]["note"]) or {}
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None
    if not isinstance(jcfg, dict) or not isinstance(acfg, dict):
        return None
    # Constrained-vs-unconstrained is the categorical difference; the period
    # itself may legitimately differ without changing WHAT is measured.
    if (jcfg.get("clock_port") is None) != (acfg.get("clock_port") is None):
        return False
    # Front end and recipe must match exactly, but only where BOTH sides
    # recorded them. A missing key must not downgrade the verdict to None:
    # None does not block (only False does), so treating "not recorded" as
    # "unknown" would make the guard weaker, not stricter, on exactly the old
    # runs it is meant to protect. Unknown is reserved for a side that recorded
    # no configuration at all, which is handled above.
    for key in ("read_cmd", "script"):
        jv, av = jcfg.get(key), acfg.get(key)
        if jv is not None and av is not None and jv != av:
            return False
    return True


def proxy_vs_judge(store: VariantStore, run_id: str,
                   metric: str = "max_delay_ns") -> float | None:
    """Spearman rho between the in-lane proxy (``qor_*``) and the judge.

    This is hypothesis H4. The threshold is
    rho >= :data:`H4_RHO_THRESHOLD`; it matters most for lane L, whose absolute
    QoR is far off the judge's but whose *ranking* is the only thing an inner
    loop needs to be right about.

    Returns ``None``, never 0.0, when rho is undefined: fewer than three
    judged variants, or no variance on one side (every candidate scoring
    identically is not evidence of agreement). A defined rho is still not an H4
    decision on its own; pass it to :func:`h4_holds` with
    :func:`judged_pairs` as ``n``.
    """
    # A clock-constrained arm reports the register-to-register critical path; an
    # unconstrained judge reports the longest combinational I/O path. On picorv32
    # that is 12.7612 vs 0.1959. Correlating them produces a plausible-looking
    # number for two different physical quantities, so refuse rather than return
    # it, this was observed live, giving rho from -0.64 to +0.92 across arms.
    if _same_timing_basis(store, run_id) is False:
        raise ValueError(
            f"refusing to correlate run {run_id}: the judge and the arm used "
            f"different timing bases (one clock-constrained, one not), so the "
            f"two columns are not the same physical quantity")
    pairs = _pairs(store, run_id, metric)
    if len(pairs) < 3:
        return None
    return spearman([p[0] for p in pairs], [p[1] for p in pairs])


def h4_holds(rho: float | None, threshold: float = H4_RHO_THRESHOLD, *,
             n: int | None = None, min_n: int = H4_MIN_N) -> bool | None:
    """``True``/``False`` for the H4 decision, ``None`` when undecidable.

    ``None`` is not a pass. An undefined rho means the run cannot speak to H4,
    and must not be counted as either supporting or refuting it.

    The sample size is half the criterion, so it is required
    rather than defaulted: ``h4_holds(rho)`` with no ``n`` answers ``None``,
    because a rho whose sample size is unknown decides nothing. Pass the number
    of judged pairs the rho was computed over (:func:`judged_pairs`).
    """
    if rho is None or n is None or n < min_n:
        return None
    return rho >= threshold


__all__ = [
    "JUDGE_ID", "JUDGE_SCRIPT", "JUDGE_SCRIPT_NAME", "JUDGE_READ_CMD",
    "H4_METRICS", "H4_RHO_THRESHOLD", "H4_MIN_N",
    "JudgeResult", "NeutralJudge", "JudgeRecord", "JudgeRunReport",
    "variant_rtl_dir", "judge_run", "proxy_vs_judge", "judged_pairs", "h4_holds",
]


if __name__ == "__main__":
    import dataclasses
    import inspect
    import tempfile

    print("=== neutral judge self-test (real tools, real design) ===")

    root = Path(__file__).resolve().parents[2]
    design = root / "designs" / "picorv32"
    assert design.is_dir(), f"missing design: {design}"

    # 1. The configuration cannot be varied, per candidate or otherwise.
    j = NeutralJudge(verbose=True)
    cfg = j.config()
    assert cfg["script"] == "baseline-2026-09-02", cfg
    assert cfg["read_cmd"] == "read_slang", cfg
    assert cfg["liberty_sha256"] == os.environ.get(
        "LIVELANE_LIBERTY_SHA256", cfg["liberty_sha256"]), \
        "judge hashed a different Liberty than configs/pdk.env pinned"
    params = set(inspect.signature(NeutralJudge.judge).parameters)
    assert not (params & {"script", "read_cmd", "liberty", "clock_port",
                          "clock_period_ns"}), \
        f"judge() exposes a per-candidate flow knob: {params}"
    try:
        j.script = "tuned-hier"        # type: ignore[attr-defined]
        raise AssertionError("NeutralJudge is mutable; the flow could be varied")
    except dataclasses.FrozenInstanceError:
        pass
    cfg["script"] = "tampered"
    cfg = j.config()
    assert cfg["script"] == "baseline-2026-09-02", \
        "a caller mutating config() changed the judge"
    print(f"    pinned: {cfg['script']} / {cfg['read_cmd']} / "
          f"liberty {str(cfg['liberty_sha256'])[:12]} / "
          f"yosys {cfg['yosys_version']}")

    # A binary that merely shares the NAME is not the binary tool_version probes,
    # and claiming PATH's version for it would be a fabricated provenance field.
    with tempfile.TemporaryDirectory() as vtd:
        impostor = Path(vtd) / "yosys"
        impostor.symlink_to(shutil.which("sh") or "/bin/sh")
        vcfg = NeutralJudge(yosys=str(impostor), verbose=False).config()
        assert vcfg["yosys_version"] is None and vcfg["yosys_git_sha"] is None, vcfg
        assert vcfg["opensta_version"] is not None, "PATH's own sta version lost"
    print("    version recorded only for the binary actually configured "
          "(same-name impostor -> None, not PATH's version)")

    try:
        NeutralJudge(liberty="/does/not/exist.lib")
        raise AssertionError("missing Liberty accepted")
    except FileNotFoundError as e:
        print(f"    Liberty presence enforced: {str(e)[:60]}...")
    try:
        NeutralJudge(clock_port="clk")
        raise AssertionError("half-specified clock accepted")
    except ValueError as e:
        print(f"    half-specified clock refused: {str(e)[:60]}...")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # 2. The real thing: picorv32 through the real Yosys/ABC/OpenSTA.
        r1 = j.judge("picorv32", filelist=design / "filelist.f",
                     workdir=tmp / "fl")
        assert r1.valid, r1.message
        assert r1.cells is not None and 1000 < r1.cells < 100_000, r1.cells
        assert r1.area_um2 is not None and r1.area_um2 > 0.0, r1.area_um2
        assert r1.max_delay_ns is not None and r1.timing_valid, r1.message
        assert r1.wall_s and r1.wall_s > 0 and r1.cpu_s and r1.cpu_s > 0
        assert r1.netlist_sha256 and r1.judged_at.endswith("+00:00")
        assert r1.config["script"] == "baseline-2026-09-02"
        assert "UNCONSTRAINED" in r1.message, \
            "an unconstrained delay must be labelled, not quoted as a critical path"
        print(f"    picorv32 (filelist): cells={r1.cells} area={r1.area_um2} um2 "
              f"delay={r1.max_delay_ns} ns slack={r1.slack_ns} ns "
              f"({r1.wall_s:.2f}s wall, {r1.cpu_s:.2f}s cpu)")

        # 3. Same design reached the other way (a stored RTL directory) must
        #    give the SAME numbers, the judge is the flow, not the entry point.
        r2 = j.judge_dir(design, "picorv32", workdir=tmp / "dir")
        assert (r2.cells, r2.area_um2) == (r1.cells, r1.area_um2), (r1, r2)
        assert r2.netlist_sha256 == r1.netlist_sha256, "judge is not deterministic"
        print(f"    from RTL dir: cells={r2.cells} area={r2.area_um2} um2, "
              f"netlist sha {r2.netlist_sha256[:12]} identical -> deterministic")

        # 4. A pinned clock reaches register paths the unconstrained run cannot.
        jc = NeutralJudge(clock_port="clk", clock_period_ns=10.0, verbose=False)
        r3 = jc.judge("picorv32", filelist=design / "filelist.f",
                      workdir=tmp / "clk")
        assert r3.valid and r3.cells == r1.cells, (r3.cells, r1.cells)
        assert r3.max_delay_ns is not None and r3.max_delay_ns > r1.max_delay_ns, \
            (r3.max_delay_ns, r1.max_delay_ns)
        assert r3.config["timing_constrained"] is True
        assert "UNCONSTRAINED" not in r3.message
        print(f"    constrained (clk @ 10ns): delay={r3.max_delay_ns} ns "
              f"slack={r3.slack_ns} ns vs unconstrained {r1.max_delay_ns} ns "
              f"(comb I/O path only)")

        # 5. Fail closed on bad input: no number, not a wrong one.
        bad = j.judge("picorv32", sources=[], workdir=tmp / "empty")
        assert not bad.valid and bad.cells is None and bad.area_um2 is None
        assert "empty design" in bad.message
        missing = j.judge_dir(tmp / "nope", "picorv32", workdir=tmp / "nope-wd")
        assert not missing.valid and missing.cells is None
        print(f"    empty input -> valid={bad.valid} cells={bad.cells}; "
              f"missing dir -> {missing.message[:44]}...")

        # 5b. A tool that FAILED must leave no number behind. OpenSTA can print
        #     a parseable partial report and still exit non-zero.
        fake_sta = tmp / "fake-sta.sh"
        fake_sta.write_text("#!/bin/sh\n"
                            "echo '   9.4210   data arrival time'\n"
                            "echo '   0.5790   slack (MET)'\n"
                            "exit 1\n")
        os.chmod(fake_sta, 0o755)
        crashed = NeutralJudge(sta=str(fake_sta), verbose=False).judge(
            "picorv32", filelist=design / "filelist.f", workdir=tmp / "sta-rc1")
        assert crashed.max_delay_ns is None and crashed.slack_ns is None, crashed
        assert not crashed.timing_valid, crashed
        assert crashed.columns["judge_max_delay_ns"] is None, crashed.columns
        assert crashed.cells == r1.cells, "a failed STA must not void the area score"
        print(f"    STA rc=1 with a parseable partial report -> delay="
              f"{crashed.max_delay_ns}, area kept ({crashed.message[:34]}...)")

        # 5c. A workdir judged twice must never be scored from the FIRST
        #     judging's netlist and stat file: tmp/'fl' still holds r1's.
        fake_yosys = tmp / "fake-yosys.sh"
        fake_yosys.write_text("#!/bin/sh\nexit 1\n")
        os.chmod(fake_yosys, 0o755)
        stale = NeutralJudge(yosys=str(fake_yosys), verbose=False).judge(
            "picorv32", filelist=design / "filelist.f", workdir=tmp / "fl")
        assert not stale.valid and stale.cells is None and stale.area_um2 is None, stale
        assert stale.max_delay_ns is None and stale.netlist_sha256 is None, stale
        assert set(stale.columns.values()) == {None}, stale.columns
        print(f"    yosys rc=1 over a workdir that already held a GOOD netlist -> "
              f"cells={stale.cells} sha={stale.netlist_sha256} (leftovers not scored)")

        # 6. Re-scoring a run: write-back, and only for accepted variants.
        store = VariantStore(tmp / "livelane.db", verbose=False)
        run = store.start_run("judge-selftest", design="picorv32", lane="L",
                              arm="L", model="scripted", seed=0)
        v_seed = store.add_variant(run, iteration_index=0, wall_offset_s=0.0,
                                   accepted=1, functional_pass=1,
                                   qor_cells=11864, qor_area_um2=88760.5,
                                   qor_max_delay_ns=130.59)
        v_bad = store.add_variant(run, iteration_index=1, parent_id=v_seed,
                                  wall_offset_s=10.0, accepted=0,
                                  lec_verdict="refuted", qor_max_delay_ns=1.0)
        rep = judge_run(store, "judge-selftest", judge=j, top="picorv32",
                        rtl_dirs={v_seed: design}, judge_workroot=tmp / "jr",
                        verbose=False)
        assert rep.n_accepted == 1 and rep.n_judged == 1 and rep.n_valid == 1
        assert rep.n_missing_rtl == 0 and rep.complete
        assert rep.n_timing_valid == 1 and rep.timing_complete
        got = store.query("SELECT judge_cells, judge_area_um2, judge_json "
                          "FROM variants WHERE id=?", (v_seed,))[0]
        assert got["judge_cells"] == r1.cells, dict(got)
        assert abs(got["judge_area_um2"] - r1.area_um2) < 1e-9
        assert JUDGE_ID in got["judge_json"] and "baseline-2026-09-02" in got["judge_json"]
        refuted = store.query("SELECT judge_cells FROM variants WHERE id=?",
                              (v_bad,))[0]
        assert refuted["judge_cells"] is None, "a LEC-refuted variant was judged"
        print(f"    judge_run wrote judge_cells={got['judge_cells']} "
              f"judge_area_um2={got['judge_area_um2']} for the accepted variant; "
              f"the refuted one stayed NULL")

        # A variant whose RTL is gone is reported missing, never invented.
        v_gone = store.add_variant(run, iteration_index=2, parent_id=v_seed,
                                   wall_offset_s=20.0, accepted=1,
                                   qor_max_delay_ns=9.0)
        rep2 = judge_run(store, "judge-selftest", judge=j, top="picorv32",
                         workroot=tmp / "no-workdirs", rtl_dirs={v_seed: design},
                         judge_workroot=tmp / "jr", verbose=False)
        assert rep2.n_missing_rtl == 1 and not rep2.complete
        assert store.query("SELECT judge_cells FROM variants WHERE id=?",
                           (v_gone,))[0]["judge_cells"] is None
        print(f"    missing RTL: {rep2.n_missing_rtl} unjudged, "
              f"complete={rep2.complete} (a partial run cannot back a claim)")

        try:
            judge_run(store, "no-such-run", judge=j, top="picorv32")
            raise AssertionError("unknown run accepted")
        except KeyError as e:
            print(f"    unknown run refused: {e}")

        # 7. H4. Synthetic judge_* values here, this tests the correlation,
        #    not the tools; the measured values are in step 2.
        for vid, (proxy, neutral) in zip(
                (v_seed, v_gone), ((130.59, 7.01), (9.0, 6.4))):
            store.update_variant(vid, qor_max_delay_ns=proxy,
                                 judge_max_delay_ns=neutral)
        assert proxy_vs_judge(store, "judge-selftest") is None, \
            "rho reported from fewer than three judged variants"
        v3 = store.add_variant(run, iteration_index=3, parent_id=v_seed,
                               wall_offset_s=30.0, accepted=1,
                               qor_max_delay_ns=200.0, judge_max_delay_ns=9.9)
        rho = proxy_vs_judge(store, "judge-selftest")
        assert rho is not None and abs(rho - 1.0) < 1e-9, rho
        n = judged_pairs(store, "judge-selftest")
        assert n == 3, n
        # The criterion is rho >= 0.8 over >= 30 candidates. Both halves
        # or no decision: a perfect rho over 3 pairs must not read as a pass.
        assert h4_holds(rho, n=n) is None, "H4 declared on 3 candidates"
        assert h4_holds(rho) is None, "H4 declared without a sample size"
        assert h4_holds(rho, n=H4_MIN_N) is True
        assert h4_holds(None, n=H4_MIN_N) is None
        store.update_variant(v3, judge_max_delay_ns=1.0)   # rank flipped
        rho_bad = proxy_vs_judge(store, "judge-selftest")
        assert rho_bad is not None and rho_bad < 0, rho_bad
        assert h4_holds(rho_bad, n=H4_MIN_N) is False
        print(f"    H4: concordant rho={rho:+.3f} holds(n=30)="
              f"{h4_holds(rho, n=H4_MIN_N)}; flipped rho={rho_bad:+.3f} "
              f"holds(n=30)={h4_holds(rho_bad, n=H4_MIN_N)}; "
              f"n={n}<{H4_MIN_N} -> {h4_holds(rho, n=n)}; pairs<3 -> None (never 0.0)")

        try:
            proxy_vs_judge(store, "judge-selftest", metric="cells; DROP TABLE variants")
            raise AssertionError("unvalidated metric name accepted")
        except ValueError as e:
            print(f"    metric allow-list holds: {str(e)[:66]}...")

        assert variant_rtl_dir(tmp / "no-workdirs", "judge-selftest", 0) is None
        (tmp / "wr" / "judge-selftest" / "seed" / "rtl").mkdir(parents=True)
        assert variant_rtl_dir(tmp / "wr", "judge-selftest", 0) == \
            tmp / "wr" / "judge-selftest" / "seed" / "rtl"
        print("    variant_rtl_dir: finds iter0's 'seed' tag, None when absent")
        store.close()

    print("=== all neutral judge self-tests passed ===")
