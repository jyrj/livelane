"""Equivalence-check gate for RTL-editing loops, backed by YosysHQ ``eqy``.

Why this exists
---------------
CHIA has no equivalence-checking node.  Searched the whole tree: ``eqy``,
``sby``, ``symbiyosys``, ``smtbmc`` and ``equivalence`` all return zero hits.
So every RTL-editing loop in the framework -- including its own ``timing_opt``
example -- accepts an agent's edit on *simulation* evidence alone.

That is exactly the failure mode the published critiques of LLM-driven RTL
optimisation describe (arXiv:2601.01765, arXiv:2507.16808): reported gains that
vanish once the edit is checked for equivalence.  We measured a concrete
instance.  Inverting the ``BGE`` comparison in picorv32
(``alu_out_0 = !alu_lts`` -> ``alu_out_0 = alu_lts``) synthesises cleanly and
scores **better** than its parent -- 75,074.50 um2 against 75,663.82 um2, with
235 fewer cells.  A loop scoring on QoR alone takes that as an improvement and
builds on it.  This gate refutes it, and names the failing partition.

Design rules, which are the whole point of the node
---------------------------------------------------
* **Fail closed.**  Anything that is not a positive proof of equivalence -- an
  error, a timeout, an unparsable log, a crashed solver -- is NOT a pass.  The
  only verdict that admits an edit is :data:`PROVEN`.
* **Refuted and undecided are different answers.**  ``eqy`` emits the SAME
  summary line and the SAME exit code (2) whether it found a counterexample or
  simply could not decide a partition; only the per-partition reason separates
  them.  Conflating them is not cosmetic: it rejects valid edits and inflates
  the measured gate-rejection rate.  This module reports :data:`REFUTED` and
  :data:`UNDECIDED` separately, and both fail closed.
* **Bounded proofs are labelled.**  A bounded-depth check is evidence, not
  proof, and :attr:`LecResult.evidence_strength` says which one you got rather
  than folding it silently into the verdict.
* **Two backends, and disagreement is a result.**  ``eqy`` is the primary; a
  second opinion (``circt-lec``, or another checker) can cross-check it with
  :func:`cross_check`.  When two independent checkers disagree that is
  reported, never averaged.

Container
---------
The node shells out to ``eqy``, which in turn drives ``yosys``, ``sby``,
``yosys-smtbmc`` and an SMT solver.  It therefore needs a worker image carrying
that stack; see ``dockerfiles/EqyDockerfile`` and bind it with a cluster node
type whose ``resources: {"eqy": 1}`` matches :func:`lec_gate`'s token.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

try:  # CHIA is optional: every verdict rule below is testable without a cluster.
    from chia.base.ChiaFunction import ChiaFunction
except Exception:  # pragma: no cover - exercised only outside a CHIA install
    def ChiaFunction(**_kwargs):  # type: ignore[misc]
        def deco(fn):
            return fn
        return deco

#: The only verdict that admits an edit: the designs were proved equivalent.
PROVEN = "proven"
#: A counterexample exists. The designs are NOT equivalent. This is a refutation.
REFUTED = "refuted"
#: The checker ran correctly but could not decide. NOT a refutation, NOT a pass.
UNDECIDED = "undecided"
#: The checker failed: crash, unparsable log, missing binary, bad config.
ERROR = "error"
#: The checker exceeded its wall-clock budget and was killed.
TIMEOUT = "timeout"
#: No check was attempted (e.g. the seed iteration has no parent).
SKIPPED = "skipped"

VALID_VERDICTS = frozenset({PROVEN, REFUTED, UNDECIDED, ERROR, TIMEOUT, SKIPPED})


@dataclass
class LecResult:
    """One equivalence check and everything needed to audit it.

    Attributes:
        verdict (str): One of :data:`VALID_VERDICTS`. Only :data:`PROVEN`
            admits an edit.
        backend (str): Which checker produced this, e.g. ``"eqy"``.
        wall_s (float): Wall-clock the check cost, charged to the caller's arm.
        bounded (bool): The proof carried a depth bound.
        depth (int | None): That bound, when there was one.
        partitions_total (int | None): Partitions ``eqy`` created.
        partitions_failed (int | None): Partitions it could not prove.
        message (str): Short human-readable summary, truncated.
        counterexample_path (str | None): Directory holding the trace, when the
            verdict is :data:`REFUTED` and the trace was found.
        returncode (int | None): The checker's exit code. ``None`` on timeout.
        log_path (str | None): Full captured stdout+stderr on disk.
        outputs_total (int | None): Top-level output bits the design has.
            Sequential backends report per-OUTPUT, not per-partition.
        outputs_checked (int | None): Output bits that reached the proof
            surface. A backend that proves 7 of 307 output bits has proved
            almost nothing, and the gate must be able to say so.
        coverage_pct (float | None): ``outputs_checked`` as a percentage, as
            the backend itself computed it.
        abstraction (str): Non-empty when the proof holds only under a stated
            abstraction. Carried into :attr:`evidence_strength` so it can never
            be silently dropped when the verdict is reported.
    """

    verdict: str
    backend: str
    wall_s: float
    bounded: bool = False
    depth: int | None = None
    partitions_total: int | None = None
    partitions_failed: int | None = None
    message: str = ""
    counterexample_path: str | None = None
    returncode: int | None = None
    log_path: str | None = None
    outputs_total: int | None = None
    outputs_checked: int | None = None
    coverage_pct: float | None = None
    abstraction: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in VALID_VERDICTS:
            raise ValueError(
                f"invalid verdict {self.verdict!r}; expected one of "
                f"{sorted(VALID_VERDICTS)}")

    @property
    def admits_edit(self) -> bool:
        """True only on a positive proof of equivalence.

        Fail-closed: :data:`REFUTED`, :data:`UNDECIDED`, :data:`ERROR`,
        :data:`TIMEOUT` and :data:`SKIPPED` all return False, so a crashed or
        unparsable check can never admit an edit.

        A *bounded* proof does admit. ``eqy``'s partitioned methodology runs a
        bounded check per partition, so requiring ``not bounded`` here would
        reject every real ``eqy`` pass and make the gate useless. The strength
        of the evidence is carried by :attr:`is_unbounded_proof` and
        :attr:`evidence_strength` and is reported alongside the verdict rather
        than silently folded into it.
        """
        return self.verdict == PROVEN

    @property
    def is_refutation(self) -> bool:
        """True only when a counterexample was actually found.

        Deliberately not ``not admits_edit``: an undecided or errored check is
        not evidence that the designs differ, and counting it as one inflates
        any rejection rate computed from these results.
        """
        return self.verdict == REFUTED

    @property
    def is_unbounded_proof(self) -> bool:
        """The stronger claim: a proof carrying no depth bound at all."""
        return self.verdict == PROVEN and not self.bounded

    @property
    def evidence_strength(self) -> str:
        """Verdict, qualified by proof depth and by any stated abstraction.

        kepler's SEC proof holds only on cycles where both outputs are
        binary-defined (its dual-rail steady-state encoding). That caveat is
        part of the claim, so it travels with the claim rather than being
        dropped at the reporting boundary.
        """
        if self.verdict != PROVEN:
            return self.verdict
        base = "proof" if not self.bounded else f"bounded-proof(depth={self.depth})"
        return f"{base}[{self.abstraction}]" if self.abstraction else base


# --- eqy log parsing ---------------------------------------------------------
# Written against eqy's real output and validated against captured logs, because
# a misparsed log here silently admits bad edits.

_OK_RE = re.compile(r"Successfully proved designs equivalent", re.I)
_FAILED_N_RE = re.compile(r"Failed to prove equivalence for (\d+)/(\d+) partitions", re.I)
_FAILED_ONE_RE = re.compile(r"Failed to prove equivalence of partition (\S+)", re.I)
_DONE_RE = re.compile(r"DONE \((\w+),\s*rc=(-?\d+)\)", re.I)

# THE distinction that decides refuted-vs-undecided. eqy emits the SAME summary
# line ("Failed to prove equivalence for N/M partitions") and the SAME exit code
# (2) whether it found a counterexample or simply could not decide, and only the
# per-partition reason separates them:
#
#   "... using strategy 'sby': partitions not equivalent"  -> a real counterexample
#   "... using strategy 'sby': equivalence unknown"        -> undecided
#
# Conflating these is not cosmetic. It rejects valid edits (measured: a
# comment-only change to picorv32 is "unknown" on its two 64-bit free-running
# counters) and it inflates any gate-rejection rate computed downstream.
_NOT_EQUIV_RE = re.compile(r"Could not prove equivalence of partition '([^']+)'"
                           r"[^\n]*?:\s*partitions not equivalent", re.I)
_UNKNOWN_RE = re.compile(r"Could not prove equivalence of partition '([^']+)'"
                         r"[^\n]*?:\s*equivalence unknown", re.I)


def parse_eqy_log(text: str) -> dict[str, object]:
    """Extract the verdict-bearing facts from an ``eqy`` log.

    Args:
        text (str): Combined stdout and stderr of an ``eqy`` run.

    Returns:
        dict: Any of ``proved`` (bool), ``partitions_failed`` (int),
        ``partitions_total`` (int), ``failed_partitions`` (list of str),
        ``not_equivalent_partitions`` (list of str -- real counterexamples),
        ``unknown_partitions`` (list of str -- undecided), ``done_status``
        (str) and ``done_rc`` (int). Keys are absent rather than ``None`` when
        the log did not carry them, so a caller can tell "not stated" from
        "stated as zero".
    """
    out: dict[str, object] = {}
    if _OK_RE.search(text):
        out["proved"] = True
    m = _FAILED_N_RE.search(text)
    if m:
        out["proved"] = False
        out["partitions_failed"] = int(m.group(1))
        out["partitions_total"] = int(m.group(2))
    fails = _FAILED_ONE_RE.findall(text)
    if fails:
        out["failed_partitions"] = fails
    ne = _NOT_EQUIV_RE.findall(text)
    unk = _UNKNOWN_RE.findall(text)
    if ne:
        out["not_equivalent_partitions"] = ne
    if unk:
        out["unknown_partitions"] = unk
    d = _DONE_RE.search(text)
    if d:
        out["done_status"] = d.group(1).upper()
        out["done_rc"] = int(d.group(2))
    return out


#: Default strategy ladder. ORDER IS A CORRECTNESS PROPERTY, not a tuning knob.
#:
#: ``eqy`` tries strategies in the order given and stops at the FIRST one that
#: decides a partition; later strategies are then never run (their status file
#: literally reads ``PASS (cached)``). So whichever strategy is listed first is
#: the one whose answer the gate reports.
#:
#: ``sat`` (Yosys's built-in ``sat -tempinduct``) MUST NOT be listed first,
#: because it returns PASS on sequential partitions that are not equivalent.
#: Measured, on the fixtures in ``test/smoke/``::
#:
#:     gold: always @(posedge clk) y <= a + b;
#:     gate: always @(posedge clk) y <= 4'b0;      // obviously NOT equivalent
#:
#:     ("induct",) first -> verdict=proven,  admits_edit=True   <-- FALSE ACCEPT
#:     ("smt",)    first -> verdict=refuted, admits_edit=False  (with a trace)
#:
#: The same pair is correctly refuted by ``sat`` when the difference is purely
#: combinational, so the unsoundness is specific to sequential partitions and is
#: data-dependent: the sibling fixture ``refuted.v`` (``a - b``) makes ``sat``
#: answer "unknown" rather than PASS, which is why a ladder led by ``induct``
#: still looked correct on that one fixture. ``eqy``'s ``sat`` strategy decides
#: PASS purely by grepping its log for "Induction step proven: SUCCESS!", and
#: with ``-set-init-undef`` the miter's ``in_gold === 1'bx ||`` escape makes
#: that assertion vacuous for a gold partition whose flops never leave X.
#:
#: ``sby`` in ``mode prove`` (k-induction driven by a real SMT solver) decided
#: every fixture here correctly, so it leads. ``induct`` is kept as a fallback
#: for partitions ``sby`` leaves undecided -- it still contributes refutations
#: and it skips memory partitions cheaply -- but it can no longer be the
#: strategy that admits an edit unless ``sby`` failed to decide first.
#:
#: RESIDUAL RISK, stated rather than hidden: a partition that ``sby`` leaves
#: undecided can still be PASSed by ``induct``. Dropping ``induct`` entirely
#: closes that hole completely, at the cost of the speed it buys and of
#: re-measuring every published gate timing. That is a project-level call.
#:
#: The SAME ladder must run in every arm of a comparison, so whatever it cannot
#: decide is a constant across arms and cannot bias the result.
#: Values ``undef_init`` accepts, matching yosys ``setundef``'s own flags.
UNDEF_INIT_VALUES: frozenset[str] = frozenset({"zero", "one"})


def auto_jobs(default: int = 1) -> int:
    """How many partition proofs this process may run concurrently.

    Derived from the runtime's own allocation, never hardcoded: inside a Ray
    worker the CPU share CHIA scheduled for this task is authoritative, because
    taking more would oversubscribe a node another task is sharing. Outside
    Ray, fall back to the machine's CPU count, and to ``default`` if even that
    is unavailable.

    Args:
        default (int): Returned when no CPU count can be determined.

    Returns:
        int: A job count >= 1, suitable for :attr:`LecGateNode.jobs`.
    """
    try:  # pragma: no cover - exercised only inside a live Ray worker
        import ray  # type: ignore
        if ray.is_initialized():
            n = ray.get_runtime_context().get_assigned_resources().get("CPU")
            if n:
                return max(1, int(n))
    except Exception:
        pass
    return max(1, os.cpu_count() or default)


DEFAULT_STRATEGIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("smt", ("use sby", "engine smtbmc yices", "depth 10")),
    ("induct", ("use sat", "depth 10")),
)


@dataclass
class LecGateNode:
    """Prove a candidate RTL edit equivalent to its parent, using ``eqy``.

    ``gold`` is the parent (known-good); ``gate`` is the candidate.

    Attributes:
        eqy (str): ``eqy`` executable, by name or absolute path.
        workdir (Path): Where the generated ``.eqy`` config, the log and
            ``eqy``'s own output tree are written.
        depth (int): Bounded depth for the single-strategy fallback. A bounded
            pass is labelled, never silently promoted to a proof.
        engine (str): SMT engine line for the single-strategy fallback.
        strategies (Sequence): Strategy ladder as ``(name, config_lines)``
            pairs; ``eqy`` tries each in order and falls through to the next
            when one cannot decide a partition. Empty means "use the single
            strategy built from ``engine``/``depth``". :data:`DEFAULT_STRATEGIES`
            is the validated ladder.
        read_cmd (str): Yosys front-end command for both sides.
        gold_read_cmd (str | None): Override the read command for the gold side.
        gate_read_cmd (str | None): Override the read command for the gate side.
        extra_gold (Sequence[str]): Extra Yosys lines appended to the gold block.
        extra_gate (Sequence[str]): Extra Yosys lines appended to the gate block.
        collect (Sequence[str]): Extra lines for the ``[collect *]`` section,
            which controls partitioning.
        undef_init (str | None): Resolve undefined flip-flop initial values to
            ``"zero"`` or ``"one"`` on BOTH sides, via ``setundef -<v> -init``.
            ``None`` leaves them undefined. See :data:`UNDEF_INIT_VALUES` and
            the note below; a proof obtained this way is labelled in
            :attr:`LecResult.abstraction`, never reported as unqualified.
        jobs (int | None): Passed to ``eqy`` as ``-j <N>``, which forwards it to
            ``make``. ``eqy`` emits one make target chain per partition, so this
            is the difference between proving partitions one at a time and
            proving them concurrently. ``None`` does not pass the flag. Use
            :func:`auto_jobs` to derive it from the runtime's own CPU
            allocation rather than hardcoding a number.
        timeout_s (float): Wall-clock budget. On expiry the whole process group
            is killed and the verdict is :data:`TIMEOUT`.
        verbose (bool): Print the command before running it.

    Note:
        **Why ``undef_init`` defaults to resolving.** ``eqy`` implements
        *safe-replacement* equivalence: the gold side is read with 3-valued
        (x-propagating) semantics, the gate side with 2-valued semantics where
        each x becomes an arbitrary unconstrained value (eqy ``docs/xprop.rst``).
        For a design whose flip-flops have no initial value, the two sides'
        registers are therefore independent unconstrained variables and the
        solver is free to start them at different values -- so a design is not
        provably equivalent even **to itself**. Measured on picorv32, whose
        ``count_cycle``/``count_instr`` are free-running 64-bit registers with
        no init (``picorv32.v:1433``): checked against an identical copy, the
        gate returned REFUTED on those two partitions, naming a counterexample
        in which gold's bit 63 was 1 and gate's was 0. That is a *false*
        refutation -- the worst failure mode this node has, because it rejects
        valid edits while looking like a careful gate.

        ``setundef -<v> -init``, applied identically to both sides, closes it:
        the same design proves 590/590 partitions. The cost is a real and
        stated narrowing of the claim -- equivalence is proved *from the
        resolved initial state*, so an edit that differs only in behaviour
        reachable from some other undefined start is not covered. That caveat
        travels with the verdict in :attr:`LecResult.abstraction` and surfaces
        in :attr:`LecResult.evidence_strength` as ``proof[undef-init=zero]``.
        Set ``undef_init=None`` to opt out and take the unqualified claim,
        accepting that designs with uninitialised state may not prove at all.
    """

    eqy: str = "eqy"
    workdir: Path = Path(".")
    depth: int = 0
    engine: str = "smtbmc yices"
    strategies: Sequence[tuple[str, Sequence[str]]] = field(default_factory=tuple)
    read_cmd: str = "read_verilog -sv"
    gold_read_cmd: str | None = None
    gate_read_cmd: str | None = None
    extra_gold: Sequence[str] = field(default_factory=tuple)
    extra_gate: Sequence[str] = field(default_factory=tuple)
    collect: Sequence[str] = field(default_factory=tuple)
    undef_init: str | None = "zero"
    jobs: int | None = None
    timeout_s: float = 1800.0
    verbose: bool = True

    def __post_init__(self) -> None:
        # ABSOLUTE, always. eqy is launched with cwd=workdir, so a RELATIVE
        # workdir makes the generated config path relative to itself and eqy
        # dies with "can't open 'wd/lec.eqy'". Its exit code is 2 -- the same
        # code it uses for a failed proof -- and its log is unparsable, so the
        # gate returns ERROR: a misconfiguration wearing the costume of a
        # cautious gate. The same trap already cost this project a 178-minute
        # sweep in the synthesis node, so it is closed here by construction
        # rather than documented.
        self.workdir = Path(self.workdir).resolve()
        # Fail on construction, not inside a solver an hour later. A typo here
        # would otherwise render an unrecognised yosys command, which eqy
        # reports as rc=2 with an unparsable log -- i.e. as ERROR, which fails
        # closed and is indistinguishable from a cautious gate.
        if self.undef_init is not None and self.undef_init not in UNDEF_INIT_VALUES:
            raise ValueError(
                f"undef_init must be None or one of {sorted(UNDEF_INIT_VALUES)}; "
                f"got {self.undef_init!r}")
        if self.jobs is not None and self.jobs < 1:
            raise ValueError(f"jobs must be >= 1 or None; got {self.jobs!r}")

    def write_config(self, gold_srcs: Sequence[str], gate_srcs: Sequence[str],
                     top: str, path: Path) -> Path:
        """Render the ``.eqy`` configuration for one check.

        Args:
            gold_srcs (Sequence[str]): Source files of the known-good design.
            gate_srcs (Sequence[str]): Source files of the candidate design.
            top (str): Top module name, identical on both sides.
            path (Path): Where to write the config.

        Returns:
            Path: ``path``, now written.
        """
        def block(name: str, srcs: Sequence[str], extra: Sequence[str]) -> str:
            read = (self.gold_read_cmd if name == "gold" else self.gate_read_cmd) \
                   or self.read_cmd
            lines = [f"[{name}]"]
            # ALL sources go into ONE read command. `read_slang` is a front-end
            # driver: each invocation elaborates independently, so emitting one
            # line per file makes every module invisible to the others and the
            # whole read fails with "Compilation failed". That silently turned
            # every multi-file equivalence check into an ERROR verdict -- which
            # fails closed, so it looked like a cautious gate rather than a
            # broken one. `read_verilog -sv a.v b.v` is equally happy with one
            # line, so this is correct for both front ends.
            # ABSOLUTE, always. eqy is launched with cwd=workdir, so a
            # relative source path is resolved against the workdir rather than
            # against the caller's cwd and yosys simply does not find the file.
            # That failure surfaces as rc=2 with an unrecognisable log, i.e. as
            # an ERROR verdict -- which fails closed and therefore looks like a
            # cautious gate rather than a misconfigured one. Resolve here so the
            # caller may pass either form.
            paths = " ".join(Path(x).resolve().as_posix() for x in srcs)
            if paths:
                # `read_slang` needs an explicit --top, or it elaborates whatever
                # it infers and `prep -top <top>` then fails with
                # "Module `<top>' not found!". `read_verilog` has no such flag
                # and is elaborated by the following `prep -top`.
                if "read_slang" in read and "--top" not in read:
                    lines.append(f"{read} --top {top} {paths}")
                else:
                    lines.append(f"{read} {paths}")
            lines.append(f"prep -top {top}")
            # AFTER prep (the flip-flops must exist to have init values) and
            # BEFORE memory_map, which is the order validated against picorv32.
            # Emitted into BOTH blocks from one field, so the two sides can
            # never be resolved differently -- an asymmetric resolution would
            # manufacture exactly the false refutation this defends against.
            if self.undef_init is not None:
                lines.append(f"setundef -{self.undef_init} -init")
            lines.append("memory_map")
            lines.extend(extra)
            return "\n".join(lines)

        cfg = [block("gold", gold_srcs, self.extra_gold), "",
               block("gate", gate_srcs, self.extra_gate), ""]
        cfg.append("[collect *]")
        cfg.extend(self.collect)
        cfg.append("")
        if self.strategies:
            for name, lines in self.strategies:
                cfg.append(f"[strategy {name}]")
                cfg.extend(lines)
                cfg.append("")
        else:
            cfg.append("[strategy sby]")
            cfg.append("use sby")
            if self.depth > 0:
                cfg.append(f"depth {self.depth}")
            cfg.append(f"engine {self.engine}")
            cfg.append("")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(cfg))
        return path

    def _argv(self, exe: str, cfg: Path) -> list[str]:
        """The exact ``eqy`` command line, so tests can assert on it.

        ``-j`` is forwarded by eqy to ``make`` (``eqy.py`` builds
        ``make{kopt}{jopt} -C <workdir> -f strategies.mk``), and eqy's
        ``strategies.mk`` carries one target chain per partition with a single
        ``all:`` depending on all of them. Without ``-j`` those chains run one
        at a time regardless of how many cores the worker holds.
        """
        argv = [exe]
        if self.jobs is not None:
            argv += ["-j", str(self.jobs)]
        return argv + ["-f", str(cfg)]

    def _run_eqy(self, exe: str, cfg: Path, wd: Path) -> tuple[int | None, str, bool]:
        """Run ``eqy`` in its own process group, killing the tree on timeout.

        ``subprocess.run(timeout=...)`` only kills the direct child. ``eqy``
        spawns ``yosys``, ``sby`` and an SMT solver; a surviving grandchild
        keeps the stdout pipe open and the parent's ``communicate()`` then
        blocks forever, so the timeout never actually fires. CHIA's own CIRCT
        node takes the same precaution (``chia/chipyard/circt.py``).

        Returns:
            tuple: ``(returncode, combined output, timed_out)``.
        """
        proc = subprocess.Popen(
            self._argv(exe, cfg), cwd=str(wd),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        try:
            out, _ = proc.communicate(timeout=self.timeout_s)
            return proc.returncode, out or "", False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):  # pragma: no cover
                pass
            try:
                out, _ = proc.communicate(timeout=30)
            except Exception:  # pragma: no cover - the pipe is already gone
                out = ""
            return None, out or "", True

    def check(self, gold_srcs: Sequence[str], gate_srcs: Sequence[str],
              top: str, *, name: str = "lec") -> LecResult:
        """Run one equivalence check and return an auditable verdict.

        Args:
            gold_srcs (Sequence[str]): Source files of the known-good design.
            gate_srcs (Sequence[str]): Source files of the candidate design.
            top (str): Top module name, identical on both sides.
            name (str): Basename for this check's config, log and output dir,
                so concurrent checks in one workdir do not collide.

        Returns:
            LecResult: Never raises for a tool failure. A missing binary, a
            crash, a timeout or an unparsable log all come back as a
            non-admitting verdict with the log path attached.
        """
        wd = Path(self.workdir)
        wd.mkdir(parents=True, exist_ok=True)

        # Fail LOUDLY on an operator error rather than closed on a tool error.
        # A missing source makes eqy exit 2 with a log this module cannot parse,
        # i.e. an ERROR verdict that is indistinguishable from a solver crash.
        missing = [str(x) for x in (*gold_srcs, *gate_srcs)
                   if not Path(x).exists()]
        if missing:
            return LecResult(ERROR, "eqy", 0.0,
                             message=f"source file(s) not found: {missing[:4]}")
        if not gold_srcs or not gate_srcs:
            return LecResult(ERROR, "eqy", 0.0,
                             message="both gold_srcs and gate_srcs are required; "
                                     "an empty side proves nothing")

        cfg = self.write_config(gold_srcs, gate_srcs, top, wd / f"{name}.eqy")
        # eqy refuses to overwrite an existing output directory.
        outdir = wd / name
        if outdir.exists():
            shutil.rmtree(outdir, ignore_errors=True)

        exe = shutil.which(self.eqy) or self.eqy
        if self.verbose:
            print(f"[LecGateNode] {' '.join(self._argv(exe, cfg))}", flush=True)

        t0 = time.monotonic()
        try:
            rc, text, timed_out = self._run_eqy(exe, cfg, wd)
        except FileNotFoundError:
            # Fail closed AND loudly: a worker whose image lacks eqy must not
            # look like a cautious gate.
            return LecResult(ERROR, "eqy", time.monotonic() - t0,
                             message=f"eqy not found at {self.eqy!r}; is this "
                                     f"worker running the chia-eqy image?")
        except OSError as e:
            # Same trap as the kepler backend: an eqy path that exists but is
            # not executable raises PermissionError, not FileNotFoundError.
            return LecResult(ERROR, "eqy", time.monotonic() - t0,
                             message=f"eqy at {self.eqy!r} could not be executed "
                                     f"({type(e).__name__}: {e})")
        wall = time.monotonic() - t0

        log = wd / f"{name}.log"
        log.write_text(text)

        if timed_out:
            return LecResult(TIMEOUT, "eqy", wall,
                             message=f"eqy exceeded {self.timeout_s}s",
                             log_path=str(log))

        p = parse_eqy_log(text)
        proved = p.get("proved")
        # Keep the tail of an unrecognised log: without it an ERROR verdict is
        # indistinguishable between "the solver crashed" and "you pointed me at
        # a file that does not exist", and both look like a cautious gate.
        tail = " ".join(text.split())[-200:] if not p else ""
        if proved is True:
            verdict = PROVEN
        elif proved is False:
            # Only a real counterexample is a refutation. A partition eqy could
            # not decide is UNDECIDED -- it still fails closed, but it is not
            # evidence that the designs differ and must not be counted as one.
            if p.get("not_equivalent_partitions"):
                verdict = REFUTED
            elif p.get("unknown_partitions"):
                verdict = UNDECIDED
            else:
                verdict = ERROR
        else:
            # Fail closed: an unrecognised log is NOT a pass, whatever rc says.
            verdict = ERROR

        ce = None
        fails = p.get("failed_partitions") or []
        if verdict == REFUTED and fails:
            cand = outdir / "strategies" / str(fails[0]) / "sby"
            if cand.exists():
                ce = str(cand)

        return LecResult(
            verdict=verdict, backend="eqy", wall_s=wall,
            bounded=self.depth > 0,
            depth=self.depth or None,
            partitions_total=p.get("partitions_total"),      # type: ignore[arg-type]
            partitions_failed=p.get("partitions_failed"),    # type: ignore[arg-type]
            message=(
                f"{p.get('done_status') or ''}"
                + (f" not_equivalent={p['not_equivalent_partitions'][:3]}"
                   if p.get("not_equivalent_partitions") else "")
                + (f" undecided={p['unknown_partitions'][:3]}"
                   if p.get("unknown_partitions") else "")
                + (f" unparsable log, tail: {tail}" if tail else "")
            )[:400],
            counterexample_path=ce,
            returncode=rc,
            log_path=str(log),
            # The resolved initial state is part of the claim, so it travels
            # with the claim. evidence_strength renders this as
            # "proof[undef-init=zero]" rather than a bare "proof".
            abstraction=(f"undef-init={self.undef_init}"
                         if self.undef_init is not None else ""),
        )


# --- kepler-formal: the second, INDEPENDENT backend --------------------------
#
# Why a second backend exists at all
# ----------------------------------
# ``eqy`` partitions at sequential elements, so it requires the two designs to
# share a sequential boundary. An agent that retimes a pipeline, merges or
# duplicates registers, or moves a stage boundary produces an edit eqy cannot
# admit -- and eqy does not fail gracefully on it. MEASURED, on a pair that is
# equivalent and confirmed so by 2000 cycles of Verilator simulation:
#
#     gold:  always @(posedge clk) r <= d;        assign q = r + 1;
#     gate:  always @(posedge clk) r <= d + 1;    assign q = r;
#
#   eqy             refuted  in 0.17s  ("partitions not equivalent: rt.q")
#   kepler-formal   proven   in 0.05s  (SEC, 100% output coverage, k = 1)
#
# eqy's answer is a FALSE refutation: a gate running eqy alone silently rejects
# that whole class of valid edit. kepler compares sequential behaviour through
# extracted transition systems and does not use internal names as a
# cross-design equivalence assumption, so it can decide it.
#
# What kepler is NOT
# ------------------
# It is not a drop-in replacement, and on the design this project optimises it
# is much weaker than eqy. MEASURED on picorv32 (v3.2.4, 3049 lines):
#
#   * The Verilog front end cannot read it at all: naja-verilog is a structural
#     netlist parser and dies on ```timescale`` (and, with
#     ``--verilog_preprocessing``, on ```assert``).
#   * The SystemVerilog front end reads it only with ``-D FORMAL`` and
#     ``-D PICORV32_REGS=picorv32_regs``; without them the SNL lowering rejects
#     the ``empty_statement`` task calls and the ``cpuregs`` initial block.
#   * Even then SEC covers 7 of 307 output bits (2.28%). picorv32 assigns
#     ``'bx`` for don't-cares in a dozen places, and kepler drops every output
#     whose cone reaches an unsupported X constant.
#   * Consequently the comment-only edit and the real BGE bug
#     (``alu_out_0 = !alu_lts`` -> ``alu_out_0 = alu_lts``) produce the
#     IDENTICAL result: "SEC partially proved equivalence at k = 1:
#     7/307 outputs proved", exit code 1, ~2.1s. kepler does not refute the bug
#     on this design. eqy does.
#
# This is not a kepler-specific weakness of our setup: kepler's OWN flagship SEC
# example (``examples/tinyrocket``, tinyrocket.v against ITSELF) also comes back
# "partially proved: 8/132 outputs" after 831s.
#
# Hence the rule this class enforces: kepler is a CROSS-CHECK, never the primary
# gate, and a partial proof is :data:`UNDECIDED` -- not a pass, and not a
# refutation either.

#: Documented SEC exit codes (docs/sec-flags-spec.md, "Bounds And Results").
#: They are NOT sufficient on their own and are used only to cross-check the
#: parsed log: rc=1 means BOTH "partially proved" and EVERY netlist-loading
#: failure (measured: missing file, bad top, unparsable source all exit 1), and
#: in LEC mode kepler exits 0 whether the designs match or not (measured on
#: examples/tinyrocket: "Difference was found." with rc=0).
KEPLER_RC_PROVED = 0
KEPLER_RC_PARTIAL = 1
KEPLER_RC_INCONCLUSIVE = 2
KEPLER_RC_COUNTEREXAMPLE = 3

# Verbatim sentences from real kepler-formal runs. These are matched
# CASE-SENSITIVELY on purpose: "No binary-defined difference was found."
# (a proof) contains the substring "difference was found", and a case-insensitive
# search for the refutation line "Difference was found." would match inside it
# and turn every SEC proof into a refutation. Capital-D "Difference was found."
# appears ONLY on a real difference, in both LEC and SEC mode.
_KEP_LEC_SAME_RE = re.compile(r"No difference was found\.")
_KEP_DIFF_RE = re.compile(r"Difference was found\.")
_KEP_SEC_PROVED_RE = re.compile(
    r"SEC proved equivalence under the ([a-z0-9_ -]+) abstraction at k = (\d+)")
_KEP_SEC_CEX_RE = re.compile(r"SEC found a counterexample at k = (\d+)")
_KEP_SEC_PARTIAL_RE = re.compile(
    r"SEC partially proved equivalence at k = (\d+): (\d+)/(\d+) outputs proved")
_KEP_SEC_COVERAGE_RE = re.compile(
    r"SEC checked-output coverage: ([0-9.]+)% \((\d+)/(\d+) covered/existing outputs\)")
_KEP_LOAD_FAIL_RE = re.compile(r"Netlist loading failed: *(.*)")
_KEP_UNSUPPORTED_RE = re.compile(r"Unsupported SystemVerilog elements encountered")


def parse_kepler_log(text: str) -> dict[str, object]:
    """Extract the verdict-bearing facts from a ``kepler-formal`` run.

    Args:
        text (str): Combined stdout and stderr of a ``kepler-formal`` run.

    Returns:
        dict: Any of ``proved`` (bool -- a full proof), ``differs`` (bool -- a
        real counterexample), ``partial`` (bool), ``outputs_proved`` (int),
        ``outputs_total`` (int), ``outputs_checked`` (int), ``coverage_pct``
        (float), ``k`` (int), ``abstraction`` (str) and ``load_error`` (str).
        Keys are absent rather than ``None`` when the log did not carry them,
        so "not stated" is distinguishable from "stated as zero".
    """
    out: dict[str, object] = {}

    m = _KEP_LOAD_FAIL_RE.search(text)
    if m:
        out["load_error"] = " ".join(m.group(1).split())[:200] or "unspecified"
    if _KEP_UNSUPPORTED_RE.search(text):
        out["unsupported_constructs"] = True

    m = _KEP_SEC_COVERAGE_RE.search(text)
    if m:
        out["coverage_pct"] = float(m.group(1))
        out["outputs_checked"] = int(m.group(2))
        out["outputs_total"] = int(m.group(3))

    m = _KEP_SEC_PARTIAL_RE.search(text)
    if m:
        out["partial"] = True
        out["k"] = int(m.group(1))
        out["outputs_proved"] = int(m.group(2))
        out.setdefault("outputs_total", int(m.group(3)))
        return out

    m = _KEP_SEC_CEX_RE.search(text)
    if m:
        out["differs"] = True
        out["k"] = int(m.group(1))
        return out

    m = _KEP_SEC_PROVED_RE.search(text)
    if m:
        out["proved"] = True
        out["abstraction"] = m.group(1).strip()
        out["k"] = int(m.group(2))
        return out

    # LEC mode says only this, and says it with exit code 0 either way.
    if _KEP_LEC_SAME_RE.search(text):
        out["proved"] = True
        return out
    if _KEP_DIFF_RE.search(text):
        out["differs"] = True
        return out
    return out


@dataclass
class KeplerBackend:
    """Second opinion on an edit, from ``kepler-formal``'s SEC/LEC engines.

    Same verdict vocabulary and the same fail-closed contract as
    :class:`LecGateNode`: only :data:`PROVEN` admits an edit, only a real
    counterexample is :data:`REFUTED`, and everything else -- a partial proof,
    an inconclusive run, a netlist that would not load, a timeout, an
    unparsable log -- is :data:`UNDECIDED`, :data:`TIMEOUT` or :data:`ERROR`.

    Attributes:
        kepler (str): The ``kepler-formal`` executable, by name or path.
        workdir (Path): Directory for the generated file lists and the log.
            kepler also drops ``miter_log_<n>.txt`` and its skipped-output
            reports in its working directory, so give each check its own.
        mode (str): ``"sec"`` (sequential, the reason this backend exists) or
            ``"lec"`` (combinational, gate-level netlists only).
        fmt (str): ``"sv"`` for RTL SystemVerilog through slang, or
            ``"verilog"`` for structural netlists through naja-verilog. RTL
            Verilog does NOT work in ``"verilog"`` mode -- that front end is a
            netlist parser and rejects compiler directives.
        engine (str): ``pdr`` | ``imc`` | ``k_induction``.
        encoding (str): ``dual_rail_steady`` | ``binary``.
        max_k (int): SEC proof/search bound. The engines are inductive, so a
            returned proof is a proof; ``max_k`` bounds the SEARCH, and running
            out of it yields "inconclusive", never a false proof.
        defines (Sequence[str]): Preprocessor defines added to the slang file
            list, e.g. ``("FORMAL", "PICORV32_REGS=picorv32_regs")``. picorv32
            does not load without exactly those two.
        liberty (Sequence[str]): Liberty libraries, for gate-level inputs.
        require_full_coverage (bool): Demote a "proved" verdict to
            :data:`UNDECIDED` when kepler reports coverage below 100%. Defence
            in depth: every observed proof came with 100% coverage, but a proof
            over a subset of outputs is not a proof that the designs agree, and
            the gate must not accept one if kepler ever prints it.
    """

    kepler: str = "kepler-formal"
    workdir: Path = Path(".")
    mode: str = "sec"
    fmt: str = "sv"
    engine: str = "pdr"
    encoding: str = "dual_rail_steady"
    max_k: int = 32
    defines: Sequence[str] = field(default_factory=tuple)
    liberty: Sequence[str] = field(default_factory=tuple)
    timeout_s: float = 1800.0
    require_full_coverage: bool = True
    verbose: bool = True

    #: Reported as :attr:`LecResult.backend`, so a cross-check names the tool.
    name: str = "kepler-formal"

    def _write_flist(self, srcs: Sequence[str], path: Path) -> Path:
        """Write a slang command file: the defines first, then absolute sources.

        Paths are absolutised because the command file is included with ``-f``
        from a temporary file kepler writes elsewhere, and a relative entry then
        resolves against the wrong directory and fails as "No such file".
        """
        lines = [f"-D {d}" for d in self.defines]
        lines += [str(Path(x).resolve()) for x in srcs]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        return path

    def _argv(self, exe: str, gold_srcs: Sequence[str], gate_srcs: Sequence[str],
              top: str, wd: Path, name: str) -> list[str]:
        argv = [exe]
        if self.fmt == "sv":
            argv.append("-sv")
        elif self.fmt == "verilog":
            argv.append("-verilog")
        else:  # pragma: no cover - guarded by check()
            raise ValueError(f"unknown fmt {self.fmt!r}")

        argv += ["-v", self.mode]
        if self.mode == "sec":
            argv += ["--sec-engine", self.engine,
                     "--sec-encoding", self.encoding,
                     "-k", str(self.max_k)]

        if self.fmt == "sv":
            g = self._write_flist(gold_srcs, wd / f"{name}.gold.f")
            t = self._write_flist(gate_srcs, wd / f"{name}.gate.f")
            argv += ["--sv_design1_flist", str(g), "--sv_design1_top", top,
                     "--sv_design2_flist", str(t), "--sv_design2_top", top]
        else:
            argv += ["--design1", *[str(Path(x).resolve()) for x in gold_srcs],
                     "--design2", *[str(Path(x).resolve()) for x in gate_srcs],
                     "--verilog_design1_top", top, "--verilog_design2_top", top]
        if self.liberty:
            argv += ["--liberty", *[str(Path(x).resolve()) for x in self.liberty]]
        return argv

    def _run(self, argv: list[str], wd: Path) -> tuple[int | None, str, bool]:
        """Run kepler in its own process group, killing the tree on timeout.

        Same precaution as :meth:`LecGateNode._run_eqy`: kepler embeds Python
        and drives SAT solvers, and a surviving child holding the stdout pipe
        open makes ``communicate()`` block past its own timeout.
        """
        proc = subprocess.Popen(argv, cwd=str(wd), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                start_new_session=True)
        try:
            out, _ = proc.communicate(timeout=self.timeout_s)
            return proc.returncode, out or "", False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):  # pragma: no cover
                pass
            try:
                out, _ = proc.communicate(timeout=30)
            except Exception:  # pragma: no cover - the pipe is already gone
                out = ""
            return None, out or "", True

    def check(self, gold_srcs: Sequence[str], gate_srcs: Sequence[str],
              top: str, *, name: str = "kepler") -> LecResult:
        """Run one check and return a verdict in the gate's own vocabulary.

        Args:
            gold_srcs (Sequence[str]): Source files of the known-good design.
            gate_srcs (Sequence[str]): Source files of the candidate design.
            top (str): Top module name, identical on both sides. kepler aligns
                the two designs by top-level terminal NAME, never by internal
                names -- which is precisely why it can cross a moved sequential
                boundary that ``eqy`` cannot.
            name (str): Basename for this check's file lists and log.

        Returns:
            LecResult: Never raises for a tool failure. ``backend`` is
            ``"kepler-formal"``; ``abstraction`` records the encoding a proof
            was obtained under; ``depth`` carries kepler's ``k``, which is the
            induction depth at which the proof CLOSED, not a bound on it --
            ``bounded`` stays False because all three SEC engines are inductive
            and ``max_k`` limits the search, not the proof.
        """
        wd = Path(self.workdir)
        wd.mkdir(parents=True, exist_ok=True)

        if self.mode not in ("sec", "lec"):
            return LecResult(ERROR, self.name, 0.0,
                             message=f"mode must be 'sec' or 'lec', not {self.mode!r}")
        if self.fmt not in ("sv", "verilog"):
            return LecResult(ERROR, self.name, 0.0,
                             message=f"fmt must be 'sv' or 'verilog', not {self.fmt!r}")
        if self.fmt == "sv" and self.mode != "sec":
            # kepler rejects this itself; say so before spending a process on it.
            return LecResult(ERROR, self.name, 0.0,
                             message="the SystemVerilog front end supports SEC only")
        if not gold_srcs or not gate_srcs:
            return LecResult(ERROR, self.name, 0.0,
                             message="both gold_srcs and gate_srcs are required; "
                                     "an empty side proves nothing")
        missing = [str(x) for x in (*gold_srcs, *gate_srcs, *self.liberty)
                   if not Path(x).exists()]
        if missing:
            return LecResult(ERROR, self.name, 0.0,
                             message=f"source file(s) not found: {missing[:4]}")

        exe = shutil.which(self.kepler) or self.kepler
        argv = self._argv(exe, gold_srcs, gate_srcs, top, wd, name)
        if self.verbose:
            print(f"[KeplerBackend] {' '.join(argv)}", flush=True)

        t0 = time.monotonic()
        try:
            rc, text, timed_out = self._run(argv, wd)
        except FileNotFoundError:
            return LecResult(ERROR, self.name, time.monotonic() - t0,
                             message=f"kepler-formal not found at {self.kepler!r}; "
                                     f"build it with scripts/build/kepler.sh")
        except OSError as e:
            # A path that EXISTS but cannot be exec'd -- not +x, a directory, a
            # half-written binary -- raises PermissionError/OSError, NOT
            # FileNotFoundError. Letting it escape would break this method's
            # never-raises contract and, in the sweep, kill a whole arm on what
            # is really just a broken install.
            return LecResult(ERROR, self.name, time.monotonic() - t0,
                             message=f"kepler-formal at {self.kepler!r} could not be "
                                     f"executed ({type(e).__name__}: {e}); "
                                     f"rebuild it with scripts/build/kepler.sh")
        wall = time.monotonic() - t0

        log = wd / f"{name}.kepler.log"
        log.write_text(text)
        if timed_out:
            return LecResult(TIMEOUT, self.name, wall,
                             message=f"kepler-formal exceeded {self.timeout_s}s",
                             log_path=str(log))

        p = parse_kepler_log(text)
        cov = p.get("coverage_pct")
        common = dict(
            backend=self.name, wall_s=wall, returncode=rc, log_path=str(log),
            outputs_total=p.get("outputs_total"),
            outputs_checked=p.get("outputs_checked"),
            coverage_pct=cov,
        )

        if p.get("load_error"):
            # A design kepler cannot ingest is an ERROR, not a cautious pass and
            # not a refutation. rc is 1 here -- the SAME code as a partial proof.
            return LecResult(ERROR, message=f"netlist loading failed: "
                                            f"{p['load_error']}"[:400], **common)

        if p.get("proved"):
            # Cross-check the parse against the documented exit code. A "proved"
            # line with any other rc means the tool and its own spec disagree,
            # and an equivalence gate must not resolve that in favour of a pass.
            if rc != KEPLER_RC_PROVED:
                return LecResult(ERROR, message=f"log says proved but rc={rc}; "
                                                f"refusing to admit on a "
                                                f"self-inconsistent run", **common)
            if self.require_full_coverage and cov is not None and cov < 100.0:
                return LecResult(
                    UNDECIDED,
                    message=(f"proof covers only {cov}% of outputs "
                             f"({p.get('outputs_checked')}/{p.get('outputs_total')}); "
                             f"a proof over a subset of outputs is not a proof"),
                    **common)
            return LecResult(PROVEN, depth=p.get("k"),  # type: ignore[arg-type]
                             abstraction=str(p.get("abstraction") or ""),
                             message=f"proved at k={p.get('k')}", **common)

        if p.get("differs"):
            if rc not in (KEPLER_RC_PROVED, KEPLER_RC_COUNTEREXAMPLE):
                # LEC exits 0 on a difference; SEC exits 3. Anything else means
                # the run did not end the way the log claims.
                return LecResult(ERROR, message=f"log says difference but rc={rc}",
                                 **common)
            return LecResult(REFUTED, depth=p.get("k"),  # type: ignore[arg-type]
                             message=f"counterexample at k={p.get('k')}", **common)

        if p.get("partial"):
            # THE verdict that keeps this backend honest. Measured on picorv32:
            # the comment-only edit and the real BGE bug both land here, so
            # calling a partial proof a pass would admit the bug and calling it
            # a refutation would reject the harmless edit.
            return LecResult(
                UNDECIDED,
                message=(f"partially proved at k={p.get('k')}: "
                         f"{p.get('outputs_proved')}/{p.get('outputs_total')} "
                         f"outputs proved, rest inconclusive"),
                **common)

        tail = " ".join(text.split())[-200:]
        return LecResult(ERROR, message=f"unrecognised kepler log (rc={rc}), "
                                        f"tail: {tail}"[:400], **common)


def cross_check(primary: LecResult, secondary: LecResult) -> dict[str, object]:
    """Compare two independent checkers; disagreement is a reportable result.

    Args:
        primary (LecResult): The gate's own verdict.
        secondary (LecResult): An independent checker's verdict on the same pair.

    Returns:
        dict: ``agree`` (bool), ``primary`` and ``secondary`` summaries, and a
        ``note`` that spells out any disagreement. Nothing is averaged or
        reconciled: two formal tools disagreeing is a finding, not noise.
    """
    agree = primary.verdict == secondary.verdict
    return {
        "agree": agree,
        "primary": {"backend": primary.backend, "verdict": primary.verdict,
                    "wall_s": primary.wall_s},
        "secondary": {"backend": secondary.backend, "verdict": secondary.verdict,
                      "wall_s": secondary.wall_s},
        "note": "" if agree else
                (f"DISAGREEMENT: {primary.backend}={primary.verdict} vs "
                 f"{secondary.backend}={secondary.verdict} -- report, do not average"),
    }


# The resource token is the ONLY binding between this function and the worker
# image that carries eqy: a cluster node type declaring
# `resources: {"eqy": 1}` with `docker.image: ghcr.io/ucb-bar/chia-eqy:latest`
# is what puts this call in a container that can actually run the tool.
@ChiaFunction(resources={"eqy": 1})
def lec_gate(gold_srcs: list[str], gate_srcs: list[str], top: str,
             eqy: str = "eqy", workdir: str = ".", depth: int = 0,
             engine: str = "smtbmc yices", timeout_s: float = 1800.0,
             strategies: list[tuple[str, list[str]]] | None = None,
             read_cmd: str = "read_verilog -sv",
             gold_read_cmd: str | None = None,
             gate_read_cmd: str | None = None,
             undef_init: str | None = "zero",
             jobs: int | None = None) -> dict:
    """Prove that a candidate RTL edit is equivalent to its parent.

    Run this before accepting any edit an agent proposes. Simulation and QoR
    cannot tell an optimisation from a functional bug: inverting picorv32's
    ``BGE`` comparison synthesises cleanly and scores *better* than the correct
    design (75,074.50 um2 vs 75,663.82 um2, 235 fewer cells), and only an
    equivalence check refutes it.

    The gate fails closed. Treat ``equivalent`` as the single decision bit:
    a timeout, a crash, an unparsable log and an undecided partition are all
    ``equivalent: false``, and only ``refuted: true`` means a counterexample was
    actually found.

    Args:
        gold_srcs (list[str]): Paths to the source files of the known-good
            (parent) design. All are read by one front-end command.
        gate_srcs (list[str]): Paths to the source files of the candidate design.
        top (str): Name of the top module. Must exist in both designs.
        eqy (str): ``eqy`` executable, by name or absolute path.
        workdir (str): Directory for the generated config, the log, and the
            checker's output tree.
        depth (int): Bounded depth for the single-strategy fallback; 0 means
            unbounded. A bounded pass is reported as a bounded proof.
        engine (str): SMT engine line for the single-strategy fallback,
            e.g. ``"smtbmc yices"`` or ``"smtbmc z3"``.
        timeout_s (float): Wall-clock budget in seconds. On expiry the checker's
            whole process group is killed and the verdict is ``"timeout"``.
        strategies (list | None): Strategy ladder as ``(name, lines)`` pairs.
            ``None`` uses the single strategy built from ``engine``/``depth``.
            The validated ladder is :data:`DEFAULT_STRATEGIES`, which decides
            strictly more partitions than either strategy alone. Its ORDER is
            load-bearing and ``smt`` must lead: ``use sat``'s induction reports
            "Induction step proven: SUCCESS!" -- which eqy turns into PASS --
            for ``y <= 4'b0`` against ``y <= a + b``, so an ``induct``-first
            ladder admits a design that is not equivalent by any reading, while
            ``sby``/smtbmc refutes the same pair in 0.2 s with a counterexample.
        read_cmd (str): Yosys front-end command for both sides. SystemVerilog
            designs need ``"read_slang"``; ``read_verilog -sv`` cannot elaborate
            them. Every recorded LiveLane run uses ``read_slang``, and the front
            end is not cosmetic -- on picorv32 it moves the critical path from
            12.7612 ns to 14.7771 ns -- so it must match the front end the QoR
            lane used.
        gold_read_cmd (str | None): Override the front end for the gold side
            only (e.g. a ``-D`` define selecting the parent variant).
        gate_read_cmd (str | None): Override the front end for the gate side.
        undef_init (str | None): Resolve undefined flip-flop initial values to
            ``"zero"`` or ``"one"`` on both sides before comparing. Designs with
            free-running or otherwise uninitialised registers are not provably
            equivalent even to themselves without this, because each side's
            undefined state is an independent unconstrained variable -- measured
            on picorv32, whose two 64-bit counters made an identical pair come
            back ``refuted``. A proof obtained this way is reported as
            ``evidence_strength="proof[undef-init=zero]"``, never as a bare
            proof. Pass ``None`` to disable and take the unqualified claim.
        jobs (int | None): Prove partitions concurrently, via ``eqy -j <N>``.
            ``eqy`` emits one make target chain per partition -- 590 of them on
            picorv32 -- and runs them serially unless this is set. ``None``
            keeps the serial default. :func:`auto_jobs` derives a value from the
            runtime's own CPU allocation.

    Returns:
        dict: ``equivalent`` (bool -- the only bit that may admit an edit),
        ``refuted`` (bool -- a counterexample was found), ``verdict`` (str: one
        of proven/refuted/undecided/error/timeout/skipped), ``backend`` (str),
        ``wall_s`` (float), ``bounded`` (bool), ``depth`` (int | None),
        ``is_unbounded_proof`` (bool), ``evidence_strength`` (str),
        ``abstraction`` (str -- empty when the claim is unqualified),
        ``partitions_total`` (int | None), ``partitions_failed`` (int | None),
        ``message`` (str), ``log_path`` (str | None) and
        ``counterexample_path`` (str | None).
    """
    node = LecGateNode(eqy=eqy, workdir=Path(workdir), depth=depth,
                       engine=engine, timeout_s=timeout_s,
                       read_cmd=read_cmd, gold_read_cmd=gold_read_cmd,
                       gate_read_cmd=gate_read_cmd,
                       undef_init=undef_init, jobs=jobs,
                       strategies=tuple((n, tuple(v)) for n, v in (strategies or ())))
    r = node.check(gold_srcs, gate_srcs, top)
    return {
        # Real bools first, so a caller can branch without parsing strings.
        "equivalent": r.admits_edit,
        "refuted": r.is_refutation,
        "verdict": r.verdict, "backend": r.backend, "wall_s": r.wall_s,
        "bounded": r.bounded, "depth": r.depth,
        "is_unbounded_proof": r.is_unbounded_proof,
        "evidence_strength": r.evidence_strength,
        "partitions_total": r.partitions_total,
        "partitions_failed": r.partitions_failed,
        "abstraction": r.abstraction,
        "message": r.message, "log_path": r.log_path,
        "counterexample_path": r.counterexample_path,
    }


__all__ = ["LecGateNode", "LecResult", "lec_gate", "cross_check",
           "parse_eqy_log", "DEFAULT_STRATEGIES", "VALID_VERDICTS",
           "UNDEF_INIT_VALUES", "auto_jobs",
           "KeplerBackend", "parse_kepler_log",
           "PROVEN", "REFUTED", "UNDECIDED", "ERROR", "TIMEOUT", "SKIPPED"]


if __name__ == "__main__":
    import tempfile

    print("=== chia.formal.lec_gate self-test ===")

    # Verbatim from real eqy runs. These two logs are byte-for-byte the same
    # shape -- same summary line, same rc -- and mean opposite things.
    undecided_log = (
        "EQY run: Could not prove equivalence of partition 'picorv32.count_cycle' "
        "using strategy 'sby': equivalence unknown\n"
        "EQY Warning: Failed to prove equivalence for 2/473 partitions:\n"
        "EQY Failed to prove equivalence of partition picorv32.count_cycle\n"
        "EQY DONE (FAIL, rc=2)")
    refuted_log = (
        "EQY run: Could not prove equivalence of partition 'nerv.next_rd' "
        "using strategy 'sby': partitions not equivalent\n"
        "EQY Warning: Failed to prove equivalence for 1/44 partitions:\n"
        "EQY Failed to prove equivalence of partition nerv.next_rd\n"
        "EQY DONE (FAIL, rc=2)")

    u, r_ = parse_eqy_log(undecided_log), parse_eqy_log(refuted_log)
    assert u["unknown_partitions"] == ["picorv32.count_cycle"]
    assert not u.get("not_equivalent_partitions")
    assert r_["not_equivalent_partitions"] == ["nerv.next_rd"]
    assert not r_.get("unknown_partitions")
    assert u["done_rc"] == r_["done_rc"] == 2, "the rc really is identical"
    print("    identical rc=2 logs separated into undecided vs refuted")

    assert parse_eqy_log("EQY Successfully proved designs equivalent")["proved"] is True
    assert parse_eqy_log("total garbage") == {}, "unparsable must yield nothing"
    print("    proven parsed; unparsable log yields no facts at all")

    for v in (REFUTED, UNDECIDED, ERROR, TIMEOUT, SKIPPED):
        assert LecResult(v, "eqy", 1.0).admits_edit is False, v
    assert LecResult(PROVEN, "eqy", 1.0).admits_edit is True
    print(f"    fail-closed: only {PROVEN!r} admits; "
          f"{sorted(VALID_VERDICTS - {PROVEN})} do not")

    assert LecResult(REFUTED, "eqy", 1.0).is_refutation is True
    for v in (UNDECIDED, ERROR, TIMEOUT, SKIPPED):
        assert LecResult(v, "eqy", 1.0).is_refutation is False, v
    print("    is_refutation is true ONLY for a real counterexample")

    b = LecResult(PROVEN, "eqy", 1.0, bounded=True, depth=10)
    assert b.admits_edit and not b.is_unbounded_proof
    assert b.evidence_strength == "bounded-proof(depth=10)"
    assert LecResult(PROVEN, "eqy", 1.0).evidence_strength == "proof"
    print("    bounded proof admits but is labelled bounded-proof(depth=10)")

    try:
        LecResult("probably_fine", "eqy", 1.0)
        raise AssertionError("bad verdict accepted")
    except ValueError as e:
        print(f"    verdict vocabulary enforced: {str(e)[:60]}...")

    d = cross_check(LecResult(PROVEN, "eqy", 1.0),
                    LecResult(REFUTED, "circt-lec", 0.5))
    assert d["agree"] is False and "DISAGREEMENT" in str(d["note"])
    print("    backend disagreement reported, never averaged")

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        for f in ("a.v", "b.v", "c.v", "x.sv", "y.sv"):
            (td_p / f).write_text("// placeholder\n")

        n = LecGateNode(workdir=td_p, strategies=DEFAULT_STRATEGIES,
                        verbose=False)
        cfg = n.write_config([str(td_p / "a.v"), str(td_p / "b.v")],
                             [str(td_p / "a.v"), str(td_p / "c.v")], "top",
                             td_p / "t.eqy")
        body = cfg.read_text()
        read_lines = [ln for ln in body.splitlines() if ln.startswith("read_verilog")]
        assert len(read_lines) == 2, f"one read per side, not per file: {read_lines}"
        assert "/a.v " in read_lines[0] and read_lines[0].endswith("/b.v"), \
            read_lines[0]
        assert "prep -top top" in body
        assert "[strategy induct]" in body and "[strategy smt]" in body
        print("    config: one read command per side, both ladder strategies present")

        # Sources are absolutised: eqy runs with cwd=workdir, so a relative
        # path would be resolved against the wrong directory.
        rel = LecGateNode(workdir=td_p, verbose=False)
        import os as _os
        cwd0 = _os.getcwd()
        _os.chdir(td)
        try:
            body2 = rel.write_config(["a.v"], ["b.v"], "t",
                                     td_p / "rel.eqy").read_text()
        finally:
            _os.chdir(cwd0)
        assert f"read_verilog -sv {Path(td).resolve()}/a.v" in body2, body2
        print("    relative source paths are absolutised against the caller's cwd")

        slang = LecGateNode(workdir=td_p, read_cmd="read_slang", verbose=False)
        s = slang.write_config([str(td_p / "x.sv")], [str(td_p / "y.sv")], "Alu",
                               td_p / "s.eqy").read_text()
        assert "read_slang --top Alu " in s, "read_slang needs an explicit --top"
        print("    read_slang gets the explicit --top it requires")

        # An operator error must be reported as one, not as a solver failure.
        gone = LecGateNode(workdir=td_p, verbose=False).check(
            [str(td_p / "a.v")], [str(td_p / "nope.v")], "t", name="gone")
        assert gone.verdict == ERROR and "not found" in gone.message, gone
        print(f"    missing source named explicitly: {gone.message[:56]}...")

        # A missing binary must fail CLOSED and say why, not raise.
        srcs = ([str(td_p / "a.v")], [str(td_p / "b.v")])
        missing = LecGateNode(eqy="definitely-not-eqy-xyz", workdir=td_p,
                              verbose=False)
        res = missing.check(*srcs, "top", name="missing")
        assert res.verdict == ERROR and not res.admits_edit and not res.is_refutation
        assert "not found" in res.message
        print(f"    missing binary -> {res.verdict!r}, admits_edit=False: "
              f"{res.message[:50]}...")

        # A tool that ignores SIGTERM must still be killed by the timeout, and
        # the verdict must be TIMEOUT rather than a hang.
        stub = td_p / "hang-eqy"
        stub.write_text("#!/bin/sh\ntrap '' TERM\nsleep 60 &\nwait\n")
        stub.chmod(0o755)
        slow = LecGateNode(eqy=str(stub), workdir=td_p, timeout_s=1.0,
                           verbose=False)
        t0 = time.monotonic()
        res = slow.check(*srcs, "top", name="slow")
        took = time.monotonic() - t0
        assert res.verdict == TIMEOUT and not res.admits_edit, res
        assert took < 20.0, f"timeout did not actually fire: {took:.1f}s"
        print(f"    SIGTERM-ignoring tool killed by process group in {took:.1f}s "
              f"-> {res.verdict!r}")

        # A tool that exits 0 with an unrecognised log is NOT a pass.
        liar = td_p / "liar-eqy"
        liar.write_text("#!/bin/sh\necho 'everything is fine'\nexit 0\n")
        liar.chmod(0o755)
        res = LecGateNode(eqy=str(liar), workdir=td_p,
                          verbose=False).check(*srcs, "t", name="liar")
        assert res.returncode == 0 and res.verdict == ERROR and not res.admits_edit
        print("    rc=0 with an unparsable log -> error, NOT a pass")

        # --- the kepler-formal backend ------------------------------------
        # Verbatim lines from real kepler-formal runs (f70c2e3, 2026-09-14).
        kep_proved = ("SEC checked-output coverage: 100.00% (9/9 covered/existing "
                      "outputs).\nNo binary-defined difference was found. SEC proved "
                      "equivalence under the dual-rail steady-state abstraction "
                      "at k = 2.")
        kep_cex = ("SEC checked-output coverage: 100.00% (9/9 covered/existing "
                   "outputs).\nDifference was found. SEC found a counterexample "
                   "at k = 1.")
        kep_partial = ("SEC checked-output coverage: 2.28% (7/307 covered/existing "
                       "outputs).\nSEC partially proved equivalence at k = 1: "
                       "7/307 outputs proved; remaining outputs are inconclusive.")

        kp = parse_kepler_log(kep_proved)
        assert kp["proved"] is True and kp["abstraction"] == "dual-rail steady-state"
        # THE trap: "No binary-defined difference was found." contains
        # "difference was found", so a case-insensitive search for the
        # refutation line matches inside the PROOF and inverts every verdict.
        assert not kp.get("differs"), "a proof must not read as a refutation"
        assert parse_kepler_log(kep_cex)["differs"] is True
        assert parse_kepler_log(kep_partial)["partial"] is True
        assert parse_kepler_log("") == {}, "unparsable must yield nothing"
        print("    kepler: proof / counterexample / partial parsed apart; "
              "case-sensitivity trap held shut")

        def _kep(text, rc, **kw):
            kb = KeplerBackend(workdir=td_p, verbose=False, **kw)
            kb._run = lambda argv, wd: (rc, text, False)  # type: ignore[method-assign]
            return kb.check([str(td_p / "a.v")], [str(td_p / "b.v")], "t",
                            name="kstub")

        assert _kep(kep_proved, 0).verdict == PROVEN
        assert _kep(kep_cex, 3).verdict == REFUTED
        # picorv32 measured: the comment-only edit AND the real BGE bug both land
        # here. Calling this a pass admits the bug; calling it a refutation
        # rejects the harmless edit. It is neither.
        pr = _kep(kep_partial, 1)
        assert pr.verdict == UNDECIDED and not pr.admits_edit and not pr.is_refutation
        assert pr.outputs_checked == 7 and pr.outputs_total == 307
        print(f"    kepler: partial proof -> {pr.verdict!r} "
              f"({pr.outputs_checked}/{pr.outputs_total} outputs), not a pass")

        # kepler exits 1 for BOTH "partially proved" and every load failure, and
        # exits 0 in LEC mode whether the designs match or not, so the exit code
        # can only ever CHECK the parsed verdict -- never produce it.
        assert _kep("Netlist loading failed: Unsupported SystemVerilog elements "
                    "encountered (3):", 1).verdict == ERROR
        assert _kep("kepler said nothing useful", 0).verdict == ERROR
        assert _kep(kep_proved, 2).verdict == ERROR, \
            "a proof line with the wrong rc must not admit"
        subset = kep_proved.replace("100.00% (9/9", "50.00% (8/16")
        assert _kep(subset, 0).verdict == UNDECIDED, \
            "a proof over a subset of outputs is not a proof"
        assert _kep(subset, 0, require_full_coverage=False).verdict == PROVEN
        print("    kepler: load failure, unparsable log, rc/log conflict and "
              "partial coverage all fail closed")

        gone_k = KeplerBackend(kepler="definitely-not-kepler-xyz", workdir=td_p,
                               verbose=False).check(
            [str(td_p / "a.v")], [str(td_p / "b.v")], "t", name="kmissing")
        assert gone_k.verdict == ERROR and not gone_k.admits_edit
        assert "scripts/build/kepler.sh" in gone_k.message
        print("    kepler: missing binary -> error, and says how to build it")

        # The measured disagreement that justifies a second backend at all.
        d2 = cross_check(LecResult(REFUTED, "eqy", 0.66),
                         LecResult(PROVEN, "kepler-formal", 0.06,
                                   abstraction="dual-rail steady-state"))
        assert d2["agree"] is False and "DISAGREEMENT" in str(d2["note"])
        print("    retimed pair: eqy=refuted vs kepler=proven reported, "
              "not averaged")

    print("=== all chia.formal.lec_gate self-tests passed ===")
