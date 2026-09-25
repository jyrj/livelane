"""Partition-level proof caching: re-prove only what an edit actually changed.

The problem, measured
---------------------
``eqy`` partitions a design and emits one make target per partition, 590 of
them on picorv32, then proves every one, every run. It has no reuse path at
all: given an existing workdir it exits with ``ERROR: Directory 'wd' already
exists``, and its only options are ``-f`` (delete it) or ``-b`` (rename it).
So an agent that edits one line pays to re-prove all 590 obligations, and
equivalence checking is 54-77% of an iteration in an agentic RTL loop.

The mechanism
-------------
Each generated rule in ``strategies.mk`` looks like::

    strategies/<partition>/<strategy>/status:
        @bash -c "cd strategies/<partition>/<strategy>; source run.sh"

with **no prerequisites**. GNU make therefore considers such a target up to date
whenever the file exists, and skips it. Writing a cached ``status`` file before
invoking make is all it takes to skip a partition, no patching of the makefile,
no change to eqy.

Soundness
---------
The cache key is the sha256 of the partition's own proof obligation, the
``.il`` netlist eqy wrote for it, which carries both the gold and the gate side
of that partition after every front-end and ``prep`` step. Two runs that produce
a byte-identical obligation are asking the solver exactly the same question, so
the earlier answer is the later answer. The key deliberately does NOT include
the design name, the file path, the workdir, or the run: an unchanged module is
reusable across designs and across runs, which is the entire point.

What is NOT cached: anything that did not end in a pass. A refutation and an
undecided result are both re-run, because they are the answers most likely to
change when a strategy, a solver version or a timeout moves, and because a
wrongly cached refutation would reject a valid edit forever.
"""

from __future__ import annotations

import hashlib
import json
import os
import functools
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: A partition is considered proven when sby left this marker.
_PASS_MARKER = "PASS"

#: Yosys's global next-free-id, written into every RTLIL dump's header.
_AUTOIDX_RE = re.compile(rb"^autoidx \d+$", re.M)

#: Directory prefixes inside ``attribute \\src "path:line.col"``. The DIRECTORY
#: is incidental, the same design checked out or copied elsewhere produces the
#: same behaviour, while the basename and line are kept, so two genuinely
#: different sources still hash apart. Without this, a cache never survives a
#: change of working directory, which is every run.
_SRC_DIR_RE = re.compile(rb'(attribute \\src ")([^"]*)(")')

#: Generated names carrying that same global counter. The prefix identifies the
#: pass and source site and is kept; only the trailing serial is renumbered.
_SERIAL_RE = re.compile(rb"(\$(?:auto|techmap|procdff|proc|abc|memory)\$[^\s]*?\$)(\d+)")


@dataclass
class CacheStats:
    """What a single incremental check reused, and what it had to pay for."""

    partitions_total: int = 0
    hits: int = 0
    misses: int = 0
    stored: int = 0
    setup_s: float = 0.0
    prove_s: float = 0.0
    wall_s: float = 0.0
    reuse_pct: float = 0.0
    verdict: str = "unknown"
    #: How many partitions landed on each verdict, in the gate's vocabulary.
    partition_verdicts: dict[str, int] = field(default_factory=dict)
    failed_partitions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class PartitionProofCache:
    """A content-addressed store of per-partition proof results.

    Args:
        root: Directory to hold the store. Created if absent.

    The layout is one directory per key, holding the ``status`` file the make
    target expects plus a small ``meta.json`` recording what produced it. A key
    is the sha256 of a partition's ``.il`` obligation, so the store is shared
    across designs and runs by construction.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def canonicalise(raw: bytes) -> bytes:
        """Strip the parts of an RTLIL dump that carry no meaning.

        Hashing the file as written gives ZERO reuse, measured: a one-line edit
        to picorv32 changed all 590 obligations. Two causes, both global
        counters rather than content:

        * ``autoidx <N>`` in the header, yosys's next-free-id, which moves
          whenever elaboration created a different number of objects ANYWHERE.
          Measured: 12359 vs 12361 on partitions the edit cannot reach.
        * generated names ending in a serial, e.g.
          ``$auto$insbuf.cc:97:execute$3972`` and
          ``$memory\\cpuregs$wrmux[10][0][0]$y$2993``, same counter, embedded
          in every cell and wire yosys minted.

        The serials are RENUMBERED by first appearance rather than deleted.
        Deleting them would map two distinct wires onto one name, making two
        different netlists hash equal, a false cache hit, which in a gate
        means admitting an unproven edit. Renumbering preserves the distinction
        while dropping the offset.

        Measured effect on a one-line ALU edit: 0% reuse raw, 12% with only
        ``autoidx`` normalised, 73.7% with serials renumbered. The residual is
        real: those partitions are in the edit's cone of influence and their
        wire COUNT differs, not just their names.
        """
        raw = _AUTOIDX_RE.sub(b"autoidx <canonical>", raw)

        def strip_dirs(m: "re.Match[bytes]") -> bytes:
            entries = m.group(2).split(b" ")
            return m.group(1) + b" ".join(
                e.rsplit(b"/", 1)[-1] for e in entries) + m.group(3)

        raw = _SRC_DIR_RE.sub(strip_dirs, raw)
        seen: dict[bytes, bytes] = {}

        def renumber(m: "re.Match[bytes]") -> bytes:
            whole = m.group(0)
            if whole not in seen:
                seen[whole] = b"%s#%d" % (m.group(1), len(seen) + 1)
            return seen[whole]

        return _SERIAL_RE.sub(renumber, raw)

    @classmethod
    def key_for(cls, obligation: Path) -> str:
        """sha256 of one partition's CANONICALISED proof obligation."""
        return hashlib.sha256(cls.canonicalise(obligation.read_bytes())).hexdigest()

    def _dir(self, key: str) -> Path:
        # Two-level fan-out: a flat directory of 590 entries per design grows
        # into the tens of thousands across a sweep and slows every lookup.
        return self.root / key[:2] / key

    def get(self, key: str) -> dict[str, Any] | None:
        d = self._dir(key)
        meta = d / "meta.json"
        if not (d / "status").exists() or not meta.exists():
            return None
        try:
            return json.loads(meta.read_text())
        except Exception:
            return None  # a corrupt entry is a miss, never a crash

    def put(self, key: str, status_file: Path, meta: dict[str, Any]) -> None:
        d = self._dir(key)
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(status_file, d / "status")
        (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    def install(self, key: str, dest_status: Path) -> bool:
        """Place a cached status where make will find it. True if installed."""
        src = self._dir(key) / "status"
        if not src.exists():
            return False
        dest_status.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_status)
        return True

    def __len__(self) -> int:
        return sum(1 for _ in self.root.glob("*/*/status"))


def _partition_obligations(workdir: Path) -> dict[str, Path]:
    """Map partition name -> its ``.il`` obligation, as eqy laid them out."""
    pdir = workdir / "partitions"
    if not pdir.is_dir():
        return {}
    return {p.stem: p for p in sorted(pdir.glob("*.il"))}


def _strategy_targets(workdir: Path) -> list[tuple[str, str, Path]]:
    """Every (partition, strategy, status path) the makefile will build.

    Parsed from ``strategies.mk`` rather than guessed from the directory tree,
    so the strategy ladder in the config is what drives reuse. A ladder change
    therefore shows up as misses, which is correct: a different strategy is a
    different question.
    """
    mk = workdir / "strategies.mk"
    if not mk.exists():
        return []
    out: list[tuple[str, str, Path]] = []
    rule = re.compile(r"^strategies/(.+)/([^/]+)/status:\s*$")
    for line in mk.read_text().splitlines():
        m = rule.match(line)
        if m:
            out.append((m.group(1), m.group(2), workdir / "strategies" /
                        m.group(1) / m.group(2) / "status"))
    return out


class NullProofCache:
    """A cache that stores nothing and returns nothing: the ``full`` mode.

    LiveLane adopts LiveHD's measurement contract verbatim, four modes, and
    speedups divide by
    **full**:

    ``full``
        every cache off. This class.
    ``cold``
        caches ON, directories fresh: full PLUS the cost of populating the
        cache.
    ``incremental``
        the same command again after a comment-only touch.
    ``edit``
        one small semantic edit over the same warm workdir.

    The distinction is not pedantry. Dividing by ``cold`` charges the baseline
    for work only the cached run benefits from, one file copy per partition,
    negligible at 590 partitions and not obviously so at 18,261, so it
    inflates every speedup by exactly the amount the cache costs to fill.

    Passing this where a :class:`PartitionProofCache` is expected gives a run
    that behaves identically except that nothing is reused and nothing is
    stored, which is the denominator.
    """

    def install(self, key: str, status: Path) -> bool:
        return False

    def get(self, key: str):
        return None

    def put(self, key: str, status: Path, meta: dict) -> None:
        return None

    def __len__(self) -> int:
        return 0


_STRATEGY_SECTION = re.compile(rb"^\[strategy\s+[^\]]*\]\s*$", re.M)


def question_digest(cfg: Path, eqy: str = "eqy") -> str:
    """A digest of the QUESTION the cache entry answers.

    A cached proof is only reusable for the same obligation AND the same
    question. `ladder_id` encodes the strategy's NAME; the question is its
    BODY, the engine, the solver, the bounded depth, plus the versions of
    the tools that decided it.

    Measured before this existed, with one strategy named `smt` in both runs:

        depth=10, smtbmc yices  ->  proven, reuse   0.00%   (cold)
        depth=40, smtbmc z3     ->  proven, reuse 100.00%   <-- served the
                                                                depth-10 proof

    A bounded depth-10 result answered a depth-40 question, and a yices result
    answered a z3 question. Unsound, and silent: the run simply got faster and
    still said "proven".

    Deliberately EXCLUDED from the digest: the workdir path and the job count.
    Neither changes the question, and including them would void the store on
    every run.
    """
    parts: list[bytes] = []
    try:
        raw = cfg.read_bytes()
    except OSError:
        raw = b""
    # Every [strategy ...] section, header and body, in declaration order,
    # order matters, because eqy reports the verdict of whichever rung settles
    # a partition first.
    hits = list(_STRATEGY_SECTION.finditer(raw))
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(raw)
        parts.append(raw[m.start():end].strip())
    parts.append(b"|tools|")
    parts.append(_tool_versions(eqy).encode())
    return hashlib.sha256(b"\n".join(parts)).hexdigest()[:16]


@functools.lru_cache(maxsize=8)
def _tool_versions(eqy: str = "eqy") -> str:
    """Versions of the tools that decide a partition, cached per process.

    A proof is only valid for the prover that produced it. yosys additionally
    embeds `__FILE__:__LINE__` in generated names, so a rebuild can change the
    obligation text itself.
    """
    out = []
    for argv in ([eqy, "--version"], ["yosys", "-V"], ["sby", "--version"]):
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=30)
            out.append((r.stdout or r.stderr or "").strip().splitlines()[0]
                       if (r.stdout or r.stderr) else "?")
        except Exception:
            out.append("?")
    return " | ".join(out)


def ladder_id(rungs: list[Path]) -> str:
    """Identity of a strategy ladder, for the cache key.

    Every rung is named, not just the first. A ladder [sat, smt] that proves a
    partition only at `smt` must NOT be reusable by a ladder [sat, something
    else]: the result was earned by a rung that is no longer there. Keying on
    the first rung alone made that a cache HIT, faster, still reporting
    "proven", and wrong.
    """
    return "|".join(r.parent.name for r in rungs)


def _ladder_targets(workdir: Path) -> dict[str, list[Path]]:
    """Every rung of every partition's ladder, in ladder order.

    A cache hit only ever satisfies the FIRST rung, because that is the only
    rule with no prerequisites. Every later rung still runs its recipe, a
    `grep`/`echo` in a subshell, once per partition, even when nothing needs
    proving:

        strategies/<p>/<rung2>/status: strategies/<p>/<rung1>/status
            @if grep PASS $^ >/dev/null ; then echo "PASS (cached)" > $@; ...

    At 5,549 partitions that is 5,549 subshells on a fully-cached run, and
    there is no proving left to hide the cost. Measured on CVA6 `tag_cmp` at
    100% reuse: **3.27s with one rung, 12.68s with two**.

    Knowing all the rungs lets a hit satisfy the whole chain, so make skips it
    rather than shelling out per partition.
    """
    mk = workdir / "strategies.mk"
    if not mk.exists():
        return {}
    rule = re.compile(r"^strategies/(.+)/([^/]+)/status:")
    out: dict[str, list[Path]] = {}
    for line in mk.read_text().splitlines():
        m = rule.match(line)
        if not m:
            continue
        part, strat = m.group(1), m.group(2)
        out.setdefault(part, []).append(
            workdir / "strategies" / part / strat / "status")
    return out


def _final_targets(workdir: Path) -> list[tuple[str, str, Path]]:
    """The LAST strategy's status per partition, where the verdict lives.

    A config may declare a ladder of strategies: a cheap one first, a heavier
    one only for what the cheap one could not settle. eqy wires them as a make
    chain, so a partition's *answer* is in the final rung, not the first. Read
    the first rung instead and a partition that the cheap strategy left
    UNKNOWN and the heavy one then proved would be reported `undecided`, and
    the whole run `not proven`.

    eqy writes `summary_targets.list` naming exactly these targets, so this is
    read rather than reconstructed. Note this is a different list from
    :func:`_strategy_targets`, which deliberately returns the FIRST rung,
    the one with no prerequisites, and therefore the only one a cached status
    can satisfy.
    """
    lst = workdir / "summary_targets.list"
    if not lst.exists():
        return []
    out: list[tuple[str, str, Path]] = []
    for line in lst.read_text().splitlines():
        line = line.strip()
        if not line.startswith("strategies/") or not line.endswith("/status"):
            continue
        body = line[len("strategies/"):-len("/status")]
        part, _, strat = body.rpartition("/")
        if part and strat:
            out.append((part, strat, workdir / line))
    return out


#: sby's own outcome vocabulary, mapped onto the gate's. Keeping FAIL and
#: UNKNOWN apart is the whole point: eqy reports both with the same summary line
#: and the same exit code, and only a FAIL is evidence that the designs differ.
#: Collapsing them rejects valid edits and inflates the measured rejection rate.
_SBY_TO_VERDICT = {
    "PASS": "proven",
    "FAIL": "refuted",
    "UNKNOWN": "undecided",
    "ERROR": "error",
    "TIMEOUT": "timeout",
}

#: Worst-first. An aggregate verdict is the worst any partition reported, and a
#: refutation outranks an undecided: one counterexample settles the pair, while
#: an undecided partition only means this strategy could not settle it.
_VERDICT_RANK = ["refuted", "error", "timeout", "undecided", "proven"]


def _partition_verdict(workdir: Path, partition: str, strategy: str) -> str:
    """This partition's outcome, in the gate's vocabulary.

    Read from sby's marker file rather than by parsing a log, because the marker
    is what sby writes last and is unambiguous. An absent marker is ``error``,
    never a pass, fail closed.
    """
    base = workdir / "strategies" / partition / strategy / partition
    for marker, verdict in _SBY_TO_VERDICT.items():
        if (base / marker).exists():
            return verdict
    status = workdir / "strategies" / partition / strategy / "status"
    if status.exists():
        head = status.read_text(errors="replace").strip().split()
        if head and head[0] in _SBY_TO_VERDICT:
            return _SBY_TO_VERDICT[head[0]]
    return "error"


def _partition_passed(workdir: Path, partition: str, strategy: str) -> bool:
    """True when sby left a PASS marker for this partition."""
    return _partition_verdict(workdir, partition, strategy) == "proven"


def incremental_check(cfg: Path, workdir: Path, cache: PartitionProofCache, *,
                      eqy: str = "eqy", jobs: int | None = None,
                      timeout_s: float = 3600.0,
                      verbose: bool = False) -> CacheStats:
    """Prove a pair, reusing every partition whose obligation is unchanged.

    Three steps, none of which modify eqy:

    1. ``eqy -m`` generates the partitions and ``strategies.mk`` and stops
       before proving anything.
    2. Every partition whose obligation hashes to a cached PASS gets its
       ``status`` file written from the store, which makes GNU make treat that
       target as already built.
    3. ``make -f strategies.mk`` runs, touching only the misses, and every new
       pass is stored.

    Args:
        cfg: The ``.eqy`` configuration.
        workdir: Work directory. Removed first, eqy refuses to reuse one.
        cache: The store.
        eqy: Executable.
        jobs: ``make -j``. None leaves make serial.
        timeout_s: Budget for the proving step.
        verbose: Print the commands.

    Returns:
        CacheStats: hits, misses, timings, and the overall verdict.
    """
    t0 = time.monotonic()
    workdir = Path(workdir).resolve()
    shutil.rmtree(workdir, ignore_errors=True)

    setup = [eqy, "-m", "-d", str(workdir), str(cfg)]
    if verbose:
        print("  $", " ".join(setup), flush=True)
    r = subprocess.run(setup, capture_output=True, text=True, timeout=timeout_s)
    setup_s = time.monotonic() - t0
    if not (workdir / "strategies.mk").exists():
        # Keep the tool's own words. "setup rc=1" alone is unactionable, and the
        # failure is usually a front-end message (a missing source, an
        # unelaborable top) that eqy printed and nobody read.
        tail = " ".join((r.stderr or r.stdout or "").split())[-400:]
        return CacheStats(verdict="error", setup_s=round(setup_s, 3),
                          wall_s=round(time.monotonic() - t0, 3),
                          failed_partitions=[f"setup rc={r.returncode}: {tail}"])

    obligations = _partition_obligations(workdir)
    targets = _strategy_targets(workdir)
    ladder = _ladder_targets(workdir)
    # The question this run is asking: strategy bodies + tool versions.
    qdigest = question_digest(cfg, eqy)
    keys: dict[str, str] = {}
    hit_ids: set[str] = set()
    for part, strat, status in targets:
        ob = obligations.get(part)
        if ob is None:
            continue
        # The key names the WHOLE ladder, not just its first rung.
        #
        # Two bugs this closes. With a ladder [sat, smt], a partition that
        # `sat` leaves UNKNOWN and `smt` then proves was stored under a key
        # naming only `sat`. So:
        #   * changing or dropping the SECOND rung still produced a hit, and
        #     reused a result that only the removed rung had ever proven,
        #     unsound, and invisible, because the run got faster and still
        #     said "proven";
        #   * the file stored was the first rung's, which says UNKNOWN, not
        #     the final rung's PASS.
        # Naming every rung means any ladder change is a miss, which is the
        # documented intent ("a different strategy is a different question").
        rungs = ladder.get(part) or [status]
        key = (PartitionProofCache.key_for(ob) + ":" + ladder_id(rungs)
               + ":" + qdigest)
        keys[f"{part}/{strat}"] = key
        if cache.install(key, status):
            hit_ids.add(f"{part}/{strat}")
            # Satisfy the REST of this partition's ladder too. Without this,
            # make still runs one subshell per later rung per partition, which
            # dominates a high-reuse run: the rungs have nothing to do but
            # copy a PASS forward, and there is no proving left to amortise
            # them against.
            #
            # Correctness: the content written is the same `PASS` the recipe
            # would have produced, and only passes are ever stored, so a hit
            # is a pass by construction. mtimes step forward along the chain
            # so make sees each rung no older than its prerequisite.
            stamp = status.stat().st_mtime if status.exists() else time.time()
            for i, rung in enumerate(rungs[1:], start=1):
                try:
                    rung.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(status, rung)
                    os.utime(rung, (stamp + i, stamp + i))
                except OSError:
                    # Fall back to letting make run the rung. Slower, never
                    # wrong.
                    break
    hits = len(hit_ids)

    misses = len(targets) - hits

    # Ask make to build ONLY the partitions that missed, instead of the
    # default `all` goal over every partition.
    #
    # The default goal makes make consider all ~N targets and, before the
    # ladder fix, shell out for each one's later rungs. Even with that fixed,
    # walking thousands of up-to-date targets is pure overhead in the regime
    # this cache exists for: an agentic loop re-proving one edit against a warm
    # cache, where 94-100% of partitions are hits.
    #
    # `all` also depends on `summary`, which re-greps every status file. The
    # verdict here is computed from the status files directly, so that pass is
    # redundant work whose result is discarded.
    #
    # Falls back to `all` if the final target of any miss cannot be named,
    # slower, never wrong.
    final_by_part = {part: status
                     for part, _strat, status in _final_targets(workdir)}
    goals: list[str] | None = []
    for part, strat, _status in targets:
        if f"{part}/{strat}" in hit_ids:
            continue
        fin = final_by_part.get(part)
        if fin is None:
            goals = None
            break
        goals.append(str(fin.relative_to(workdir)))

    t1 = time.monotonic()
    if goals is not None and not goals:
        # Every partition hit. Only passes are ever stored, so the verdict is
        # already determined and there is nothing for make to build.
        pr = None
        if verbose:
            print("  $ (make skipped: every partition hit)", flush=True)
    else:
        mk = ["make", "-f", "strategies.mk", "-C", str(workdir)]
        if jobs:
            mk.insert(1, f"-j{jobs}")
        if goals:
            mk += goals
        if verbose:
            shown = " ".join(mk[:6]) + (f" ...({len(goals)} goals)"
                                        if goals else "")
            print("  $", shown, flush=True)
        pr = subprocess.run(mk, capture_output=True, text=True,
                            timeout=timeout_s)
    prove_s = time.monotonic() - t1

    stored = 0
    failed: list[str] = []
    verdicts: dict[str, int] = {}

    # Store into the cache from the FIRST rung (the only target a cached status
    # can satisfy, because it alone has no prerequisites), but take the verdict
    # from the LAST rung, which is where a ladder puts the answer. With a
    # single strategy the two lists are identical and this is a no-op.
    final = {part: (strat, status)
             for part, strat, status in _final_targets(workdir)}
    for part, strat, status in targets:
        k = keys.get(f"{part}/{strat}")
        v_strat, v_status = final.get(part, (strat, status))
        if not v_status.exists():
            failed.append(part)
            verdicts["error"] = verdicts.get("error", 0) + 1
            continue
        if f"{part}/{strat}" in hit_ids:
            # A hit is a pass by construction, only passes are ever stored.
            # The PASS marker itself is NOT restored (it lives in sby's own
            # output tree), so checking for it here would misread every cached
            # partition as a failure and report a fully-reused run as
            # not-proven. Measured: that bug turned a 100%-hit run into
            # "not-proven" with zero failing partitions to point at.
            verdicts["proven"] = verdicts.get("proven", 0) + 1
            continue
        v = _partition_verdict(workdir, part, v_strat)
        verdicts[v] = verdicts.get(v, 0) + 1
        if v != "proven":
            # Never cached. A refutation or an undecided result is exactly what
            # a strategy, solver or timeout change should be free to revisit.
            failed.append(part)
            continue
        if k and cache.get(k) is None:
            # Store the FINAL rung's status, the file that actually carries
            # the PASS. `status` is the first rung's, which on a ladder may
            # say UNKNOWN for a partition a later rung proved; caching that
            # and replaying it into every rung left a workdir whose own
            # `make summary` reported failure while this function returned
            # "proven".
            store_from = v_status if v_status.exists() else status
            cache.put(k, store_from, {"partition": part, "strategy": v_strat,
                                  "verdict": "pass", "stored_at": time.time()})
            stored += 1

    total = len(targets)
    return CacheStats(
        partitions_total=total, hits=hits, misses=misses, stored=stored,
        setup_s=round(setup_s, 3), prove_s=round(prove_s, 3),
        wall_s=round(time.monotonic() - t0, 3),
        reuse_pct=round(100.0 * hits / total, 2) if total else 0.0,
        verdict=next((v for v in _VERDICT_RANK if verdicts.get(v)), "error"),
        partition_verdicts=verdicts,
        failed_partitions=sorted(set(failed))[:10],
    )


__all__ = [
    "NullProofCache","PartitionProofCache", "CacheStats", "incremental_check"]
