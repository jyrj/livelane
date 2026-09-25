"""The agent-facing tool surface types.

Blinding requirement: the agent
must not be able to tell which evaluator lane it is running in.  Behind
``try_candidate`` sits either the Yosys/ABC/OpenSTA stack (optionally delayed) or
``lhd synth``; behind ``read_timing`` sits either OpenSTA or ``timing.json``.  If
either leaked its provenance, every arm comparison would be confounded by the
agent behaving differently when it knows it is in the fast lane.

So each report carries the internal fields the harness needs for the ledger AND
an :meth:`to_agent_dict` that emits only the lane-neutral subset.  The split is
enforced by :data:`LANE_REVEALING_FIELDS` and tested, rather than left to
reviewer discipline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, ClassVar

# Any of these, shown to the agent, would identify the lane or the treatment.
LANE_REVEALING_FIELDS: frozenset[str] = frozenset({
    "source",           # 'yosys+opensta' vs 'lhd'
    "tool_versions",
    "wall_s",           # lane L is seconds, lane S is minutes
    "cpu_s",
    "peak_rss_kb",
    "injected_delay_s",
    "observed_latency_s",
    "backend",
    "log_paths",
    "argv",
    "workdir",
})


@dataclass
class _AgentVisible:
    """Base: split internal bookkeeping from what the agent is allowed to see."""

    #: Fields never serialized to the agent, beyond the global blocklist.
    _internal: ClassVar[frozenset[str]] = frozenset()

    def to_agent_dict(self) -> dict[str, Any]:
        hidden = LANE_REVEALING_FIELDS | self._internal
        out: dict[str, Any] = {}
        for f in fields(self):
            if f.name.startswith("_") or f.name in hidden:
                continue
            out[f.name] = getattr(self, f.name)
        return out

    def to_agent_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_agent_dict(), indent=indent, sort_keys=True, default=str)

    def to_record(self) -> dict[str, Any]:
        """Everything, for the harness ledger."""
        return asdict(self)


@dataclass
class EditResult(_AgentVisible):
    """Result of ``edit_rtl(path, old, new)``."""

    applied: bool
    path: str
    message: str = ""
    occurrences: int = 0
    lines_changed: int = 0
    # internal
    diff: str | None = None
    verilog_sha256: str | None = None
    _internal: ClassVar[frozenset[str]] = frozenset({"diff", "verilog_sha256"})


@dataclass
class TimingReport(_AgentVisible):
    """Worst-case path: slack, endpoint, and the RTL line that drives it.

    Identical shape whether it came from OpenSTA (lane S) or ``timing.json``
    (lane L).
    """

    top: str
    slack_ns: float | None
    max_delay_ns: float | None
    clock_period_ns: float | None = None
    endpoint: str | None = None
    startpoint: str | None = None
    rtl_file: str | None = None
    rtl_line: int | None = None
    path_summary: str = ""
    valid: bool = True
    message: str = ""
    # internal
    source: str | None = None
    raw_json: str | None = None
    wall_s: float | None = None
    _internal: ClassVar[frozenset[str]] = frozenset({"raw_json"})


@dataclass
class QorReport(_AgentVisible):
    """Cell count, area, and max delay for the current design."""

    top: str
    cells: int | None
    area_um2: float | None
    max_delay_ns: float | None
    slack_ns: float | None = None
    valid: bool = True
    message: str = ""
    # internal
    source: str | None = None
    raw_json: str | None = None
    wall_s: float | None = None
    cpu_s: float | None = None
    peak_rss_kb: int | None = None
    _internal: ClassVar[frozenset[str]] = frozenset({"raw_json"})

    def improves_on(self, parent: "QorReport", area_epsilon: float = 0.02) -> bool:
        """Primary objective: shorter worst-case delay, with area as a guardrail.

        An edit that shortens the path but
        inflates area past ``area_epsilon`` is not counted as an improvement.
        """
        if self.max_delay_ns is None or parent.max_delay_ns is None:
            return False
        if self.max_delay_ns >= parent.max_delay_ns:
            return False
        if self.area_um2 is not None and parent.area_um2:
            if self.area_um2 > parent.area_um2 * (1.0 + area_epsilon):
                return False
        return True


@dataclass
class FunctionalResult(_AgentVisible):
    """Shared Verilator oracle, the same gate in every arm."""

    passed: bool
    tests_run: int = 0
    tests_failed: int = 0
    first_failure: str | None = None
    message: str = ""
    # internal
    wall_s: float | None = None
    log_paths: list[str] = field(default_factory=list)


@dataclass
class LecVerdict(_AgentVisible):
    """Equivalence against the parent. Runs in EVERY arm, on every candidate."""

    verdict: str                     # proven | refuted | error | skipped | timeout
    message: str = ""
    counterexample: str | None = None
    # internal
    backend: str | None = None       # eqy | circt-lec | lhd-lec
    wall_s: float | None = None
    log_paths: list[str] = field(default_factory=list)

    VALID: ClassVar[frozenset[str]] = frozenset(
        {"proven", "refuted", "error", "skipped", "timeout"}
    )

    def __post_init__(self) -> None:
        if self.verdict not in self.VALID:
            raise ValueError(f"unknown LEC verdict {self.verdict!r}; expected one of {sorted(self.VALID)}")

    @property
    def is_proven(self) -> bool:
        return self.verdict == "proven"


@dataclass
class CandidateVerdict(_AgentVisible):
    """What ``try_candidate`` returns: build, simulate, prove, score."""

    accepted: bool
    stage: str                        # compile | functional | equivalence | qor | done
    message: str = ""
    functional: FunctionalResult | None = None
    lec: LecVerdict | None = None
    qor: QorReport | None = None
    improved: bool | None = None
    # internal
    variant_id: int | None = None
    observed_latency_s: float | None = None
    injected_delay_s: float = 0.0
    _internal: ClassVar[frozenset[str]] = frozenset({"variant_id"})

    def to_agent_dict(self) -> dict[str, Any]:
        d = super().to_agent_dict()
        for k in ("functional", "lec", "qor"):
            v = d.get(k)
            if isinstance(v, _AgentVisible):
                d[k] = v.to_agent_dict()
        return d


if __name__ == "__main__":
    print("=== report blinding self-test ===")

    # The SAME agent-visible payload must come out of both lanes.
    lane_s = QorReport(top="picorv32", cells=6691, area_um2=75664.0,
                       max_delay_ns=9.4, source="yosys+abc+opensta",
                       wall_s=2.81, cpu_s=2.6, peak_rss_kb=57000,
                       raw_json='{"yosys": true}')
    lane_l = QorReport(top="picorv32", cells=6691, area_um2=75664.0,
                       max_delay_ns=9.4, source="lhd",
                       wall_s=0.9, cpu_s=0.8, peak_rss_kb=210000,
                       raw_json='{"lhd": true}')
    a, b = lane_s.to_agent_dict(), lane_l.to_agent_dict()
    assert a == b, f"LANE LEAK: {a} != {b}"
    print(f"    lane S and lane L emit identical agent views: {a}")

    for k in ("source", "wall_s", "cpu_s", "peak_rss_kb", "raw_json"):
        assert k not in a, f"{k} leaked to the agent"
    assert lane_s.to_record()["source"] == "yosys+abc+opensta", "ledger must keep provenance"
    print("    ledger retains source/timing; agent view does not")

    # Objective: delay down, area guarded.
    parent = QorReport("t", 100, 1000.0, 10.0)
    assert QorReport("t", 101, 1010.0, 9.0).improves_on(parent), "clear win rejected"
    assert not QorReport("t", 101, 1010.0, 10.5).improves_on(parent), "slower accepted"
    assert not QorReport("t", 400, 1500.0, 9.0).improves_on(parent), "area blowup accepted"
    assert QorReport("t", 100, 1019.0, 9.9).improves_on(parent), "within-epsilon area rejected"
    assert not QorReport("t", 100, None, None).improves_on(parent), "missing data accepted"
    print("    improves_on: delay primary, area guardrail at 2%")

    try:
        LecVerdict(verdict="probably_fine")
        raise AssertionError("bad verdict accepted")
    except ValueError as e:
        print(f"    LEC verdict vocabulary enforced: {e}")

    cv = CandidateVerdict(
        accepted=True, stage="done",
        functional=FunctionalResult(passed=True, tests_run=12, wall_s=3.0),
        lec=LecVerdict("proven", backend="eqy", wall_s=1.2),
        qor=lane_s, improved=True, variant_id=7,
        observed_latency_s=632.8, injected_delay_s=600.0,
    )
    d = cv.to_agent_dict()
    assert "variant_id" not in d and "injected_delay_s" not in d and "observed_latency_s" not in d
    assert "backend" not in d["lec"] and "source" not in d["qor"]
    assert d["lec"]["verdict"] == "proven" and d["qor"]["max_delay_ns"] == 9.4
    print(f"    nested reports scrubbed too: {json.dumps(d, sort_keys=True)[:120]}...")
    print("=== all blinding self-tests passed ===")
