"""The agent's tool surface, and a deterministic agent that exercises it.

The tool surface is fixed: identical method names,
signatures and docstrings in every arm, so the agent cannot infer which evaluator
lane it is running in.  The agent proposes edits; it never scores them, never
writes to the variant tree, and never sees a timing number.

:class:`ScriptedAgent` implements the same protocol with no model behind it.  It
exists so the entire pipeline, gates, delay injection, persistence, analysis,
can be validated deterministically and for zero dollars before any LLM seat is
attached.  A mechanism bug found by a scripted run costs nothing; the same bug
found halfway through a paid sweep costs the sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable


@dataclass
class Proposal:
    """One exact-match edit to one RTL file."""

    path: str
    old: str
    new: str
    note: str = ""

    def is_noop(self) -> bool:
        return self.old == self.new


@dataclass
class AgentContext:
    """Everything the agent is allowed to see about the current state.

    Deliberately narrow. It carries no wall-clock, no tool name, no injected
    delay and no lane identifier; see ``tests/unit/test_blinding.py``.
    """

    design: str
    top: str
    iteration: int
    files: list[str]
    #: Current QoR, already scrubbed via ``to_agent_dict()``.
    qor: dict[str, Any]
    #: Worst-path summary, scrubbed.
    timing: dict[str, Any]
    #: Prior attempts and what they scored, from ``VariantStore.history``.
    history: list[dict[str, Any]]
    #: Read access to the working design.
    read_file: Any = field(repr=False, default=None)
    #: The file declaring the top module. The agent is shown THIS, not whichever
    #: file happens to sort first.
    primary_file: str | None = None


@runtime_checkable
class Agent(Protocol):
    name: str

    def propose(self, ctx: AgentContext) -> Proposal | None:
        """Return the next edit to try, or None to stop."""
        ...


@dataclass
class ScriptedAgent:
    """A deterministic agent: replays a fixed list of edits.

    Used to validate the harness end-to-end without an LLM. Returns ``None`` when
    the script is exhausted, which the loop treats as a clean stop.
    """

    edits: Sequence[Proposal]
    name: str = "scripted"
    _i: int = 0

    def propose(self, ctx: AgentContext) -> Proposal | None:
        if self._i >= len(self.edits):
            return None
        p = self.edits[self._i]
        self._i += 1
        return p


@dataclass
class NoOpAgent:
    """Proposes nothing. Measures the harness's own per-iteration overhead."""

    name: str = "noop"

    def propose(self, ctx: AgentContext) -> Proposal | None:
        return None


__all__ = ["Agent", "AgentContext", "Proposal", "ScriptedAgent", "NoOpAgent"]
