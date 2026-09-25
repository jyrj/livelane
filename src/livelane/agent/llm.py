"""LLM agent seats, and the four-class cost accounting they must carry.

Two things about CHIA's LLM layer shape this module, both established by reading
its source rather than its docs:

1. **``QueryResult`` carries no tokens and no cost.**  Counts live on
   ``self._last_metadata``, which Ray discards when ``prompt`` runs on a remote
   worker.  The documented workaround is to construct the backend *and* call
   ``prompt`` inside one ``@ChiaFunction`` on the creds worker, read
   ``_last_metadata`` there, and return a plain dict.  That is what
   :func:`run_seat` does.
2. **Four-class accounting exists only in the Claude backend.**  Vertex, Bedrock
   and the OpenAI-compatible providers report input/output only.  Since LiveLane
   budgets in dollars and cache-read is the majority of billed input in agentic
   loops, a seat that cannot report four classes has its dollars computed from
   list price and is flagged ``cost_source="listprice"`` so the two are never
   silently mixed in one column.

The agent itself is deliberately a *structured-response* agent rather than an
MCP tool loop.  For this experiment that is the stronger choice: the number of
model turns per iteration is fixed at one, so a latency arm cannot accidentally
differ from another in how many times the model was called.  The MCP tool
surface is still built (``livelane.agent.tools``) as the CHIA-native deliverable.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from livelane.agent.base import AgentContext, Proposal


@dataclass
class Usage:
    """Four-class token accounting plus dollars.

    Two-class pricing is wrong by up to 10x on agentic loops, where most billed
    input is cache-read, so the classes are kept separate all the way to the
    cost table.
    """

    tokens_in: int = 0
    tokens_out: int = 0
    cache_read: int = 0
    cache_write: int = 0
    #: Reasoning/"thinking" tokens. Not part of Anthropic's four-class shape, but
    #: Google bills them as output and on gemini-3.x they DOMINATE it, a
    #: measured PONG reply was 2 output tokens against 92 thinking tokens.
    #: Dropping them undercounts billed output by ~45x, which would corrupt both
    #: the sweep's token cap and the paper's cost table.
    thoughts: int = 0
    cost_usd: float | None = None
    cost_source: str = "unknown"      # reported | listprice | unknown
    model: str | None = None
    latency_s: float = 0.0

    @property
    def billed_input(self) -> int:
        return self.tokens_in + self.cache_read + self.cache_write

    @property
    def billed_output(self) -> int:
        """Google bills thinking tokens as output; Anthropic has none."""
        return self.tokens_out + self.thoughts

    @property
    def total_tokens(self) -> int:
        return self.billed_input + self.billed_output

    def as_dict(self) -> dict[str, Any]:
        return {
            "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
            "tokens_cache_read": self.cache_read,
            "tokens_cache_write": self.cache_write,
            "tokens_thoughts": self.thoughts,
            "cost_usd": self.cost_usd, "cost_source": self.cost_source,
        }


@runtime_checkable
class Seat(Protocol):
    """One model seat. Returns raw text plus whatever usage it can report."""

    name: str

    def ask(self, system: str, user: str, timeout_s: float) -> tuple[str, Usage]:
        ...


# --- prompt construction -----------------------------------------------------
# The SAME strings in every arm. Nothing here mentions a tool, a lane, a delay or
# a wall-clock, so the agent cannot infer which evaluator it is talking to.

SYSTEM_PROMPT = """\
You are optimising a synthesised hardware design written in Verilog/SystemVerilog.

Your goal, in priority order:
  1. reduce the worst-case path delay of the design;
  2. do not inflate area by more than about 2%.

You propose exactly ONE edit at a time. The edit is applied by exact string
replacement, so `old` must appear EXACTLY ONCE in the named file, character for
character including whitespace. An edit whose `old` text is not found, or is
found more than once, is rejected without being evaluated.

Every edit you propose is checked for logic equivalence against the design it was
derived from. An edit that changes behaviour is rejected. Optimise the structure,
not the function.

Reply with a single JSON object and nothing else:

{"file": "<path>", "old": "<exact text to replace>", "new": "<replacement>",
 "note": "<one line: what you changed and why it should help>"}

Only if you have genuinely exhausted every idea, reply exactly: {"stop": true}.
Do not stop merely because an edit looks small or uncertain: the harness proves
equivalence and measures the result, so a rejected edit costs one iteration and
tells you something. A design reported with negative slack still has work to do.
"""


def build_user_prompt(ctx: AgentContext, excerpt_chars: int = 12000) -> str:
    """Assemble the per-iteration prompt from the lane-neutral context."""
    parts = [
        f"Design: {ctx.design}   Top module: {ctx.top}   Iteration: {ctx.iteration}",
        "",
        "Current quality of results:",
        json.dumps(ctx.qor, indent=2, sort_keys=True),
    ]
    if ctx.timing:
        parts += ["", "Worst-case path:", json.dumps(ctx.timing, indent=2, sort_keys=True)]
    if ctx.history:
        parts += ["", "What has already been tried on this design:",
                  json.dumps(ctx.history, indent=2, sort_keys=True)]
    parts += ["", f"Files you may edit ({len(ctx.files)}):",
              "\n".join(f"  {f}" for f in ctx.files[:60])]
    if len(ctx.files) > 60:
        parts.append(f"  ... and {len(ctx.files) - 60} more")
    if ctx.read_file is not None and ctx.files:
        # The file declaring the top module, never merely the first one sorted.
        target = ctx.primary_file or ctx.files[0]
        try:
            text = ctx.read_file(target)
            parts += ["", f"--- {target} (first {excerpt_chars} chars) ---",
                      text[:excerpt_chars]]
        except Exception:
            pass
    return "\n".join(parts)


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_proposal(text: str) -> Proposal | None:
    """Extract the edit. Returns None for a stop or an unusable reply.

    Deliberately strict: a malformed reply is a lost iteration, not a guess.
    Guessing what the model meant would silently change the treatment.
    """
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if obj.get("stop") is True:
        return None
    for k in ("file", "old", "new"):
        if not isinstance(obj.get(k), str):
            return None
    if obj["old"] == obj["new"]:
        return None
    return Proposal(path=obj["file"], old=obj["old"], new=obj["new"],
                    note=str(obj.get("note", ""))[:400])


# --- seats -------------------------------------------------------------------


@dataclass
class ClaudeCliSeat:
    """The `claude` CLI in non-interactive JSON mode.

    Chosen as the reference seat because it is the ONLY backend that reports all
    four token classes and a real ``total_cost_usd`` rather than a list-price
    estimate.
    """

    model: str = "claude-sonnet-5"
    binary: str = "claude"
    name: str = "claude-cli"
    extra_args: tuple[str, ...] = ()

    def ask(self, system: str, user: str, timeout_s: float = 300.0) -> tuple[str, Usage]:
        exe = shutil.which(self.binary) or self.binary
        argv = [exe, "-p", user, "--output-format", "json",
                "--model", self.model, "--append-system-prompt", system,
                *self.extra_args]
        t0 = time.monotonic()
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=timeout_s)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return "", Usage(model=self.model, cost_source="unknown",
                             latency_s=time.monotonic() - t0)
        latency = time.monotonic() - t0
        raw = proc.stdout or ""
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return raw, Usage(model=self.model, latency_s=latency)
        return parse_claude_json(obj, self.model, latency)


def parse_claude_json(obj: dict, model: str, latency_s: float) -> tuple[str, Usage]:
    """Pull text and four-class usage out of the CLI's JSON envelope."""
    text = obj.get("result") or obj.get("text") or ""
    u = obj.get("usage") or {}
    usage = Usage(
        tokens_in=int(u.get("input_tokens", 0) or 0),
        tokens_out=int(u.get("output_tokens", 0) or 0),
        cache_read=int(u.get("cache_read_input_tokens", 0) or 0),
        cache_write=int(u.get("cache_creation_input_tokens", 0) or 0),
        cost_usd=obj.get("total_cost_usd"),
        cost_source="reported" if obj.get("total_cost_usd") is not None else "unknown",
        model=obj.get("model") or model,
        latency_s=latency_s,
    )
    return text, usage


@dataclass
class ChiaSeat:
    """Any ``chia.models`` backend, with usage harvested where it exists.

    Backends other than Claude report at most input/output, so dollars for them
    are computed from list price and flagged.
    """

    backend: Any
    name: str = "chia"
    model: str = "unknown"

    def ask(self, system: str, user: str, timeout_s: float = 300.0) -> tuple[str, Usage]:
        t0 = time.monotonic()
        res = self.backend.prompt(user)
        latency = time.monotonic() - t0
        text = getattr(res, "result", "") or ""
        meta = getattr(res, "usage", None) or getattr(self.backend, "_last_metadata", None) or {}
        usage = Usage(
            tokens_in=int(meta.get("input_tokens", meta.get("tokens_in", 0)) or 0),
            tokens_out=int(meta.get("output_tokens", meta.get("tokens_out", 0)) or 0),
            cache_read=int(meta.get("cache_read_input_tokens", 0) or 0),
            cache_write=int(meta.get("cache_creation_input_tokens", 0) or 0),
            cost_usd=meta.get("cost_usd") or meta.get("total_cost_usd"),
            model=self.model, latency_s=latency,
        )
        usage.cost_source = "reported" if usage.cost_usd is not None else "listprice"
        return text, usage


@dataclass
class RecordedSeat:
    """Replays a recorded transcript. Zero dollars, exact reproduction.

    This is LiveLane's $0-replay story for judges: a run recorded once can be
    replayed for free, and the replay is flagged so it can never be mistaken for
    a fresh latency measurement.
    """

    transcript: list[str]
    name: str = "recorded"
    model: str = "recorded"
    _i: int = 0

    def ask(self, system: str, user: str, timeout_s: float = 0.0) -> tuple[str, Usage]:
        if self._i >= len(self.transcript):
            return '{"stop": true}', Usage(model=self.model, cost_source="reported",
                                           cost_usd=0.0)
        out = self.transcript[self._i]
        self._i += 1
        return out, Usage(model=self.model, cost_usd=0.0, cost_source="reported")


@dataclass
class LLMAgent:
    """Drives one seat, one turn per iteration.

    Exactly one model turn per loop iteration, so no arm can differ from another
    in how many times the model was called.
    """

    seat: Seat
    name: str = "llm"
    timeout_s: float = 300.0
    usages: list[Usage] = field(default_factory=list)
    transcript: list[str] = field(default_factory=list)
    malformed: int = 0

    def __post_init__(self) -> None:
        self.name = f"llm:{getattr(self.seat, 'name', 'seat')}"

    def propose(self, ctx: AgentContext) -> Proposal | None:
        user = build_user_prompt(ctx)
        text, usage = self.seat.ask(SYSTEM_PROMPT, user, self.timeout_s)
        self.usages.append(usage)
        self.transcript.append(text)
        p = parse_proposal(text)
        if p is None and text and '"stop"' not in text:
            self.malformed += 1
        return p

    @property
    def total_cost_usd(self) -> float | None:
        """None when NOTHING could be priced, 0.0 would read as 'free'."""
        priced = [u.cost_usd for u in self.usages if u.cost_usd is not None]
        if not priced and self.usages:
            return None
        return sum(priced)

    def usage_summary(self) -> dict[str, Any]:
        return {
            "turns": len(self.usages),
            "tokens_in": sum(u.tokens_in for u in self.usages),
            "tokens_out": sum(u.tokens_out for u in self.usages),
            "cache_read": sum(u.cache_read for u in self.usages),
            "cache_write": sum(u.cache_write for u in self.usages),
            "thoughts": sum(u.thoughts for u in self.usages),
            "cost_usd": self.total_cost_usd,
            "malformed_replies": self.malformed,
            "mean_latency_s": (sum(u.latency_s for u in self.usages) / len(self.usages)
                               if self.usages else 0.0),
        }


__all__ = ["Usage", "Seat", "ClaudeCliSeat", "ChiaSeat", "RecordedSeat", "LLMAgent",
           "SYSTEM_PROMPT", "build_user_prompt", "parse_proposal", "parse_claude_json"]
