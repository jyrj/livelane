"""The Vertex AI Gemini seat, and the reason its dollars are never called real.

Three facts about Vertex shaped every decision in this module, all of them
established by reading the installed ``google-genai`` source (2.8.0) and by
probing the live endpoint rather than by trusting documentation:

1. **Vertex reports tokens, never dollars.**  ``usage_metadata`` carries
   ``prompt_token_count`` / ``candidates_token_count`` and *sometimes*
   ``cached_content_token_count``; there is no cost field anywhere in the
   response.  So a cost here is always an arithmetic product of a list price we
   typed in by hand, and it is flagged ``cost_source="listprice"`` so it can
   never be pooled in one column with the Claude CLI's ``total_cost_usd``, which
   is a real billed number.  A model absent from the price table gets
   ``cost_usd=None`` and ``"unknown"``, an unpriced call is recorded as
   unpriced, never as free, because a zero would quietly deflate $/improvement.
2. **Location is load-bearing.**  ``client.models.list()`` enumerates publisher
   models globally but availability is regional: ``gemini-3.8-flash`` answers
   only on ``global`` and 404s in ``us-central1``.  A 404 is therefore a
   *configuration* error, not weather, and retrying it just burns wall-clock in
   a latency experiment whose whole dependent variable is wall-clock.  Fatal
   errors (auth, permission, 404, bad request) propagate immediately; only
   genuinely transient ones (429/5xx/timeouts) are retried, and when those are
   finally exhausted the turn returns empty text so one bad iteration costs one
   iteration instead of the sweep.
3. **The SDK does not retry by default.**  ``_api_client.retry_args(None)``
   returns ``stop_after_attempt(1)``, so the backoff below is the only backoff;
   the per-request ``retry_options`` is pinned to one attempt anyway so that a
   future SDK default cannot silently double the retry count and corrupt the
   latency distribution we are measuring.

The seat implements the same :class:`livelane.agent.llm.Seat` protocol as the
other seats, ``ask(system, user, timeout_s) -> (text, Usage)``, so a sweep
can swap seats without any other code knowing which vendor answered.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from livelane.agent.llm import Usage

DEFAULT_LOCATION = "global"
DEFAULT_MODEL = "gemini-3.8-flash"


# --- list price ---------------------------------------------------------------
# !!! LIST PRICE, TYPED BY HAND, NOT REPORTED BY THE API !!!
# MUST be re-checked against current Vertex AI pricing
# (https://cloud.google.com/vertex-ai/generative-ai/pricing) before ANY cost
# table, figure or $/improvement number derived from it is published. Vertex
# returns tokens only; every dollar below is our multiplication, not Google's.
#
# Prices are USD per 1,000,000 tokens, text modality, pay-as-you-go.
# `cache_read` is the discounted rate for tokens the API reports as
# `cached_content_token_count`; where it is None the call is priced as UNKNOWN
# rather than guessed, because charging cached tokens at the full input rate
# would silently overstate spend and charging them at zero would understate it.
#
# gemini-3.8-flash is DELIBERATELY ABSENT: its list price was not verified, and
# an invented price is worse than no price. Fill it in (with the date checked)
# or pass VertexSeat(price=ModelPrice(...)) explicitly.


@dataclass(frozen=True)
class ModelPrice:
    """One SKU's list price, in USD per million tokens.

    ``tier_max_prompt_tokens`` exists because Gemini Pro SKUs step to a higher
    rate above a prompt-size threshold; a prompt past the threshold is priced as
    unknown rather than at the cheap tier we happen to have typed in.
    """

    usd_per_mtok_in: float
    usd_per_mtok_out: float
    usd_per_mtok_cache_read: float | None = None
    tier_max_prompt_tokens: int | None = None
    checked: str = "unverified"


VERTEX_LIST_PRICE_USD_PER_MTOK: dict[str, ModelPrice] = {
    "gemini-2.5-flash": ModelPrice(0.30, 2.50, 0.075, None, "2026-09-04 (recheck)"),
    "gemini-2.5-flash-lite": ModelPrice(0.10, 0.40, 0.025, None, "2026-09-04 (recheck)"),
    "gemini-2.5-pro": ModelPrice(1.25, 10.00, 0.3125, 200_000, "2026-09-04 (recheck)"),
    # --- UNVERIFIED, and labelled as such -----------------------------------
    # These two are the SKUs LiveLane's recorded sweeps actually ran on, so
    # without them every published cost is None. They are taken from third-party
    # pricing summaries because cloud.google.com's pricing page could not be
    # retrieved in full (its tables are truncated by the fetcher). Treat them as
    # provisional: `cost_source` still reads "listprice", but `checked` names the
    # provenance, and PRICES_PATH below lets a verified figure replace them
    # without editing this file.
    "gemini-3.1-pro-preview": ModelPrice(
        2.00, 12.00, None, 200_000,
        "UNVERIFIED 2026-09-17, third-party summary; above 200k is 4.00/18.00"),
    "gemini-3.8-flash": ModelPrice(
        0.75, 3.75, None, None,
        "UNVERIFIED 2026-09-17, third-party summary; INTRODUCTORY rate, "
        "reverts to 1.50/7.50 on 2027-01-01"),
}

#: Overrides loaded at import from JSON, so a verified price never requires a
#: code edit. Set ``LIVELANE_PRICES_JSON`` to point elsewhere. Schema:
#:
#:     {"gemini-3.8-flash": {"usd_per_mtok_in": 0.75, "usd_per_mtok_out": 3.75,
#:                           "usd_per_mtok_cache_read": null,
#:                           "tier_max_prompt_tokens": null,
#:                           "checked": "2026-09-17 cloud.google.com"}}
#:
#: An entry here REPLACES the built-in one for that key, which is the supported
#: way to correct a provisional rate or to add a SKU this file has never heard
#: of. Nothing is inferred: a model absent from both stays unpriced.
PRICES_PATH_ENV = "LIVELANE_PRICES_JSON"


def load_price_overrides(path: str | os.PathLike[str] | None = None) -> dict[str, ModelPrice]:
    """Read price overrides from JSON. Returns {} when there are none.

    Fails LOUD on a malformed file rather than silently falling back to the
    built-in table: a typo in a price file would otherwise change every reported
    dollar figure with nothing in the output to show it happened.

    Args:
        path: JSON file. Defaults to ``$LIVELANE_PRICES_JSON``, then to
            ``configs/prices.json`` beside the repo if it exists.

    Returns:
        dict: model key -> :class:`ModelPrice`.

    Raises:
        ValueError: the file exists but is not a JSON object of price objects.
    """
    import json

    p = path or os.environ.get(PRICES_PATH_ENV)
    if not p:
        default = Path(__file__).resolve().parents[3] / "configs" / "prices.json"
        if not default.exists():
            return {}
        p = default
    p = Path(p)
    if not p.exists():
        raise ValueError(f"{PRICES_PATH_ENV}={p} does not exist")
    raw = json.loads(p.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: expected a JSON object of model -> price")
    out: dict[str, ModelPrice] = {}
    for key, v in raw.items():
        if not isinstance(v, dict) or "usd_per_mtok_in" not in v \
                or "usd_per_mtok_out" not in v:
            raise ValueError(
                f"{p}: entry {key!r} needs at least usd_per_mtok_in and "
                f"usd_per_mtok_out")
        out[normalise_model(key)] = ModelPrice(
            usd_per_mtok_in=float(v["usd_per_mtok_in"]),
            usd_per_mtok_out=float(v["usd_per_mtok_out"]),
            usd_per_mtok_cache_read=(None if v.get("usd_per_mtok_cache_read") is None
                                     else float(v["usd_per_mtok_cache_read"])),
            tier_max_prompt_tokens=(None if v.get("tier_max_prompt_tokens") is None
                                    else int(v["tier_max_prompt_tokens"])),
            checked=str(v.get("checked", "override")),
        )
    return out


def normalise_model(model: str) -> str:
    """Reduce a Vertex model reference to its price-table key.

    Strips only the resource path and an ``@version`` suffix. Preview and dated
    SKUs are NOT folded onto their GA name: they are separately priced, and a
    fuzzy prefix match would price one at the other's rate.
    """
    m = (model or "").strip().lower()
    for prefix in ("publishers/google/models/", "models/"):
        if m.startswith(prefix):
            m = m[len(prefix):]
    return m.split("@", 1)[0]


def estimate_cost_usd(model: str, tokens_in: int, tokens_out: int,
                      cache_read: int = 0, thoughts: int = 0,
                      price: ModelPrice | None = None) -> tuple[float | None, str]:
    """Return ``(cost_usd, cost_source)`` from list price, or ``(None, "unknown")``.

    Thinking tokens are billed as output by Google but are reported separately
    from ``candidates_token_count``, so they are priced at the output rate here
    while :class:`Usage.tokens_out` keeps the raw candidates count. Cost and
    ``tokens_out * rate`` therefore differ on a thinking model, deliberately,
    because the billed quantity is the larger one.

    Fails closed: an unknown SKU, an unpriced cache class that actually has
    cached tokens, or a prompt past the priced tier all yield ``None``.
    """
    key = normalise_model(model)
    p = price
    if p is None:
        p = load_price_overrides().get(key) or VERTEX_LIST_PRICE_USD_PER_MTOK.get(key)
    if p is None:
        return None, "unknown"
    if p.tier_max_prompt_tokens is not None and tokens_in + cache_read > p.tier_max_prompt_tokens:
        return None, "unknown"
    if cache_read > 0 and p.usd_per_mtok_cache_read is None:
        return None, "unknown"
    cost = (
        tokens_in * p.usd_per_mtok_in
        + (tokens_out + thoughts) * p.usd_per_mtok_out
        + cache_read * (p.usd_per_mtok_cache_read or 0.0)
    ) / 1_000_000.0
    return cost, "listprice"


# --- error classification -----------------------------------------------------
# Retry weather, never configuration. 408/429/5xx are load; 401/403/404/400 are
# a wrong project, a missing role, or a model name that does not exist in this
# location, all of which stay wrong for every one of the retries.

RETRYABLE_HTTP_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})


def is_transient(exc: BaseException) -> bool:
    """True only for errors a second attempt could plausibly survive.

    Unknown exception types are treated as fatal. In a wall-clock experiment the
    cost of not retrying something retryable is one lost iteration; the cost of
    retrying something permanent is minutes of dead time inside every iteration.
    """
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code in RETRYABLE_HTTP_CODES
    name = type(exc).__name__
    if name in {"ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout",
                "PoolTimeout", "TimeoutException", "RemoteProtocolError",
                "ReadError", "ConnectionError", "TimeoutError"}:
        return True
    return isinstance(exc, (ConnectionError, TimeoutError))


# --- usage extraction ---------------------------------------------------------


def usage_from_response(response: Any, model: str, latency_s: float,
                        price: ModelPrice | None = None) -> tuple[Usage, int]:
    """Build a :class:`Usage` from a ``GenerateContentResponse``.

    Returns the usage plus the thinking-token count, which has no home in the
    four-class ``Usage`` (that shape is Anthropic's) but is needed for cost and
    worth logging. Every count is read as "absent means absent": a missing
    ``usage_metadata`` yields zeros and an unknown cost, never a made-up one.
    """
    meta = getattr(response, "usage_metadata", None)
    raw_in = getattr(meta, "prompt_token_count", None)
    raw_out = getattr(meta, "candidates_token_count", None)
    tokens_in = int(raw_in or 0)
    tokens_out = int(raw_out or 0)
    cache_read = int(getattr(meta, "cached_content_token_count", 0) or 0)
    thoughts = int(getattr(meta, "thoughts_token_count", 0) or 0)
    # Vertex counts cached tokens INSIDE prompt_token_count; the four-class
    # ledger wants them disjoint, or billed_input double-counts the cache.
    tokens_in = max(0, tokens_in - cache_read)
    # An UNREPORTED count is not a zero count. Every call has a prompt, so an
    # absent prompt_token_count means the meter is missing, not that the turn
    # was free, and pricing those zeros would book a billed call at $0.00,
    # deflating $/improvement exactly the way an invented price would inflate
    # it. Same for an absent candidates count sitting next to a candidate that
    # visibly exists: that cost is unknown, not merely input-only.
    unreported = raw_in is None or (raw_out is None
                                    and bool(getattr(response, "candidates", None)))
    if unreported:
        cost, source = None, "unknown"
    else:
        cost, source = estimate_cost_usd(model, tokens_in, tokens_out, cache_read,
                                         thoughts, price)
    usage = Usage(
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_read=cache_read,
        cache_write=0,          # Vertex bills cache creation separately and does
                                # not report it per-call; 0 here means unreported.
        thoughts=thoughts,
        cost_usd=cost,
        cost_source=source,
        model=model,
        latency_s=latency_s,
    )
    return usage, thoughts


# --- the seat -----------------------------------------------------------------


@dataclass
class VertexSeat:
    """A Gemini model on Vertex AI, spoken to through ``google.genai``.

    Defaults are read from the environment (``GOOGLE_CLOUD_PROJECT``,
    ``GOOGLE_CLOUD_LOCATION``, ``LIVELANE_VERTEX_MODEL``), so no account value
    is hard-coded anywhere.
    """

    model: str = field(default_factory=lambda: os.environ.get("LIVELANE_VERTEX_MODEL", DEFAULT_MODEL))
    project: str | None = field(default_factory=lambda: os.environ.get("GOOGLE_CLOUD_PROJECT") or None)
    location: str = field(default_factory=lambda: os.environ.get("GOOGLE_CLOUD_LOCATION") or DEFAULT_LOCATION)
    name: str = "vertex"
    temperature: float | None = None
    #: Extended-thinking budget in tokens. 0 disables thinking, a positive
    #: value caps it, None leaves the model's own default alone.
    #: This is an EXPERIMENTAL VARIABLE, not a tuning knob: it must be
    #: identical across every arm of a comparison and recorded with the run,
    #: because it changes T_llm and therefore the ladder's dynamic range.
    thinking_budget: int | None = None
    max_output_tokens: int | None = None
    max_attempts: int = 4
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 30.0
    #: Set explicitly to price a SKU that is not in the table. Still "listprice".
    price: ModelPrice | None = None
    #: Diagnostics from the most recent call; never used for scoring.
    last_error: str | None = field(default=None, init=False, repr=False)
    last_finish_reason: str | None = field(default=None, init=False, repr=False)
    last_thoughts_tokens: int = field(default=0, init=False, repr=False)
    last_attempts: int = field(default=0, init=False, repr=False)
    _client: Any = field(default=None, init=False, repr=False)

    def client(self) -> Any:
        """Lazily build the Vertex client, so constructing a seat needs no creds."""
        if self._client is None:
            from google import genai

            if not self.project:
                raise RuntimeError(
                    "GOOGLE_CLOUD_PROJECT is unset; export it or set it in configs/gcp.local.env")
            self._client = genai.Client(vertexai=True, project=self.project,
                                        location=self.location)
        return self._client

    def _config(self, system: str, timeout_s: float) -> Any:
        from google.genai import types

        return types.GenerateContentConfig(
            # The system prompt goes in its own slot, not glued to the user text:
            # the arms must differ only in evaluator latency, and concatenation
            # would change the token layout as well as the role structure.
            system_instruction=system,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            http_options=types.HttpOptions(
                timeout=max(1, int(timeout_s * 1000)),
                # One attempt: the backoff in ask() is the only backoff.
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    def ask(self, system: str, user: str, timeout_s: float = 300.0) -> tuple[str, Usage]:
        """One turn. Returns ``("", Usage)`` if transient failures are exhausted.

        Fatal errors are raised, not swallowed: a 404 model name or a missing
        role would otherwise turn a whole sweep into a long row of empty turns
        that looks like a model that had nothing to say.
        """
        client = self.client()
        t0 = time.monotonic()
        deadline = t0 + timeout_s
        self.last_error = None
        self.last_finish_reason = None
        self.last_thoughts_tokens = 0
        self.last_attempts = 0

        for attempt in range(1, max(1, self.max_attempts) + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.last_error = "deadline exceeded before attempt"
                break
            self.last_attempts = attempt
            try:
                resp = client.models.generate_content(
                    model=self.model,
                    contents=user,
                    config=self._config(system, remaining),
                )
            except Exception as exc:  # noqa: BLE001, re-raised unless transient
                if not is_transient(exc):
                    raise
                self.last_error = f"{type(exc).__name__}: {exc}"
                sleep_s = self._backoff(attempt)
                if attempt >= self.max_attempts or time.monotonic() + sleep_s >= deadline:
                    break
                time.sleep(sleep_s)
                continue

            latency = time.monotonic() - t0
            usage, thoughts = usage_from_response(resp, self.model, latency, self.price)
            self.last_thoughts_tokens = thoughts
            cands = getattr(resp, "candidates", None) or []
            if cands:
                self.last_finish_reason = str(getattr(cands[0], "finish_reason", None))
            # An empty answer is NOT retried: it is usually a MAX_TOKENS cut or a
            # safety stop, both of which repeat. The tokens were still billed, so
            # the real usage is returned with the empty text.
            return (getattr(resp, "text", None) or ""), usage

        return "", Usage(model=self.model, cost_source="unknown",
                         latency_s=time.monotonic() - t0)

    def _backoff(self, attempt: int) -> float:
        """Exponential with jitter, capped. Jitter de-synchronises parallel arms."""
        base = min(self.max_backoff_s, self.initial_backoff_s * (2.0 ** (attempt - 1)))
        return base * random.uniform(0.5, 1.0)


__all__ = ["VertexSeat", "ModelPrice", "VERTEX_LIST_PRICE_USD_PER_MTOK",
           "estimate_cost_usd", "normalise_model", "is_transient",
           "usage_from_response", "DEFAULT_MODEL", "DEFAULT_LOCATION"]


if __name__ == "__main__":
    # Self-test. Everything below the divider except the final block is offline
    # and free; the last block makes exactly ONE billed call.
    import types as _pytypes

    from livelane.agent.llm import Seat

    print("=== VertexSeat self-test ===")

    assert normalise_model("publishers/google/models/Gemini-2.5-Flash") == "gemini-2.5-flash"
    assert normalise_model("models/gemini-2.5-pro@001") == "gemini-2.5-pro"
    # A dated preview SKU must NOT collapse onto the GA name it prefixes.
    assert normalise_model("gemini-2.5-pro-preview-05-06") != "gemini-2.5-pro"

    # Exact list-price arithmetic, checked by hand: 1M in + 1M out on 2.5-flash.
    cost, src = estimate_cost_usd("gemini-2.5-flash", 1_000_000, 1_000_000)
    assert src == "listprice" and abs(cost - 2.80) < 1e-9, (cost, src)
    # Thinking tokens are billed at the output rate even though tokens_out omits them.
    cost_t, _ = estimate_cost_usd("gemini-2.5-flash", 0, 0, 0, 1_000_000)
    assert abs(cost_t - 2.50) < 1e-9, cost_t

    # Fail-closed paths: never a fabricated dollar.
    assert estimate_cost_usd(DEFAULT_MODEL, 10, 10) == (None, "unknown"), \
        "gemini-3.8-flash must stay unpriced until its list price is verified"
    assert estimate_cost_usd("no-such-model", 10, 10) == (None, "unknown")
    assert estimate_cost_usd("gemini-2.5-pro", 300_000, 10)[0] is None, "past priced tier"
    nocache = ModelPrice(1.0, 2.0, None)
    assert estimate_cost_usd("x", 10, 10, cache_read=5, price=nocache)[0] is None
    assert estimate_cost_usd("x", 10, 10, price=nocache)[1] == "listprice"

    # Retry classification: weather retries, configuration does not.
    class _Err(Exception):
        def __init__(self, code: int) -> None:
            super().__init__(str(code))
            self.code = code

    for c in (408, 429, 500, 502, 503, 504):
        assert is_transient(_Err(c)), c
    for c in (400, 401, 403, 404, 422):
        assert not is_transient(_Err(c)), c
    assert is_transient(TimeoutError("slow")) and is_transient(ConnectionError("down"))
    assert not is_transient(ValueError("nonsense")), "unknown errors must not be retried"

    # Usage extraction, with cached tokens carved out of the prompt count so
    # billed_input does not double-count them.
    stub = _pytypes.SimpleNamespace(
        usage_metadata=_pytypes.SimpleNamespace(
            prompt_token_count=1000, candidates_token_count=40,
            cached_content_token_count=600, thoughts_token_count=7),
        text="hi", candidates=[])
    u, th = usage_from_response(stub, "gemini-2.5-flash", 1.5)
    assert (u.tokens_in, u.cache_read, u.tokens_out, th) == (400, 600, 40, 7), u
    assert u.billed_input == 1000, u.billed_input
    assert u.cost_source == "listprice" and u.cost_usd is not None

    # A response with no usage_metadata reports zeros and an unknown cost, not a
    # zero cost. Asserted on a PRICED SKU as well: on the unpriced default this
    # passes for the wrong reason (unknown model) and hides a fabricated $0.00.
    bare = _pytypes.SimpleNamespace(usage_metadata=None, text=None, candidates=[])
    for _m in (DEFAULT_MODEL, "gemini-2.5-flash"):
        u0, _ = usage_from_response(bare, _m, 0.1)
        assert u0.tokens_in == 0 and u0.cost_usd is None and u0.cost_source == "unknown", _m

    # Metadata present but every count null, same rule: unreported is not free.
    _null = _pytypes.SimpleNamespace(
        usage_metadata=_pytypes.SimpleNamespace(
            prompt_token_count=None, candidates_token_count=None,
            cached_content_token_count=None, thoughts_token_count=None),
        text="", candidates=[])
    assert usage_from_response(_null, "gemini-2.5-flash", 0.1)[0].cost_usd is None

    # A candidate came back but its output tokens went uncounted: price it as
    # unknown, not as a listprice figure that silently omits billed output.
    _half = _pytypes.SimpleNamespace(
        usage_metadata=_pytypes.SimpleNamespace(
            prompt_token_count=500, candidates_token_count=None,
            cached_content_token_count=None, thoughts_token_count=None),
        text="hi", candidates=[_pytypes.SimpleNamespace(finish_reason="STOP")])
    _uh, _ = usage_from_response(_half, "gemini-2.5-flash", 0.1)
    assert _uh.tokens_in == 500 and _uh.cost_usd is None and _uh.cost_source == "unknown"

    # Retry behaviour, driven through a stub client so it costs nothing: a
    # transient error is retried up to max_attempts and then LOSES THE TURN
    # rather than raising, while a fatal one is re-raised on the first attempt.
    _calls = {"n": 0}

    class _Flaky:
        def __init__(self, code: int) -> None:
            self.code = code

        def generate_content(self, **kw: Any) -> Any:
            _calls["n"] += 1
            raise _Err(self.code)

    flaky = VertexSeat(model="gemini-2.5-flash", project="p", max_attempts=3,
                       initial_backoff_s=0.01, max_backoff_s=0.05)
    flaky._client = _pytypes.SimpleNamespace(models=_Flaky(503))
    text_f, usage_f = flaky.ask("s", "u", timeout_s=30.0)
    assert text_f == "" and _calls["n"] == 3, (text_f, _calls)
    assert usage_f.cost_usd is None and usage_f.cost_source == "unknown"

    _calls["n"] = 0
    fatal = VertexSeat(model="gemini-2.5-flash", project="p", max_attempts=3,
                       initial_backoff_s=0.01)
    fatal._client = _pytypes.SimpleNamespace(models=_Flaky(403))
    try:
        fatal.ask("s", "u", timeout_s=30.0)
        raise AssertionError("a 403 must propagate, not be retried away")
    except _Err:
        assert _calls["n"] == 1, _calls

    # A backoff longer than the caller's budget must not be slept through.
    _calls["n"] = 0
    slow = VertexSeat(model="gemini-2.5-flash", project="p", max_attempts=8,
                      initial_backoff_s=5.0, max_backoff_s=5.0)
    slow._client = _pytypes.SimpleNamespace(models=_Flaky(429))
    _t0 = time.monotonic()
    assert slow.ask("s", "u", timeout_s=1.0)[0] == ""
    assert time.monotonic() - _t0 < 2.0, "overran the caller's timeout budget"

    seat = VertexSeat()
    assert isinstance(seat, Seat), "VertexSeat must satisfy the Seat protocol"
    print(f"    offline checks passed; seat={seat.name} model={seat.model} "
          f"project={seat.project} location={seat.location}")

    # --- ONE real, billed call -------------------------------------------------
    if os.environ.get("LIVELANE_NO_NET") == "1":
        print("    SKIPPED live call (LIVELANE_NO_NET=1) -- NOT a pass")
    else:
        text, usage = seat.ask("You are a test harness. Reply with exactly one word.",
                               "Reply with exactly: PONG", timeout_s=120.0)
        print(f"    live reply={text!r} attempts={seat.last_attempts} "
              f"finish={seat.last_finish_reason}")
        print(f"    tokens_in={usage.tokens_in} tokens_out={usage.tokens_out} "
              f"cache_read={usage.cache_read} thoughts={seat.last_thoughts_tokens} "
              f"cost_usd={usage.cost_usd} cost_source={usage.cost_source} "
              f"latency={usage.latency_s:.2f}s")
        assert text.strip(), "empty reply from the live seat"
        assert usage.tokens_in > 0, "no prompt tokens reported"
        assert usage.model == seat.model
        assert usage.cost_source in {"listprice", "unknown"}, usage.cost_source

    print("=== VertexSeat self-test passed ===")
