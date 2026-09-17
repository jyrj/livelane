"""Controlled feedback-latency injection for any CHIA evaluator.

Why this exists
---------------
CHIA's paper argues from Amdahl's law that parallelism cannot fix *evaluation
latency*, and every agentic hardware loop in the literature treats evaluator
wall-clock as a fixed cost.  Nobody varies it and measures the effect, because
doing so normally means owning two evaluators of different speed.

``DelayNode`` removes that requirement.  It wraps *any* evaluator so the loop
observes a controlled amount of extra latency with the tools left literally
identical, which lets a CHIA flow answer two questions it cannot answer today:

* "what does my result look like if feedback takes 10x longer?"
* read the other way: "how much could a faster evaluator buy me, before anyone
  builds one?"

Semantics
---------
``ADDITIVE`` (default): ``observed = real + seconds``.  ``DelayNode(0)`` is
exactly the undelayed evaluator, so it doubles as the control arm at zero cost.

``FLOOR``: ``observed = max(real, seconds)`` -- "an evaluator that always takes
at least this long".  Note it becomes a no-op whenever the real evaluation
already exceeds ``seconds``, so on a slow design a small delay silently stops
being a treatment.  ADDITIVE is the safer default for experiments.

Failures are delayed by default.  A slow evaluator is slow whether the candidate
passes or fails, and exempting failures quietly discounts the cheapest
iterations of the slowest arms.

Placement
---------
:func:`delay_seconds` is declared ``num_cpus=0`` and asks for no custom
resource.  Both are deliberate: the node does nothing but block, so charging it
a Ray CPU slot (the ``ray.remote`` default is ``num_cpus=1``) would make a
long injected delay evict real work from the cluster.  Being resource-free also
means it needs no image of its own -- it runs in whichever container the caller
is already using.  A caller that wants it pinned can still say
``delay_seconds.options(resources={"tok": 1})``.

Usage
-----
::

    from chia.base.delay import DelayNode, DelayMode

    delay = DelayNode(seconds=600)              # or DelayNode(0) for the control

    # 1. wrap a call
    result, record = delay.around_call(evaluate_candidate, design, top)

    # 2. or use it as a node in a graph
    get(delay_seconds.chia_remote(600.0))
"""

from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

try:  # CHIA is optional: the semantics are unit-testable without a cluster.
    from chia.base.ChiaFunction import ChiaFunction
except Exception:  # pragma: no cover - exercised only outside a CHIA install
    def ChiaFunction(**_kwargs):  # type: ignore[misc]
        def deco(fn):
            return fn
        return deco

T = TypeVar("T")


class DelayMode(str, enum.Enum):
    """How an injected delay combines with the evaluator's real wall-clock."""

    #: ``observed = real + seconds``.
    ADDITIVE = "additive"
    #: ``observed = max(real, seconds)``.
    FLOOR = "floor"


@dataclass
class DelayRecord:
    """What actually happened on one injection, for the caller's ledger.

    Attributes:
        requested_s (float): The delay the node was configured with.
        mode (str): :class:`DelayMode` value in force for this injection.
        real_elapsed_s (float): Wall-clock the wrapped work actually took.
        slept_s (float): Wall-clock actually spent sleeping. Differs from
            ``requested_s`` under ``FLOOR``, on cancellation, and when
            ``virtual`` is set.
        succeeded (bool): Whether the wrapped work reported success.
        interrupted (bool): The sleep was cut short by :meth:`DelayNode.cancel`.
        virtual (bool): The delay was accounted but never slept. A record with
            this set is NOT a latency measurement.
    """

    requested_s: float
    mode: str
    real_elapsed_s: float
    slept_s: float
    succeeded: bool
    interrupted: bool = False
    virtual: bool = False

    @property
    def observed_latency_s(self) -> float:
        """Total latency the caller waited: real work plus injected delay."""
        return self.real_elapsed_s + self.slept_s


@dataclass
class DelayNode:
    """Inject a controlled amount of extra feedback latency around real work.

    Attributes:
        seconds (float): Delay to inject. ``0.0`` makes the node an exact
            identity, which is what the control arm of an experiment uses.
        mode (DelayMode): ``ADDITIVE`` (default) or ``FLOOR``.
        apply_on_failure (bool): Delay failed work too. Default True; turning it
            off biases an experiment toward whichever arm fails more often.
        virtual (bool): Account for the delay without spending wall-clock. For
            cached/replayed runs only -- every record it produces is flagged
            ``virtual=True`` so analysis can drop it.
        verbose (bool): Print each injection as it starts.
    """

    seconds: float = 0.0
    mode: DelayMode = DelayMode.ADDITIVE
    apply_on_failure: bool = True
    virtual: bool = False
    verbose: bool = True

    records: list[DelayRecord] = field(default_factory=list, repr=False)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def __post_init__(self) -> None:
        if self.seconds < 0:
            raise ValueError(f"delay seconds must be >= 0, got {self.seconds}")

    # -- introspection --------------------------------------------------------
    @property
    def is_identity(self) -> bool:
        """True when this node adds nothing at all (the control arm)."""
        return self.seconds == 0.0

    @property
    def label(self) -> str:
        """Short arm label, e.g. ``delay(600s,additive)``."""
        s = int(self.seconds) if float(self.seconds).is_integer() else self.seconds
        return f"delay({s}s,{self.mode.value})"

    def sleep_for(self, real_elapsed_s: float, succeeded: bool = True) -> float:
        """How long this node would sleep, without sleeping.

        Args:
            real_elapsed_s (float): Wall-clock the real work took.
            succeeded (bool): Whether the real work succeeded.

        Returns:
            float: Seconds that :meth:`apply` would sleep for.
        """
        if self.seconds == 0.0:
            return 0.0
        if not succeeded and not self.apply_on_failure:
            return 0.0
        if self.mode is DelayMode.ADDITIVE:
            return self.seconds
        return max(0.0, self.seconds - real_elapsed_s)

    # -- control --------------------------------------------------------------
    def cancel(self) -> None:
        """Cut a pending delay short. The resulting record is ``interrupted``."""
        self._cancel.set()

    def reset(self) -> None:
        """Clear a previous :meth:`cancel` so later injections sleep again."""
        self._cancel.clear()

    # -- application ----------------------------------------------------------
    def apply(self, real_elapsed_s: float, succeeded: bool = True) -> DelayRecord:
        """Sleep for this node's delay and record what happened.

        Args:
            real_elapsed_s (float): Wall-clock the real work took, used by
                ``FLOOR`` mode and recorded either way.
            succeeded (bool): Whether the real work succeeded.

        Returns:
            DelayRecord: The injection record, also appended to :attr:`records`.
        """
        want = self.sleep_for(real_elapsed_s, succeeded)
        slept, interrupted = 0.0, False
        if want > 0.0 and self.virtual:
            slept = want
        elif want > 0.0:
            if self.verbose:
                print(f"[DelayNode] injecting {want:.1f}s "
                      f"({self.mode.value}; real={real_elapsed_s:.2f}s)", flush=True)
            t0 = time.monotonic()
            interrupted = self._cancel.wait(timeout=want)
            slept = time.monotonic() - t0
        rec = DelayRecord(self.seconds, self.mode.value, real_elapsed_s, slept,
                          succeeded, interrupted, self.virtual)
        self.records.append(rec)
        return rec

    def around_call(self, fn: Callable[..., T], *args: Any,
                    succeeded: Callable[[T], bool] = lambda r: True,
                    **kwargs: Any) -> tuple[T, DelayRecord]:
        """Run ``fn``, then inject the delay -- never before.

        Order is not cosmetic: the delay models a *slow evaluator*, so the caller
        must wait for the real work and then keep waiting. Injecting first would
        model a queue, which is a different treatment.

        A raising ``fn`` is still delayed and still recorded (a crashed
        evaluation consumed the evaluator's latency), then the exception is
        re-raised unchanged.

        Args:
            fn (Callable): The real work to run.
            *args: Positional arguments forwarded to ``fn``.
            succeeded (Callable): Maps ``fn``'s return value to a success bool,
                used only by ``apply_on_failure``.
            **kwargs: Keyword arguments forwarded to ``fn``.

        Returns:
            tuple: ``(fn's return value, DelayRecord)``.
        """
        t0 = time.monotonic()
        ok = True
        try:
            result = fn(*args, **kwargs)
            ok = bool(succeeded(result))
            return result, self.apply(time.monotonic() - t0, ok)
        except BaseException:
            self.apply(time.monotonic() - t0, False)
            raise

    # -- aggregate ------------------------------------------------------------
    @property
    def total_injected_s(self) -> float:
        """Sum of wall-clock actually slept across every injection so far."""
        return sum(r.slept_s for r in self.records)

    def summary(self) -> dict[str, Any]:
        """Ledger-ready summary of every injection this node has made.

        Returns:
            dict: ``requested_s``, ``mode``, ``calls``, ``total_injected_s``,
            ``interrupted`` (count) and ``virtual``.
        """
        return {
            "requested_s": self.seconds,
            "mode": self.mode.value,
            "calls": len(self.records),
            "total_injected_s": self.total_injected_s,
            "interrupted": sum(1 for r in self.records if r.interrupted),
            "virtual": self.virtual,
        }


# num_cpus=0 is load-bearing, not a micro-optimisation. ray.remote defaults to
# num_cpus=1, so a 600 s injected delay would hold a CPU slot for ten minutes
# and evict real work from the cluster -- which would make the injected latency
# perturb throughput as well, confounding exactly the measurement this node
# exists to make. The node only blocks; it must cost nothing schedulable.
@ChiaFunction(num_cpus=0)
def delay_seconds(seconds: float) -> float:
    """Sleep for a fixed number of seconds as a step in a CHIA graph.

    Use this when the delay belongs *between* two nodes of a graph rather than
    around a single Python call; use :class:`DelayNode` when it belongs around
    one call and you want a record of it.

    Holds no CPU and no custom resource, so it neither burns cluster capacity
    nor needs a container image of its own.

    Args:
        seconds (float): How long to sleep. Must be >= 0; ``0.0`` returns
            immediately and is the control arm.

    Returns:
        float: Wall-clock seconds actually slept, measured with a monotonic
        clock so it is immune to NTP steps.

    Raises:
        ValueError: If ``seconds`` is negative.
    """
    if seconds < 0:
        raise ValueError(f"delay seconds must be >= 0, got {seconds}")
    t0 = time.monotonic()
    time.sleep(seconds)
    return time.monotonic() - t0


__all__ = ["DelayNode", "DelayMode", "DelayRecord", "delay_seconds"]


if __name__ == "__main__":
    print("=== chia.base.delay self-test ===")

    # ADDITIVE: observed = real + seconds, and the node is the control at 0.
    d = DelayNode(seconds=0.25, verbose=False)
    t0 = time.monotonic()
    val, rec = d.around_call(lambda: sum(range(100000)))
    took = time.monotonic() - t0
    assert val == sum(range(100000))
    assert 0.25 <= took < 1.0, f"additive delay not injected: {took}"
    assert rec.slept_s >= 0.24, rec
    print(f"    ADDITIVE: real={rec.real_elapsed_s:.3f}s slept={rec.slept_s:.3f}s "
          f"observed={rec.observed_latency_s:.3f}s")

    ident = DelayNode(0.0, verbose=False)
    assert ident.is_identity
    t0 = time.monotonic()
    ident.around_call(lambda: 1)
    assert time.monotonic() - t0 < 0.05, "DelayNode(0) is not an identity"
    print("    DelayNode(0) is an exact identity (the control arm)")

    # FLOOR collapses once real work exceeds the floor -- the documented trap.
    f = DelayNode(seconds=1.0, mode=DelayMode.FLOOR, verbose=False)
    assert abs(f.sleep_for(0.25) - 0.75) < 1e-9, f.sleep_for(0.25)
    assert f.sleep_for(2.0) == 0.0, "FLOOR must be a no-op past the floor"
    print("    FLOOR: sleep_for(0.25)=0.75, sleep_for(2.0)=0.0 (no-op past floor)")

    # Failures are delayed by default; opting out is explicit.
    assert DelayNode(1.0, verbose=False).sleep_for(0.1, succeeded=False) == 1.0
    assert DelayNode(1.0, apply_on_failure=False,
                     verbose=False).sleep_for(0.1, succeeded=False) == 0.0
    print("    failures delayed by default; apply_on_failure=False opts out")

    # A raising evaluator is still charged its latency, and the error propagates.
    boom = DelayNode(seconds=0.15, verbose=False)
    try:
        boom.around_call(lambda: (_ for _ in ()).throw(RuntimeError("tool died")))
        raise AssertionError("exception was swallowed")
    except RuntimeError as e:
        assert str(e) == "tool died"
    assert len(boom.records) == 1 and boom.records[0].slept_s >= 0.14
    assert boom.records[0].succeeded is False
    print(f"    crashed evaluation still charged {boom.records[0].slept_s:.3f}s")

    # cancel() cuts a long sleep short and says so.
    c = DelayNode(seconds=30.0, verbose=False)
    threading.Timer(0.2, c.cancel).start()
    t0 = time.monotonic()
    r = c.apply(0.0)
    assert r.interrupted and time.monotonic() - t0 < 5.0, r
    print(f"    cancel() interrupted a 30s sleep after {r.slept_s:.3f}s")

    # virtual accounting sleeps for nothing and flags every record.
    v = DelayNode(seconds=600.0, virtual=True, verbose=False)
    t0 = time.monotonic()
    rv = v.apply(1.0)
    assert time.monotonic() - t0 < 0.05 and rv.slept_s == 600.0 and rv.virtual
    print("    virtual=True accounts 600s without spending it, and flags it")

    # The graph-step form validates its input and really sleeps.
    t0 = time.monotonic()
    slept = delay_seconds(0.2)
    assert 0.2 <= slept < 1.0 and time.monotonic() - t0 >= 0.2, slept
    try:
        delay_seconds(-1.0)
        raise AssertionError("negative delay accepted")
    except ValueError as e:
        print(f"    delay_seconds rejects negatives: {e}")

    try:
        DelayNode(seconds=-1.0)
        raise AssertionError("negative DelayNode accepted")
    except ValueError:
        pass

    print("=== all chia.base.delay self-tests passed ===")
