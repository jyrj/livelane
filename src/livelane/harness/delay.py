"""Controlled feedback-latency injection, LiveLane's independent variable.

The primary experiment is arm ``I(d)``: the *identical* lane-S tool stack with
``d`` seconds of artificial delay added to every evaluation.  Because the tools
are literally the same binaries on the same inputs, any outcome difference
between ``I(0)`` and ``I(600)`` is attributable to latency and to nothing else.
That is the whole point, and it is why this module is deliberately boring and
heavily asserted.

Latency semantics (fixed)
------------------------
``ADDITIVE`` is the primary mode: ``observed = real + d``.

The alternative, ``FLOOR`` (``observed = max(real, d)``), was considered and
rejected as the primary.  Both satisfy the required ``I(0) == S``, and on the
small blocks they are indistinguishable (real is ~3 s, so ``30 + 3`` vs ``30``).
But on the medium block real evaluation is ~1448 s, where ``FLOOR`` at every
``d`` in {30, 120, 600} collapses to *no delay at all* and the arm silently
stops being an experiment.  ``ADDITIVE`` keeps the treatment well-defined at
every design scale.  ``FLOOR`` is retained only for a planned robustness check.

Failures are delayed too.  A slow evaluator is slow whether the candidate
passes or fails, and exempting failures would hand the high-delay arms a
discount precisely on their cheapest iterations, biasing the headline result
in favour of the hypothesis.  ``apply_on_failure=False`` exists to make that
choice explicit and testable, never as the default.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


# ONE DelayMode for the whole project. This module and
# ``chia_livelane.delay_node`` both express the same treatment; defining the enum
# twice made ``mode is DelayMode.FLOOR`` silently false when a policy built with
# one class met a comparison against the other. Import, do not redefine.
from chia_livelane.delay_node import DelayMode  # noqa: E402,F401


@dataclass(frozen=True)
class DelayPolicy:
    """How much latency to inject, and under what rules."""

    seconds: float = 0.0
    mode: DelayMode = DelayMode.ADDITIVE
    apply_on_failure: bool = True
    # Replay/analysis only: account for the delay without spending wall-clock.
    # Any run using this is NOT a valid latency measurement and is flagged as such.
    virtual: bool = False

    def __post_init__(self) -> None:
        if self.seconds < 0:
            raise ValueError(f"delay seconds must be >= 0, got {self.seconds}")

    @property
    def arm_name(self) -> str:
        """Canonical arm label, e.g. 'I(0)', 'I(600)'."""
        s = int(self.seconds) if float(self.seconds).is_integer() else self.seconds
        return f"I({s})"

    @property
    def is_identity(self) -> bool:
        """I(0) is exactly lane S: no sleep, no accounting difference."""
        return self.seconds == 0.0

    def sleep_for(self, real_elapsed_s: float, *, succeeded: bool = True) -> float:
        """Seconds to sleep after an evaluation that really took ``real_elapsed_s``."""
        if self.seconds == 0.0:
            return 0.0
        if not succeeded and not self.apply_on_failure:
            return 0.0
        if self.mode is DelayMode.ADDITIVE:
            return self.seconds
        if self.mode is DelayMode.FLOOR:
            return max(0.0, self.seconds - real_elapsed_s)
        raise AssertionError(f"unhandled delay mode {self.mode!r}")


@dataclass
class DelayRecord:
    """What actually happened, for the per-iteration ledger."""

    policy_seconds: float
    mode: str
    real_elapsed_s: float
    slept_s: float
    virtual: bool
    succeeded: bool
    interrupted: bool = False

    @property
    def observed_latency_s(self) -> float:
        """What the agent actually waited for."""
        return self.real_elapsed_s + self.slept_s


class DelayInjector:
    """Applies a :class:`DelayPolicy`, interruptibly.

    A 600 s sleep must not make a run un-cancellable, so the wait is done on a
    :class:`threading.Event` rather than ``time.sleep``: setting ``cancel`` cuts
    it short and the record says the delay was interrupted, so that iteration
    can be excluded from analysis instead of quietly under-counting latency.
    """

    def __init__(self, policy: DelayPolicy, *, verbose: bool = True) -> None:
        self.policy = policy
        self.verbose = verbose
        self._cancel = threading.Event()
        self.records: list[DelayRecord] = []

    def cancel(self) -> None:
        self._cancel.set()

    def reset(self) -> None:
        self._cancel.clear()

    def apply(self, real_elapsed_s: float, *, succeeded: bool = True) -> DelayRecord:
        want = self.policy.sleep_for(real_elapsed_s, succeeded=succeeded)
        interrupted = False
        slept = 0.0

        if want > 0.0 and self.policy.virtual:
            slept = want  # accounted, not spent
            if self.verbose:
                print(f"      [delay] VIRTUAL {want:.1f}s (replay: no wall-clock spent)", flush=True)
        elif want > 0.0:
            if self.verbose:
                print(
                    f"      [delay] injecting {want:.1f}s "
                    f"({self.policy.mode.value}, real={real_elapsed_s:.2f}s)",
                    flush=True,
                )
            t0 = time.monotonic()
            # Event.wait returns True only if the event was set -> cancelled.
            interrupted = self._cancel.wait(timeout=want)
            slept = time.monotonic() - t0

        rec = DelayRecord(
            policy_seconds=self.policy.seconds,
            mode=self.policy.mode.value,
            real_elapsed_s=real_elapsed_s,
            slept_s=slept,
            virtual=self.policy.virtual,
            succeeded=succeeded,
            interrupted=interrupted,
        )
        self.records.append(rec)
        return rec

    @contextmanager
    def around(self, *, succeeded_getter=lambda: True) -> Iterator[None]:
        """Time a block, then inject the delay after it, never before.

        Order matters: the delay models a *slow evaluator*, so the agent must
        wait for real work and then keep waiting. Sleeping first would let the
        real work overlap nothing and would not be the same treatment.
        """
        t0 = time.monotonic()
        try:
            yield
        finally:
            real = time.monotonic() - t0
            self.apply(real, succeeded=bool(succeeded_getter()))


# The fixed arm ladder.
INJECTION_LADDER_S: tuple[float, ...] = (0.0, 30.0, 120.0, 600.0)


def ladder_policies(
    mode: DelayMode = DelayMode.ADDITIVE, virtual: bool = False
) -> list[DelayPolicy]:
    return [DelayPolicy(seconds=s, mode=mode, virtual=virtual) for s in INJECTION_LADDER_S]


if __name__ == "__main__":
    print("=== delay self-test ===")

    # I(0) must be exactly lane S: zero sleep, no observable difference.
    p0 = DelayPolicy(0.0)
    assert p0.is_identity and p0.sleep_for(3.0) == 0.0 and p0.arm_name == "I(0)"

    # ADDITIVE adds regardless of how long the real evaluation took.
    pa = DelayPolicy(30.0, DelayMode.ADDITIVE)
    assert pa.sleep_for(3.0) == 30.0
    assert pa.sleep_for(1448.0) == 30.0, "additive must stay a treatment on slow designs"

    # FLOOR degenerates on the slow design, the documented reason it is not primary.
    pf = DelayPolicy(600.0, DelayMode.FLOOR)
    assert pf.sleep_for(3.0) == 597.0
    assert pf.sleep_for(1448.0) == 0.0, "floor collapses on the medium block, as documented"

    # Failures are delayed by default; opting out must be explicit.
    assert DelayPolicy(30.0).sleep_for(2.0, succeeded=False) == 30.0
    assert DelayPolicy(30.0, apply_on_failure=False).sleep_for(2.0, succeeded=False) == 0.0

    try:
        DelayPolicy(-1.0)
        raise AssertionError("negative delay must be rejected")
    except ValueError as e:
        print(f"    negative delay rejected: {e}")

    # Real sleep is really spent.
    inj = DelayInjector(DelayPolicy(0.4))
    t0 = time.monotonic()
    rec = inj.apply(0.1)
    spent = time.monotonic() - t0
    assert 0.35 < spent < 1.0, f"expected ~0.4s sleep, spent {spent}"
    assert abs(rec.observed_latency_s - 0.5) < 0.15, rec

    # Virtual mode accounts without spending, and is flagged.
    vinj = DelayInjector(DelayPolicy(600.0, virtual=True))
    t0 = time.monotonic()
    vrec = vinj.apply(1.0)
    assert time.monotonic() - t0 < 0.5, "virtual delay must not sleep"
    assert vrec.slept_s == 600.0 and vrec.virtual is True

    # A long delay stays cancellable.
    cinj = DelayInjector(DelayPolicy(600.0))
    threading.Timer(0.3, cinj.cancel).start()
    t0 = time.monotonic()
    crec = cinj.apply(0.0)
    spent = time.monotonic() - t0
    assert spent < 5.0, f"cancel did not interrupt a 600s delay: {spent}"
    assert crec.interrupted is True, "interrupted delays must be flagged, not silently short"
    print(f"    600s delay cancelled after {spent:.2f}s, flagged interrupted")

    # The context manager delays AFTER the work, not before.
    order: list[str] = []
    inj2 = DelayInjector(DelayPolicy(0.2), verbose=False)
    with inj2.around():
        order.append("work")
    order.append("after")
    assert order == ["work", "after"]
    assert inj2.records[-1].real_elapsed_s < 0.1

    assert [p.arm_name for p in ladder_policies()] == ["I(0)", "I(30)", "I(120)", "I(600)"]
    print("=== all delay self-tests passed ===")
