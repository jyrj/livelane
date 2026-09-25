"""Latency injection: the independent variable."""

import threading
import time

import pytest

from chia_livelane.delay_node import DelayMode, DelayNode
from livelane.harness.delay import (INJECTION_LADDER_S, DelayPolicy,
                                    DelayInjector, ladder_policies)


def test_identity_arm_is_exactly_lane_s():
    p = DelayPolicy(0.0)
    assert p.is_identity and p.sleep_for(3.0) == 0.0 and p.arm_name == "I(0)"


def test_additive_stays_a_treatment_on_slow_designs():
    p = DelayPolicy(30.0, DelayMode.ADDITIVE)
    assert p.sleep_for(3.0) == 30.0
    assert p.sleep_for(1448.0) == 30.0


def test_floor_collapses_on_slow_designs_which_is_why_it_is_not_primary():
    p = DelayPolicy(600.0, DelayMode.FLOOR)
    assert p.sleep_for(3.0) == 597.0
    assert p.sleep_for(1448.0) == 0.0


def test_failures_are_delayed_by_default():
    assert DelayPolicy(30.0).sleep_for(2.0, succeeded=False) == 30.0
    assert DelayPolicy(30.0, apply_on_failure=False).sleep_for(2.0, succeeded=False) == 0.0


def test_negative_delay_rejected():
    with pytest.raises(ValueError):
        DelayPolicy(-1.0)
    with pytest.raises(ValueError):
        DelayNode(seconds=-1.0)


def test_virtual_delay_is_accounted_but_not_spent_and_is_flagged():
    inj = DelayInjector(DelayPolicy(600.0, virtual=True), verbose=False)
    t0 = time.monotonic()
    rec = inj.apply(1.0)
    assert time.monotonic() - t0 < 0.5
    assert rec.slept_s == 600.0 and rec.virtual is True


def test_long_delay_is_cancellable_and_flagged():
    inj = DelayInjector(DelayPolicy(600.0), verbose=False)
    threading.Timer(0.2, inj.cancel).start()
    t0 = time.monotonic()
    rec = inj.apply(0.0)
    assert time.monotonic() - t0 < 5.0
    assert rec.interrupted is True


def test_delay_is_applied_after_the_work_not_before():
    order = []
    inj = DelayInjector(DelayPolicy(0.1), verbose=False)
    with inj.around():
        order.append("work")
    order.append("after")
    assert order == ["work", "after"]
    assert inj.records[-1].real_elapsed_s < 0.1


def test_fixed_ladder():
    assert INJECTION_LADDER_S == (0.0, 30.0, 120.0, 600.0)
    assert [p.arm_name for p in ladder_policies()] == ["I(0)", "I(30)", "I(120)", "I(600)"]


def test_node_delays_a_crashed_evaluation_too():
    d = DelayNode(seconds=0.1, verbose=False)
    with pytest.raises(RuntimeError):
        d.around_call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert len(d.records) == 1 and d.records[0].succeeded is False
