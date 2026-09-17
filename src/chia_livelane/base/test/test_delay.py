"""Unit tests for the delay node: the treatment it applies must be exact.

A latency-injection node is only useful if the injected latency is the *only*
thing that changes, so these tests pin the arithmetic, the control arm, the
failure path and the accounting rather than just checking that it sleeps.
"""

import threading
import time

import pytest

from chia_livelane.base.delay import (DelayMode, DelayNode, DelayRecord,
                                      delay_seconds)


class TestDelayArithmetic:
    def test_additive_adds_to_real_time(self):
        assert DelayNode(2.0, verbose=False).sleep_for(0.5) == 2.0

    def test_additive_is_independent_of_real_time(self):
        d = DelayNode(2.0, verbose=False)
        assert d.sleep_for(0.0) == d.sleep_for(100.0) == 2.0

    def test_floor_tops_up_to_the_floor(self):
        d = DelayNode(1.0, mode=DelayMode.FLOOR, verbose=False)
        assert d.sleep_for(0.25) == pytest.approx(0.75)

    def test_floor_becomes_a_noop_past_the_floor(self):
        # The documented trap: on a slow design a small FLOOR delay silently
        # stops being a treatment at all.
        d = DelayNode(1.0, mode=DelayMode.FLOOR, verbose=False)
        assert d.sleep_for(2.0) == 0.0

    def test_zero_is_an_exact_identity(self):
        d = DelayNode(0.0, verbose=False)
        assert d.is_identity
        assert d.sleep_for(5.0) == 0.0

    def test_negative_delay_rejected(self):
        with pytest.raises(ValueError):
            DelayNode(-1.0)


class TestFailureHandling:
    def test_failures_are_delayed_by_default(self):
        # Exempting failures would quietly discount the cheapest iterations of
        # the slowest arms.
        assert DelayNode(3.0, verbose=False).sleep_for(0.1, succeeded=False) == 3.0

    def test_failures_can_be_exempted_explicitly(self):
        d = DelayNode(3.0, apply_on_failure=False, verbose=False)
        assert d.sleep_for(0.1, succeeded=False) == 0.0
        assert d.sleep_for(0.1, succeeded=True) == 3.0

    def test_raising_work_is_still_charged_and_still_raises(self):
        d = DelayNode(0.05, verbose=False)

        def boom():
            raise RuntimeError("tool died")

        with pytest.raises(RuntimeError, match="tool died"):
            d.around_call(boom)
        assert len(d.records) == 1
        assert d.records[0].succeeded is False
        assert d.records[0].slept_s >= 0.04


class TestOrdering:
    def test_delay_comes_after_the_real_work(self):
        # The delay models a slow evaluator, not a queue: the caller must wait
        # for the real work and then keep waiting.
        order = []
        d = DelayNode(0.05, verbose=False)
        d.around_call(lambda: order.append("work"))
        order.append("after")
        assert order == ["work", "after"]
        assert d.records[0].real_elapsed_s < d.records[0].slept_s * 10


class TestAccounting:
    def test_observed_latency_is_real_plus_slept(self):
        r = DelayRecord(5.0, "additive", 2.0, 5.0, True)
        assert r.observed_latency_s == 7.0

    def test_summary_counts_every_call(self):
        d = DelayNode(0.01, verbose=False)
        for _ in range(3):
            d.apply(0.0)
        s = d.summary()
        assert s["calls"] == 3
        assert s["requested_s"] == 0.01
        assert s["mode"] == "additive"
        assert s["total_injected_s"] == pytest.approx(d.total_injected_s)

    def test_virtual_accounts_without_spending(self):
        d = DelayNode(600.0, virtual=True, verbose=False)
        t0 = time.monotonic()
        r = d.apply(1.0)
        assert time.monotonic() - t0 < 0.5
        assert r.slept_s == 600.0
        # Every virtual record is flagged so analysis can drop it: a run using
        # this is NOT a latency measurement.
        assert r.virtual is True

    def test_label_identifies_the_arm(self):
        assert DelayNode(600.0).label == "delay(600s,additive)"
        assert DelayNode(1.5, mode=DelayMode.FLOOR).label == "delay(1.5s,floor)"


class TestCancellation:
    def test_cancel_cuts_a_long_sleep_short_and_says_so(self):
        d = DelayNode(30.0, verbose=False)
        threading.Timer(0.1, d.cancel).start()
        t0 = time.monotonic()
        r = d.apply(0.0)
        assert time.monotonic() - t0 < 5.0
        assert r.interrupted is True

    def test_reset_re_arms_the_sleep(self):
        d = DelayNode(0.05, verbose=False)
        d.cancel()
        assert d.apply(0.0).interrupted is True
        d.reset()
        assert d.apply(0.0).interrupted is False


class TestGraphStep:
    def test_delay_seconds_sleeps_and_reports_monotonic_time(self):
        t0 = time.monotonic()
        slept = delay_seconds(0.05)
        assert 0.05 <= slept < 2.0
        assert time.monotonic() - t0 >= 0.05

    def test_delay_seconds_rejects_negatives(self):
        with pytest.raises(ValueError):
            delay_seconds(-1.0)

    def test_delay_seconds_holds_no_cpu_slot(self):
        # num_cpus=0 is load-bearing: ray.remote defaults to 1, and a 600s
        # injected delay holding a CPU slot would evict real work from the
        # cluster and perturb the very throughput being measured.
        opts = getattr(delay_seconds, "_chia_options", None)
        if opts is None:          # running without CHIA installed
            pytest.skip("chia not installed; decorator is the no-op fallback")
        assert opts.get("num_cpus") == 0
