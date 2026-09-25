"""Which instances' TIMINGS are disqualified by contention on the box.

One rule, one place. It lives here rather than inside the harness because the
harness is not the only thing that divides two wall-clocks: the front-end
comparator does too, and a guard that exists at one call site and not the other
is exactly how a contaminated number ships. Both import this.

The rule is: an instance's PEAK load against the run's MEDIAN peak, at 1.5x.

The obvious alternative, a >=2x rise within one instance, was tried and got
both directions wrong on the same pair of runs:

  * ibex #167 rose only 1.85x, under the threshold, but rose to 22.2 against a
    run baseline of 13.3 because a recursive grep was started on the same box.
    Its wall came out 45.0s against 21-22s for its neighbours. Passed.
  * ibex #48 "rose" 9.8x, from 1.3 to 12.7, purely because it was the FIRST
    instance and sampled an idle box before the harness's own 12 jobs started.
    12.7 is that run's normal. Flagged.

A within-instance ratio cannot tell "something else started" from "we started",
and it is blind to an instance that was contended for its whole duration, whose
start/end ratio is 1.0. Comparing against the run's own steady state handles
all three.
"""
from __future__ import annotations

# A run needs a few instances before its median peak means anything. Below
# this, nothing is disqualified, and callers say so rather than printing a
# clean run.
MIN_BASELINE = 4
THRESHOLD = 1.5


def peak(row: dict) -> float | None:
    """This instance's own load: the sample taken at its END.

    NOT max(start, end), which is what this did and which double-counted.
    Instances run sequentially, so an instance's `load_at_start` IS its
    predecessor's `load_at_end`, literally the same reading. In the
    hierarchical arm every row's start equals the previous row's end to the
    hundredth: 13.21, 13.30, 12.09, 12.03, 22.22, 15.63 ... Folding it in means
    one spike disqualifies TWO instances, and it did: #167 genuinely ran hot
    (end 22.22, wall 45.0s against ~21s for its neighbours) and #176 was
    disqualified for inheriting #167's closing sample, although its own end
    was 15.63 and its wall 21.9s was entirely normal for its size.

    `load_at_start` says nothing about this instance, it is sampled before
    any of its work has run. Only `load_at_end` contains it.

    Caveat that cannot be removed with these samples: load1 is a 1-minute
    average, so for an instance shorter than a minute the end sample is still
    blended with its predecessor's tail. That makes the guard conservative
    (it can still flag the successor of a contended instance) rather than
    permissive, which is the right direction for a guard on timings.
    """
    b = row.get("load_at_end") or {}
    v = b.get("load1")
    return v if v else None


def baseline(rows: list[dict]) -> float | None:
    """The run's own steady state, or None if too few samples to say.

    A true median. `peaks[len // 2]` is the UPPER-middle element for even n,
    which biases the baseline up and so makes the guard permissive, the exact
    mistake hwebench.py's own mid() helper carries a comment warning about
    ("the upper-middle value would bias every figure up"), rewritten here from
    scratch a day later.
    """
    peaks = sorted(x for x in (peak(r) for r in rows) if x)
    if len(peaks) < MIN_BASELINE:
        return None
    n = len(peaks)
    return peaks[n // 2] if n % 2 else (peaks[n // 2 - 1] + peaks[n // 2]) / 2.0


def spread(rows: list[dict]) -> float | None:
    """max/min of the run's peaks. A sanity check ON the baseline itself.

    The median baseline assumes the contended instances are a MINORITY. They
    need not be. With peaks [12, 12, 12, 22, 22, 22] the median is 17, every
    instance sits within 1.29x of it, and a run that was half contended is
    reported clean with no caveat, the contamination has been absorbed into
    the yardstick.

    This cannot be fixed with these samples: there is no outside reference for
    what the box's idle load should be. But it can be DECLARED. A wide spread
    with nothing flagged is exactly the signature of that case, and callers
    say so rather than printing a clean run.
    """
    peaks = [x for x in (peak(r) for r in rows) if x]
    return max(peaks) / min(peaks) if peaks and min(peaks) > 0 else None


def drifted(rows: list[dict]) -> tuple[dict[object, str], float | None]:
    """Map of instance number -> why its timings are disqualified."""
    base = baseline(rows)
    out: dict[object, str] = {}
    if base is None or base <= 0:
        return out, base
    for r in rows:
        pk = peak(r)
        if pk and pk / base >= THRESHOLD:
            out[r.get("number")] = (f"peak load {pk:.1f} vs {base:.1f} median "
                                    f"for the run ({pk / base:.2f}x)")
    return out, base


def baseline_is_suspect(rows: list[dict]) -> str | None:
    """Why this run's own baseline should not be trusted, if it should not."""
    sp = spread(rows)
    if sp is None:
        return None
    drift, base = drifted(rows)
    if sp >= THRESHOLD and not drift:
        return (f"load across this run spans {sp:.2f}x (min to max) yet NO "
                f"instance exceeded {THRESHOLD:.1f}x of the median. That is "
                f"the signature of a run where contention was not a minority, "
                f"so the median has absorbed it. Treat every timing here as "
                f"unvetted.")
    return None
