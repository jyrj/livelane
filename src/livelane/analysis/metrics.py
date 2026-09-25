"""The metrics and statistics the paper's figures are computed from.

Everything here is deliberately independent of how a run was produced: it reads
the variant tree and returns numbers.  That separation matters because the
analysis plan is fixed in advance and must be runnable on
the data without the code that produced it being in the loop.

Three rules are enforced rather than documented:

* **Virtual-delay runs are excluded from every latency result.**  A replayed run
  accounts for delay without spending it, so including one would silently
  understate the treatment.
* **Interrupted delays are excluded and counted.**  An iteration whose injected
  delay was cancelled did not receive the treatment.
* **No p-value without an effect size and a confidence interval.**  The bootstrap
  here is the only inferential tool, and it always returns an interval.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Callable, Sequence

# --- primitives --------------------------------------------------------------


def best_so_far(points: Sequence[tuple[float, float]],
                lower_is_better: bool = True) -> list[tuple[float, float]]:
    """Monotone running best over (time, value), sorted by time."""
    out: list[tuple[float, float]] = []
    best: float | None = None
    for t, v in sorted(points, key=lambda p: p[0]):
        if v is None:
            continue
        if best is None or (v < best if lower_is_better else v > best):
            best = v
        out.append((t, best))
    return out


def auc(curve: Sequence[tuple[float, float]], horizon_s: float,
        baseline: float | None = None, lower_is_better: bool = True) -> float:
    """Area between the best-so-far curve and the seed baseline, to a horizon.

    Reported instead of an endpoint because diminishing returns are expected and
    an endpoint hides them.  The curve is held flat after its last point out to
    ``horizon_s`` so every arm is integrated over the SAME horizon, otherwise a
    slow arm would be rewarded for having a shorter curve.
    """
    if not curve:
        return 0.0
    pts = list(curve)
    if baseline is None:
        baseline = pts[0][1]
    if pts[-1][0] < horizon_s:
        pts.append((horizon_s, pts[-1][1]))
    total = 0.0
    for (t0, v0), (t1, _) in zip(pts, pts[1:]):
        dt = min(t1, horizon_s) - t0
        if dt <= 0:
            continue
        gain = (baseline - v0) if lower_is_better else (v0 - baseline)
        total += gain * dt
    return total


def time_to_first_improvement(curve: Sequence[tuple[float, float]],
                              lower_is_better: bool = True) -> float | None:
    """Wall-clock at which the best-so-far first beats the seed. None if never."""
    if not curve:
        return None
    base = curve[0][1]
    for t, v in curve:
        if (v < base) if lower_is_better else (v > base):
            return t
    return None


def time_to_threshold(curve: Sequence[tuple[float, float]], threshold: float,
                      lower_is_better: bool = True) -> float | None:
    for t, v in curve:
        if (v <= threshold) if lower_is_better else (v >= threshold):
            return t
    return None


def value_at_time(curve: Sequence[tuple[float, float]], t: float) -> float | None:
    """Best-so-far at wall-clock ``t``, the iso-time comparison."""
    val = None
    for ct, cv in curve:
        if ct <= t:
            val = cv
        else:
            break
    return val


def saturation_point(curve: Sequence[tuple[float, float]],
                     tol: float = 1e-9) -> tuple[float, float] | None:
    """The (time, value) after which the curve never improves again.

    This is the quantity the headline actually turns on. "More latency means
    fewer iterations per hour" is arithmetic; the non-obvious question is where
    the search stops paying, because any latency small enough to reach saturation
    inside the budget costs nothing in final quality.
    """
    if not curve:
        return None
    last = curve[-1][1]
    for t, v in curve:
        if abs(v - last) <= tol:
            return (t, v)
    return curve[-1]


# --- bootstrap ---------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.point:.4g} [{self.lo:.4g}, {self.hi:.4g}] (n={self.n})"

    @property
    def excludes_zero(self) -> bool:
        return (self.lo > 0) or (self.hi < 0)


def bootstrap(values: Sequence[float], *, statistic: Callable[[Sequence[float]], float] = statistics.mean,
              resamples: int = 10_000, alpha: float = 0.05,
              seed: int = 0) -> Interval:
    """Percentile bootstrap CI. Seeded, so a reported interval is reproducible."""
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    if len(vals) == 1:
        return Interval(vals[0], vals[0], vals[0], 1)
    rng = random.Random(seed)
    n = len(vals)
    stats = []
    for _ in range(resamples):
        stats.append(statistic([vals[rng.randrange(n)] for _ in range(n)]))
    stats.sort()
    lo = stats[int((alpha / 2) * resamples)]
    hi = stats[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return Interval(statistic(vals), lo, hi, n)


def bootstrap_difference(a: Sequence[float], b: Sequence[float], *,
                         resamples: int = 10_000, alpha: float = 0.05,
                         seed: int = 0) -> Interval:
    """CI on mean(a) - mean(b). The effect size, always reported with its CI."""
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if not a or not b:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    rng = random.Random(seed)
    diffs = []
    for _ in range(resamples):
        ra = [a[rng.randrange(len(a))] for _ in range(len(a))]
        rb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        diffs.append(statistics.mean(ra) - statistics.mean(rb))
    diffs.sort()
    lo = diffs[int((alpha / 2) * resamples)]
    hi = diffs[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return Interval(statistics.mean(a) - statistics.mean(b), lo, hi, len(a) + len(b))


# --- rank correlation (H4: does the fast proxy rank like the neutral judge?),


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman rho with average ranks for ties. None if undefined."""
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None

    def ranks(vals: Sequence[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    n = len(pairs)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


# --- monotonicity (H1 over the ordered ladder) -------------------------------


def is_monotone_decreasing(values: Sequence[float], tol: float = 0.0) -> bool:
    return all(b <= a + tol for a, b in zip(values, values[1:]))


def kendall_tau_ordered(values: Sequence[float]) -> float | None:
    """Tau against the index order.

    H1 is registered as a claim about the ORDERED ladder (0, 30, 120, 600 s), so
    it is tested over the whole ladder rather than by comparing endpoints.
    """
    n = len(values)
    if n < 2:
        return None
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = values[j] - values[i]
            if d < 0:
                conc += 1
            elif d > 0:
                disc += 1
    total = conc + disc
    return None if total == 0 else (conc - disc) / total


@dataclass
class ArmSummary:
    """Everything reported per experimental cell."""

    arm: str
    design: str
    model: str
    delay_s: float
    seeds: list[int] = field(default_factory=list)
    iterations: int = 0
    accepted: int = 0
    verified_improvements: int = 0
    wall_s: float = 0.0
    injected_s: float = 0.0
    llm_s: float = 0.0
    tool_s: float = 0.0
    gate_s: float = 0.0
    cost_usd: float = 0.0
    evaluator_cpu_s: float = 0.0
    lec_refuted: int = 0
    lec_undecided: int = 0
    excluded_virtual: int = 0
    excluded_interrupted: int = 0

    @property
    def improvements_per_hour(self) -> float:
        return 0.0 if self.wall_s <= 0 else self.verified_improvements * 3600.0 / self.wall_s

    @property
    def iterations_per_hour(self) -> float:
        return 0.0 if self.wall_s <= 0 else self.iterations * 3600.0 / self.wall_s

    @property
    def gate_rejection_rate(self) -> float:
        """H5. Undecided partitions are NOT rejections and are excluded."""
        denom = self.iterations - self.lec_undecided
        return 0.0 if denom <= 0 else self.lec_refuted / denom

    @property
    def time_split(self) -> dict[str, float]:
        total = self.llm_s + self.tool_s + self.gate_s + self.injected_s
        if total <= 0:
            return {}
        return {"llm": self.llm_s / total, "tool": self.tool_s / total,
                "gate": self.gate_s / total, "injected": self.injected_s / total}

    @property
    def usd_per_improvement(self) -> float | None:
        return None if self.verified_improvements == 0 else self.cost_usd / self.verified_improvements


def predicted_speedup(t_llm_s: float, t_tool_fast_s: float,
                      t_tool_slow_s: float) -> float:
    """(T_llm + T_slow) / (T_llm + T_fast), fixed in advance.

    Registered as the quantity to check BEFORE claiming any latency effect: if
    the model's own turn time dominates, no evaluator speedup can move the
    result much, and that bound is arithmetic rather than empirical.
    """
    denom = t_llm_s + t_tool_fast_s
    return float("inf") if denom <= 0 else (t_llm_s + t_tool_slow_s) / denom


__all__ = ["best_so_far", "auc", "time_to_first_improvement", "time_to_threshold",
           "value_at_time", "saturation_point", "bootstrap", "bootstrap_difference",
           "Interval", "spearman", "is_monotone_decreasing", "kendall_tau_ordered",
           "ArmSummary", "predicted_speedup"]
