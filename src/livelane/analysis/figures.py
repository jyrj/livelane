"""The paper's three figures, rendered straight from the measurement DB.

Plotting is kept downstream of :mod:`livelane.analysis.metrics` and of
:class:`livelane.db.store.VariantStore` and is allowed to compute *nothing* of
its own: every number on every axis is either a column read out of the ledger or
a value returned by ``metrics``.  That is deliberate.  A figure is the artefact a
reviewer actually looks at, so if the plotting layer were permitted its own
arithmetic there would be two definitions of every headline quantity and no way
to tell which one produced the picture.

Three consequences, enforced here rather than documented:

* **Nothing is interpolated, extended or averaged across seeds.**  A best-so-far
  curve is a step function sampled at the instants variants were persisted;
  drawing it beyond its last observation, or averaging two seeds onto a common
  time grid, would paint values that were never measured.  Seeds are therefore
  drawn as separate lines sharing one colour, and a curve stops where the data
  stops.
* **Virtual-delay runs are refused.**  ``arm_delay_virtual = 1`` means a replayed
  run that accounted for delay without spending it, so it
  has no place on a wall-clock axis.  They are dropped with a warning, never
  silently mixed in.
* **A run with no usable data is skipped and logged, never imputed.**  An empty
  arm must look empty.

The reporting emphasis:
"more latency means fewer iterations per hour" is arithmetic, so Figure 1's
payload is not the throughput ordering but the *saturation point* of each curve,
the instant after which more time stops buying quality.  Every curve is therefore
marked at its saturation point, because the critical latency ``d*`` above which
the budget can no longer reach the plateau is the number a practitioner can act
on.

Run:  python -m livelane.analysis.figures --demo
      python -m livelane.analysis.figures --db var/runs/livelane.db --out var/figures
"""

from __future__ import annotations

import argparse
import logging
import math
import statistics
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless workstation and CI; must precede the pyplot import

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from livelane.analysis import metrics
from livelane.db.store import VariantStore
from livelane.harness.delay import INJECTION_LADDER_S
from livelane.state.reports import QorReport

LOG = logging.getLogger("livelane.analysis.figures")

# --- palette -----------------------------------------------------------------
#
# Okabe-Ito, the standard eight-colour set that survives deuteranopia,
# protanopia and tritanopia.  No series is distinguished from another by a
# red/green contrast anywhere in this module, and every categorical encoding
# carries a redundant non-colour channel (hatch for areas, marker for lines) so
# the figures still read when printed in greyscale.

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

#: Ordered blue -> orange ramp for the delay ladder.  Ordered, not categorical:
#: the delay axis has a direction and the colours must show it.
DELAY_RAMP = [OKABE_ITO["blue"], OKABE_ITO["sky"], OKABE_ITO["purple"],
              OKABE_ITO["orange"], OKABE_ITO["vermillion"], OKABE_ITO["black"]]

#: The fixed ladder, re-exported rather than restated.  Colour is
#: assigned by a delay's position *here*, not by its position within whatever
#: subset a given figure happens to draw, so ``d = 600`` is the same colour in
#: Figure 1 and in an appendix figure that omits an arm.  Importing the harness'
#: constant keeps that mapping tied to the ladder the experiment actually ran:
#: if the ladder is ever extended, the palette follows it without a second edit.
DELAY_LADDER: tuple[float, ...] = INJECTION_LADDER_S

#: Figure 2's four time classes.  Hatches are the greyscale fallback.
SPLIT_STYLE: dict[str, tuple[str, str, str]] = {
    # key: (label, colour, hatch)
    "llm": ("LLM turn", OKABE_ITO["blue"], ""),
    "tool": ("evaluator", OKABE_ITO["sky"], "//"),
    "gate": ("equivalence gate", OKABE_ITO["purple"], "xx"),
    "delay": ("injected delay", OKABE_ITO["orange"], ".."),
}

MODEL_STYLE = [
    (OKABE_ITO["blue"], "o"),
    (OKABE_ITO["orange"], "s"),
    (OKABE_ITO["purple"], "^"),
    (OKABE_ITO["sky"], "D"),
    (OKABE_ITO["black"], "v"),
]

#: Figure 1 spends colour on the delay ladder, so a second model seat on the
#: same axes has to be separated by line style or it is an invisible confound.
MODEL_DASH = ["-", "--", ":", "-.", (0, (3, 1, 1, 1, 1, 1))]

#: Metrics that may be plotted, mirroring ``VariantStore.best_so_far``'s
#: allow-list.  The column name is interpolated into SQL, so an unknown metric is
#: an error rather than a query.
QOR_METRICS: dict[str, tuple[str, bool]] = {
    # column: (axis label, lower_is_better)
    "qor_max_delay_ns": ("worst-case path delay (ns)", True),
    "qor_area_um2": ("area (um^2)", True),
    "qor_cells": ("cells", True),
    "qor_slack_ns": ("slack (ns)", False),
    "judge_max_delay_ns": ("judge worst-case path delay (ns)", True),
    "judge_area_um2": ("judge area (um^2)", True),
}

DEMO_BANNER = "SYNTHETIC DEMO DATA -- NOT A RESULT"


# --- run metadata ------------------------------------------------------------


@dataclass(frozen=True)
class RunMeta:
    """The per-run facts every figure needs, read once from ``runs``."""

    run_id: str
    design: str
    lane: str
    arm: str
    delay_s: float
    virtual: bool
    model: str
    seed: int
    status: str
    wall_s: float | None

    @property
    def label(self) -> str:
        return f"{self.arm}/{self.model}/s{self.seed}"


def _run_wall_s(store: VariantStore, run_id: str) -> float | None:
    """Measured duration = the largest ``wall_offset_s`` persisted by the run.

    ``loop.py`` writes both the variant and the iteration row *after* that
    iteration's LLM turn, evaluation, gate and injected delay have all completed,
    so the last stamp is the moment the run's final iteration finished.  It is a
    slight under-count (teardown after the last row is charged to nobody) but it
    is the same under-count in every arm, and unlike ``ended_at - started_at`` it
    comes from the same monotonic clock as the x-axis of Figure 1.  ``None`` when
    the run persisted nothing, the caller must then drop the run, not guess.
    """
    row = store.query(
        """SELECT MAX(t) AS wall_s FROM (
               SELECT MAX(wall_offset_s) AS t FROM variants   WHERE run_id=?
               UNION ALL
               SELECT MAX(wall_offset_s) AS t FROM iterations WHERE run_id=?)""",
        (run_id, run_id),
    )
    return None if not row or row[0]["wall_s"] is None else float(row[0]["wall_s"])


def load_runs(store: VariantStore, run_ids: Sequence[str] | None = None, *,
              include_virtual: bool = False) -> list[RunMeta]:
    """Metadata for ``run_ids`` (or every run in the DB), virtual runs dropped.

    Unknown run ids are warned about and skipped rather than raising: a figure
    over nine of ten cells is still publishable, a traceback is not.
    """
    if run_ids is None:
        rows = store.query("SELECT * FROM runs ORDER BY arm_delay_s, model, seed")
    else:
        wanted = list(dict.fromkeys(run_ids))
        holes = ",".join("?" * len(wanted))
        rows = store.query(
            f"SELECT * FROM runs WHERE run_id IN ({holes}) "
            f"ORDER BY arm_delay_s, model, seed", wanted)
        found = {r["run_id"] for r in rows}
        for missing in [r for r in wanted if r not in found]:
            LOG.warning("run %s is not in %s -- skipped", missing, store.path)

    out: list[RunMeta] = []
    for r in rows:
        if r["arm_delay_virtual"] and not include_virtual:
            LOG.warning("run %s is a VIRTUAL-delay replay: excluded from every "
                        "latency figure", r["run_id"])
            continue
        out.append(RunMeta(
            run_id=r["run_id"], design=r["design"], lane=r["lane"], arm=r["arm"],
            delay_s=float(r["arm_delay_s"]), virtual=bool(r["arm_delay_virtual"]),
            model=r["model"], seed=int(r["seed"]), status=r["status"],
            wall_s=_run_wall_s(store, r["run_id"])))
    return out


def _warn_if_mixed(runs: Sequence[RunMeta], attr: str, what: str) -> None:
    vals = sorted({getattr(r, attr) for r in runs})
    if len(vals) > 1:
        LOG.warning("this figure mixes %d %s (%s); they are not comparable on one "
                    "set of axes", len(vals), what, ", ".join(map(str, vals)))


# --- data extraction ---------------------------------------------------------


def accepted_curve(store: VariantStore, run_id: str,
                   metric: str = "qor_max_delay_ns") -> list[tuple[float, float]]:
    """Best-so-far over ACCEPTED variants only, via :func:`metrics.best_so_far`.

    Rejected candidates are excluded at the SQL level: a variant the equivalence
    gate refuted is not a result, however good its QoR looked.
    """
    if metric not in QOR_METRICS:
        raise ValueError(f"refusing to plot unknown metric {metric!r}; "
                         f"known: {sorted(QOR_METRICS)}")
    _, lower_is_better = QOR_METRICS[metric]
    rows = store.query(
        f"""SELECT wall_offset_s, {metric} AS v FROM variants
            WHERE run_id=? AND accepted=1 AND {metric} IS NOT NULL
            ORDER BY wall_offset_s""",
        (run_id,))
    pts = [(float(r["wall_offset_s"]), float(r["v"])) for r in rows]
    return metrics.best_so_far(pts, lower_is_better=lower_is_better)


def _seed_baseline(store: VariantStore, run_id: str) -> QorReport | None:
    """The run's seed evaluation, against which improvements are counted.

    The study defines a verified improvement against *the run's seed
    baseline*, not against the parent, so the baseline is iteration 0 and nothing
    else.  Pinning it there is not pedantry.  ``loop.seed()`` persists iteration 0
    with ``status='eval-failed'``, ``accepted=0`` and a NULL ``qor_max_delay_ns``
    whenever the baseline evaluation could not be parsed, the exact state the
    house rule requires, and the next-earliest scored variant is then an *agent
    edit*.  Falling through to it would silently re-base the run onto the agent's
    own work and report the remaining edits as improvements over the seed.  No
    scored seed => no improvement can be counted for this run at all; the caller
    drops the run rather than counting against a substitute.
    """
    rows = store.query(
        """SELECT qor_cells, qor_area_um2, qor_max_delay_ns, qor_slack_ns
           FROM variants
           WHERE run_id=? AND iteration_index=0 AND qor_max_delay_ns IS NOT NULL
           ORDER BY wall_offset_s LIMIT 1""",
        (run_id,))
    if not rows:
        return None
    r = rows[0]
    return QorReport(top="", cells=r["qor_cells"], area_um2=r["qor_area_um2"],
                     max_delay_ns=r["qor_max_delay_ns"], slack_ns=r["qor_slack_ns"])


@dataclass
class ImprovementCount:
    """Verified improvements for one run, plus why rows were not counted."""

    run_id: str
    verified: int = 0
    considered: int = 0
    no_functional_record: int = 0
    not_proven: int = 0
    no_baseline: bool = False


def verified_improvements(store: VariantStore, run_id: str, *,
                          area_epsilon: float = 0.02,
                          require_functional: bool = True) -> ImprovementCount:
    """Count variants meeting the study's definition of an improvement.

    The definition: (a) passes the shared Verilator oracle, (b) is proven
    equivalent by the common gate, (c) strictly improves the primary objective
    against the run's seed baseline.  Clause (c) is delegated to
    :meth:`QorReport.improves_on` so the objective, and its area guardrail,
    has exactly one definition in the codebase.

    Fails closed on clause (a): ``functional_pass IS NULL`` means the oracle left
    no record, which is not a pass.  Runs affected are reported in the returned
    counts so an all-zero figure can never be mistaken for "the agent found
    nothing".  ``require_functional=False`` relaxes this and is a *departure from
    the study's definition* that must be stated wherever it is used.
    """
    base = _seed_baseline(store, run_id)
    if base is None:
        LOG.warning("run %s has no scored seed variant: no improvement can be "
                    "counted against a baseline that does not exist", run_id)
        return ImprovementCount(run_id, no_baseline=True)

    rows = store.query(
        """SELECT functional_pass, lec_verdict, qor_cells, qor_area_um2,
                  qor_max_delay_ns, qor_slack_ns
           FROM variants
           WHERE run_id=? AND accepted=1 AND iteration_index > 0""",
        (run_id,))
    c = ImprovementCount(run_id, considered=len(rows))
    for r in rows:
        if r["functional_pass"] is None:
            c.no_functional_record += 1
            if require_functional:
                continue
        elif not r["functional_pass"]:
            continue
        if r["lec_verdict"] != "proven":
            c.not_proven += 1
            continue
        cand = QorReport(top="", cells=r["qor_cells"], area_um2=r["qor_area_um2"],
                         max_delay_ns=r["qor_max_delay_ns"], slack_ns=r["qor_slack_ns"])
        if cand.improves_on(base, area_epsilon):
            c.verified += 1
    return c


@dataclass
class SplitTotals:
    """Summed iteration clocks for one group of runs (Figure 2's bar)."""

    key: str
    runs: list[str] = field(default_factory=list)
    iterations: int = 0
    llm_s: float = 0.0
    tool_s: float = 0.0
    gate_s: float = 0.0
    delay_s: float = 0.0
    orch_s: float = 0.0
    interrupted: int = 0
    delay_nominal_s: float = 0.0

    def per_iteration(self) -> dict[str, float]:
        n = self.iterations
        if n <= 0:
            return {}
        return {"llm": self.llm_s / n, "tool": self.tool_s / n,
                "gate": self.gate_s / n, "delay": self.delay_s / n}


def time_split(store: VariantStore, run: RunMeta) -> SplitTotals:
    """Per-iteration time split for one run, interrupted delays excluded.

    ``delay_interrupted = 1`` means the treatment was cancelled mid-sleep, so
    that iteration did not receive the arm it is labelled with; those rows are
    excluded and counted (metrics module rule 2), never averaged in.
    """
    rows = store.query(
        """SELECT t_llm_ms, t_tool_ms, t_orch_ms, t_gate_ms, t_delay_ms,
                  injected_delay_s, delay_interrupted
           FROM iterations WHERE run_id=?""",
        (run.run_id,))
    t = SplitTotals(key=run.arm, runs=[run.run_id])
    for r in rows:
        if r["delay_interrupted"]:
            t.interrupted += 1
            continue
        t.iterations += 1
        t.llm_s += (r["t_llm_ms"] or 0.0) / 1000.0
        t.tool_s += (r["t_tool_ms"] or 0.0) / 1000.0
        t.gate_s += (r["t_gate_ms"] or 0.0) / 1000.0
        t.delay_s += (r["t_delay_ms"] or 0.0) / 1000.0
        t.orch_s += (r["t_orch_ms"] or 0.0) / 1000.0
        t.delay_nominal_s += r["injected_delay_s"] or 0.0
    return t


# --- shared chrome -----------------------------------------------------------


def _time_unit(max_t: float) -> tuple[float, str]:
    """Rescale the wall-clock axis so a two-hour run is readable at print size."""
    if max_t <= 300.0:
        return 1.0, "s"
    if max_t <= 3 * 3600.0:
        return 60.0, "min"
    return 3600.0, "h"


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def _finish(fig: plt.Figure, out_path: str | Path, *, watermark: str | None,
            footnote: str | None) -> Path:
    """Stamp, save and close.  Every figure leaves through here.

    The crop box is computed *before* the watermark is drawn.  ``bbox_inches
    ="tight"`` grows the canvas to contain every artist, so a large rotated
    watermark would otherwise pad each figure with several inches of white and
    shrink the plot itself at print size.  The caption is hard-wrapped for the
    same reason: one long line silently doubles the figure's width.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if footnote:
        width = max(60, int(fig.get_size_inches()[0] * 23))
        fig.text(0.01, -0.012, textwrap.fill(footnote, width), fontsize=6.5,
                 color="0.35", ha="left", va="top", linespacing=1.5)
    fig.canvas.draw()
    box = fig.get_tightbbox().padded(0.08)
    if watermark:
        # Shrink the banner to fit the crop that was just computed: a watermark
        # whose ends are cropped off reads as a corrupted figure rather than as a
        # warning, and the warning is the whole point of it.
        t = fig.text(0.5, 0.5, watermark, fontsize=40, color="0.86", alpha=0.55,
                     ha="center", va="center", rotation=24, zorder=0,
                     fontweight="bold")
        span = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
        t.set_fontsize(40 * min(1.0, 0.94 * box.width / max(span, 1e-6)))
    fig.savefig(out, dpi=200, bbox_inches=box, facecolor="white")
    plt.close(fig)
    LOG.info("wrote %s (%d bytes)", out, out.stat().st_size)
    return out


def _title(ax_or_fig, title: str, subtitle: str | None = None) -> None:
    ax_or_fig.suptitle(title, fontsize=12, fontweight="bold", y=0.99)
    if subtitle:
        ax_or_fig.text(0.5, 0.945, subtitle, fontsize=8.5, color="0.3",
                       ha="center", va="top")


def _delay_colours(delays: Sequence[float]) -> dict[float, str]:
    """Colour by position on the fixed ladder, not within this subset.

    A figure that omits an arm must not recolour the arms it keeps: if ``d=600``
    were sky blue in an appendix figure and orange in Figure 1, a reader
    comparing them would be misled by the palette alone.

    Every ladder colour stays reserved even in a figure that omits that arm, so
    an off-ladder delay can never borrow one; it takes the next ramp entry
    nothing else is using.  Wrapping blindly would eventually hand two different
    delays the same colour on the same axes, which is a misread rather than a
    blemish, so an exhausted ramp is reported instead of absorbed.
    """
    out: dict[float, str] = {}
    used = {DELAY_RAMP[i % len(DELAY_RAMP)] for i in range(len(DELAY_LADDER))}
    spare = len(DELAY_LADDER)
    for d in sorted(set(delays)):
        if d in DELAY_LADDER:
            out[d] = DELAY_RAMP[DELAY_LADDER.index(d) % len(DELAY_RAMP)]
            continue
        for k in range(len(DELAY_RAMP)):
            cand = DELAY_RAMP[(spare + k) % len(DELAY_RAMP)]
            if cand not in used:
                out[d], spare = cand, spare + k + 1
                used.add(cand)
                break
        else:
            LOG.warning("%d delay level(s) but only %d palette entries: d=%gs "
                        "REUSES a colour already on these axes -- distinguish it "
                        "by the legend, not by eye", len(set(delays)),
                        len(DELAY_RAMP), d)
            out[d] = DELAY_RAMP[spare % len(DELAY_RAMP)]
            spare += 1
    return out


# --- Figure 1 ----------------------------------------------------------------


def fig1_best_so_far(store: VariantStore, run_ids: Sequence[str] | None,
                     out_path: str | Path, *,
                     metric: str = "qor_max_delay_ns",
                     title: str = "Best-so-far QoR vs wall-clock",
                     subtitle: str | None = None,
                     watermark: str | None = None) -> Path:
    """The money chart: best-so-far QoR against wall-clock, one colour per delay.

    Seeds within a delay share a colour and are drawn as separate lines; the
    alternative, averaging them, would require resampling step functions onto
    a common grid, i.e. inventing values between observations.

    Each curve is marked at :func:`metrics.saturation_point`, the instant after
    which it never improves again.  That marker, not the
    ordering of the curves, is the finding: an arm whose marker lands inside the
    budget paid nothing for its latency.
    """
    if metric not in QOR_METRICS:
        raise ValueError(f"refusing to plot unknown metric {metric!r}")
    ylabel, lower_is_better = QOR_METRICS[metric]
    runs = load_runs(store, run_ids)
    _warn_if_mixed(runs, "design", "designs")

    curves: list[tuple[RunMeta, list[tuple[float, float]]]] = []
    for r in runs:
        c = accepted_curve(store, r.run_id, metric)
        if not c:
            LOG.warning("run %s (%s) has no accepted variant with %s: nothing "
                        "drawn for it", r.run_id, r.label, metric)
            continue
        curves.append((r, c))

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    _style_axes(ax)

    if not curves:
        LOG.warning("figure 1: no run had plottable data; drawing an empty frame")
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                va="center", color="0.4", fontsize=12)
        ax.set_xlabel("wall-clock")
        ax.set_ylabel(ylabel)
        _title(fig, title, subtitle)
        return _finish(fig, out_path, watermark=watermark, footnote=None)

    div, unit = _time_unit(max(c[-1][0] for _, c in curves))
    colours = _delay_colours([r.delay_s for r, _ in curves])
    models = sorted({r.model for r, _ in curves})
    dashes = {m: MODEL_DASH[i % len(MODEL_DASH)] for i, m in enumerate(models)}
    saturations: list[tuple[float, float]] = []

    for r, c in curves:
        col = colours[r.delay_s]
        xs = [t / div for t, _ in c]
        ys = [v for _, v in c]
        ax.plot(xs, ys, drawstyle="steps-post", color=col, linewidth=1.7,
                linestyle=dashes[r.model], alpha=0.9, solid_joinstyle="round",
                zorder=3)
        ax.plot(xs, ys, linestyle="none", marker="o", markersize=2.6,
                color=col, alpha=0.55, zorder=3)
        sat = metrics.saturation_point(c)
        if sat is not None:
            st, sv = sat
            saturations.append((r.delay_s, st))
            ax.plot([st / div], [sv], marker="*", markersize=13, color=col,
                    markeredgecolor="white", markeredgewidth=0.7, zorder=5)

    handles = [Line2D([], [], color=colours[d], linewidth=2.0,
                      label=f"d = {d:g} s")
               for d in sorted(colours)]
    handles.append(Line2D([], [], color="0.35", marker="*", markersize=11,
                          linestyle="none", label="saturation (no later gain)"))
    if len(models) > 1:
        handles += [Line2D([], [], color="0.35", linewidth=1.6,
                           linestyle=dashes[m], label=m) for m in models]
    # Placed OUTSIDE the axes. Best-so-far curves are step functions that stay
    # high for a long time in the slow arms, so an in-axes legend sits directly
    # on top of the d=600 trace and hides the data it is labelling.
    ax.legend(handles=handles, frameon=False, fontsize=8.5,
              loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    ax.set_xlabel(f"wall-clock since run start ({unit})")
    ax.set_ylabel(f"best-so-far {ylabel}")
    _title(fig, title, subtitle or
           f"accepted variants only, {len(curves)} run(s), "
           f"{len(colours)} injected-delay level(s)")

    foot = ("best-so-far over equivalence-gate-accepted variants only; curves "
            "are step functions drawn to their last observation and are never "
            "extended or averaged across seeds.")
    if saturations:
        by_delay: dict[float, list[float]] = {}
        for d, st in saturations:
            by_delay.setdefault(d, []).append(st)
        # "over curves", not "over seeds": by_delay pools every curve drawn at
        # that delay, which is one per (model, seed) pair whenever more than one
        # model seat is on the axes.
        foot += ("  saturation (median over curves): " +
                 ", ".join(f"d={d:g}s at {statistics.median(v) / div:.1f}{unit}"
                           for d, v in sorted(by_delay.items())))
    return _finish(fig, out_path, watermark=watermark, footnote=foot)


# --- Figure 2 ----------------------------------------------------------------


def fig2_time_split(store: VariantStore, run_ids: Sequence[str] | None,
                    out_path: str | Path, *,
                    title: str = "Where an iteration goes",
                    subtitle: str | None = None,
                    watermark: str | None = None) -> Path:
    """Stacked horizontal bars of the per-iteration time split, one bar per arm.

    Two panels over the same rows, because one alone misleads.  Absolute seconds
    (left) show that an ``I(600)`` iteration costs two orders of magnitude more
    wall-clock than an ``I(0)`` one, but at that scale the fast arms collapse to
    a sliver; the normalised panel (right) shows what the plan actually wants to
    display, the same agent, mostly idle at ``d=600`` and continuously working
    at ``d=0``.  The share panel is drawn with
    :attr:`metrics.ArmSummary.time_split` so the fractions are the same ones the
    tables report.
    """
    runs = load_runs(store, run_ids)
    if not runs:
        LOG.warning("figure 2: no non-virtual runs selected")
    # This figure pools iterations into one bar per arm, so a mixed-design
    # selection silently averages two circuits' evaluator costs together.
    _warn_if_mixed(runs, "design", "designs")

    multi_model = len({r.model for r in runs}) > 1
    groups: dict[str, SplitTotals] = {}
    order: list[str] = []
    interrupted_total = 0
    for r in runs:
        t = time_split(store, r)
        if t.iterations == 0:
            LOG.warning("run %s (%s) has no usable iteration row: no bar drawn",
                        r.run_id, r.label)
            interrupted_total += t.interrupted
            continue
        key = f"{r.arm} - {r.model}" if multi_model else r.arm
        g = groups.get(key)
        if g is None:
            g = groups[key] = SplitTotals(key=key)
            order.append(key)
        g.runs.append(r.run_id)
        g.iterations += t.iterations
        g.llm_s += t.llm_s
        g.tool_s += t.tool_s
        g.gate_s += t.gate_s
        g.delay_s += t.delay_s
        g.orch_s += t.orch_s
        g.interrupted += t.interrupted
        interrupted_total += t.interrupted

    # Sort bars by the arm's mean injected delay: the ladder is ordered and the
    # figure has to show the ladder, not the dict insertion order.
    order.sort(key=lambda k: (groups[k].delay_s / max(groups[k].iterations, 1), k))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 0.62 * max(len(order), 3) + 2.3),
                             sharey=True,
                             gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.12})
    for ax in axes:
        _style_axes(ax)
        ax.grid(axis="y", visible=False)

    if not order:
        for ax in axes:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                    va="center", color="0.4", fontsize=12)
        _title(fig, title, subtitle)
        return _finish(fig, out_path, watermark=watermark, footnote=None)

    ys = list(range(len(order)))
    orch_dropped = 0.0
    widest = max(sum(groups[k].per_iteration().values()) for k in order)
    for ax, mode in zip(axes, ("abs", "share")):
        for i, key in enumerate(order):
            g = groups[key]
            per = g.per_iteration()
            if mode == "abs":
                vals = per
            else:
                summ = metrics.ArmSummary(arm=key, design="", model="", delay_s=0.0,
                                          llm_s=g.llm_s, tool_s=g.tool_s,
                                          gate_s=g.gate_s, injected_s=g.delay_s)
                frac = summ.time_split
                vals = {"llm": frac.get("llm", 0.0) * 100.0,
                        "tool": frac.get("tool", 0.0) * 100.0,
                        "gate": frac.get("gate", 0.0) * 100.0,
                        "delay": frac.get("injected", 0.0) * 100.0}
            left = 0.0
            for cls, (_, colour, hatch) in SPLIT_STYLE.items():
                w = vals.get(cls, 0.0)
                if w <= 0:
                    continue
                ax.barh(i, w, left=left, height=0.62, color=colour, hatch=hatch,
                        edgecolor="white", linewidth=0.6, zorder=3)
                left += w
            if mode == "abs":
                ax.text(left + widest * 0.015, i, f"{left:.1f} s  x{g.iterations}",
                        va="center", fontsize=7.5, color="0.25")
                orch_dropped = max(orch_dropped, g.orch_s / max(g.llm_s + g.tool_s +
                                                               g.gate_s + g.delay_s, 1e-9))
            elif vals.get("delay", 0.0) >= 1.0:
                # Only label a bar that has an idle segment to label. "0% idle"
                # written across the working segments would be actively confusing.
                ax.text(min(left, 100.0) - 1.5, i,
                        f"{vals['delay']:.0f}% idle", va="center",
                        ha="right", fontsize=7.5, color="white", fontweight="bold")

    axes[0].set_yticks(ys, order, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("mean wall-clock per iteration (s)")
    # headroom for the "Ns xM" annotations; an all-zero-clock arm would
    # otherwise ask matplotlib for a singular axis and get a silent rescale.
    axes[0].set_xlim(0, widest * 1.30 if widest > 0 else 1.0)
    axes[1].set_xlabel("share of iteration time (%)")
    axes[1].set_xlim(0, 100)

    # Bars are sorted by ascending delay and the top bar is therefore always the
    # shortest, so the top-right of the absolute panel is structurally free.
    handles = [Patch(facecolor=c, hatch=h, edgecolor="white", label=lab)
               for lab, c, h in SPLIT_STYLE.values()]
    axes[0].legend(handles=handles, frameon=True, framealpha=0.9,
                   edgecolor="none", fontsize=8.5, loc="upper right")
    _title(fig, title, subtitle or
           f"{sum(groups[k].iterations for k in order)} iterations over "
           f"{len(runs)} run(s)")

    foot = ("per-iteration means over all iterations in the arm; "
            "delay-interrupted iterations excluded"
            + (f" ({interrupted_total} excluded)" if interrupted_total else " (0 excluded)")
            + ".")
    if orch_dropped > 0.01:
        LOG.warning("orchestration time is %.1f%% of the largest bar's total and "
                    "is NOT drawn (the four registered classes are LLM/tool/gate/"
                    "delay); state this if the figure is published",
                    orch_dropped * 100.0)
        foot += (f"  orchestration overhead (up to {orch_dropped * 100:.1f}% of a "
                 f"bar) is not shown.")
    return _finish(fig, out_path, watermark=watermark, footnote=foot)


# --- Figure 3 ----------------------------------------------------------------


def fig3_improvements_vs_delay(store: VariantStore, run_ids: Sequence[str] | None,
                               out_path: str | Path, *,
                               area_epsilon: float = 0.02,
                               require_functional: bool = True,
                               resamples: int = 10_000,
                               alpha: float = 0.05,
                               boot_seed: int = 0,
                               title: str = "Verified improvements per hour vs injected delay",
                               subtitle: str | None = None,
                               watermark: str | None = None) -> Path:
    """H1: verified improvements/hour against injected delay, one line per model.

    The x-axis is ``symlog``, not ``log``: the control arm is ``d = 0`` and a log
    axis cannot show it, so the usual dodges are to drop the control or to plot
    it at some invented epsilon.  ``symlog`` with a linear threshold below the
    smallest real delay puts ``d = 0`` at its true position and keeps the ladder
    logarithmic above it, with ticks labelled at the delays that were actually
    run.

    Error bars are percentile bootstrap intervals over the seeds in each
    (model, delay) cell, from :func:`metrics.bootstrap`.  A cell with one seed
    gets a zero-width bar, which is honest: one seed is a point, not a spread.
    """
    runs = load_runs(store, run_ids)
    _warn_if_mixed(runs, "design", "designs")

    cells: dict[tuple[str, float], list[float]] = {}
    no_functional = 0
    dropped = 0
    for r in runs:
        if r.wall_s is None or r.wall_s <= 0:
            LOG.warning("run %s (%s) has no measured wall-clock: dropped from "
                        "figure 3 rather than dividing by a guess", r.run_id, r.label)
            dropped += 1
            continue
        c = verified_improvements(store, r.run_id, area_epsilon=area_epsilon,
                                  require_functional=require_functional)
        if c.no_baseline:
            dropped += 1
            continue
        no_functional += c.no_functional_record
        summ = metrics.ArmSummary(arm=r.arm, design=r.design, model=r.model,
                                  delay_s=r.delay_s,
                                  verified_improvements=c.verified, wall_s=r.wall_s)
        cells.setdefault((r.model, r.delay_s), []).append(summ.improvements_per_hour)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    _style_axes(ax)

    if not cells:
        LOG.warning("figure 3: no run produced a countable improvement rate")
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                va="center", color="0.4", fontsize=12)
        ax.set_xlabel("injected evaluator delay d (s)")
        ax.set_ylabel("verified improvements / hour")
        _title(fig, title, subtitle)
        return _finish(fig, out_path, watermark=watermark, footnote=None)

    models = sorted({m for m, _ in cells})
    delays = sorted({d for _, d in cells})
    positive = [d for d in delays if d > 0]
    linthresh = min(positive) if positive else 1.0

    for i, model in enumerate(models):
        colour, marker = MODEL_STYLE[i % len(MODEL_STYLE)]
        xs, ys, lo, hi = [], [], [], []
        for d in delays:
            vals = cells.get((model, d))
            if not vals:
                LOG.warning("cell (model=%s, d=%gs) has no run: no point plotted",
                            model, d)
                continue
            ci = metrics.bootstrap(vals, resamples=resamples, alpha=alpha,
                                   seed=boot_seed)
            if math.isnan(ci.point):
                LOG.warning("cell (model=%s, d=%gs) bootstrapped to NaN: skipped",
                            model, d)
                continue
            xs.append(d)
            ys.append(ci.point)
            down, up = ci.point - ci.lo, ci.hi - ci.point
            if down < 0 or up < 0:
                LOG.warning("cell (model=%s, d=%gs): bootstrap point %.4g lies "
                            "outside [%.4g, %.4g]; error bar clamped at zero",
                            model, d, ci.point, ci.lo, ci.hi)
            lo.append(max(down, 0.0))
            hi.append(max(up, 0.0))
        if not xs:
            LOG.warning("model %s has no plottable cell: no line drawn", model)
            continue
        ax.errorbar(xs, ys, yerr=[lo, hi], color=colour, marker=marker,
                    markersize=6, linewidth=1.8, capsize=3.5, capthick=1.0,
                    elinewidth=1.0, label=model, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.6)

    ax.set_xscale("symlog", linthresh=linthresh, linscale=0.6)
    ax.set_xticks(delays, [f"{d:g}" for d in delays])
    ax.minorticks_off()
    ax.set_xlim(-linthresh * 0.35, max(delays) * 1.6 if max(delays) > 0 else 1.0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(f"injected evaluator delay d (s), symlog below {linthresh:g} s")
    ax.set_ylabel("verified improvements / hour")
    ax.legend(frameon=False, fontsize=8.5, title="model seat",
              title_fontsize=8.5)
    _title(fig, title, subtitle or
           f"{len(models)} model seat(s), {len(delays)} delay level(s), "
           f"{sum(len(v) for v in cells.values())} run(s)")

    foot = (f"verified improvement = functional oracle pass + equivalence proven "
            f"+ strictly beats the run's seed on worst-case delay with area "
            f"guardrail +{area_epsilon * 100:g}%; "
            f"error bars are {int((1 - alpha) * 100)}% percentile bootstrap CIs "
            f"({resamples} resamples, seed {boot_seed}) over seeds. "
            f"x is symlog with a linear segment below {linthresh:g} s so the "
            f"d = 0 control sits at its true position rather than at an "
            f"invented epsilon.")
    if not require_functional:
        foot += "  NOTE: functional-oracle clause RELAXED -- departs from the study's definition."
    elif no_functional:
        foot += (f"  {no_functional} accepted variant(s) had no functional-oracle "
                 f"record and were not counted.")
    if dropped:
        foot += f"  {dropped} run(s) dropped for lacking a baseline or a wall-clock."
    return _finish(fig, out_path, watermark=watermark, footnote=foot)


# --- driver ------------------------------------------------------------------


def render_all(store: VariantStore, run_ids: Sequence[str] | None,
               out_dir: str | Path, *, metric: str = "qor_max_delay_ns",
               tag: str = "", watermark: str | None = None,
               subtitle: str | None = None) -> list[Path]:
    """Render figures 1-3 into ``out_dir``; returns the three paths in order."""
    d = Path(out_dir)
    suffix = f"_{tag}" if tag else ""
    return [
        fig1_best_so_far(store, run_ids, d / f"fig1_best_so_far{suffix}.png",
                         metric=metric, watermark=watermark, subtitle=subtitle),
        fig2_time_split(store, run_ids, d / f"fig2_time_split{suffix}.png",
                        watermark=watermark, subtitle=subtitle),
        fig3_improvements_vs_delay(store, run_ids,
                                   d / f"fig3_improvements_vs_delay{suffix}.png",
                                   watermark=watermark, subtitle=subtitle),
    ]


# --- synthetic demo ----------------------------------------------------------


def build_demo_store(*, delays: Sequence[float] = (0.0, 30.0, 120.0, 600.0),
                     models: Sequence[str] = ("gemini-3.8-flash", "claude-sonnet-5"),
                     seeds: Sequence[int] = (1, 2),
                     budget_wall_s: float = 3600.0,
                     budget_iters: int = 40) -> VariantStore:
    """A tiny SYNTHETIC store, so the plotting code is exercised before real data.

    The generator is not a simulation of the experiment and must never be read as
    a prediction of it.  It exists to drive every branch of the plotting code,
    saturating curves, rejected variants, an interrupted delay, a model with a
    slower turn, an empty run, while real runs are still being collected.  It
    is seeded, so the demo figures are byte-stable across invocations.
    """
    import random

    store = VariantStore(":memory:", verbose=False)
    plateau_ns, start_ns = 8.2, 10.4

    for model in models:
        t_llm = 4.0 if "flash" in model else 22.0
        for d in delays:
            for s in seeds:
                # Seeded from a string, not from hash(): str hashing is salted
                # per process, so hash() would make the demo figures differ
                # between invocations and the self-test's numbers unquotable.
                rng = random.Random(f"livelane-demo|{model}|{d:g}|{s}")
                rid = f"demo-{model.split('-')[0]}-d{int(d)}-s{s}"
                run = store.start_run(
                    rid, design="Alu", lane="S", arm=f"I({int(d)})", model=model,
                    seed=s, arm_delay_s=d, budget_wall_s=budget_wall_s,
                    budget_iters=budget_iters, note="SYNTHETIC demo data")

                t = 0.0
                gains = 0
                best = start_ns
                area = 1000.0
                root = store.add_variant(
                    run, iteration_index=0, wall_offset_s=0.0, accepted=1,
                    functional_pass=1, lec_verdict="skipped",
                    qor_max_delay_ns=start_ns, qor_area_um2=area, qor_cells=900,
                    edit_note="seed")
                store.add_iteration(run, iteration_index=0, variant_id=root,
                                    wall_offset_s=0.0, t_tool_ms=3800.0,
                                    t_delay_ms=d * 1000.0, injected_delay_s=d,
                                    delay_mode="additive", evaluator_cpu_s=3.4)
                parent = root

                for i in range(1, budget_iters + 1):
                    t_tool = 3.8 + rng.uniform(-0.3, 0.4)
                    t_gate = 1.9 + rng.uniform(-0.2, 0.3)
                    llm = t_llm * rng.uniform(0.85, 1.2)
                    t += llm + t_tool + t_gate + d
                    if t > budget_wall_s:
                        break

                    # Diminishing returns: each gain closes a shrinking fraction
                    # of the remaining distance to a hard plateau.
                    roll = rng.random()
                    interrupted = (i == 3 and d >= 120.0 and s == 2)
                    if roll < 0.18:                       # equivalence failure
                        verdict, accepted, delay_ns = "refuted", 0, best - 0.4
                    elif roll < 0.34:                     # accepted, no gain
                        verdict, accepted = "proven", 1
                        delay_ns = best + rng.uniform(0.05, 0.5)
                    else:
                        verdict, accepted = "proven", 1
                        step = (best - plateau_ns) * rng.uniform(0.25, 0.45)
                        delay_ns = max(plateau_ns, best - step)
                        if delay_ns < best - 1e-9:
                            gains += 1
                            best = delay_ns
                    area_i = area * rng.uniform(0.995, 1.015)

                    vid = store.add_variant(
                        run, iteration_index=i, parent_id=parent, wall_offset_s=t,
                        accepted=accepted, functional_pass=1, lec_verdict=verdict,
                        lec_backend="eqy", lec_wall_s=t_gate,
                        qor_max_delay_ns=round(delay_ns, 4),
                        qor_area_um2=round(area_i, 2), qor_cells=900 + i,
                        edit_note=f"synthetic edit {i}")
                    store.add_iteration(
                        run, iteration_index=i, variant_id=vid, wall_offset_s=t,
                        t_llm_ms=llm * 1000.0, t_tool_ms=t_tool * 1000.0,
                        t_gate_ms=t_gate * 1000.0, t_orch_ms=140.0,
                        t_delay_ms=(0.0 if interrupted else d * 1000.0),
                        injected_delay_s=d, delay_mode="additive",
                        delay_interrupted=int(interrupted),
                        tokens_in=1800, tokens_out=340, tokens_cache_read=9000,
                        cost_usd=0.004, cost_source="reported", evaluator_cpu_s=3.4)
                    if accepted:
                        parent = vid
                store.finish_run(rid)

    # A run that produced nothing: the empty-arm path must warn, not invent.
    empty = store.start_run("demo-empty", design="Alu", lane="S", arm="I(0)",
                            model=models[0], seed=99, arm_delay_s=0.0,
                            note="SYNTHETIC: crashed before its first variant")
    store.finish_run("demo-empty", status="error")
    # A replayed run: must be excluded from every latency figure.
    store.start_run("demo-virtual", design="Alu", lane="S", arm="I(600)",
                    model=models[0], seed=98, arm_delay_s=600.0,
                    arm_delay_virtual=True, note="SYNTHETIC replay")
    store.finish_run("demo-virtual")
    del empty
    return store


def _demo(out_dir: Path, metric: str) -> int:
    print("=== livelane.analysis.figures self-test (SYNTHETIC data) ===")
    store = build_demo_store()
    runs = load_runs(store, None)
    assert all(not r.virtual for r in runs), "a virtual-delay run reached the figures"
    assert "demo-virtual" not in {r.run_id for r in runs}
    print(f"    synthetic store: {len(store.query('SELECT run_id FROM runs'))} runs, "
          f"{len(runs)} after excluding virtual-delay replays")

    paths = render_all(store, None, out_dir, metric=metric, tag="demo",
                       watermark=DEMO_BANNER,
                       subtitle=DEMO_BANNER + " -- generated to exercise the "
                                "plotting code before real runs exist")
    for p in paths:
        size = p.stat().st_size
        assert p.exists() and size > 20_000, f"{p} is only {size} bytes"
        print(f"    wrote {p}  ({size:,} bytes)")

    # Properties of the pipeline, not of the picture.
    curve = accepted_curve(store, runs[0].run_id)
    assert curve, "the demo store must produce a plottable curve"
    assert all(curve[i][1] >= curve[i + 1][1] for i in range(len(curve) - 1)), \
        "best-so-far must be monotone non-increasing"
    sat = metrics.saturation_point(curve)
    assert sat is not None and sat[1] == curve[-1][1], "saturation must sit on the plateau"
    print(f"    curve {runs[0].label}: {len(curve)} points, "
          f"{curve[0][1]:.2f} -> {curve[-1][1]:.2f} ns, saturates at t+{sat[0]:.0f}s")

    empty = accepted_curve(store, "demo-empty")
    assert empty == [], "an empty run must yield no points at all"

    # Throughput must fall with d: same improvements, longer iterations. If this
    # ever fails, the DB read or the rate arithmetic is wrong, not the physics.
    rates: dict[float, list[float]] = {}
    for r in runs:
        if r.wall_s is None or r.wall_s <= 0:
            continue
        c = verified_improvements(store, r.run_id)
        rates.setdefault(r.delay_s, []).append(
            metrics.ArmSummary(arm=r.arm, design=r.design, model=r.model,
                               delay_s=r.delay_s, verified_improvements=c.verified,
                               wall_s=r.wall_s).improvements_per_hour)
    ladder = [sum(v) / len(v) for _, v in sorted(rates.items())]
    print("    improvements/hour by d: " +
          ", ".join(f"d={d:g}s {sum(v) / len(v):.1f}" for d, v in sorted(rates.items())))
    assert metrics.is_monotone_decreasing(ladder), ladder

    ci = metrics.bootstrap(rates[sorted(rates)[0]])
    assert ci.n == len(rates[sorted(rates)[0]]) and ci.lo <= ci.point <= ci.hi, ci
    print(f"    bootstrap at d=0: {ci}")

    store.close()
    print("=== all figure self-tests passed (output is SYNTHETIC) ===")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[3]
    ap.add_argument("--demo", action="store_true",
                    help="render all three figures from a synthetic in-memory store")
    ap.add_argument("--db", type=Path, help="path to livelane.db")
    ap.add_argument("--runs", default=None,
                    help="comma-separated run ids (default: every run in the DB)")
    ap.add_argument("--out", type=Path, default=root / "var" / "figures")
    ap.add_argument("--metric", default="qor_max_delay_ns", choices=sorted(QOR_METRICS))
    ap.add_argument("--tag", default="", help="suffix for the output filenames")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if a.demo or not a.db:
        if not a.demo:
            ap.error("either --demo or --db is required")
        return _demo(Path(a.out), a.metric)

    # VariantStore is the harness' WRITER: it creates whatever it is pointed at.
    # An analysis run must never manufacture the ledger it claims to read, or a
    # mistyped --db produces three publishable-looking "no data" figures, an
    # empty .db in the measurement directory, and exit 0.
    if not a.db.is_file():
        ap.error(f"no measurement DB at {a.db}; refusing to create one")
    store = VariantStore(a.db, verbose=False)
    try:
        ids = [x.strip() for x in a.runs.split(",") if x.strip()] if a.runs else None
        paths = render_all(store, ids, a.out, metric=a.metric, tag=a.tag)
    finally:
        store.close()
    for p in paths:
        print(f"wrote {p} ({p.stat().st_size:,} bytes)")
    return 0


__all__ = ["fig1_best_so_far", "fig2_time_split", "fig3_improvements_vs_delay",
           "render_all", "load_runs", "accepted_curve", "verified_improvements",
           "time_split", "build_demo_store", "RunMeta", "ImprovementCount",
           "SplitTotals", "QOR_METRICS", "OKABE_ITO", "DELAY_LADDER"]


if __name__ == "__main__":
    raise SystemExit(main())
