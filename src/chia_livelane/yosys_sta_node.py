"""Open-source QoR evaluation as a CHIA node: Yosys + ABC + OpenSTA.

CHIA ships a Cadence Genus path, which needs a licence server. That makes every
CHIA RTL loop unreproducible for anyone outside a licensed site, including the
reviewers of a paper that uses one. This node is the open-source equivalent.

Why this delegates to ``LaneSEvaluator`` instead of driving ``YosysStaLane``
---------------------------------------------------------------------------
An earlier version of this node built its own Yosys read line. That is one line
of code and it silently changed the answer: ``read_verilog -sv`` and
``read_slang`` on the SAME picorv32, same recipe, same Liberty, same clock give

    read_slang        6563 cells   73992.2 um2   12.7612 ns
    read_verilog -sv  6691 cells   75663.8 um2   14.7771 ns

a 15.8% difference in the critical path, the very quantity the experiment
optimises, from the front end alone. The lane evaluator already resolves
``read_slang --top <top> -F <filelist>`` correctly and reports its own
:meth:`config`, so the node delegates rather than reimplementing. One code path,
one front end, no divergence.

The returned dict carries that ``config`` back with the numbers, so any consumer
can check two results share a timing basis before comparing them.

Returns a plain dict so the result crosses Ray's object store cleanly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:  # CHIA is optional: the node is unit-testable without a cluster.
    from chia.base.ChiaFunction import ChiaFunction
except Exception:  # pragma: no cover
    def ChiaFunction(**_kwargs):  # type: ignore[misc]
        def deco(fn):
            return fn
        return deco


#: Resource token for the synthesis lane. Bind it in the cluster YAML
#: (``available_node_types.<type>.resources.synth``) to cap how many Yosys
#: processes run at once, synthesis is memory-hungry, and oversubscribing it
#: is the fastest way to make wall-clock measurements meaningless.
SYNTH_RESOURCE = "synth"


@ChiaFunction(resources={SYNTH_RESOURCE: 1})
def yosys_sta_qor(sources: list[str], top: str, *, yosys: str, sta: str,
                  liberty: str, workdir: str = ".",
                  clock_port: str | None = None,
                  clock_period_ns: float | None = None,
                  read_cmd: str = "read_slang",
                  script_name: str = "baseline-2026-09-02",
                  filelist: str | None = None,
                  timeout_s: float = 7200.0) -> dict[str, Any]:
    """CHIA node form of the Yosys+OpenSTA lane.

    ``sources`` are absolute paths on the worker; pass ``filelist`` instead for
    a multi-file design. ``liberty`` MUST be the same file in every arm of a
    comparison, and so must ``read_cmd`` and ``script_name``, see the module
    docstring for what happens when the front end differs.

    A SEQUENTIAL design MUST be given ``clock_port``/``clock_period_ns``, or
    OpenSTA reports the longest unconstrained combinational I/O path instead of
    the register-to-register critical path (0.196 ns against a real 12.761 ns on
    picorv32).
    """
    from livelane.evaluators import LaneSEvaluator
    from livelane.nodes.yosys_sta import SCRIPTS

    ev = LaneSEvaluator(
        yosys=yosys, sta=sta, liberty=liberty, script=SCRIPTS[script_name],
        read_cmd=read_cmd, clock_port=clock_port,
        clock_period_ns=clock_period_ns, timeout_s=timeout_s,
    )
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    qor, timing = ev.evaluate(list(sources), top, wd, filelist=filelist)
    return {
        "valid": qor.valid,
        "message": qor.message,
        "cells": qor.cells,
        "area_um2": qor.area_um2,
        "max_delay_ns": qor.max_delay_ns,
        "slack_ns": qor.slack_ns,
        "cpu_s": ev.cpu_s,
        "wall_s": qor.wall_s,
        "peak_rss_kb": qor.peak_rss_kb,
        "timing_ok": timing is not None,
        # The basis these numbers were measured on. Two results are comparable
        # only if this dict matches.
        "config": ev.config(),
    }


__all__ = ["yosys_sta_qor", "SYNTH_RESOURCE"]
