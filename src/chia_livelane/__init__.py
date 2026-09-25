"""CHIA nodes for equivalence checking, open-source QoR, and controlled latency.

Each node is a plain ``@ChiaFunction``: call it directly to run in process, or
``.chia_remote(...)`` to dispatch it onto a worker whose cluster entry declares
its resource token.

* :mod:`chia_livelane.formal.lec_gate`, prove or refute a candidate RTL edit
  against its parent. Only a positive proof admits an edit; a crash, a timeout,
  a missing binary, an unparsable log and an undecided partition are all
  non-admitting, and a refutation is reported separately from an undecided
  result because only the first is evidence that the designs differ.
* :mod:`chia_livelane.formal.second_stage`, settle an edit the partitioned
  gate refused: an unbounded proof of the edited instance, both copies powered
  up to the same arbitrary state with reset asserted, and an edit to a package
  proven where its changed types reach.
* :mod:`chia_livelane.vlsi.yosys_sta`, cells, area, worst slack and the
  critical path from Yosys + ABC + OpenSTA against any Liberty, with no
  commercial licence in the loop.
* :mod:`chia_livelane.vlsi.tool_run`, the instrumented subprocess runner the
  QoR node uses: wall-clock, CPU time and peak RSS charged to the specific
  child, so concurrent runs cannot contaminate each other's numbers.
* :mod:`chia_livelane.sim.verilator`, run a design against its own testbench;
  cheap evidence to reject with, never to accept on.
* :mod:`chia_livelane.base.delay`, add a controlled delay to any evaluator, so
  a loop can be run at a chosen feedback latency with everything else fixed.

``chia_livelane.lec_gate`` and ``chia_livelane.delay_node`` re-export the gate
and the delay node under their older import paths. ``chia_livelane.yosys_sta_node``
wraps LiveLane's own QoR evaluator and needs the ``livelane`` package; every
other node runs in a bare CHIA environment, which
``chia_livelane/tests/test_bare_chia_env.py`` checks.

The layout mirrors ``chia``'s own (``base``/``vlsi``/``sim``, plus ``formal``
for tools that prove things about a design rather than implement it).
"""

__all__ = ["base", "formal", "vlsi", "delay_node", "lec_gate", "yosys_sta_node"]
