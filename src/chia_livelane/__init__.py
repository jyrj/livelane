"""CHIA nodes for equivalence checking, open-source QoR, and controlled latency.

Each node is a plain ``@ChiaFunction``: call it directly to run in process, or
``.chia_remote(...)`` to dispatch it onto a worker whose cluster entry declares
its resource token.

* :mod:`chia_livelane.formal.lec_gate` -- prove or refute a candidate RTL edit
  against its parent. Only a positive proof admits an edit; a crash, a timeout,
  a missing binary, an unparsable log and an undecided partition are all
  non-admitting, and a refutation is reported separately from an undecided
  result because only the first is evidence that the designs differ.
* :mod:`chia_livelane.vlsi.yosys_sta` -- cells, area, worst slack and the
  critical path from Yosys + ABC + OpenSTA against any Liberty, with no
  commercial licence in the loop.
* :mod:`chia_livelane.vlsi.tool_run` -- the instrumented subprocess runner the
  QoR node uses: wall-clock, CPU time and peak RSS charged to the specific
  child, so concurrent runs cannot contaminate each other's numbers.
* :mod:`chia_livelane.base.delay` -- add a controlled delay to any evaluator, so
  a loop can be run at a chosen feedback latency with everything else fixed.

The layout mirrors ``chia``'s own (``base``/``vlsi``, plus ``formal`` for tools
that prove things about a design rather than implement it).
"""

__all__ = ["base", "formal", "vlsi"]
