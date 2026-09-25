"""Formal verification nodes for CHIA.

Staged as ``chia/formal/``: a new sibling of ``chia/vlsi`` for tools that prove
things about a design rather than implement it.  Today that is one node, an
equivalence-check gate backed by YosysHQ ``eqy``.

The re-export below matches ``chia/vlsi/__init__.py``, which lifts its package's
public names to the package root.
"""

from chia_livelane.formal.lec_gate import (  # noqa: F401
    DEFAULT_STRATEGIES, ERROR, PROVEN, REFUTED, SKIPPED, TIMEOUT, UNDECIDED,
    VALID_VERDICTS, LecGateNode, LecResult, cross_check, lec_gate,
    parse_eqy_log,
)
