"""Formal verification nodes: tools that prove things about a design.

Today that is one node, an equivalence-check gate backed by YosysHQ ``eqy``,
with an optional second backend for cross-checking.
"""

from chia_livelane.formal.lec_gate import (  # noqa: F401
    DEFAULT_STRATEGIES, ERROR, PROVEN, REFUTED, SKIPPED, TIMEOUT, UNDECIDED,
    UNDEF_INIT_VALUES, VALID_VERDICTS, KeplerBackend, LecGateNode, LecResult,
    auto_jobs, cross_check, lec_gate, parse_eqy_log, parse_kepler_log,
)
