"""Back-compatible alias for :mod:`chia_livelane.formal.lec_gate`.

The gate lives in ``chia_livelane/formal/lec_gate.py``. LiveLane's loop, sweep
and gate scripts import ``chia_livelane.lec_gate``, so this module re-exports
the same names from there. New code should import from
:mod:`chia_livelane.formal.lec_gate`.
"""

from __future__ import annotations

from chia_livelane.formal.lec_gate import (DEFAULT_STRATEGIES, ERROR, PROVEN,
                                           REFUTED, SKIPPED, TIMEOUT,
                                           UNDECIDED, UNDEF_INIT_VALUES,
                                           VALID_VERDICTS, KeplerBackend,
                                           LecGateNode, LecResult, auto_jobs,
                                           cross_check, lec_gate,
                                           parse_eqy_log, parse_kepler_log)

__all__ = ["LecGateNode", "LecResult", "lec_gate", "cross_check",
           "parse_eqy_log", "DEFAULT_STRATEGIES", "VALID_VERDICTS",
           "KeplerBackend", "parse_kepler_log",
           "UNDEF_INIT_VALUES", "auto_jobs",
           "PROVEN", "REFUTED", "UNDECIDED", "ERROR", "TIMEOUT", "SKIPPED"]
