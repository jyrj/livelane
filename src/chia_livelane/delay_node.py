"""Back-compatible alias for :mod:`chia_livelane.base.delay`.

The delay node lives in ``chia_livelane/base/delay.py``. LiveLane's harness,
its tests and its measurement scripts import ``chia_livelane.delay_node``, so
this module re-exports the same names from there. New code should import from
:mod:`chia_livelane.base.delay`.
"""

from __future__ import annotations

from chia_livelane.base.delay import (DelayMode, DelayNode, DelayRecord,
                                      delay_seconds)

__all__ = ["DelayNode", "DelayMode", "DelayRecord", "delay_seconds"]
