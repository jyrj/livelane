"""Machine load, recorded beside every timing, and a guard against measuring
on a busy box.

A duration without its load is not interpretable. Measured on this machine, the
same cold picorv32 equivalence proof took **23.9s** idle and **55.7s** at load
58 on 24 cores: identical work, 2.3x apart, with nothing in the result to say
which one you were looking at.

Verdicts, partition counts and reuse percentages are unaffected by load, so a
soundness result measured on a busy machine is still a soundness result. Times
are not, and neither is any speedup computed from them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = ["HostLoad", "host_load", "load_warning"]


@dataclass(frozen=True)
class HostLoad:
    load1: float | None = None
    load5: float | None = None
    load15: float | None = None
    cpus: int | None = None

    @property
    def per_cpu(self) -> float | None:
        """Runnable processes per core. ~1.0 means saturated."""
        if self.load1 is None or not self.cpus:
            return None
        return self.load1 / self.cpus

    def as_dict(self) -> dict:
        d = {"load1": self.load1, "load5": self.load5, "load15": self.load15,
             "cpus": self.cpus}
        return {k: v for k, v in d.items() if v is not None}


def host_load() -> HostLoad:
    try:
        one, five, fifteen = os.getloadavg()
    except (OSError, AttributeError):        # not available on every platform
        return HostLoad(cpus=os.cpu_count())
    return HostLoad(round(one, 2), round(five, 2), round(fifteen, 2),
                    os.cpu_count())


def load_warning(threshold: float = 0.5) -> str | None:
    """A sentence to print, or None when the machine is quiet enough.

    *threshold* is load per core. The default of 0.5 is deliberately low: by
    the time the machine is merely "busy" the timings are already unusable, and
    the cost of a spurious warning is one line of output.

    This warns rather than refuses. Refusing would be wrong, a run whose
    point is the verdict is perfectly valid on a loaded box, and blocking it
    would trade a real result for a clean number nobody asked for.
    """
    hl = host_load()
    pc = hl.per_cpu
    if pc is None or pc < threshold:
        return None
    return (f"load {hl.load1} on {hl.cpus} cores ({pc:.2f} per core): timings "
            f"from this run are upper bounds and are not comparable with "
            f"timings taken on a quiet machine. Verdicts, partition counts and "
            f"reuse percentages are unaffected.")
