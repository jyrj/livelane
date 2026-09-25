"""Lane L: driving LiveHD's ``lhd`` and parsing its result envelope.

Everything here is written against the envelope produced by ``write_result``
(``lhd/lhd_result.cpp:665``) and the fused ``synth`` path
(``lhd/lhd_kernel_synth.cpp:116``), read from the pinned checkout rather than
from documentation.  The protocol below mirrors LiveHD's own end-to-end test,
``lhd/tests/lhd_synth_test.sh``, which is the reference for what a correct
invocation and a correct warm/cold comparison look like.

Three source-verified facts shape this module:

1. **There is no top-level ``-o``.**  ``-o``/``--output`` belongs to
   ``lhd pyrope fmt`` alone (``lhd_options.cpp:605``).  Passing ``-o net.v`` to
   ``compile``/``synth`` silently produces nothing.  Outputs come from
   ``--emit KIND:PATH`` / ``--emit-dir KIND:DIR`` / ``--result-json``.
2. **Always pass ``--result-json``.**  Without it the envelope goes to stdout and
   obeys ``--diag-fmt``, which is *pretty human text on a tty*
   (``lhd_options.cpp:16``).  Parsing stdout is a trap.
3. **The compile cache is Pyrope-only.**  ``compile_cache.present/enabled`` are
   set only inside ``if (opts.language == "pyrope")``
   (``lhd_kernel_compile.cpp:1736``), so a SystemVerilog design gets **no**
   ``incremental.compile`` member at all, whatever ``--workdir`` or
   ``--set lhd.incremental`` say.  Only the ABC and STA caches reuse for SV.
   LiveLane's designs are SystemVerilog, so this is not an edge case, it is the
   normal path, and it is why the paper must not call lane L "incremental"
   without qualification.

Additionally, every incremental tier requires a **user-named** ``--workdir``:
``workdir()`` mints an mkdtemp scratch directory when none is given and sets
``workdir_scratch=true`` (``lhd_kernel_common.cpp:412``), and each cache gate
tests ``!workdir.empty() && !workdir_scratch``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from livelane.harness.run import ToolRun, run_tool
from livelane.state.reports import LecVerdict, QorReport, TimingReport

#: Readers accepted by ``--reader`` (lhd_options.cpp:438). Order is the fallback
#: ladder: slang is the default; yosys-slang is the escape hatch for SV that
#: slang rejects; yosys-verilog is plain-Verilog only.
READERS: tuple[str, ...] = ("slang", "yosys-slang", "yosys-verilog")

#: lhd's default Liberty basename (lhd.hpp:521), resolved under $HAGENT_TECH_DIR
#: unless --set synth.liberty=PATH overrides it (lhd_kernel_synth.cpp:94).
DEFAULT_LIBERTY = "sky130_fd_sc_hd__tt_025C_1v80.lib"


class LhdError(RuntimeError):
    pass


@dataclass
class LhdEnvelope:
    """The parsed ``--result-json`` object."""

    raw: dict[str, Any]
    path: Path | None = None

    # --- generic envelope fields (lhd_result.cpp:665) ---
    @property
    def status(self) -> str:
        return str(self.raw.get("status", ""))

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    @property
    def command(self) -> str:
        return str(self.raw.get("command", ""))

    @property
    def run_id(self) -> str | None:
        """Content hash, deterministic, never wall clock (lhd.hpp:215)."""
        return self.raw.get("run_id")

    @property
    def exit_code(self) -> int | None:
        return self.raw.get("exit_code")

    @property
    def error(self) -> dict[str, Any] | None:
        return self.raw.get("error")

    @property
    def diagnostics(self) -> dict[str, Any]:
        return self.raw.get("diagnostics_count") or {}

    @property
    def phases(self) -> dict[str, float]:
        """``phases[{name, ms}]`` flattened, the per-pass time split."""
        return {p["name"]: float(p["ms"]) for p in (self.raw.get("phases") or [])
                if "name" in p and "ms" in p}

    # --- incremental telemetry -----------------------------------------------
    @property
    def incremental(self) -> dict[str, Any]:
        """The single ``incremental`` member mirroring every reuse tier.

        Present even when a tier is off (``enabled: false`` plus zero counters),
        which is how we distinguish a disabled tier from an older binary that
        never emitted the member at all (lhd_result.cpp:728).
        """
        return self.raw.get("incremental") or {}

    def tier(self, name: str) -> dict[str, Any]:
        """``compile`` | ``abc`` | ``sta`` counters, or {} when absent."""
        return self.incremental.get(name) or {}

    @property
    def compile_cache_engaged(self) -> bool:
        """True only if the Pyrope-only compile tier actually ran.

        For SystemVerilog input this is always False, by construction.
        """
        return bool(self.tier("compile").get("enabled")) and (
            self.tier("compile").get("hits") or 0
        ) > 0

    def reuse_summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for t in ("compile", "abc", "sta"):
            d = self.tier(t)
            if d:
                out[t] = {
                    "enabled": d.get("enabled"),
                    "hits": d.get("hits"),
                    "misses": d.get("misses"),
                    "regions": d.get("regions"),
                }
        return out

    # --- qor payload (fused `synth`) -----------------------------------------
    @property
    def qor(self) -> dict[str, Any]:
        return self.raw.get("qor") or {}

    @property
    def abc(self) -> dict[str, Any]:
        """``qor.abc``, the pass.abc report (kind ``abc-map``)."""
        return self.qor.get("abc") or {}

    @property
    def sta(self) -> dict[str, Any]:
        """``qor.sta``, the pass.opentimer report (kind ``sta``)."""
        return self.qor.get("sta") or {}

    @property
    def lec(self) -> dict[str, Any]:
        return self.raw.get("lec") or {}

    # --- sim payload ---------------------------------------------------------
    @property
    def tests(self) -> list[dict[str, Any]]:
        """Per-test records from ``lhd sim``.

        Each carries ``test`` and ``status``, and on a failure also ``cycle``,
        ``failing_assert``, ``prp_file``, ``line`` and ``msg``, the assert is
        checked by running, not formally, so a failure names the cycle it first
        disagreed at.
        """
        return list(self.raw.get("tests") or [])

    @property
    def tests_failed(self) -> list[str]:
        """Names of the tests that did not pass."""
        return [t.get("test", "?") for t in self.tests
                if str(t.get("status", "")).lower() != "pass"]

    @property
    def sim_ok(self) -> bool:
        """True only when the envelope passed AND at least one test ran.

        Fail closed on an empty test list: ``lhd sim`` reports ``pass`` for a
        file whose ``test`` blocks it found none of, and treating "nothing ran"
        as "nothing failed" would silently admit an unverified design.
        """
        return self.ok and bool(self.tests) and not self.tests_failed

    def to_qor_report(self, top: str) -> QorReport:
        """Map lhd's envelope onto the lane-neutral :class:`QorReport`.

        ``abc.total`` carries ``{regions, gates, area, module_gates,
        module_area, max_delay, critical_region}`` (pass_abc.cpp:395).
        """
        total = self.abc.get("total") or {}
        max_delay = total.get("max_delay")
        # Prefer the whole-design STA critical path when opentimer ran: R3 in
        # LiveHD's own plan says the ABC per-region estimate is the cheap signal
        # and opentimer is what a change must not regress.
        designs = self.sta.get("designs") or []
        sta_delay = None
        if designs:
            vals = [d.get("max_delay") for d in designs if d.get("max_delay") is not None]
            if vals:
                sta_delay = max(vals)
        return QorReport(
            top=top,
            cells=total.get("gates"),
            area_um2=total.get("area"),
            max_delay_ns=sta_delay if sta_delay is not None else max_delay,
            valid=self.ok,
            message="" if self.ok else str((self.error or {}).get("message", "")),
            source="lhd",
            raw_json=json.dumps(self.qor, sort_keys=True) if self.qor else None,
        )

    def to_timing_report(self, top: str) -> TimingReport:
        """``sta.designs[]{module, max_delay, critical_pin, path[], endpoints[]}``."""
        designs = self.sta.get("designs") or []
        worst = None
        for d in designs:
            if d.get("max_delay") is None:
                continue
            if worst is None or d["max_delay"] > worst["max_delay"]:
                worst = d
        if worst is None:
            return TimingReport(top=top, slack_ns=None, max_delay_ns=None,
                                valid=False, message="no STA designs in envelope",
                                source="lhd")
        return TimingReport(
            top=top,
            slack_ns=None,  # lhd reports delay, not slack against a period
            max_delay_ns=worst.get("max_delay"),
            endpoint=worst.get("critical_pin"),
            path_summary=" -> ".join(str(x) for x in (worst.get("path") or [])[:8]),
            valid=self.ok,
            source="lhd",
            raw_json=json.dumps(self.sta, sort_keys=True),
        )

    def to_lec_verdict(self) -> LecVerdict:
        """``res.lec = {verdict: proven|refuted|unknown, solver, bounded, bound}``.

        lhd's ``unknown`` is mapped to ``error``: LiveLane's gate must never treat
        "the solver could not decide" as a pass (lhd_kernel_formal.cpp:5094).
        """
        l = self.lec
        v = str(l.get("verdict", "")) or "error"
        if v == "unknown":
            v = "error"
        if v not in LecVerdict.VALID:
            v = "error"
        bounded = l.get("bounded")
        msg = f"solver={l.get('solver')}"
        if bounded:
            msg += f" BOUNDED(bound={l.get('bound')}) -- not a full proof"
        return LecVerdict(verdict=v, message=msg, backend="lhd-lec")


@dataclass
class Lhd:
    """A pinned ``lhd`` binary plus the invariants every invocation needs."""

    binary: str
    workdir: Path
    liberty: str | None = None
    reader: str = "slang"
    incremental: bool = True
    extra_sets: dict[str, str] = field(default_factory=dict)
    log_dir: Path | None = None
    env: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.reader not in READERS:
            raise ValueError(f"unknown reader {self.reader!r}; lhd accepts {READERS}")
        self.workdir = Path(self.workdir)
        if not str(self.workdir):
            raise ValueError("a USER-NAMED --workdir is mandatory: every incremental "
                             "tier is gated on !workdir_scratch")

    # --- argv construction ---------------------------------------------------
    def _common(self, result_json: Path) -> list[str]:
        argv = ["-q", "--result-json", str(result_json),
                "--workdir", str(self.workdir),
                "--reader", self.reader]
        argv += ["--set", f"lhd.incremental={'true' if self.incremental else 'false'}"]
        if self.liberty:
            argv += ["--set", f"synth.liberty={self.liberty}"]
        for k, v in self.extra_sets.items():
            argv += ["--set", f"{k}={v}"]
        return argv

    def _common_sim(self, result_json: Path) -> list[str]:
        """Global flags ``sim`` accepts, deliberately NOT :meth:`_common`.

        ``sim`` rejects ``--reader``, and it does so in the worst possible way:
        measured, ``lhd sim <tb> --reader slang`` exits **11 with zero bytes on
        both stdout and stderr** and writes no ``--result-json``. Reusing
        :meth:`_common` here would therefore surface a malformed invocation as
        an unreadable design rather than as a usage error. ``synth.liberty`` is
        omitted for the same reason it is meaningless: nothing is mapped to a
        library on the simulation path.
        """
        argv = ["-q", "--result-json", str(result_json),
                "--workdir", str(self.workdir)]
        argv += ["--set", f"lhd.incremental={'true' if self.incremental else 'false'}"]
        for k, v in self.extra_sets.items():
            argv += ["--set", f"{k}={v}"]
        return argv

    def _invoke(self, args: Sequence[str], label: str,
                timeout_s: float | None) -> tuple[LhdEnvelope, ToolRun]:
        self.workdir.mkdir(parents=True, exist_ok=True)
        rj = self.workdir / f"{label}.result.json"
        if rj.exists():
            rj.unlink()
        argv = [self.binary, *args, *self._common(rj)]
        run = run_tool(argv, timeout_s=timeout_s, log_dir=self.log_dir,
                       label=label, env=self.env)
        if not rj.exists():
            raise LhdError(
                f"lhd produced no --result-json at {rj} (rc={run.returncode}, "
                f"timed_out={run.timed_out}); stderr: {run.stderr_path}"
            )
        env = LhdEnvelope(raw=json.loads(rj.read_text()), path=rj)
        return env, run

    def synth(self, sources: Sequence[str | os.PathLike[str]], top: str, *,
              emit_verilog: str | os.PathLike[str] | None = None,
              stats: bool = True, timeout_s: float | None = 3600.0,
              label: str = "lhd-synth") -> tuple[LhdEnvelope, ToolRun]:
        """Fused compile + color + abc + opentimer (lhd_kernel_synth.cpp:116).

        Produces ``<workdir>/synth/{lg,net,qor.json,timing.json}``.
        """
        args: list[str] = ["synth", *[str(s) for s in sources], "--top", top]
        if stats:
            args.append("--stats")
        if emit_verilog is not None:
            args += ["--emit", f"verilog:{emit_verilog}"]
        return self._invoke(args, label, timeout_s)

    def compile(self, sources: Sequence[str | os.PathLike[str]], top: str, *,
                emit_dir: str | os.PathLike[str] | None = None,
                timeout_s: float | None = 1800.0,
                label: str = "lhd-compile") -> tuple[LhdEnvelope, ToolRun]:
        args: list[str] = ["compile", *[str(s) for s in sources], "--top", top]
        if emit_dir is not None:
            args += ["--emit-dir", f"lg:{emit_dir}"]
        return self._invoke(args, label, timeout_s)

    def sim(self, sources: Sequence[str | os.PathLike[str]], *,
            test: str | None = None, setup_only: bool = False,
            run_only: bool = False, args: dict[str, str] | None = None,
            seed: int | None = None, restart_cycle: int | None = None,
            list_tests: bool = False, timeout_s: float | None = 1800.0,
            label: str = "lhd-sim") -> tuple[LhdEnvelope, ToolRun]:
        """Build and run a C++ simulation of a design's ``test`` blocks.

        The LAST source must hold the ``test`` blocks; earlier positionals are
        the design, or ``ln:DIR``/``lg:DIR`` artifacts a previous compile
        emitted, so a design compiled once simulates without re-reading source.

        ``setup_only`` and ``run_only`` split the two costs, and the split is
        the whole reason to expose this node. Measured on a small design, one
        host core: the simulation itself is ~2 ms while the host C++ build that
        precedes it is ~3,000 ms, and a re-run whose sources are byte-identical
        serves from cache at ~3 ms. An edit to the source invalidates that
        cache and the build returns (~1,950 ms measured on a one-line change),
        so a caller that edits every iteration should expect the build cost,
        not the cached one, and should measure rather than assume.

        Args:
            sources: ``.prp`` sources and/or ``ln:``/``lg:`` artifact dirs; the
                last entry holds the ``test`` blocks.
            test: Run a single named test rather than every one.
            setup_only: Elaborate and generate the C++ driver, do not build/run.
            run_only: Build (if needed) and run against an existing setup.
            args: ``test name(params)`` parameters, passed as ``--<key> <value>``.
            seed: Seed for the generated binary.
            restart_cycle: Resume the run from this cycle instead of from reset.
            list_tests: Enumerate the test blocks and exit.
            timeout_s: Wall-clock budget for the whole invocation.
            label: Log/result-json label.

        Returns:
            tuple: the parsed envelope and the :class:`ToolRun` that produced it.

        Raises:
            ValueError: if both ``setup_only`` and ``run_only`` are set.
        """
        if setup_only and run_only:
            raise ValueError("setup_only and run_only are mutually exclusive")
        argv: list[str] = ["sim", *[str(s) for s in sources]]
        if test:
            argv.append(test)
        if list_tests:
            argv.append("--list-tests")
        if setup_only:
            argv.append("--setup-only")
        if run_only:
            argv.append("--run-only")
        for k, v in (args or {}).items():
            argv += ["--arg", f"{k}={v}"]
        if seed is not None:
            argv += ["--seed", str(seed)]
        if restart_cycle is not None:
            argv += ["--restart-cycle", str(restart_cycle)]

        self.workdir.mkdir(parents=True, exist_ok=True)
        rj = self.workdir / f"{label}.result.json"
        if rj.exists():
            rj.unlink()
        cmd = [self.binary, *argv, *self._common_sim(rj)]
        run = run_tool(cmd, timeout_s=timeout_s, log_dir=self.log_dir,
                       label=label, env=self.env)
        if not rj.exists():
            raise LhdError(
                f"lhd sim produced no --result-json at {rj} (rc={run.returncode}, "
                f"timed_out={run.timed_out}); stderr: {run.stderr_path}"
            )
        return LhdEnvelope(raw=json.loads(rj.read_text()), path=rj), run

    def lec(self, impl: str, ref: str, *, impl_top: str | None = None,
            ref_top: str | None = None, timeout_s: float | None = 1800.0,
            label: str = "lhd-lec") -> tuple[LhdEnvelope, ToolRun]:
        args: list[str] = ["lec", "--impl", impl, "--ref", ref]
        if impl_top:
            args += ["--impl-top", impl_top]
        if ref_top:
            args += ["--ref-top", ref_top]
        return self._invoke(args, label, timeout_s)


def find_lhd(root: str | os.PathLike[str] | None = None) -> str | None:
    """Locate the built binary without guessing at a PATH entry."""
    root = Path(root or os.environ.get("LIVELANE_ROOT", "."))
    for cand in (root / "tools" / "bin" / "lhd",
                 root / "thirdparty" / "livehd" / "bazel-bin" / "lhd" / "lhd"):
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand)
    return None


if __name__ == "__main__":
    print("=== lhd envelope parser self-test (no binary required) ===")
    # Shaped exactly like the envelope lhd_synth_test.sh asserts on.
    sample = {
        "schema_version": 1, "tool": "lhd", "command": "synth", "status": "pass",
        "run_id": "0123456789abcdef", "exit_code": 0,
        "phases": [{"name": "pass.color", "ms": 12.5}, {"name": "pass.abc", "ms": 310.0},
                   {"name": "pass.opentimer", "ms": 88.0}],
        "incremental": {
            "compile": {"enabled": True, "hits": 4, "misses": 0},
            "abc": {"enabled": True, "regions": 3, "hits": 3, "misses": 0, "store_failed": 0},
            "sta": {"enabled": True, "hits": 1, "misses": 0},
        },
        "diagnostics_count": {"errors": 0, "warnings": 2},
        "qor": {"schema_version": 1, "kind": "synth", "top": "dut.top",
                "abc": {"kind": "abc-map",
                        "total": {"regions": 3, "gates": 6691, "area": 75664.0,
                                  "max_delay": 9.1, "critical_region": "r2"},
                        "incremental": {"hits": 3, "misses": 0}},
                "sta": {"kind": "sta", "time_unit": "ns",
                        "designs": [{"module": "top", "max_delay": 9.42,
                                     "critical_pin": "u_alu/Z",
                                     "path": ["a", "b", "c"], "endpoints": ["e"]},
                                    {"module": "sub", "max_delay": 3.1,
                                     "critical_pin": "u_x/Z", "path": []}]}},
    }
    e = LhdEnvelope(raw=sample)
    assert e.ok and e.command == "synth" and e.run_id == "0123456789abcdef"
    assert e.phases["pass.abc"] == 310.0
    print(f"    phases: {e.phases}")
    print(f"    reuse:  {e.reuse_summary()}")
    assert e.compile_cache_engaged is True

    q = e.to_qor_report("dut.top")
    # STA whole-design critical path must win over the ABC per-region estimate.
    assert q.max_delay_ns == 9.42, q
    assert q.cells == 6691 and q.area_um2 == 75664.0 and q.source == "lhd"
    assert "source" not in q.to_agent_dict(), "lane leak"
    print(f"    qor -> {q.to_agent_dict()}")

    t = e.to_timing_report("dut.top")
    assert t.max_delay_ns == 9.42 and t.endpoint == "u_alu/Z", t
    print(f"    timing -> endpoint={t.endpoint} delay={t.max_delay_ns}ns")

    # SystemVerilog input: no compile tier at all (Pyrope-only cache).
    sv = dict(sample)
    sv["incremental"] = {"abc": {"enabled": True, "regions": 3, "hits": 3, "misses": 0}}
    esv = LhdEnvelope(raw=sv)
    assert esv.tier("compile") == {}, "SV must have no compile tier"
    assert esv.compile_cache_engaged is False
    print(f"    SystemVerilog reuse (no compile tier): {esv.reuse_summary()}")

    # An undecided solver must never read as a pass.
    for verdict, want in (("proven", "proven"), ("refuted", "refuted"),
                          ("unknown", "error"), ("weird", "error")):
        ev = LhdEnvelope(raw={"status": "pass",
                              "lec": {"verdict": verdict, "solver": "z3"}})
        got = ev.to_lec_verdict()
        assert got.verdict == want, f"{verdict} -> {got.verdict}, want {want}"
        assert got.is_proven == (want == "proven")
    print("    lec: unknown/unrecognised map to 'error', never to a pass")

    b = LhdEnvelope(raw={"status": "pass",
                         "lec": {"verdict": "proven", "solver": "z3",
                                 "bounded": True, "bound": 12}}).to_lec_verdict()
    assert "BOUNDED" in b.message, b
    print(f"    bounded proof flagged: {b.message}")

    try:
        Lhd(binary="/nonexistent", workdir=Path("/tmp/x"), reader="not-a-reader")
        raise AssertionError("bad reader accepted")
    except ValueError as ex:
        print(f"    reader validated: {ex}")

    l = Lhd(binary="/opt/lhd", workdir=Path("/tmp/wd"), liberty="/pdk/sky130.lib")
    argv = [l.binary, "synth", "a.sv", "--top", "T", "--stats",
            *l._common(Path("/tmp/wd/x.json"))]
    assert "-o" not in argv, "there is no top-level -o in lhd"
    assert "--result-json" in argv and "--workdir" in argv
    assert "lhd.incremental=true" in argv and "synth.liberty=/pdk/sky130.lib" in argv
    print(f"    argv: {' '.join(argv[1:])}")
    print("=== all lhd parser self-tests passed ===")
