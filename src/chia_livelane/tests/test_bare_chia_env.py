"""Proof that each node runs in a bare CHIA environment, with no ``livelane``.

Why this file exists
--------------------
A CHIA user adopting these nodes will have ``chia`` and nothing else.  Every other test in
this package runs inside the LiveLane checkout, where ``livelane`` happens to be
importable, so none of them can tell the difference between "this node is
self-contained" and "this node quietly imports the research harness".  A missing
import would then surface as a CI failure in *someone else's* repository.

So each test here spawns a subprocess that is a bare CHIA environment by
construction:

1. the package is copied to a scratch directory and imported from there,
   so nothing depends on this repository's ``src/`` layout;
2. this repository's ``src`` directory, which the editable install puts on
   ``sys.path``, is removed from the child's path;
3. a ``sys.meta_path`` finder raises on ANY attempt to import ``livelane`` or a
   submodule of it, so the proof does not rest on path hygiene alone: if a node
   reaches for the harness at any point, the test fails loudly and names it.

The child then exercises the node for real, including a real ``eqy`` run when
the binary is present, rather than merely importing it, because an import that
succeeds says nothing about a node whose dependency is inside a function body.
``test_livelane_wrapper_needs_livelane`` demonstrates exactly that failure
mode on the one module in this package that still has it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]      # .../src/chia_livelane
SRC_DIR = PKG_ROOT.parent                           # .../src

# Installed into the child before anything else. Two independent barriers: the
# development checkout is off sys.path, AND any import of livelane raises.
_BOOTSTRAP = '''
import sys

_DEV_SRC = {dev_src!r}
_STAGE = {stage!r}


class _NoLiveLane:
    """Refuse to import the research harness, wherever it may be installed."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "livelane" or fullname.startswith("livelane."):
            raise ModuleNotFoundError(
                "BARE CHIA ENV: {{}} is not available to a CHIA user"
                .format(fullname))
        return None


sys.meta_path.insert(0, _NoLiveLane())
sys.path[:] = [p for p in sys.path if p and Path_resolve(p) != _DEV_SRC]
sys.path.insert(0, _STAGE)
'''


def _bare_env(tmp_path: Path, body: str) -> subprocess.CompletedProcess:
    """Run ``body`` in a subprocess that has CHIA but not LiveLane.

    Args:
        tmp_path (Path): Scratch directory; the package is copied into it.
        body (str): Python source to run after the barriers are installed.

    Returns:
        subprocess.CompletedProcess: With text stdout/stderr, never raising, so
        a failing child's output can be shown in the assertion message.
    """
    stage = tmp_path / "bare"
    stage.mkdir(exist_ok=True)
    dest = stage / "chia_livelane"
    if not dest.exists():
        shutil.copytree(PKG_ROOT, dest,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    prelude = (
        "import os\n"
        "def Path_resolve(p):\n"
        "    try:\n"
        "        return os.path.realpath(p)\n"
        "    except Exception:\n"
        "        return p\n"
        + _BOOTSTRAP.format(dev_src=str(SRC_DIR.resolve()), stage=str(stage))
    )
    script = prelude + "\n" + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=600,
                          cwd=str(tmp_path))


def _ok(res: subprocess.CompletedProcess) -> str:
    assert res.returncode == 0, (
        f"bare CHIA environment failed (rc={res.returncode})\n"
        f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}")
    return res.stdout


class TestTheHarnessIsReallyGone:
    def test_livelane_cannot_be_imported(self, tmp_path):
        # If this ever passes silently, every other test in this file is
        # meaningless, so assert the barrier itself first.
        res = _bare_env(tmp_path, """
            try:
                import livelane
            except ModuleNotFoundError as e:
                print("BLOCKED:", e)
            else:
                raise SystemExit("livelane was importable; the barrier failed")
            print("OK")
        """)
        out = _ok(res)
        assert "BLOCKED: BARE CHIA ENV" in out

    def test_chia_itself_is_present(self, tmp_path):
        # The environment must be bare of LiveLane but NOT bare of CHIA, or the
        # tests below would pass against the no-op decorator fallback.
        res = _bare_env(tmp_path, """
            from chia.base.ChiaFunction import ChiaFunction
            print("CHIA:", ChiaFunction.__module__)
        """)
        assert "CHIA: chia.base.ChiaFunction" in _ok(res)


class TestDelayNodeInBareEnv:
    def test_imports_and_injects_real_latency(self, tmp_path):
        res = _bare_env(tmp_path, """
            import time
            from chia_livelane.base.delay import (DelayMode, DelayNode,
                                                  delay_seconds)

            d = DelayNode(seconds=0.2, verbose=False)
            t0 = time.monotonic()
            val, rec = d.around_call(lambda: 6 * 7)
            took = time.monotonic() - t0
            assert val == 42, val
            assert 0.2 <= took < 5.0, took
            assert rec.slept_s >= 0.19, rec

            assert DelayNode(0.0, verbose=False).is_identity
            assert DelayNode(1.0, mode=DelayMode.FLOOR,
                             verbose=False).sleep_for(2.0) == 0.0
            assert delay_seconds(0.01) >= 0.01
            print("DELAY OK slept=%.3f" % rec.slept_s)
        """)
        assert "DELAY OK" in _ok(res)

    def test_decorator_is_the_real_chiafunction(self, tmp_path):
        # The module falls back to a no-op decorator when CHIA is absent. In a
        # real CHIA environment the fallback must NOT be what is in force, or
        # the node would silently lose its scheduling options.
        res = _bare_env(tmp_path, """
            from chia_livelane.base.delay import delay_seconds
            opts = getattr(delay_seconds, "_chia_options", None)
            assert opts is not None, "fallback decorator in force under real CHIA"
            assert opts.get("num_cpus") == 0, opts
            assert hasattr(delay_seconds, "chia_remote")
            print("DECORATOR OK", opts)
        """)
        assert "DECORATOR OK" in _ok(res)


class TestLecGateInBareEnv:
    def test_verdict_logic_and_subprocess_path(self, tmp_path):
        # Exercises the real subprocess path with stub checkers, so the fail-
        # closed behaviour is proven end to end and not just at the parser.
        res = _bare_env(tmp_path, """
            import os, stat
            from pathlib import Path
            from chia_livelane.formal.lec_gate import (ERROR, PROVEN, REFUTED,
                                                       TIMEOUT, UNDECIDED,
                                                       LecGateNode,
                                                       parse_eqy_log)

            wd = Path("wd"); wd.mkdir(exist_ok=True)
            gold = wd / "gold.v"; gold.write_text("// placeholder\\n")
            gate = wd / "gate.v"; gate.write_text("// placeholder\\n")
            SRC = ([str(gold)], [str(gate)])

            def stub(name, body):
                # ABSOLUTE. check() runs the checker with cwd=workdir, so a
                # relative executable path would be resolved against workdir
                # rather than against the caller's cwd.
                p = (wd / name).resolve()
                p.write_text("#!/bin/sh\\n" + body)
                p.chmod(0o755)
                return str(p)

            proven = stub("ok", "echo 'EQY Successfully proved designs "
                                "equivalent'\\nexit 0\\n")
            refute = stub("no", "echo \\"EQY run: Could not prove equivalence of "
                                "partition 'nerv.next_rd' using strategy 'sby': "
                                "partitions not equivalent\\"\\n"
                                "echo 'EQY Warning: Failed to prove equivalence "
                                "for 1/44 partitions:'\\n"
                                "echo 'EQY Failed to prove equivalence of "
                                "partition nerv.next_rd'\\nexit 2\\n")
            unknown = stub("dunno", "echo \\"EQY run: Could not prove equivalence "
                                    "of partition 'p.count' using strategy 'sby': "
                                    "equivalence unknown\\"\\n"
                                    "echo 'EQY Warning: Failed to prove "
                                    "equivalence for 1/2 partitions:'\\n"
                                    "echo 'EQY Failed to prove equivalence of "
                                    "partition p.count'\\nexit 2\\n")
            liar = stub("liar", "echo 'all good'\\nexit 0\\n")

            def run(exe, name, **kw):
                return LecGateNode(eqy=exe, workdir=wd, verbose=False,
                                   **kw).check(*SRC, "t", name=name)

            r = run(proven, "p")
            assert r.verdict == PROVEN and r.admits_edit, r
            r = run(refute, "r")
            assert r.verdict == REFUTED and not r.admits_edit and r.is_refutation, r
            r = run(unknown, "u")
            assert r.verdict == UNDECIDED and not r.admits_edit, r
            assert not r.is_refutation, "undecided must not count as a refutation"
            r = run(liar, "l")
            assert r.verdict == ERROR and not r.admits_edit, r
            r = run("definitely-not-eqy", "m")
            assert r.verdict == ERROR and "not found" in r.message, r

            # An operator error (missing source) must be reported as one,
            # rather than reaching eqy and coming back as an unparsable rc=2.
            r = LecGateNode(eqy=proven, workdir=wd, verbose=False).check(
                [str(gold)], [str(wd / "absent.v")], "t", name="absent")
            assert r.verdict == ERROR and "absent.v" in r.message, r
            print("GATE OK")
        """)
        assert "GATE OK" in _ok(res)

    @pytest.mark.skipif(shutil.which("eqy") is None,
                        reason="eqy not on PATH; the stub-based test still runs")
    def test_real_eqy_proves_and_refutes(self, tmp_path):
        # The strongest form of the proof: a real solver, in a bare CHIA
        # environment, on a design pair whose answer is known by construction.
        res = _bare_env(tmp_path, """
            from pathlib import Path
            from chia_livelane.formal.lec_gate import (DEFAULT_STRATEGIES,
                                                       PROVEN, REFUTED,
                                                       LecGateNode)

            wd = Path("lec"); wd.mkdir(exist_ok=True)
            (wd / "gold.v").write_text(
                "module tiny(input clk, input [3:0] a, input [3:0] b,"
                " output reg [3:0] y);\\n"
                "  always @(posedge clk) y <= a + b;\\nendmodule\\n")
            # Commuted operands: different text, same function.
            (wd / "ok.v").write_text(
                "module tiny(input clk, input [3:0] a, input [3:0] b,"
                " output reg [3:0] y);\\n"
                "  wire [3:0] s = b + a;\\n"
                "  always @(posedge clk) y <= s;\\nendmodule\\n")
            # A real functional bug.
            (wd / "bad.v").write_text(
                "module tiny(input clk, input [3:0] a, input [3:0] b,"
                " output reg [3:0] y);\\n"
                "  always @(posedge clk) y <= a - b;\\nendmodule\\n")

            def check(cand, name):
                n = LecGateNode(workdir=wd, strategies=DEFAULT_STRATEGIES,
                                timeout_s=300.0, verbose=False)
                return n.check([str(wd / "gold.v")], [str(wd / cand)], "tiny",
                               name=name)

            good = check("ok.v", "real_ok")
            assert good.verdict == PROVEN, good
            assert good.admits_edit and not good.is_refutation

            bad = check("bad.v", "real_bad")
            assert bad.verdict == REFUTED, bad
            assert bad.is_refutation and not bad.admits_edit
            print("REAL EQY OK proven=%.2fs refuted=%.2fs"
                  % (good.wall_s, bad.wall_s))
        """)
        assert "REAL EQY OK" in _ok(res)

    def test_ladder_prefers_a_counterexample_over_an_undecided_strategy(self, tmp_path):
        # Real ladder behaviour: `sat` reports the partition unknown and `sby`
        # then refutes it, so ONE log carries both reasons for the SAME
        # partition. Returning UNDECIDED there would discard a real refutation.
        res = _bare_env(tmp_path, """
            from chia_livelane.formal.lec_gate import parse_eqy_log
            log = (
                "EQY run: Could not prove equivalence of partition 'tiny.y' "
                "using strategy 'induct': equivalence unknown\\n"
                "EQY run: Could not prove equivalence of partition 'tiny.y' "
                "using strategy 'sby': partitions not equivalent\\n"
                "EQY Warning: Failed to prove equivalence for 1/1 partitions:\\n"
                "EQY Failed to prove equivalence of partition tiny.y\\n"
                "EQY DONE (FAIL, rc=2)")
            p = parse_eqy_log(log)
            assert p["not_equivalent_partitions"] == ["tiny.y"], p
            assert p["unknown_partitions"] == ["tiny.y"], p
            print("LADDER OK")
        """)
        assert "LADDER OK" in _ok(res)


class TestYosysStaInBareEnv:
    def test_parsers_and_guards(self, tmp_path):
        res = _bare_env(tmp_path, """
            from pathlib import Path
            from chia_livelane.vlsi.yosys_sta import (SCRIPTS, YosysStaNode,
                                                      parse_sta_report,
                                                      parse_yosys_stat,
                                                      yosys_sta_qor)

            cells, area = parse_yosys_stat(
                "     7692 4.32E+04 cells\\n"
                "   Chip area for module '\\\\Alu': 43193.926400\\n")
            assert cells == 7692 and abs(area - 43193.9264) < 1e-6, (cells, area)
            assert parse_yosys_stat("nothing") == (None, None)

            p = parse_sta_report(
                "Startpoint: a\\nEndpoint: b\\n"
                "             9.4210   data arrival time\\n"
                "             0.5790   slack (MET)\\n")
            assert abs(p["max_delay_ns"] - 9.421) < 1e-9, p

            lib = Path("fake.lib"); lib.write_text("library(x){}")
            n = YosysStaNode(yosys="yosys", sta="sta", liberty=str(lib),
                             workdir=Path("wd"), verbose=False)
            assert n.constrained is False
            assert "set_max_delay" in n._sta_script(Path("n.v"), "t")
            n2 = YosysStaNode(yosys="yosys", sta="sta", liberty=str(lib),
                              workdir=Path("wd"), clock_period_ns=10.0,
                              clock_port="clk", verbose=False)
            assert n2.constrained and "create_clock" in n2._sta_script(
                Path("n.v"), "t")

            try:
                YosysStaNode(yosys="yosys", sta="sta", liberty="/nope.lib",
                             workdir=Path("wd"))
            except FileNotFoundError:
                pass
            else:
                raise SystemExit("missing Liberty accepted")

            assert set(SCRIPTS) == {"baseline-flat", "tuned-hier"}
            print("YOSYS_STA OK")
        """)
        assert "YOSYS_STA OK" in _ok(res)

    def test_instrumented_runner_measures_in_bare_env(self, tmp_path):
        res = _bare_env(tmp_path, """
            from chia_livelane.vlsi.tool_run import ToolNotFound, run_tool

            r = run_tool(["/bin/sh", "-c", "sleep 0.2"], verbose=False)
            assert r.ok and 0.15 < r.wall_s < 5.0, r
            r = run_tool(["/bin/sh", "-c", "exit 7"], verbose=False)
            assert r.returncode == 7 and not r.ok, r
            try:
                run_tool(["definitely-not-a-real-binary-xyz"], verbose=False)
            except ToolNotFound:
                pass
            else:
                raise SystemExit("missing binary did not raise")
            print("TOOL_RUN OK")
        """)
        assert "TOOL_RUN OK" in _ok(res)

    def test_node_falls_back_to_the_local_tool_run(self, tmp_path):
        # Installed inside CHIA the node imports `chia.vlsi.tool_run`; run from
        # this checkout it falls back to the sibling module. Both paths have to
        # work.
        res = _bare_env(tmp_path, """
            import chia_livelane.vlsi.yosys_sta as m
            assert m.run_tool.__module__.endswith("tool_run"), m.run_tool.__module__
            print("FALLBACK OK ->", m.run_tool.__module__)
        """)
        assert "FALLBACK OK" in _ok(res)


class TestWhatNeedsLivelane:
    def test_livelane_wrapper_needs_livelane(self, tmp_path):
        # chia_livelane/yosys_sta_node.py imports livelane.nodes.yosys_sta
        # inside the function body, so it IMPORTS fine and only fails when
        # called, which is precisely the failure a CHIA user would hit at
        # run time rather than at import time. Pinned here so the
        # distinction between it and chia_livelane.vlsi.yosys_sta is asserted
        # rather than remembered.
        res = _bare_env(tmp_path, """
            import chia_livelane.yosys_sta_node as m   # import succeeds
            fn = getattr(m.yosys_sta_qor, "_chia_original", m.yosys_sta_qor)
            try:
                fn([], "t", yosys="yosys", sta="sta", liberty="/nope.lib")
            except ModuleNotFoundError as e:
                assert "livelane" in str(e), e
                print("NEEDS LIVELANE, as documented:", e)
            else:
                raise SystemExit(
                    "chia_livelane.yosys_sta_node ran without livelane; "
                    "update chia_livelane/__init__.py, it is now self-contained")
        """)
        assert "NEEDS LIVELANE" in _ok(res)
