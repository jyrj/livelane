"""``livelane``, one entry point for the whole pipeline.

The project is deliberately one pipeline rather than a pile of scripts, so there
is one command with subcommands that run it end to end:

    livelane doctor                 # is the environment actually usable?
    livelane gate g0|g1|g2|g4       # the validation gates
    livelane calibrate              # lane-S calibration
    livelane sweep --dry-run        # the experiment matrix, without spending
    livelane sweep                  # the experiment
    livelane judge <run_id>         # re-score finals through the ONE neutral flow
    livelane figures                # the paper's figures from the database
    livelane status                 # what has been run so far

``doctor`` exists because every failure this project has hit so far was an
environment problem wearing a tool's clothes: a ccache wrapper Bazel could not
write through, a CMake libdir that differs on RedHat, a Gemini model that is
listed globally and only answers on ``global``. It checks the things that have
actually broken, not the things that theoretically could.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from pathlib import Path


def _root() -> Path:
    env = os.environ.get("LIVELANE_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2]


# --- doctor ------------------------------------------------------------------

def cmd_doctor(args: argparse.Namespace) -> int:
    root = _root()
    ok = True

    def check(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        mark = "OK  " if good else "FAIL"
        print(f"  [{mark}] {label}" + (f": {detail}" if detail else ""))

    print("=== tools (must resolve to OUR prefix, not the host) ===")
    for t in ("yosys", "yosys-abc", "sta", "eqy", "sby", "verilator", "lhd"):
        p = root / "tools" / "bin" / t
        found = shutil.which(t) or ""
        check(t, p.exists(),
              "ours" if found.startswith(str(root)) else f"resolves to {found or 'nothing'}")

    print("\n=== PDK (both lanes MUST share one Liberty) ===")
    lib = os.environ.get("LIVELANE_LIBERTY", "")
    check("LIVELANE_LIBERTY set", bool(lib))
    check("Liberty exists", bool(lib) and Path(lib).exists(), lib[-60:] if lib else "")
    check("PDK version pinned", bool(os.environ.get("CIEL_PDK_VERSION")),
          os.environ.get("CIEL_PDK_VERSION", "")[:16])

    print("\n=== python ===")
    check("interpreter <3.14", sys.version_info < (3, 14),
          f"{sys.version_info.major}.{sys.version_info.minor} "
          f"(ray 2.54 has no cp314 wheels)")
    for mod in ("chia", "google.genai"):
        try:
            __import__(mod)
            check(f"import {mod}", True)
        except Exception as e:
            check(f"import {mod}", False, str(e)[:50])

    print("\n=== vertex seat ===")
    proj = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    loc = os.environ.get("GOOGLE_CLOUD_LOCATION", "")
    check("GOOGLE_CLOUD_PROJECT", bool(proj), proj)
    # gemini-3.x answers ONLY on `global`; this has already bitten once.
    check("GOOGLE_CLOUD_LOCATION == global", loc == "global",
          f"{loc!r}: gemini-3.x answers only on 'global'")
    check("ADC present",
          Path.home().joinpath(".config/gcloud/application_default_credentials.json").exists())

    print("\n=== database ===")
    db = root / "var" / "livelane.db"
    if db.exists():
        con = sqlite3.connect(db)
        runs = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        var = con.execute("SELECT COUNT(*) FROM variants").fetchone()[0]
        con.close()
        check("variant store", True, f"{runs} runs, {var} variants")
    else:
        check("variant store", True, "not created yet (expected before first sweep)")

    print(f"\n=== {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} ===")
    return 0 if ok else 1


# --- status ------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> int:
    root = _root()
    db = root / "var" / "livelane.db"
    if not db.exists():
        print("no database yet: nothing has been swept")
        return 0
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT design, arm, model, seed, status,
                  (SELECT COUNT(*) FROM variants v WHERE v.run_id = r.run_id) AS variants,
                  (SELECT SUM(accepted) FROM variants v WHERE v.run_id = r.run_id) AS accepted,
                  (SELECT COALESCE(SUM(cost_usd),0) FROM iterations i WHERE i.run_id = r.run_id) AS usd
           FROM runs r ORDER BY design, arm, model, seed""").fetchall()
    if not rows:
        print("database exists but contains no runs")
        con.close()
        return 0
    print(f"{'design':<20} {'arm':<8} {'model':<20} {'seed':>4} {'status':<9} "
          f"{'vars':>5} {'acc':>4} {'usd':>8}")
    print("-" * 84)
    total = 0.0
    for r in rows:
        total += r["usd"] or 0.0
        print(f"{r['design']:<20} {r['arm']:<8} {r['model']:<20} {r['seed']:>4} "
              f"{r['status']:<9} {r['variants']:>5} {str(r['accepted'] or 0):>4} "
              f"{r['usd'] or 0.0:>8.4f}")
    print("-" * 84)
    print(f"{'TOTAL':<60}{total:>24.4f}")
    con.close()
    return 0


# --- delegating subcommands --------------------------------------------------

def cmd_sweep(args: argparse.Namespace) -> int:
    import importlib
    m = importlib.import_module("livelane.sweep")
    sys.argv = ["sweep", *args.rest]
    return m.main()


def cmd_judge(args: argparse.Namespace) -> int:
    try:
        from livelane.judge import judge_run  # noqa: F401
    except ImportError as e:
        print(f"judge module not available: {e}")
        return 1
    from livelane.db.store import VariantStore
    from livelane.judge import judge_run
    store = VariantStore(_root() / "var" / "livelane.db", verbose=False)
    n = judge_run(store, args.run_id)
    print(f"judged {n} variants of {args.run_id}")
    store.close()
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    try:
        import livelane.analysis.figures as F
    except ImportError as e:
        print(f"figures module not available: {e}")
        return 1
    sys.argv = ["figures", *args.rest]
    return F.main() if hasattr(F, "main") else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="livelane", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check the environment").set_defaults(fn=cmd_doctor)
    sub.add_parser("status", help="what has been run").set_defaults(fn=cmd_status)

    s = sub.add_parser("sweep", help="run the experiment matrix")
    s.add_argument("rest", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_sweep)

    j = sub.add_parser("judge", help="re-score a run's finals through the neutral flow")
    j.add_argument("run_id")
    j.set_defaults(fn=cmd_judge)

    f = sub.add_parser("figures", help="render the paper's figures")
    f.add_argument("rest", nargs=argparse.REMAINDER)
    f.set_defaults(fn=cmd_figures)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
