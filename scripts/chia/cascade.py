"""The edit cascade, run end to end as a real CHIA graph.

An RTL-editing loop has to decide whether to keep an edit. There are three
sources of evidence, they cost different amounts, and, this is the point,
they do not agree:

    simulate  ~1 s     cheap, and bounded by whatever its stimulus covers
    score     ~10 s    tells you if it is FASTER, never if it is CORRECT
    prove     ~15 s    the only stage that can refute

This script runs all three over one pair of designs, on Ray, with each stage on
its own resource token, and prints what each stage concluded. Run it on a pair
whose answer is known by construction and the disagreement is the result:

    python scripts/chia/cascade.py --design picorv32 --inject-bge-bug

`--inject-bge-bug` inverts picorv32's BGE comparison
(``alu_out_0 = !alu_lts`` -> ``alu_out_0 = alu_lts``). That edit is a functional
bug. Simulation passes it with a byte-identical trace, synthesis scores it
BETTER than the correct design, and only the formal gate refutes it, which is
the whole argument for having a formal gate in the loop.

Without the flag the candidate is a copy of the parent, so every stage should
agree that it is equivalent; that is the control.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

BGE_GOOD = "\t\t\t\talu_out_0 = !alu_lts;"
BGE_BAD = "\t\t\t\talu_out_0 = alu_lts;"

DESIGNS = {
    # name: (rtl, testbench, tb_top, synth_top, clock port, period ns, read cmd)
    "picorv32": ("designs/picorv32/picorv32.v",
                 "thirdparty/picorv32/testbench_ez.v",
                 "testbench", "picorv32", "clk", 10.0, "read_slang"),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", default="picorv32", choices=sorted(DESIGNS))
    ap.add_argument("--inject-bge-bug", action="store_true",
                    help="make the candidate functionally wrong, on purpose")
    ap.add_argument("--workdir", default=str(ROOT / "var" / "cascade"))
    ap.add_argument("--read-cmd", default=None,
                    help="override the yosys front end; the front end alone "
                         "moves picorv32 from 6563 cells to 6691")
    ap.add_argument("--script", default="baseline-flat",
                    help="synthesis recipe name")
    ap.add_argument("--address", default=None, help="existing Ray cluster")
    a = ap.parse_args()

    rtl, tb, tb_top, top, clk, period, read_cmd = DESIGNS[a.design]
    read_cmd = a.read_cmd or read_cmd
    rtl, tb = ROOT / rtl, ROOT / tb
    for p in (rtl, tb):
        if not p.exists():
            log(f"FATAL: missing {p}")
            return 2
    liberty = os.environ.get("LIVELANE_LIBERTY")
    if not liberty or not Path(liberty).exists():
        log(f"FATAL: LIVELANE_LIBERTY unset or missing ({liberty!r}); source env.sh")
        return 2

    wd = Path(a.workdir)
    shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True, exist_ok=True)
    parent_v, cand_v = wd / "parent.v", wd / "candidate.v"
    src = rtl.read_text()
    parent_v.write_text(src)
    if a.inject_bge_bug:
        if src.count(BGE_GOOD) != 1:
            log(f"FATAL: expected exactly one BGE site, found {src.count(BGE_GOOD)}")
            return 2
        cand_v.write_text(src.replace(BGE_GOOD, BGE_BAD, 1))
        log("candidate: BGE comparison INVERTED (a functional bug, injected)")
    else:
        cand_v.write_text(src)
        log("candidate: byte-identical copy of the parent (control)")

    from chia_livelane.formal.lec_gate import auto_jobs, lec_gate
    from chia_livelane.sim.verilator import verilator_sim
    from chia_livelane.vlsi.yosys_sta import yosys_sta_qor

    nodes = {"verilator_sim": verilator_sim, "yosys_sta_qor": yosys_sta_qor,
             "lec_gate": lec_gate}
    needed: dict[str, float] = {}
    for name, fn in nodes.items():
        opts = getattr(fn, "_chia_options", None)
        if opts is None:
            log(f"FATAL: {name} is the fallback stub, not a real ChiaFunction")
            return 2
        want = 2.0 if name == "yosys_sta_qor" else 1.0
        for res, qty in (opts.get("resources") or {}).items():
            needed[res] = max(needed.get(res, 0.0), float(qty) * want)
    log(f"front end: {read_cmd!r}  recipe: {a.script!r}")
    log(f"node-declared resource tokens: {needed}")

    import ray
    from chia.base.ChiaFunction import get
    if a.address:
        ray.init(address=a.address)
    else:
        ray.init(resources={k: v for k, v in needed.items()}, log_to_driver=False)
    avail = ray.cluster_resources()
    missing = {r: q for r, q in needed.items() if avail.get(r, 0.0) < q}
    if missing:
        # A Ray task asking for a resource the cluster never declared does not
        # fail: it sits PENDING forever, with no error and no log line.
        log(f"FATAL: cluster missing tokens {missing}")
        return 2
    log("preflight: every node-declared token is satisfied")

    results: dict[str, dict] = {}

    log("stage 1/3  simulate (cheap) ...")
    t = time.monotonic()
    sim_p, sim_c = get(verilator_sim.chia_remote(
        [str(tb), str(parent_v)], str(wd / "sim"), top=tb_top,
        verilator="verilator", jobs=8, name="parent")), None
    sim_c = get(verilator_sim.chia_remote(
        [str(tb), str(cand_v)], str(wd / "sim"), top=tb_top,
        verilator="verilator", jobs=8, name="candidate"))
    results["simulate"] = {"parent": sim_p, "candidate": sim_c,
                           "wall_s": time.monotonic() - t}
    log(f"  parent   passed={sim_p['passed']} trace={sim_p['trace_sha256'][:16]}")
    log(f"  candidate passed={sim_c['passed']} trace={sim_c['trace_sha256'][:16]}")

    log("stage 2/3  score (synthesis + timing) ...")
    t = time.monotonic()
    ref_p = yosys_sta_qor.chia_remote([str(parent_v)], top, liberty,
                                      workdir=str(wd / "qor-parent"),
                                      clock_port=clk, clock_period_ns=period,
                                      read_cmd=read_cmd, script=a.script)
    ref_c = yosys_sta_qor.chia_remote([str(cand_v)], top, liberty,
                                      workdir=str(wd / "qor-candidate"),
                                      clock_port=clk, clock_period_ns=period,
                                      read_cmd=read_cmd, script=a.script)
    qor_p, qor_c = get(ref_p), get(ref_c)
    results["score"] = {"parent": qor_p, "candidate": qor_c,
                        "wall_s": time.monotonic() - t}
    log(f"  parent    cells={qor_p['cells']} area={qor_p['area_um2']} "
        f"delay={qor_p['max_delay_ns']}")
    log(f"  candidate cells={qor_c['cells']} area={qor_c['area_um2']} "
        f"delay={qor_c['max_delay_ns']}")

    log("stage 3/3  prove (formal equivalence) ...")
    t = time.monotonic()
    lec = get(lec_gate.chia_remote([str(parent_v)], [str(cand_v)], top,
                                   workdir=str(wd / "lec"), read_cmd=read_cmd,
                                   jobs=auto_jobs()))
    results["prove"] = {"verdict": lec, "wall_s": time.monotonic() - t}
    log(f"  verdict={lec['verdict']} equivalent={lec['equivalent']} "
        f"refuted={lec['refuted']} evidence={lec['evidence_strength']}")

    # --- what each stage concluded ------------------------------------------
    same_trace = sim_p["trace_sha256"] == sim_c["trace_sha256"]
    better = (qor_c["area_um2"] or 0) < (qor_p["area_um2"] or 0)
    rows = [
        ("simulate", results["simulate"]["wall_s"],
         "PASS" if sim_c["passed"] else "FAIL",
         "identical trace" if same_trace else "trace differs"),
        ("score", results["score"]["wall_s"],
         "BETTER" if better else "not better",
         f"{qor_c['area_um2']} vs {qor_p['area_um2']} um2"),
        ("prove", results["prove"]["wall_s"],
         "REFUTED" if lec["refuted"] else
         ("PROVEN" if lec["equivalent"] else lec["verdict"].upper()),
         lec["message"][:60]),
    ]
    print("\n=== what each stage concluded ===")
    print(f"  {'stage':10s} {'cost':>8s}  {'verdict':10s} detail")
    for n, w, v, d in rows:
        print(f"  {n:10s} {w:7.2f}s  {v:10s} {d}")

    admits = sim_c["passed"] and not lec["refuted"]
    print()
    if a.inject_bge_bug:
        caught = lec["refuted"]
        print(f"  injected functional bug -> "
              f"{'CAUGHT by the formal gate' if caught else 'NOT CAUGHT (!!)'}")
        print(f"  a loop accepting on simulation alone would have accepted it: "
              f"{sim_c['passed']}")
        ok = caught and sim_c["passed"]
    else:
        ok = admits and lec["equivalent"]
        print(f"  control pair -> every stage agrees equivalent: {ok}")

    out = wd / "cascade-summary.json"
    out.write_text(json.dumps({"design": a.design,
                               "injected_bge_bug": a.inject_bge_bug,
                               "stages": results, "expected": ok}, indent=2,
                              default=str))
    log(f"wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
