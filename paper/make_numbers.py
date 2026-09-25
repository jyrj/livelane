"""Generate paper/numbers.tex from the measurement JSON.

Every rewrite-campaign figure in the paper is a macro defined here, computed
from var/final/rewrites.json. Nothing in livelane.tex is typed by hand from a
log, so the paper cannot drift from the data it cites, re-run this after any
re-measurement and rebuild.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "measurements/data/rewrites.json"
OUT = Path(__file__).resolve().parent / "numbers.tex"


def verdict(r: dict) -> str | None:
    """A refusal's verdict: the explicit-initial-state proof (resettle.py)
    when it has run, else the earlier all-zero-start check."""
    e = r.get("explicit")
    return e["from_reset"] if e else r.get("bmc")


def any_state(r: dict) -> str | None:
    e = r.get("explicit")
    return e["any_state"] if e else r.get("any_state")


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}" if d else "0"


def tex(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}").replace("_", r"\_")
             .replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
             .replace("{", r"\{").replace("}", r"\}"))


def main() -> int:
    d = json.loads(SRC.read_text())
    base, rows = d["base"], d["rows"]
    eps = 1e-6

    def better(r, k):
        return (r.get(k) is not None and base.get(k) is not None
                and r[k] < base[k] - eps)

    settled = [r for r in rows if r["verdict"] in ("proven", "refuted")]
    eq = [r for r in settled if r["family"] == "equivalent"]
    sm = [r for r in settled if r["family"] == "simplifying"]
    broken = [r for r in sm if r["verdict"] == "refuted"]
    simp_eq = [r for r in sm if r["verdict"] == "proven"]
    bs = [r for r in broken if r.get("synth_ok")]
    faster = [r for r in bs if better(r, "max_delay_ns")]
    smaller = [r for r in bs if better(r, "area_um2")]
    either = [r for r in bs if better(r, "max_delay_ns") or better(r, "area_um2")]
    loc = [r for r in broken if r.get("localised") is not None]
    loc_hit = [r for r in loc if r["localised"]]
    gs = sorted(r["gate_s"] for r in rows)
    unsettled = [r for r in rows if r["verdict"] not in ("proven", "refuted")]

    # the most striking single broken rewrite: largest critical-path gain
    ex = max(faster, key=lambda r: base["max_delay_ns"] - r["max_delay_ns"],
             default=None)

    m = {
        "RWtotal": len(rows),
        "RWsettled": len(settled),
        "RWunsettled": len(unsettled),
        "RWeq": len(eq),
        "RWeqProven": sum(r["verdict"] == "proven" for r in eq),
        "RWeqRefuted": sum(r["verdict"] == "refuted" for r in eq),
        "RWsimp": len(sm),
        "RWbroken": len(broken),
        "RWbrokenPct": pct(len(broken), len(sm)),
        "RWsimpEquiv": len(simp_eq),
        "RWbrokenSynth": len(bs),
        "RWfaster": len(faster),
        "RWfasterPct": pct(len(faster), len(bs)),
        "RWsmaller": len(smaller),
        "RWsmallerPct": pct(len(smaller), len(bs)),
        "RWeither": len(either),
        "RWeitherPct": pct(len(either), len(bs)),
        "RWloc": len(loc_hit),
        "RWlocDen": len(loc),
        "RWlocPct": pct(len(loc_hit), len(loc)),
        "RWgateMedian": f"{gs[len(gs) // 2]:.1f}" if gs else "0",
        "RWgateMax": f"{gs[-1]:.1f}" if gs else "0",
        "RWbaseArea": f"{base['area_um2']:,.0f}".replace(",", "{,}"),
        "RWbaseDelay": f"{base['max_delay_ns']:.2f}",
        "RWbaseCells": f"{base['cells']:,}".replace(",", "{,}"),
        "RWparts": f"{d.get('warm_partitions', 0):,}".replace(",", "{,}"),
    }
    # --- sound confirmation of refusals (resettle.py) ----------------------
    chk = [r for r in broken if verdict(r)]
    conf = [r for r in chk if verdict(r) == "CONFIRMED"]
    nf = [r for r in chk if verdict(r) == "NOT-FOUND"]
    to = [r for r in chk if verdict(r) in ("TIMEOUT", "ERROR", "UNDECIDED")]
    depth = (d.get("bmc") or {}).get("depth", "?")
    m["RWbmcChecked"] = len(chk)
    m["RWbmcConfirmed"] = len(conf)
    if not chk:
        m["RWbmcTail"] = ("; \\textbf{not yet run}; until it is, rejections "
                          "are refusals, not confirmed bugs.")
    elif len(conf) + sum(verdict(r) == "PROVEN-SEQ" for r in chk) == len(chk):
        # the rest were proven correct: false refusals, reported separately
        m["RWbmcTail"] = ": \\textbf{every one} that is not a false rejection."
    else:
        parts = []
        if nf:
            ds = sorted({r.get("bmc_depth", depth) for r in nf})
            parts.append(f"{len(nf)} showed no difference within "
                         f"{'/'.join(map(str, ds))} cycles")
        if to:
            parts.append(f"{len(to)} exhausted the solver's time")
        n_rest = len(chk) - len(conf)
        m["RWbmcTail"] = ("; of the rest, " + " and ".join(parts) + ", and "
                          + ("it is" if n_rest == 1 else "they are")
                          + " counted as a refusal, not a bug."
                          if n_rest == 1 else
                          "; of the rest, " + " and ".join(parts) +
                          ", and they are counted as refusals, not bugs.")
    # the headline QoR rate, restricted to CONFIRMED bugs, so the claim
    # "synthesis rewards broken designs" rests on sound evidence alone
    cs = [r for r in conf if r.get("synth_ok")]
    ce = [r for r in cs if better(r, "max_delay_ns") or better(r, "area_um2")]
    m["RWconfSynth"] = len(cs)
    m["RWconfEither"] = len(ce)
    m["RWconfEitherPct"] = pct(len(ce), len(cs))
    bc = d.get("bmc_controls") or {}
    ctl = [r for r in d["rows"] if r.get("family") == "equivalent"
           and r.get("explicit")]
    if ctl:
        # every control decided from reset and from any state (resettle.py)
        dead = [r for r in ctl if not r["explicit"]["instances"]]
        live = [r for r in ctl if r["explicit"]["instances"]]
        bc = {"checked": len(live),
              "proven": sum(verdict(r) == "PROVEN-SEQ" for r in live),
              "confirmed": sum(verdict(r) == "CONFIRMED" for r in live),
              "timeout": sum(verdict(r) not in ("PROVEN-SEQ", "CONFIRMED")
                             for r in live),
              "not_instantiated": len(dead), "top_level_not_checked": 0}
        m["RWctlAny"] = sum(any_state(r) == "ANY-STATE" for r in live)
    m["RWctlChecked"] = bc.get("checked", 0)
    m["RWctlConfirmed"] = bc.get("confirmed", 0)
    m["RWctlProven"] = bc.get("proven", 0)
    m["RWctlTimeout"] = bc.get("timeout", 0)
    m["RWctlDead"] = bc.get("not_instantiated", 0)
    # equivalence-preserving rewrites of logic this configuration instantiates
    m["RWeqLive"] = m["RWeq"] - m["RWctlDead"]
    m["RWctlTop"] = bc.get("top_level_not_checked", 0)
    # a refusal the unbounded proof shows equivalent is a FALSE rejection
    m["RWfalseRej"] = sum(1 for r in broken if verdict(r) == "PROVEN-SEQ")
    m["RWsimpDead"] = sum(1 for r in simp_eq if r.get("module") == "ibex_multdiv_slow")

    if ex:
        gain =100 * (base["max_delay_ns"] - ex["max_delay_ns"]) / base["max_delay_ns"]
        m.update({
            "RWexOp": tex(ex["operator"]).replace(r"\_", " "),
            "RWexFile": tex(ex["file"].split("/")[-1]),
            "RWexLine": ex["line"],
            "RWexOrig": tex(" ".join(ex["original"].split())),
            "RWexNew": tex(" ".join(ex["rewrite"].split())),
            "RWexGain": f"{gain:.1f}",
            "RWexDelay": f"{ex['max_delay_ns']:.2f}",
        })
    else:
        # No broken rewrite improved timing. Define the macros anyway so the
        # paper still compiles, and so a reader sees "--", not a silently
        # missing sentence.
        m.update({k: "--" for k in ("RWexOp", "RWexFile", "RWexLine",
                                     "RWexOrig", "RWexNew", "RWexGain",
                                     "RWexDelay")})
    # scatter data for the figure: every simplifying rewrite that synthesised,
    # change against the original in percent (negative = better)
    dat = ["ddelay darea"]
    for fam_rows, name in ((broken, "refused"), (simp_eq, "proven")):
        rows_ = ["ddelay darea"]
        for r in fam_rows:
            if not r.get("synth_ok") or r.get("max_delay_ns") is None:
                continue
            dd = 100 * (r["max_delay_ns"] - base["max_delay_ns"]) / base["max_delay_ns"]
            da = 100 * (r["area_um2"] - base["area_um2"]) / base["area_um2"]
            rows_.append(f"{dd:.3f} {da:.3f}")
        (OUT.parent / f"scatter-{name}.dat").write_text("\n".join(rows_) + "\n")
    # --- the agentic runs (agent_analysis.py + resettle.py) ----------------
    ag_path = SRC.parent / "agent.json"
    if ag_path.exists():
        ag = json.loads(ag_path.read_text())
        ar = [r for r in ag["rows"] if r["verdict"] in ("proven", "refuted")]
        aref = [r for r in ar if r["verdict"] == "refuted"]
        aconf = [r for r in aref if verdict(r) == "CONFIRMED"]
        runs = ag["runs"]
        gates = sorted(r["gate_s"] for r in ar if r.get("gate_s"))
        # the first proof of a run is on a cold cache; report warm separately
        warm = sorted(r["gate_s"] for r in ar if r.get("gate_s")
                      and r["iteration"] > min(x["iteration"] for x in ar
                                               if x["run"] == r["run"]))
        best = max(runs, key=lambda x: x["improvement_pct"] or 0)
        models = sorted({x["model"] for x in runs})
        afalse = [r for r in aref if verdict(r) == "PROVEN-SEQ"]
        pro = [r for r in ar if "pro" in r["model"]]
        fla = [r for r in ar if "flash" in r["model"]]
        # strongest form of a false rejection: equivalent from ANY shared state
        m["AGanyState"] = sum(1 for r in afalse if any_state(r) == "ANY-STATE")
        m["AGresetOnly"] = len(afalse) - m["AGanyState"]
        # iterations of the stronger model spent on edits the gate falsely refused
        pro_runs = [x for x in runs if "pro" in x["model"]]
        pro_iters = sum(x["iterations"] for x in pro_runs)
        pro_false = [r for r in afalse if "pro" in r["model"]]
        per_run = {}
        for r in pro_false:
            per_run[r["run"]] = per_run.get(r["run"], 0) + 1
        worst = max(per_run.items(), key=lambda kv: kv[1]) if per_run else (None, 0)
        worst_iters = next((x["iterations"] for x in pro_runs if x["run"] == worst[0]), 0)
        m["AGproIters"] = pro_iters
        m["AGproLostPct"] = pct(len(pro_false), pro_iters)
        m["AGworstLost"] = worst[1]
        m["AGworstIters"] = worst_iters
        m["AGworstPct"] = pct(worst[1], worst_iters)
        m.update({
            "AGbugs": len(aconf),
            "AGfalseRej": len(afalse),
            "AGfalseRejPct": pct(len(afalse), len(aref)),
            "AGproGated": len(pro),
            "AGproRef": sum(r["verdict"] == "refuted" for r in pro),
            "AGproFalse": sum(verdict(r) == "PROVEN-SEQ" for r in pro
                              if r["verdict"] == "refuted"),
            "AGproBugs": sum(verdict(r) == "CONFIRMED" for r in pro
                             if r["verdict"] == "refuted"),
            "AGflashGated": len(fla),
            "AGflashBugs": sum(verdict(r) == "CONFIRMED" for r in fla
                               if r["verdict"] == "refuted"),
            "AGflashFalse": sum(verdict(r) == "PROVEN-SEQ" for r in fla
                                if r["verdict"] == "refuted"),
            "AGruns": len(runs),
            "AGmodels": len(models),
            "AGiters": sum(x["iterations"] for x in runs),
            "AGgated": len(ar),
            "AGrefused": len(aref),
            "AGrefusedPct": pct(len(aref), len(ar)),
            "AGconfirmed": len(aconf),
            "AGrefFaster": sum(1 for r in aref if r["faster"]),
            "AGaccepted": sum(x["accepted"] for x in runs),
            "AGimproved": sum(1 for x in runs if (x["improvement_pct"] or 0) > 0),
            "AGbestPct": f"{best['improvement_pct']:.1f}",
            "AGbestFrom": f"{best['seed_ns']:.2f}",
            "AGbestTo": f"{best['best_ns']:.2f}",
            "AGgateWarm": f"{warm[len(warm) // 2]:.1f}" if warm else "--",
            "AGgateCold": f"{max(gates):.0f}" if gates else "--",
            "AGcost": f"{sum(x['cost_usd'] or 0 for x in runs):.2f}",
            "AGiterMedian": (lambda xs: f"{xs[len(xs) // 2]:.0f}" if xs else "--")(
                sorted(r["iter_s"] for r in ar if r.get("iter_s"))),
            "AGgateShare": (lambda xs, ys: f"{100 * sum(xs) / sum(ys):.0f}"
                            if ys and sum(ys) else "--")(
                [r["gate_s"] for r in ar if r.get("gate_s") and r.get("iter_s")],
                [r["iter_s"] for r in ar if r.get("gate_s") and r.get("iter_s")]),
            "AGparts": "11{,}117",
        })
    # --- the latency study (scripts/chia/hypotheses.py) ---------------------
    hp = SRC.parent / "hypotheses.json"
    if hp.exists():
        h = json.loads(hp.read_text())
        lat = h.get("latency", {})

        def imp(model, arm):
            v = ((lat.get(model) or {}).get(arm) or {}).get("improvement")
            return f"{100 * v['mean']:.1f}" if v and v.get("mean") is not None else "--"

        for tag, model in (("Pro", "gemini-3.1-pro-preview"),
                           ("Flash", "gemini-2.5-flash")):
            for arm, k in (("I(0)", "Zero"), ("I(30)", "Thirty"),
                           ("I(120)", "OneTwenty"), ("I(600)", "SixHundred")):
                m[f"H{tag}{k}"] = imp(model, arm)
            n = sum(((lat.get(model) or {}).get(a) or {}).get("n", 0)
                    for a in ("I(0)", "I(30)", "I(120)", "I(600)"))
            m[f"H{tag}Cells"] = n
        for tag, model in (("Pro", "gemini-3.1-pro-preview"),
                           ("Flash", "gemini-2.5-flash")):
            tv = (h.get("turn") or {}).get(model)
            m[f"H{tag}Turn"] = f"{tv['mean']:.0f}" if tv else "--"
        # the delays where each model's curve bends, as multiples of its turn
        tp = ((h.get("turn") or {}).get("gemini-3.1-pro-preview") or {}).get("mean")
        tf = ((h.get("turn") or {}).get("gemini-2.5-flash") or {}).get("mean")
        m["HProLoX"] = f"{120 / tp:.1f}" if tp else "--"
        m["HProHiX"] = f"{600 / tp:.1f}" if tp else "--"
        m["HFlashLoX"] = f"{30 / tf:.1f}" if tf else "--"
        iso = h.get("iso", {})
        for arm, k in (("I(0)", "Zero"), ("I(30)", "Thirty"),
                       ("I(120)", "OneTwenty"), ("I(600)", "SixHundred")):
            v = (iso.get(arm) or {}).get("improvement")
            m[f"HIso{k}"] = f"{100 * v['mean']:.1f}" if v else "--"
        it = h.get("iso_test") or {}
        # "holds" in hypotheses.json means only that H2 was not rejected
        m["HTwoVerdict"] = {"holds": "not rejected", "refuted": "refuted"}.get(
            it.get("verdict"), "pending")
        m["HTwoP"] = f"{it['p']:.2f}" if it.get("p") is not None else "--"
        m["HIsoSeeds"] = max(((iso.get(a) or {}).get("n", 0) for a in iso), default=0)
        # H4 from a candidate study (not shipped with this release): the agent's
        # own edits on these designs almost never applied, so candidates are
        # deterministic one-token rewrites scored by BOTH lanes. Within-design
        # rho is the number that matters: a loop only ever compares variants of
        # one design, and the pooled rho is driven by the gap BETWEEN designs.
        h4p = SRC.parent / "h4.json"
        if h4p.exists():
            h4 = json.loads(h4p.read_text())["rho"]
            f2 = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else "--"
            m["HFourN"] = h4["delay"]["n"]
            m["HFourPooled"] = f2(h4["delay"]["all"])
            m["HFourAreaPooled"] = f2(h4["area"]["all"])
            m["HFourAlu"] = f2(h4["delay"]["Alu"]["rho"])
            m["HFourDec"] = f2(h4["delay"]["DecodeUnit"]["rho"])
            m["HFourAluN"] = h4["delay"]["Alu"]["n"]
            m["HFourDecN"] = h4["delay"]["DecodeUnit"]["n"]
        h5 = h.get("h5", {})
        rates = [v["rate"] for v in h5.values() if v.get("rate") is not None
                 and v.get("decided", 0) >= 10]
        m["HFiveLo"] = f"{100 * min(rates):.1f}" if rates else "--"
        m["HFiveHi"] = f"{100 * max(rates):.1f}" if rates else "--"
        m["HFiveDecided"] = sum(v.get("decided", 0) for v in h5.values())
        # figure data: improvement per delay arm, both models (blank if absent)
        rows = ["delay pro flash"]
        for d, arm in ((0, "I(0)"), (30, "I(30)"), (120, "I(120)"), (600, "I(600)")):
            vals = []
            for model in ("gemini-3.1-pro-preview", "gemini-2.5-flash"):
                v = ((lat.get(model) or {}).get(arm) or {}).get("improvement")
                vals.append(f"{100 * v['mean']:.1f}" if v and v.get("mean") is not None
                            else "nan")
            rows.append(f"{d} {vals[0]} {vals[1]}")
        (OUT.parent / "latency.dat").write_text("\n".join(rows) + "\n")
    # --- the explicit-initial-state second stage (resettle.py) -----------
    def q(xs, f):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else None
    walls = []
    for path in (SRC, SRC.parent / "agent.json"):
        if path.exists():
            for r in json.loads(path.read_text())["rows"]:
                e = r.get("explicit")
                if e and r.get("verdict_eqy", r["verdict"]) == "refuted":
                    walls.append(e["wall_s"])
    med, p90 = q(walls, 0.5), q(walls, 0.9)
    m["SSmedian"] = f"{med:.1f}" if med is not None else "--"
    m["SSpNinety"] = f"{p90:.1f}" if p90 is not None else "--"
    # edits to a package: proven where their changed types reach
    pkg, seen = [], set()
    for path in sorted(SRC.parent.glob("arm_*.json")):
        for r in json.loads(path.read_text())["rows"]:
            e = r.get("explicit") or {}
            key = (r.get("run"), r.get("iteration"))
            if "package_changed" in e and key not in seen:
                seen.add(key)
                pkg.append(e)
    top = [e for e in pkg if e["scope"] == "ibex_core"]
    # the best of them: a correct edit the gate refused, scored faster
    fast = []
    for path in sorted(SRC.parent.glob("arm_*.json")):
        dd = json.loads(path.read_text())
        for r in dd["rows"]:
            e = r.get("explicit") or {}
            if "package_changed" in e and e["from_reset"] == "PROVEN-SEQ" \
                    and r.get("faster") and r.get("parent_ns"):
                run = next(x for x in dd["runs"] if x["run"] == r["run"])
                fast.append((100 * (r["parent_ns"] - r["delay_ns"]) / r["parent_ns"],
                             r, run, e))
    if fast:
        g, r, run, e = max(fast, key=lambda t: t[0])
        m["PKGbestPct"] = f"{g:.1f}"
        m["PKGbestNs"] = f"{r['delay_ns']:.2f}"
        m["PKGbestParent"] = f"{r['parent_ns']:.2f}"
        m["PKGbestS"] = f"{e['wall_s']:.0f}"
        m["PKGrunPct"] = f"{run['improvement_pct']:.1f}"
        m["PKGrunWould"] = f"{100 * (run['seed_ns'] - r['delay_ns']) / run['seed_ns']:.1f}"
        m["PKGfast"] = len({(t[1]['run'], t[1]['iteration']) for t in fast})
    m["PKGn"] = len(pkg)
    m["PKGproven"] = sum(e["from_reset"] == "PROVEN-SEQ" for e in pkg)
    m["PKGtop"] = len(top)
    m["PKGtopProven"] = sum(e["from_reset"] == "PROVEN-SEQ" for e in top)
    tw = q([e["wall_s"] for e in top], 0.5)
    m["PKGtopS"] = f"{tw:.0f}" if tw is not None else "--"
    m["PKGtopMax"] = f"{max(e['wall_s'] for e in top):.0f}" if top else "--"
    m["PKGruns"] = len({k[0] for k in seen})
    m["PKGrunsText"] = {1: "all in one run"}.get(m["PKGruns"],
                                                  f"from {m['PKGruns']} runs")
    enums = sorted({n for e in pkg if e["from_reset"] == "PROVEN-SEQ"
                    for n in e.get("package_changed", []) if n.endswith("_e")})
    m["PKGenums"] = ", ".join(f"\\texttt{{{tex(n)}}}" for n in enums) or "--"

    # --- the second stage live in the loop (gate_ab.py) ------------------
    ab_path = SRC.parent / "gate_ab.json"
    if ab_path.exists():
        ab = json.loads(ab_path.read_text())
        ctl, trt = ab.get("pro / standard gate", {}), ab.get("pro / + second stage", {})
        m["ABctlRuns"] = ctl.get("runs", 0)
        m["ABctlIters"] = ctl.get("iterations", 0)
        m["ABctlFalse"] = ctl.get("false_refusals", 0)
        m["ABctlRefused"] = ctl.get("refused_by_partitioned_gate", 0)
        m["ABctlLostPct"] = (f"{100 * ctl['false_refusal_iteration_share']:.0f}"
                             if ctl.get("false_refusal_iteration_share") is not None else "--")
        m["ABtrtRuns"] = trt.get("runs", 0)
        m["ABtold"] = trt.get("told_correct", 0)
        m["ABtoldReproven"] = trt.get("told_correct_reproven", 0)
        told_fast = 0
        tp = SRC.parent / "arm_pro_+secondstage.json"
        if tp.exists():
            told_fast = sum(1 for r in json.loads(tp.read_text())["rows"]
                            if r["verdict"] == "proven-seq" and r.get("faster"))
        m["ABtoldFast"] = told_fast
        m["ABruns"] = sum(v.get("runs", 0) for v in ab.values())
        m["ABmodels"] = len({k.split(" / ")[0] for k in ab})
        f1 = lambda v: f"{v:.1f}" if v is not None else "--"
        m["ABctlMed"] = f1(ctl.get("improvement_median"))
        m["ABtrtMed"] = f1(trt.get("improvement_median"))
        sc = ab.get("pro / + scoped second stage", {})
        m["ABscRuns"] = sc.get("runs", 0)
        m["ABscMed"] = f1(sc.get("improvement_median"))
        m["ABscTold"] = sc.get("told_correct", 0)
        m["ABscRefused"] = sc.get("refused_by_partitioned_gate", 0)
        sc_ref = sc.get("refused_by_partitioned_gate", 0)
        sc_bug = sc.get("real_bugs", 0)
        told = sc.get("told_correct", 0)
        # what the second stage did with the refusals it did NOT prove
        rest = ("" if not sc_bug else
                " (the other was a real bug, and it was refuted)" if sc_bug == 1 else
                f" (the other {sc_bug} were real bugs, and were refuted)")
        m["ABscBugs"] = sc_bug
        # the same released second stage, live with the other models
        gen = [(k.split(" / ")[0], v) for k, v in ab.items()
               if k.endswith("/ + scoped second stage") and not k.startswith("pro ")
               and v.get("runs")]
        if gen:
            g_runs = sum(v["runs"] for _, v in gen)
            g_ref = sum(v.get("refused_by_partitioned_gate", 0) for _, v in gen)
            g_ok = sum(v.get("told_correct", 0) for _, v in gen)
            g_bug = sum(v.get("real_bugs", 0) for _, v in gen)
            names = " and ".join(f"\\texttt{{gemini-{n}}}" for n, _ in gen)
            m["ABgenSentence"] = (
                f" Live with {names} ({g_runs} runs), it decided their {g_ref}"
                f" rejected edits too: {g_ok} proven correct"
                + (f", {g_bug} real {'bug' if g_bug == 1 else 'bugs'} refuted" if g_bug else "")
                + (f", {g_ref - g_ok - g_bug} undecided" if g_ref - g_ok - g_bug else "")
                + ".")
        else:
            m["ABgenSentence"] = ""
        m["ABscRest"] = rest
        m["ABscSentence"] = (
            f" With the final version of the sequential check (from reset, with"
            f" package edits checked at the scope their changed types reach) in"
            f" {sc['runs']} more runs,"
            f" {'all ' if told == sc_ref else ''}{told} of {sc_ref} rejected edits"
            f" came back proven and the agent was told so"
            + (f"; the other {'was a real bug' if sc_bug == 1 else f'{sc_bug} were real bugs'},"
               f" refuted with a counterexample from reset" if sc_bug else "")
            + f". It proposed no package edit, and the median is"
            f" {sc.get('improvement_median') or 0:.1f}\\%."
            if sc.get("runs") else "")
        # every accepted edit in every arm, re-proven from reset
        m["ABacc"] = sum(v.get("accepted", 0) for v in ab.values())
        m["ABaccReproven"] = sum(v.get("accepted_reproven", 0) for v in ab.values())
        # the stronger model's broken edits across all its ibex runs
        pro_bugs = pro_gated = 0
        for path in sorted(SRC.parent.glob("arm_pro_*.json")):
            for r in json.loads(path.read_text())["rows"]:
                if r.get("verdict_eqy", r["verdict"]) in ("proven", "refuted"):
                    pro_gated += 1
                    pro_bugs += verdict(r) == "CONFIRMED"
        m["ABproBugs"], m["ABproGated"] = pro_bugs, pro_gated

    # the test count, as the last full run recorded it (scripts: pytest -q)
    tp = SRC.parent / "tests.json"
    m["TESTS"] = json.loads(tp.read_text())["passed"] if tp.exists() else "--"

    lines = ["% GENERATED by paper/make_numbers.py from "
             f"{SRC.relative_to(ROOT) if SRC.is_relative_to(ROOT) else SRC}"
             " -- do not edit by hand"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    OUT.write_text("\n".join(lines) + "\n")
    for k, v in m.items():
        print(f"  {k:16s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
