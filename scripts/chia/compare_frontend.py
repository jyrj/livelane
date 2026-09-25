"""Compare two corpus runs of the same instances under different FRONT-END flags.

A strategy-ladder comparison is the wrong tool for this and would give a false alarm.
It requires reuse to be identical and failing-partition NAMES to match exactly,
because a strategy ladder changes neither. A front-end flag changes both by
construction: `read_slang --keep-hierarchy` keeps module boundaries, so the
design is cut into different partitions with module-scoped names. Running the
ladder comparator here prints "reuse differs" and "DIFFERENT failing
partitions" on every instance and exits 2, which reads exactly like a
soundness failure and is not one.

So this comparator states its gate explicitly:

  GATE: a verdict disagreement stops the flag PENDING AN EXPLANATION.

  Not "one of them missed a real bug", which is how this was first worded and
  is stronger than eqy's mechanism supports. Every partition is proven as a
  FLAT miter (`flatten -wb` in eqy.py:901) with FREE inputs at its boundary, so
  a cut point is a conservative abstraction and no partition's proof assumes
  another partition is equivalent. A disagreement therefore has two readings:

    1. the arm that proved missed a real difference, unsound;
    2. the arm that refuted reported a FALSE difference at an over-conservative
       cut, and since flat produces ~1.5x MORE partitions, flat is the more
       conservative arm, so this reading is not the unlikely one.

  This comparator cannot tell them apart. The explanation has to come from the
  per-partition no-cache control or a distinguishing input.

  REPORTED, NOT GATED: partition count, reuse, and timings. They are expected
  to differ. They say what the flag costs or saves; they do not decide it.

  LOCALISATION: both arms name the partitions that failed. The corpus knows
  which FILE each PR changed, and in ibex a file's module is its basename, so
  we can ask an objective question neither arm was told the answer to:

      does the failing set sit inside the module the PR actually changed?

  A "yes" is a stronger claim than "refuted". "Refuted" says the designs
  differ; "refuted at `controller_i.csr_save_cause_o`" says WHERE, and the fix
  file says whether that is right.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loadguard import (drifted as load_drifted, MIN_BASELINE,  # noqa: E402
                       baseline_is_suspect)
from localisation import Tree, build as build_nets  # noqa: E402

# ibex instantiates as `<module> [#(params)] <instance> (` at indent >= 2.
# Params may nest one level. A module we cannot map is reported unmapped
# rather than silently counted as a miss.
# hwebench.py recorded `failed` as failed_partitions[:5] until 2026-09-18.
# A list of exactly this length with no `n_failed` beside it is a PREFIX of
# unknown length, not a set: seven of twelve flat instances came back with
# exactly five.
LEGACY_CAP = 5
# Verdicts that actually decided something. 'timeout', 'unknown' and 'error'
# are non-answers; two of them matching is not agreement.
SETTLED = {"proven", "refuted"}

_INST = re.compile(
    r"^\s{2,}(\w+)\s*(?:#\s*\((?:[^()]|\([^()]*\))*\)\s*)?(\w+)\s*\(", re.M)


def instance_names(rtl, sha: str | None = None,
                   repo: Path | None = None) -> dict[str, set[str]]:
    """module -> the instance names it is instantiated under, AT `sha`.

    Delegates to localisation.py so there is one scanner, not two. The version
    this replaces had its own regex, its own `ibex_`/`prim_` prefix filter, and
    read the WORKING TREE, the same three defects localisation.py had, in a
    second copy. Reading HEAD is the one that bites: each instance's partition
    names describe the hierarchy at its own base.sha, and ibex's has changed.
    """
    tree = Tree(repo, sha, rtl) if repo is not None else rtl
    nets = build_nets(tree)
    out: dict[str, set[str]] = {}
    for inst, mod in nets.inst_module.items():
        out.setdefault(mod, set()).add(inst)
    # An instance name that maps to more than one module cannot identify a
    # module, so it is dropped rather than used to score a localisation hit.
    for inst in nets.collisions:
        for mod in list(out):
            out[mod].discard(inst)
    return out


def _parents(failed: list[str]) -> set[str]:
    """The distinct instance paths implicated, i.e. names minus the signal."""
    return {p.rsplit(".", 1)[0] for p in failed if "." in p}


def _by_number(path: Path) -> dict[int, dict]:
    return {r["number"]: r
            for r in json.loads(path.read_text()).get("rows", [])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a", help="results JSON, arm A (baseline)")
    ap.add_argument("b", help="results JSON, arm B (the flag under test)")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--repo", default="thirdparty/ibex",
                    help="git checkout to read each instance's own sha from")
    ap.add_argument("--rtl", default="thirdparty/ibex/rtl",
                    help="fallback tree when a row carries no sha")
    args = ap.parse_args()

    A, B = _by_number(Path(args.a)), _by_number(Path(args.b))
    both = sorted(set(A) & set(B))
    if not both:
        print("no instances measured under both arms yet")
        return 1
    rtl_dir, repo_dir = Path(args.rtl), Path(args.repo)
    inst_cache: dict = {}

    def names_for(row: dict) -> dict[str, set[str]]:
        sha = row.get("sha")
        if sha not in inst_cache:
            inst_cache[sha] = instance_names(rtl_dir, sha, repo_dir)
        return inst_cache[sha]
    # Same guard the harness applies, for the same reason: a ratio of two
    # wall-clocks taken on a contended box is not a measurement. An instance
    # contended in EITHER arm disqualifies the pair, the ratio has a
    # contaminated number in it whichever side it sits on.
    da, base_a = load_drifted(list(A.values()))
    db, base_b = load_drifted(list(B.values()))
    dirty = set(da) | set(db)

    na, nb = args.name_a, args.name_b
    print(f"{'inst':>6}  {na:<9} {nb:<9} {'agree':<6} "
          f"{'parts A':>8} {'parts B':>8} {'ratio':>6}  "
          f"{'wall A':>7} {'wall B':>7} {'ratio':>6}")
    unsound, part_ratio, wall_ratio, setup_ratio = [], [], [], []
    drift_skipped: list[int] = []
    no_failures: list[int] = []
    capped: list[int] = []
    asymmetric: list = []
    nonsettling: list = []
    loc = {na: [0, 0], nb: [0, 0]}   # [hits, mapped]
    unmapped, rows = [], []
    for n in both:
        x, y = A[n], B[n]
        sx, sy = x.get("status"), y.get("status")
        if sx != "ok" or sy != "ok":
            # Asymmetric coverage is NOT the same as "both sides skipped it".
            # An instance that one arm measured and the other did not is a hole
            # in the comparison, and the gate used to print "every instance
            # agrees" over a set that silently excluded it.
            note = ("(neither arm measured it)" if sx == sy
                    else "** ASYMMETRIC: one arm measured it, the other did "
                         "not **")
            if sx != sy:
                asymmetric.append((n, sx, sy))
            print(f"{n:>6}  {str(sx):<9} {str(sy):<9} {note}")
            continue
        xm, ym = x["measure"], y["measure"]
        # A verdict that did not settle is not an opinion to agree with.
        # 'timeout' == 'timeout' is two non-answers, and counting it as
        # agreement lets a flag ship on instances where nothing was decided.
        unsettled = [v for v in (xm["verdict"], ym["verdict"])
                     if v not in SETTLED]
        if unsettled:
            nonsettling.append((n, xm["verdict"], ym["verdict"]))
            print(f"{n:>6}  {xm['verdict']:<9} {ym['verdict']:<9} "
                  f"{'n/a':<6} (verdict did not settle; not counted as "
                  f"agreement)")
            continue
        ok_v = xm["verdict"] == ym["verdict"]
        if not ok_v:
            unsound.append((n, xm["verdict"], ym["verdict"]))
        pr = xm["partitions"] / ym["partitions"]
        wr = xm["wall_s"] / ym["wall_s"]
        part_ratio.append(pr)          # counts, not times: load cannot move them
        if n in dirty:
            drift_skipped.append(n)
        else:
            wall_ratio.append(wr)
            setup_ratio.append(xm["setup_s"] / ym["setup_s"])
        print(f"{n:>6}  {xm['verdict']:<9} {ym['verdict']:<9} "
              f"{'yes' if ok_v else '** NO **':<6} "
              f"{xm['partitions']:>8} {ym['partitions']:>8} {pr:>5.2f}x  "
              f"{xm['wall_s']:>7.1f} {ym['wall_s']:>7.1f} {wr:>5.2f}x"
              f"{'  <-- load drift, timings not counted' if n in dirty else ''}")

        # Localisation: which module did the PR change, and did each arm's
        # failing set land inside it?
        mod = Path(x.get("file", "")).stem
        names = names_for(x).get(mod, set())
        if not names:
            unmapped.append((n, mod))
            continue
        # An instance that PROVED has no failing set, so it cannot localise
        # anything. Counting it as a miss is not a conservative choice, it is a
        # wrong one: #1780 proves on both arms, and scoring it 0/1 on each
        # turned flat's 8/11 into 7/12 and hierarchy's perfect score into
        # 11/12. Excluded from the denominator and named.
        if not (xm.get("failed") or ym.get("failed")):
            no_failures.append(n)
            continue
        # Truncation guard. hwebench.py used to record only the first five
        # failing partitions, so a list of exactly LEGACY_CAP may be a prefix
        # of an unknown longer list. A "hit" off a prefix is still a hit, the
        # signal was really reported, but a MISS off a prefix proves nothing,
        # and `len(_parents(...))` off a prefix is a floor, not a count.
        truncated = any(len(m.get("failed") or []) == LEGACY_CAP
                        and "n_failed" not in m for m in (xm, ym))
        if truncated:
            capped.append(n)
        row = [n, mod, truncated]
        for tag, m in ((na, xm), (nb, ym)):
            failed = m.get("failed") or []
            hit = any(f".{i}." in p or p.startswith(f"{i}.")
                      for p in failed for i in names)
            loc[tag][1] += 1
            loc[tag][0] += int(hit)
            row += [hit, len(_parents(failed))]
        rows.append(row)

    print(f"\n--- GATE: verdict agreement ---")
    if unsound:
        print(f"  !! {len(unsound)} VERDICT DISAGREEMENT(S). The flag does "
              f"not ship.")
        for n, va, vb in unsound:
            print(f"     #{n}: {na}={va}  {nb}={vb}")
    else:
        n_cmp = len(part_ratio)
        print(f"  every instance agrees on the verdict ({n_cmp} compared)")
    if asymmetric:
        print(f"  !! {len(asymmetric)} instance(s) measured on ONE arm only, "
              f"so the arms were not compared on them: "
              + ", ".join(f"#{n} {na}={a} {nb}={b}" for n, a, b in asymmetric))
    if nonsettling:
        print(f"  !! {len(nonsettling)} instance(s) whose verdict did not "
              f"settle on at least one arm; excluded from the gate rather "
              f"than counted as agreement: "
              + ", ".join(f"#{n} {a}/{b}" for n, a, b in nonsettling))
    for tag, vals in ((na, list(A.values())), (nb, list(B.values()))):
        why = baseline_is_suspect(vals)
        if why:
            print(f"  !! {tag}: {why}")
    if base_a is None or base_b is None:
        print(f"  !! fewer than {MIN_BASELINE} load samples on at least one "
              f"arm, so NO timing was vetted for contention. The timing "
              f"ratios below are UNVETTED, not clean.")

    if rows:
        print(f"\n--- LOCALISATION: did the failing set land in the module "
              f"the PR changed? ---")
        print(f"{'inst':>6}  {'changed module':<26} "
              f"{na+' in-module':>14} {na+' modules':>12} "
              f"{nb+' in-module':>14} {nb+' modules':>12}")
        for n, mod, trunc, ha, pa, hb, pb in rows:
            print(f"{n:>6}  {mod:<26} {('yes' if ha else 'no'):>14} "
                  f"{pa:>12} {('yes' if hb else 'no'):>14} {pb:>12}"
                  f"{'   <-- list truncated at ' + str(LEGACY_CAP) if trunc else ''}")
        for tag in (na, nb):
            h, m = loc[tag]
            print(f"  {tag}: {h}/{m} localised into the changed module")
        mp = [r[3] for r in rows]
        mq = [r[5] for r in rows]
        print(f"  distinct implicated instances: {na} median "
              f"{statistics.median(mp):.1f}, {nb} median "
              f"{statistics.median(mq):.1f}  (lower is a tighter answer)")
    if no_failures:
        print(f"  ({len(no_failures)} instance(s) with no failing partitions "
              f"on either arm are excluded from BOTH denominators -- a proven "
              f"instance cannot localise anything: {no_failures})")
    if capped:
        print(f"  !! {len(capped)} instance(s) carry a failing list of exactly "
              f"{LEGACY_CAP} with no n_failed, i.e. a TRUNCATED prefix of "
              f"unknown length: {capped}. A 'no' on those proves nothing and "
              f"the module counts are floors. Re-measure before quoting.")
    if unmapped:
        print(f"  ({len(unmapped)} instance(s) whose changed module could not "
              f"be mapped to an instance name and are NOT counted: "
              f"{[f'#{n} {m}' for n, m in unmapped]})")

    if part_ratio:
        print(f"\n--- COST (reported, not gated) ---")
        print(f"  partitions  {na}/{nb} median "
              f"{statistics.median(part_ratio):.2f}x   "
              f"({len(part_ratio)} instances; counts, so load cannot move them)")
        if wall_ratio:
            note = (f"({len(wall_ratio)} timing-clean; {len(drift_skipped)} "
                    f"dropped for load drift: {drift_skipped})"
                    if drift_skipped else f"({len(wall_ratio)} instances)")
            print(f"  eqy -m      {na}/{nb} median "
                  f"{statistics.median(setup_ratio):.2f}x   {note}")
            print(f"  wall        {na}/{nb} median "
                  f"{statistics.median(wall_ratio):.2f}x   {note}")
        else:
            print(f"  no timing-clean instance in either arm "
                  f"({len(drift_skipped)} dropped for load drift) -- no "
                  f"timing ratio reported")
        print(f"  reuse is expected to differ between arms and is not "
              f"compared: the two arms do not share partitions.")
    return 2 if unsound else 0


if __name__ == "__main__":
    raise SystemExit(main())
