"""Second stage of the equivalence gate: settle a refusal before believing it.

The partition-based gate (``eqy``) cuts both designs at signals it matches
between them and proves each piece with those cuts as FREE inputs. It cannot
falsely accept. It can falsely reject: an edit that redefines a matched
internal net while preserving every output is refused on that net.

Measured on a whole ``ibex_core``: of 21 edits production agents proposed and
the partitioned gate refused, 17 were correct, all 15 from the stronger
model, whose fetch-FIFO rewrites substitute ``out_valid_o``'s definition into
``pop_fifo``. The gate cuts at ``out_valid_o``, frees it, and refuses
``fifo_i.pop_fifo``. Believing that refusal tells the agent its correct work is
wrong.

This stage proves the EDITED INSTANCE against its parent with an unbounded
sequential engine (``abc pdr``), and when PDR cannot decide, searches for a
counterexample with bounded model checking (``abc bmc3``). Three properties
make its verdict usable as the gate's:

* **Instance scope, real parameters.** Each instance is cut out of the WHOLE
  design elaborated with hierarchy kept, so it carries the parameters the design
  actually gives it. Checked standalone at default parameters, one agent edit
  to ``ibex_counter`` is "proven" although it breaks the core's ``minstret``
  counter, and another yields a counterexample at widths the core never uses.
* **Every instance.** A module instantiated N times is proven only if all N
  are; one diverging instance is a counterexample.
* **One common initial state.** ``setundef -zero -init`` on both copies, so BMC
  and PDR cannot start gold and gate in different states and "find" a
  difference that is only an artefact of the start.

A proof at instance scope, with free inputs at the instance boundary, covers
every input the rest of the design could supply, so replacing the instance
preserves the design's behaviour: it is safe to ACCEPT on. A counterexample
shows only that the instance computes something different, not that the rest
of the design ever drives that input, so a counterexample keeps the edit
refused, which is where it already was.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

try:  # CHIA is optional, exactly as for lec_gate: verdict logic stays testable
    from chia.base.ChiaFunction import ChiaFunction
except Exception:  # pragma: no cover - exercised only outside a CHIA install
    def ChiaFunction(**_kwargs):  # type: ignore[misc]
        def deco(fn):
            return fn
        return deco

PROVEN = "PROVEN-SEQ"
CONFIRMED = "CONFIRMED"
UNDECIDED = "UNDECIDED"


def _tool(name: str) -> str:
    tools = os.environ.get("LIVELANE_TOOLS")
    return str(Path(tools) / "bin" / name) if tools else name


def instances(srcs: list[str], top: str, module: str, includes: list[str],
              defines: list[str]) -> list[str]:
    """Specialised module names for every instance of ``module`` under ``top``.

    With ``--keep-hierarchy`` read_slang names each instance's module
    ``<module>$<instance path>``, parameters baked in.
    """
    inc = " ".join(f"-I {i}" for i in includes)
    dfs = " ".join(f"-D {d}" for d in defines)
    p = subprocess.run(
        [_tool("yosys"), "-p",
         f"read_slang --keep-hierarchy {inc} {dfs} {' '.join(srcs)} "
         f"--top {top}; hierarchy -top {top}; ls"],
        capture_output=True, text=True, timeout=600)
    out = []
    for ln in p.stdout.splitlines():
        ln = ln.strip()
        if ln.startswith(f"{module}$"):
            out.append(ln)
    return out


def _sby(gold: list[str], gate: list[str], top: str, inst: str,
         includes: list[str], defines: list[str], engine: str,
         depth: int) -> str:
    inc = " ".join(f"-I {i}" for i in includes)
    dfs = " ".join(f"-D {d}" for d in defines)

    def side(files, name):
        return (f"read_slang --keep-hierarchy {inc} {dfs} {' '.join(files)} "
                f"--top {top}\n"
                f"hierarchy -top {top}\nhierarchy -top {inst}\n"
                f"proc\nflatten\nopt_clean\nmemory -nomap\nmemory_map\n"
                f"chformal -remove\nsetundef -zero -init\nrename {inst} {name}\n"
                f"design -stash {name}\n")

    mode = "prove" if engine == "abc pdr" else "bmc"
    head = f"[options]\nmode {mode}\n"
    if mode == "bmc":
        head += f"depth {depth}\n"
    return (head + "multiclock off\n\n" + f"[engines]\n{engine}\n\n[script]\n"
            + side(gold, "gold") + side(gate, "gate")
            + "design -copy-from gold -as gold gold\n"
              "design -copy-from gate -as gate gate\n"
              "miter -equiv -flatten -make_assert gold gate miter\n"
              "hierarchy -top miter\n")


def _descendants(pid: int) -> list[int]:
    kids: dict[int, list[int]] = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            ppid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        kids.setdefault(ppid, []).append(int(d.name))
    out, todo = [], [pid]
    while todo:
        for k in kids.get(todo.pop(), []):
            out.append(k)
            todo.append(k)
    return out


def _kill_tree(pid: int) -> None:
    for q in [pid] + _descendants(pid):
        for kill in (lambda: os.killpg(q, signal.SIGKILL),
                     lambda: os.kill(q, signal.SIGKILL)):
            try:
                kill()
            except (ProcessLookupError, PermissionError):
                pass


def _run(job: str, wd: Path, timeout_s: float) -> str:
    wd.mkdir(parents=True, exist_ok=True)
    # sby's own timeout stops its solvers cleanly; the hard kill below is the
    # fallback. sby runs each solver in a process group of its own, through a
    # shell, so killing sby, or sby's group, leaves the solver running with
    # no one to read it: every descendant is found and killed explicitly.
    job = job.replace("[options]\n",
                      f"[options]\ntimeout {max(1, int(timeout_s) - 5)}\n", 1)
    (wd / "job.sby").write_text(job)
    p = subprocess.Popen([_tool("sby"), "-f", "job.sby"], cwd=wd,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, start_new_session=True)
    try:
        log, _ = p.communicate(timeout=timeout_s + 30)
    except subprocess.TimeoutExpired:
        _kill_tree(p.pid)
        p.communicate()
        return "TIMEOUT"
    _kill_tree(p.pid)          # any solver sby left behind on its way out
    if "DONE (FAIL" in log:
        return "FAIL"
    if "DONE (PASS" in log:
        return "PASS"
    return "TIMEOUT" if "timeout" in log.lower() else "ERROR"


def settle(gold_srcs: list[str], gate_srcs: list[str], top: str, module: str,
           includes: list[str], defines: list[str], workdir: str,
           pdr_timeout_s: float = 120.0, bmc_depth: int = 20,
           bmc_timeout_s: float = 120.0) -> dict:
    """Decide a refused edit: proven, confirmed broken, or undecided.

    Args:
        gold_srcs: Full source list of the parent design.
        gate_srcs: The same list with the edited file replaced.
        top: Top module of the design (the scope parameters come from).
        module: Name of the edited module.
        includes: Include directories.
        defines: Preprocessor defines.
        workdir: Scratch directory.

    Returns:
        dict: ``verdict`` (``PROVEN-SEQ``, safe to accept; ``CONFIRMED``,
        a counterexample exists; ``UNDECIDED``, keep the refusal),
        ``instances`` with per-instance results, and ``wall_s``.
    """
    t0 = time.monotonic()
    wd = Path(workdir)
    insts = instances(gold_srcs, top, module, includes, defines)
    if not insts:
        return {"verdict": UNDECIDED, "instances": [],
                "wall_s": round(time.monotonic() - t0, 2),
                "note": f"{module} is not instantiated under {top}"}
    per = []
    for inst in insts:
        tag = re.sub(r"[^A-Za-z0-9_.]", "_", inst)   # sby shells out; no `$`
        r = _run(_sby(gold_srcs, gate_srcs, top, inst, includes, defines,
                      "abc pdr", 0), wd / f"pdr-{tag}", pdr_timeout_s)
        if r not in ("PASS", "FAIL"):
            b = _run(_sby(gold_srcs, gate_srcs, top, inst, includes, defines,
                          "abc bmc3", bmc_depth), wd / f"bmc-{tag}",
                     bmc_timeout_s)
            r = "FAIL" if b == "FAIL" else r
        per.append({"instance": inst, "result": r})
    rs = [x["result"] for x in per]
    if "FAIL" in rs:
        verdict = CONFIRMED
    elif all(x == "PASS" for x in rs):
        verdict = PROVEN
    else:
        verdict = UNDECIDED
    return {"verdict": verdict, "instances": per,
            "wall_s": round(time.monotonic() - t0, 2)}


def prove_any_state(gold_srcs: list[str], gate_srcs: list[str], top: str,
                    module: str, includes: list[str], defines: list[str],
                    workdir: str, seq: int = 2,
                    timeout_s: float = 300.0) -> dict:
    """Weak, state-agnostic check (``equiv_induct``), kept for ``init="zero"``.

    ``equiv_make`` matches registers and signals by name; ``equiv_induct``
    assumes every match held in the previous cycles and proves it holds in the
    next. That shows agreement, once reached, persists, it never checks that
    agreement is reached, which is the weak equivalence Yosys itself documents.
    For a proof from every common power-up state use ``settle_explicit``.
    """
    t0 = time.monotonic()
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    inc = " ".join(f"-I {i}" for i in includes)
    dfs = " ".join(f"-D {d}" for d in defines)
    per = []
    for inst in instances(gold_srcs, top, module, includes, defines):
        def side(files, name):
            return (f"read_slang --keep-hierarchy {inc} {dfs} {' '.join(files)} "
                    f"--top {top}\nhierarchy -top {top}\nhierarchy -top {inst}\n"
                    f"proc\nflatten\nopt_clean\nmemory -nomap\nmemory_map\n"
                    f"async2sync\nopt_clean\nrename {inst} {name}\n"
                    f"design -stash {name}\n")
        ys = (side(gold_srcs, "gold") + side(gate_srcs, "gate")
              + "design -copy-from gold -as gold gold\n"
                "design -copy-from gate -as gate gate\n"
                "equiv_make gold gate equiv\nhierarchy -top equiv\n"
                f"equiv_simple -seq {seq}\nequiv_induct -seq {seq}\n"
                "equiv_status -assert\n")
        tag = re.sub(r"[^A-Za-z0-9_.]", "_", inst)
        f = wd / f"{tag}.ys"
        f.write_text(ys)
        try:
            p = subprocess.run([_tool("yosys"), "-q", "-s", str(f)],
                               capture_output=True, text=True, timeout=timeout_s)
            r = "PASS" if p.returncode == 0 else "FAIL"
        except subprocess.TimeoutExpired:
            r = "TIMEOUT"
        per.append({"instance": inst, "result": r})
    rs = [x["result"] for x in per]
    return {"verdict": "ANY-STATE" if rs and all(x == "PASS" for x in rs)
            else "REACHABLE-ONLY" if "FAIL" in rs else "UNDECIDED",
            "instances": per, "wall_s": round(time.monotonic() - t0, 2)}


# --------------------------------------------------------------------------
# Initial-state-explicit proofs.
#
# ``settle`` starts both copies from all-zero. That is the reset state only
# when every reset value is zero: an edit that re-encodes a state machine
# one-hot (RESET = 'h001) starts in an ILLEGAL state and yields a
# counterexample that no reset ever reaches. ``prove_any_state`` uses
# ``equiv_induct``, which proves only that agreement on every matched signal,
# once reached, persists, it never checks that agreement is reached.
#
# The functions below build the gold/gate miter with NO initial values and
# state the initial condition on the first cycle:
#
#   shared  every register present in both copies (same name, same width)
#           powers up to the SAME arbitrary value, in the first cycle the
#           gate reads the gold register; registers only one copy has power
#           up arbitrary and independently
#   reset   the reset input is assumed asserted in the first cycle
#
# ``from-reset`` = shared + reset: whatever the silicon powers up to, after
# reset the copies never diverge. ``any-state`` = shared alone: they never
# diverge from ANY common state, reset or not. PDR proves either unboundedly;
# a counterexample to ``from-reset`` is reachable from reset.
# --------------------------------------------------------------------------

_REG = re.compile(r"\s*cell \$(?:dff|dffe|adff|adffe|sdff|sdffe|sdffce|aldff|"
                  r"aldffe|dffsr|dffsre) (\S+)")
_FORMAL_ONLY = re.compile(r"cell \$(?:anyseq|anyconst|anyinit|allseq|allconst|"
                          r"initstate) \$flatten")
_CLK = re.compile(r"\s*connect \\CLK (.+)$")
_RST = re.compile(r"^\s*wire (?:width 1 )?input \d+ \\in_(rst_ni|rst_n|rstn|"
                  r"reset_n|resetn)$", re.M)


def constrain_miter(il: str, shared: bool, reset: bool,
                    correspond: bool = False) -> tuple[str, dict]:
    """State the first cycle of a flattened ``miter`` module (RTLIL).

    With ``correspond`` every matched register pair is also ASSERTED equal in
    every cycle. That is a stronger property than the miter's, so a pass still
    proves the miter; and when the edit keeps each register's next-state
    function, the stronger property is inductive in one step, which is what
    makes a whole-core proof tractable.

    Returns the new RTLIL and a summary: registers per side, how many were
    shared, the ones that could not be (``unmatched``), and the reset port.
    """
    lines = il.split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "module \\miter")
    end = next(i for i in range(start, len(lines)) if lines[i] == "end")
    regs: dict[str, dict[str, tuple[str, int, int]]] = {"gold": {}, "gate": {}}
    i = start
    while i < end:
        m = _REG.match(lines[i])
        if m:
            w, q, qline, j = 1, None, None, i + 1
            while lines[j].strip() != "end":
                s = lines[j].strip()
                if s.startswith("parameter \\WIDTH "):
                    w = int(s.split()[-1])
                elif s.startswith("connect \\Q "):
                    q, qline = s[len("connect \\Q "):], j
                j += 1
            side = re.match(r"\$flatten\\(gold|gate)\.(.*)", m.group(1))
            if side and q:
                regs[side.group(1)][side.group(2)] = (q, w, qline)
            i = j
        i += 1
    gold, gate = regs["gold"], regs["gate"]
    pairs = [(gold[k], gate[k]) for k in sorted(gold)
             if k in gate and gold[k][1] == gate[k][1]]
    unmatched = sorted((set(gold) ^ set(gate))
                       | {k for k in gold if k in gate and gold[k][1] != gate[k][1]})
    rm = _RST.search("\n".join(lines[start:end]))
    rst = f"\\in_{rm.group(1)}" if rm else None

    decl: list[str] = []
    add = ["  wire \\__is", "  wire \\__nis",
           "  cell $initstate \\__isc", "    connect \\Y \\__is", "  end",
           "  cell $not \\__nisc", "    parameter \\A_SIGNED 0",
           "    parameter \\A_WIDTH 1", "    parameter \\Y_WIDTH 1",
           "    connect \\A \\__is", "    connect \\Y \\__nis", "  end"]

    def first_cycle(k: str, cond: str) -> None:   # assume(initstate -> cond)
        add.extend([
            f"  wire \\__a{k}", f"  cell $or \\__oc{k}",
            "    parameter \\A_SIGNED 0", "    parameter \\B_SIGNED 0",
            "    parameter \\A_WIDTH 1", "    parameter \\B_WIDTH 1",
            "    parameter \\Y_WIDTH 1", "    connect \\A \\__nis",
            f"    connect \\B {cond}", f"    connect \\Y \\__a{k}", "  end",
            f"  cell $assume \\__as{k}", f"    connect \\A \\__a{k}",
            "    connect \\EN 1'1", "  end"])

    if shared:
        # Shared STRUCTURALLY, not by assumption: in the first cycle the gate
        # reads the gold register's value through a mux on $initstate. Equal
        # logic on equal signals then hashes to one node in every unrolled
        # frame; with an assumption instead, a solver must PROVE two copies
        # of a multiplier equal before it can take a single BMC step.
        for n, ((qa, w, _), (qb, _, qline)) in enumerate(pairs):
            lines[qline] = lines[qline].replace(f"connect \\Q {qb}",
                                                f"connect \\Q \\__rq{n}")
            decl.append(f"  wire width {w} \\__rq{n}")   # used above `add`
            add.extend([
                f"  cell $mux \\__im{n}",
                f"    parameter \\WIDTH {w}", f"    connect \\A \\__rq{n}",
                f"    connect \\B {qa}", "    connect \\S \\__is",
                f"    connect \\Y {qb}", "  end"])
            if correspond:
                add.extend([
                    f"  wire \\__eq{n}", f"  cell $eq \\__eqc{n}",
                    "    parameter \\A_SIGNED 0", "    parameter \\B_SIGNED 0",
                    f"    parameter \\A_WIDTH {w}", f"    parameter \\B_WIDTH {w}",
                    "    parameter \\Y_WIDTH 1", f"    connect \\A {qa}",
                    f"    connect \\B {qb}", f"    connect \\Y \\__eq{n}", "  end",
                    f"  cell $assert \\__ca{n}",
                    f"    connect \\A \\__eq{n}", "    connect \\EN 1'1", "  end"])
    if reset and rst:
        add.extend(["  wire \\__nrst", "  cell $not \\__nrstc",
                    "    parameter \\A_SIGNED 0", "    parameter \\A_WIDTH 1",
                    "    parameter \\Y_WIDTH 1", f"    connect \\A {rst}",
                    "    connect \\Y \\__nrst", "  end"])
        first_cycle("rst", "\\__nrst")
    return ("\n".join(lines[:start + 1] + decl + lines[start + 1:end] + add
                      + lines[end:]),
            {"gold_regs": len(gold), "gate_regs": len(gate),
             "shared": len(pairs) if shared else 0, "unmatched": unmatched,
             "reset_port": rst[1:] if (reset and rst) else None})


def clock_domains(il: str) -> set[tuple[str, str]]:
    """Every (clock signal, edge) that a ``$dff`` in the miter uses.

    The sequential proofs run with one global step per cycle, which is the
    design's behaviour only if every register ticks on the same clock edge.
    """
    out, clk, pol, in_dff = set(), None, "1", False
    for ln in il.split("\n"):
        t = ln.strip()
        if t.startswith("cell "):
            in_dff, clk, pol = t.startswith("cell $dff "), None, "1"
        elif in_dff and t.startswith("parameter \\CLK_POLARITY "):
            pol = t.split()[-1]
        elif in_dff and t.startswith("connect \\CLK "):
            clk = t[len("connect \\CLK "):]
        elif t == "end" and in_dff:
            out.add((clk, pol))
            in_dff = False
    return out


def cut_registers(il: str) -> tuple[str | None, dict]:
    """Replace every register pair of a miter by ONE free value per cycle.

    Each gold/gate register pair (same name, same width) is removed; a single
    ``$anyseq`` drives both copies' register outputs, and the copies' next
    states are asserted equal. What remains has no state, so one cycle is the
    whole proof: from ANY common state, both copies produce the same outputs
    and the same next state, by induction, they never diverge. Unchanged
    logic now reads identical signals in both copies and merges away, which is
    what makes a whole-core check cheap.

    Returns ``(None, info)`` when some register has no partner: the cut would
    then leave state behind, and the caller must use a sequential proof.
    Requires plain ``$dff`` registers (``async2sync; dffunmap``).
    """
    lines = il.split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "module \\miter")
    end = next(i for i in range(start, len(lines)) if lines[i] == "end")
    cells: dict[str, dict[str, tuple[int, int, str, str, int]]] = {"gold": {}, "gate": {}}
    other_ff = 0
    i = start
    while i < end:
        m = re.match(r"\s*cell \$(\w+) (\S+)", lines[i])
        if m:
            j = i + 1
            while lines[j].strip() != "end":
                j += 1
            if m.group(1) == "dff":
                w, q, d, clk, pol = 1, None, None, None, "1"
                for ln in lines[i + 1:j]:
                    t = ln.strip()
                    if t.startswith("parameter \\WIDTH "):
                        w = int(t.split()[-1])
                    elif t.startswith("parameter \\CLK_POLARITY "):
                        pol = t.split()[-1]
                    elif t.startswith("connect \\Q "):
                        q = t[len("connect \\Q "):]
                    elif t.startswith("connect \\D "):
                        d = t[len("connect \\D "):]
                    elif t.startswith("connect \\CLK "):
                        clk = t[len("connect \\CLK "):]
                side = re.match(r"\$flatten\\(gold|gate)\.(.*)", m.group(2))
                if side and q and d:
                    cells[side.group(1)][side.group(2)] = (i, j, q, d, w, (clk, pol))
                else:
                    other_ff += 1
            elif "ff" in m.group(1) or m.group(1) in ("dlatch", "sr"):
                other_ff += 1
            i = j
        i += 1
    gold, gate = cells["gold"], cells["gate"]
    # a pair must agree in width AND in clock and edge: one cycle of a cut is
    # one tick of ONE clock, and a register moved to the other edge is not the
    # same register
    unmatched = sorted((set(gold) ^ set(gate))
                       | {k for k in gold if k in gate
                          and gold[k][4:6] != gate[k][4:6]})
    if len({v[5] for v in list(gold.values()) + list(gate.values())}) > 1:
        other_ff += 1                      # more than one clock or edge
    info = {"pairs": len(set(gold) & set(gate)), "unmatched": unmatched,
            "other_ff": other_ff}
    if unmatched or other_ff:
        return None, info
    drop = set()
    add = []
    for n, k in enumerate(sorted(gold)):
        (gi, gj, gq, gd, w, _), (ti, tj, tq, td, _, _) = gold[k], gate[k]
        for a_, b_ in ((gi, gj), (ti, tj)):
            while a_ > start and lines[a_ - 1].strip().startswith("attribute "):
                a_ -= 1          # a cell's attributes precede it
            drop.update(range(a_, b_ + 1))
        add.extend([
            f"  cell $anyseq \\__cq{n}", f"    parameter \\WIDTH {w}",
            f"    connect \\Y {gq}", "  end",
            f"  connect {tq} {gq}",
            f"  wire \\__cd{n}", f"  cell $eq \\__cde{n}",
            "    parameter \\A_SIGNED 0", "    parameter \\B_SIGNED 0",
            f"    parameter \\A_WIDTH {w}", f"    parameter \\B_WIDTH {w}",
            "    parameter \\Y_WIDTH 1", f"    connect \\A {gd}",
            f"    connect \\B {td}", f"    connect \\Y \\__cd{n}", "  end",
            f"  cell $assert \\__cda{n}", f"    connect \\A \\__cd{n}",
            "    connect \\EN 1'1", "  end"])
    body = [ln for k, ln in enumerate(lines[start:end], start) if k not in drop]
    return "\n".join(lines[:start] + body + add + lines[end:]), info


def _prove_comb(il: str, wd: Path, timeout_s: float) -> str:
    """One-cycle proof of a stateless miter (after ``cut_registers``)."""
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "m.il").write_text(il)
    job = ("[options]\nmode bmc\ndepth 1\n\n[engines]\nsmtbmc yices\n\n"
           "[script]\nread_rtlil m.il\nhierarchy -top miter\nopt -fast\n"
           "opt_clean\n\n[files]\nm.il\n")
    return _run(job, wd, timeout_s)


def _miter_il(gold: list[str], gate: list[str], top: str, inst: str,
              includes: list[str], defines: list[str], wd: Path,
              timeout_s: float = 600.0) -> str | None:
    """The flattened gold/gate miter of one instance, with no initial values."""
    inc = " ".join(f"-I {i}" for i in includes)
    dfs = " ".join(f"-D {d}" for d in defines)

    def side(files, name):
        # Synthesis ignores formal statements and declaration initialisers,
        # so the proof must too: an `assume` in the edited RTL would otherwise
        # constrain the instance's free inputs, and an initialiser would pin
        # the "arbitrary" power-up state the proofs quantify over.
        return (f"read_slang --keep-hierarchy {inc} {dfs} {' '.join(files)} "
                f"--top {top}\nhierarchy -top {top}\nhierarchy -top {inst}\n"
                f"proc\nflatten\nopt_clean\nmemory -nomap\nmemory_map\n"
                f"chformal -remove\nsetattr -unset init\n"
                f"rename {inst} {name}\ndesign -stash {name}\n")

    wd.mkdir(parents=True, exist_ok=True)
    out = wd / "miter.il"
    ys = (side(gold, "gold") + side(gate, "gate")
          + "design -copy-from gold -as gold gold\n"
            "design -copy-from gate -as gate gate\n"
            "miter -equiv -flatten -make_assert gold gate miter\n"
            "hierarchy -top miter\nasync2sync\ndffunmap\nopt_clean\n"
            f"write_rtlil {out}\n")
    (wd / "miter.ys").write_text(ys)
    try:
        p = subprocess.run([_tool("yosys"), "-q", "-s", str(wd / "miter.ys")],
                           capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return None
    if p.returncode or not out.exists():
        return None
    il = out.read_text()
    if _FORMAL_ONLY.search(il):
        return None      # a free value the design itself declares: no verdict
    return il


def _prove_il(il: str, wd: Path, engine: str, depth: int,
              timeout_s: float, extra: str = "") -> str:
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "m.il").write_text(il)
    mode = "prove" if engine == "abc pdr" else "bmc"
    job = (f"[options]\nmode {mode}\n" + (f"depth {depth}\n" if mode == "bmc" else "")
           + f"\n[engines]\n{engine}\n\n[script]\nread_rtlil m.il\n"
             f"hierarchy -top miter\n{extra}\n[files]\nm.il\n")
    return _run(job, wd, timeout_s)


_NETS: dict[tuple, dict | None] = {}
_NETS_LOCK = threading.Lock()


def _flat_nets(srcs: list[str], top: str, includes: list[str],
               defines: list[str], wd: Path) -> dict | None:
    """Net names -> bit ids of the whole design flattened (aliases share ids)."""
    key = (tuple(srcs), top, tuple(includes), tuple(defines))
    with _NETS_LOCK:
        if key in _NETS:
            return _NETS[key]
    inc = " ".join(f"-I {i}" for i in includes)
    dfs = " ".join(f"-D {d}" for d in defines)
    wd.mkdir(parents=True, exist_ok=True)
    out = wd / "flat.json"
    try:
        p = subprocess.run(
            [_tool("yosys"), "-q", "-p",
             f"read_slang --keep-hierarchy {inc} {dfs} {' '.join(srcs)} --top {top}; "
             f"hierarchy -top {top}; proc; flatten; write_json {out}"],
            capture_output=True, text=True, timeout=600)
        nets = None
        if p.returncode == 0 and out.exists():
            mod = json.loads(out.read_text())["modules"][top]
            nets = {"nets": {k: v["bits"] for k, v in mod["netnames"].items()},
                    "ports": set(mod["ports"])}
            out.unlink()
    except (subprocess.TimeoutExpired, ValueError, KeyError):
        nets = None
    with _NETS_LOCK:
        _NETS[key] = nets
    return nets


def reset_is_design_reset(srcs: list[str], top: str, inst: str, port: str,
                          includes: list[str], defines: list[str],
                          wd: Path) -> bool:
    """Is the instance's reset input the design's own reset input?

    Asserting an instance's reset in the first cycle stands for asserting the
    DESIGN's reset. That holds only if the parent wires the one to the other;
    an instance whose reset is tied off, gated or synchronised is never reset
    the way the assumption says, and must be proven without it.
    """
    if inst == top:
        return True
    nets = _flat_nets(srcs, top, includes, defines, wd)
    if not nets:
        return False
    path = inst.split("$", 1)[1] if "$" in inst else inst
    path = path[len(top) + 1:] if path.startswith(top + ".") else path
    mine = nets["nets"].get(f"{path}.{port}")
    return any(mine is not None and nets["nets"].get(p) == mine
               for p in nets["ports"] if _RST.match(f"  wire input 1 \\in_{p}"))


def settle_explicit(gold_srcs: list[str], gate_srcs: list[str], top: str,
                    module: str, includes: list[str], defines: list[str],
                    workdir: str, pdr_timeout_s: float = 240.0,
                    bmc_depth: int = 20, bmc_timeout_s: float = 240.0) -> dict:
    """Decide an edit from reset AND from any shared state, per instance.

    Returns ``from_reset`` (``PROVEN-SEQ`` / ``CONFIRMED`` / ``UNDECIDED``:
    after reset, from any common power-up state) and ``any_state``
    (``ANY-STATE`` / ``REACHABLE-ONLY`` / ``UNDECIDED``: from any common state,
    no reset), with per-instance detail.
    """
    t0 = time.monotonic()
    wd = Path(workdir)
    insts = [top] if module == top else instances(gold_srcs, top, module,
                                                  includes, defines)
    per = []
    for inst in insts:
        tag = re.sub(r"[^A-Za-z0-9_.]", "_", inst)
        il = _miter_il(gold_srcs, gate_srcs, top, inst, includes, defines,
                       wd / tag)
        if il is None:
            per.append({"instance": inst, "reset": "ERROR", "any": "ERROR"})
            continue
        row = {"instance": inst}
        if len(clock_domains(il)) > 1:
            # one global step per cycle is this design's behaviour only with
            # one clock and one edge; anything else gets no verdict here
            row.update({"reset": "UNDECIDED", "any": "UNDECIDED",
                        "note": "more than one clock or clock edge"})
            per.append(row)
            continue
        rm = _RST.search(il)
        port = rm.group(1) if rm else None
        use_reset = bool(port) and reset_is_design_reset(
            gold_srcs, top, inst, port, includes, defines, wd / "nets")
        row["reset_assumed"] = use_reset
        # Fast path: registers correspond one-to-one. Cut them and prove one
        # cycle; a pass proves both properties at once (any common state
        # includes every state after reset). A failure proves nothing, the
        # edit may change what a register holds on unreachable states only,
        # so fall through to the sequential proofs.
        c, cinfo = cut_registers(il)
        if c is not None:
            r = _prove_comb(c, wd / tag / "comb", pdr_timeout_s)
            row["register_cut"] = r
            if r == "PASS":
                row.update({"reset": "PASS", "any": "PASS", "unmatched": [],
                            "reset_port": None})
                per.append(row)
                continue
        for key, reset in (("reset", use_reset), ("any", False)):
            if key == "any" and row.get("reset") == "FAIL":
                # a trace from reset starts in some common power-up state
                row["any"] = "FAIL"
                continue
            c, info = constrain_miter(il, shared=True, reset=reset)
            r = _prove_il(c, wd / tag / f"pdr-{key}", "abc pdr", 0, pdr_timeout_s)
            if r not in ("PASS", "FAIL"):
                # A refutation needs ONE legitimate power-up state, not all of
                # them. All-zero is one; searching from it lets identical logic
                # constant-fold, where arbitrary register values leave a solver
                # proving two multipliers equal before its first step. A trace
                # found here is a trace in silicon, reachable from reset.
                z, _ = constrain_miter(il, shared=False, reset=reset)
                b = _prove_il(z, wd / tag / f"bmc-{key}", "abc bmc3", bmc_depth,
                              bmc_timeout_s, extra="setundef -zero -init\n")
                r = "FAIL" if b == "FAIL" else r
            row[key] = r
            row["unmatched"] = info["unmatched"]
            row["reset_port"] = row.get("reset_port") or info["reset_port"]
        per.append(row)
    rs = [x["reset"] for x in per]
    an = [x["any"] for x in per]
    from_reset = (UNDECIDED if not per else CONFIRMED if "FAIL" in rs
                  else PROVEN if all(x == "PASS" for x in rs) else UNDECIDED)
    any_state = ("ANY-STATE" if per and all(x == "PASS" for x in an)
                 else "REACHABLE-ONLY" if from_reset == PROVEN and "FAIL" in an
                 else UNDECIDED)
    return {"from_reset": from_reset, "any_state": any_state, "instances": per,
            "wall_s": round(time.monotonic() - t0, 2)}


def refute_at_top(gold_srcs: list[str], gate_srcs: list[str], top: str,
                  includes: list[str], defines: list[str], workdir: str,
                  depth: int = 20, timeout_s: float = 600.0) -> str:
    """Search for a counterexample of the WHOLE design, from reset.

    An instance-scope refutation shows the edited module computes something
    different for SOME input; the rest of the design may never supply it (a
    rewrite dropping `|| pmp_err_q` is refuted at instance scope and correct in
    a core whose PMP is disabled). A trace of the whole design, all-zero
    power-up, reset asserted in the first cycle, is a bug in silicon.

    Returns ``FAIL`` (a whole-design counterexample exists), ``PASS`` (none
    within ``depth`` cycles), or ``TIMEOUT``/``ERROR``.
    """
    wd = Path(workdir)
    il = _miter_il(gold_srcs, gate_srcs, top, top, includes, defines, wd)
    if il is None:
        return "ERROR"
    z, _ = constrain_miter(il, shared=False, reset=True)
    return _prove_il(z, wd / "bmc", "abc bmc3", depth, timeout_s,
                     extra="setundef -zero -init\n")


_ITEM = re.compile(
    r"typedef\s+(?:enum|struct|union)\b[^;{]*\{.*?\}\s*(?P<t>\w+)\s*;"
    r"|\b(?:localparam|parameter)\b[^;=]*?\b(?P<p>\w+)\s*=[^;]*;"
    r"|\bfunction\b[^;(]*?\b(?P<f>\w+)\s*\(.*?\bendfunction\b", re.S)


# comments are stripped first; `endmodule` has no word boundary before "module"
_MODULE = re.compile(r"\b(?:module|macromodule)\s+(?:automatic\s+|static\s+)?(\w+)")


def _unnest(text: str) -> str:
    """Drop bracketed content, so commas inside (), [], {} are not seen."""
    prev = None
    while prev != text:
        prev, text = text, re.sub(r"\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\}", " ", text)
    return text


def _norm(text: str) -> str:
    text = re.sub(r"//[^\n]*|/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"\s+", " ", text).strip()


def _items(text: str) -> tuple[dict[str, str], str]:
    """A package's named items (typedefs, parameters, functions) -> their
    text, and everything else (the residue)."""
    text = _norm(text)
    items, rest, last = {}, [], 0
    for m in _ITEM.finditer(text):
        name = m.group("t") or m.group("p") or m.group("f")
        items[name] = m.group(0)
        rest.append(text[last:m.start()])
        last = m.end()
    rest.append(text[last:])
    return items, _norm(" ".join(rest))


def edit_scope(gold_pkg: Path, cand_pkg: Path, srcs: list[Path],
               top: str) -> tuple[str, list[str]]:
    """Where to prove an edit to a PACKAGE, which has no instance of its own.

    The package's named items (typedefs, parameters, functions) are diffed,
    and the change is closed over dependence: an item that names a changed
    item, or one of its enum members, has changed too (a struct holding a
    re-encoded enum). The modules that name anything in that closure are the
    edit's reach. Exactly one such module is proven at its instances; any
    other reach, or any change outside a named item, is proven at ``top``.
    A module proven equivalent alone changes nothing its neighbours can see,
    so the reach needs only to cover every module whose OWN logic changed.

    Returns the module to prove and the changed names.
    """
    g, g_rest = _items(gold_pkg.read_text())
    c, c_rest = _items(cand_pkg.read_text())
    changed = {k for k in set(g) | set(c) if g.get(k) != c.get(k)}
    if not changed or g_rest != c_rest:
        return top, sorted(changed)
    # a parameter item declaring several names (`A = 1, B = 2;`) is keyed by
    # its first name only: its other names cannot be traced, so prove at top
    for k in changed:
        for side in (g, c):
            t = side.get(k, "")
            if re.match(r"\s*(?:localparam|parameter)\b", t) and \
                    re.search(r",\s*[A-Za-z_]\w*\s*=(?!=)", _unnest(t)):
                return top, sorted(changed)

    def words(t: str) -> set[str]:
        # sized literals first: the `h001` of `10'h001` is not an identifier
        t = re.sub(r"\d*\s*'\s*[sS]?[bBoOdDhH]\s*[0-9a-fA-FxXzZ_?]+", " ", t)
        return set(re.findall(r"\b[A-Za-z_]\w*\b", t))

    def reach(names: set[str]) -> set[str]:
        out = set(names)
        for k in names:     # an enum's members are names in their own right
            for side in (g, c):
                body = side.get(k, "")
                if body.startswith("typedef") and "enum" in body.split("{")[0]:
                    out |= words(body[body.find("{") + 1:body.rfind("}")])
        return out

    names = reach(changed)
    while True:             # close over dependence inside the package
        more = {k for k in set(g) | set(c) if k not in changed
                and (words(g.get(k, "")) | words(c.get(k, ""))) & (names - {k})}
        if not more:
            break
        changed |= more
        names = reach(changed)
    pm = re.search(r"\bpackage\s+(\w+)", _norm(gold_pkg.read_text()))
    pkg_name = pm.group(1) if pm else None
    users = set()
    for s in srcs:
        if s.name == gold_pkg.name:
            continue
        t = s.read_text(errors="ignore")
        nt = _norm(t)
        # another package's identifiers are its own namespace: `RESET` in
        # prim_foo_pkg is not ours unless that package names ours
        if pkg_name and re.search(r"\bpackage\s+\w+", nt) and \
                pkg_name not in words(nt):
            continue
        if words(nt) & names:
            mods = _MODULE.findall(nt)
            if not mods:
                # another package, an interface, an include: whatever it
                # defines from the changed names reaches modules we cannot see
                return top, sorted(changed)
            users.update(mods)
    if len(users) == 1:
        return users.pop(), sorted(changed)
    return top, sorted(changed)


@ChiaFunction(resources={"eqy": 1})
def second_stage(gold_srcs: list[str], gate_srcs: list[str], top: str,
                 module: str, includes: list[str], defines: list[str],
                 workdir: str, any_state: bool = False,
                 init: str = "reset") -> dict:
    """CHIA node: settle an edit the partitioned gate refused.

    Run this after ``lec_gate`` returns ``refuted``, on the same sources.
    It proves the edited module's instances against the parent at the
    parameters the design gives them, and returns ``PROVEN-SEQ`` (the refusal
    was false; the edit may be accepted on this proof), ``CONFIRMED`` (a
    counterexample trace exists) or ``UNDECIDED`` (keep the refusal).

    ``init`` states where both copies start:

    * ``"reset"``, the same arbitrary power-up state, reset asserted in the
      first cycle (``settle_explicit``). An edit to a package is proven where
      its changed types reach (``edit_scope``). Use this.
    * ``"zero"``, both copies from all-zero (``settle``), as the second
      stage ran in the release's reported runs (``loop.py --init zero``).
      It can accept an edit that relies on un-reset flops powering up to
      zero; see ``settle_explicit``.

    With ``any_state=True`` a ``PROVEN-SEQ`` must also hold from every
    common state, reset or not.

    Args:
        gold_srcs (list[str]): Source list of the parent design.
        gate_srcs (list[str]): Same list with the edited file replaced.
        top (str): Top module (where parameters come from).
        module (str): Name of the edited module (or package).
        includes (list[str]): Include directories.
        defines (list[str]): Preprocessor defines.
        workdir (str): Scratch directory.
        any_state (bool): Also require equivalence from any common state.
        init (str): ``"reset"`` or ``"zero"``.

    Returns:
        dict: ``verdict``, per-instance results and ``wall_s``.
    """
    if init == "reset":
        edited = [(Path(g), Path(c)) for g, c in zip(gold_srcs, gate_srcs)
                  if Path(g).read_bytes() != Path(c).read_bytes()]
        scope = module
        if len(edited) == 1:
            text = edited[0][1].read_text(errors="ignore")
            if re.search(r"^\s*package\s+\w+", text, re.M) and \
                    not re.search(r"^\s*module\s+", text, re.M):
                scope, _ = edit_scope(edited[0][0], edited[0][1],
                                      [Path(x) for x in gold_srcs], top)
        e = settle_explicit(gold_srcs, gate_srcs, top, scope, includes,
                            defines, workdir)
        r = {"verdict": e["from_reset"], "any_state": e["any_state"],
             "scope": scope, "instances": e["instances"], "wall_s": e["wall_s"]}
        if any_state and r["verdict"] == PROVEN and e["any_state"] != "ANY-STATE":
            r["verdict"] = UNDECIDED
            r["note"] = "equivalent from reset only"
        return r
    r = settle(gold_srcs, gate_srcs, top, module, includes, defines, workdir)
    if any_state and r["verdict"] == PROVEN:
        a = prove_any_state(gold_srcs, gate_srcs, top, module, includes,
                            defines, str(Path(workdir) / "any-state"))
        r["any_state"] = a["verdict"]
        if a["verdict"] != "ANY-STATE":
            r["verdict"] = UNDECIDED
            r["note"] = "equivalent from the common reset state only"
    return r
