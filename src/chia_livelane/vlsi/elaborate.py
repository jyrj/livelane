"""Work out how to elaborate a design, by asking the front end.

A real RTL checkout does not tell you how to read it. Its filelist is often
stale, its include paths move between commits, and some of the modules it
instantiates are technology primitives the integrator is expected to supply.
Hardcoding one recipe per design works until the design moves; across the
pinned base commits of a benchmark it is wrong on most of them.

This module treats elaboration as a fixpoint instead. Feed the front end what
you have, read what it says is missing, resolve that against the checkout, and
retry. Three kinds of complaint are handled:

``'foo.svh': No such file or directory``
    Find the header in the tree and add its directory as an include path.

``unknown module 'foo'`` / ``unknown package 'foo'``
    Find the file that declares it and add it to the source list. When several
    files declare it, integrations routinely ship their own behavioural copy
    of a clock gate, the choice is deterministic and recorded.

``encountered unsupported SVA feature``
    Escalate through the macros designs conventionally use to exclude
    simulation-only assertion code, one at a time, and only while SVA is
    actually blocking. A design that needs none is never silently abstracted.

Everything resolved is returned in ``notes`` so a run can be reproduced and
audited. For equivalence checking the soundness argument is symmetry: both
sides of a miter are elaborated from the identical source list, include paths
and define set, so a substituted primitive or an excluded assertion cannot make
two different designs look equal.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Elaboration", "resolve", "seed_sources"]

# An include name may be subdirectory-qualified, CVA6's cvfpu writes
# `include "common_cells/registers.svh"`, so the name carries a separator and
# the directory to add is the path MINUS that suffix, not the file's parent.
_MISSING_INC = re.compile(
    r"'([\w./-]+\.(?:svh|sv|vh|h))': No such file or directory")
_UNKNOWN_MOD = re.compile(r"unknown module '(\w+)'")
# slang says "unknown class or package 'x'", not "unknown package 'x'".
_UNKNOWN_PKG = re.compile(r"unknown (?:class or )?package '(\w+)'")
_SVA_ERR = re.compile(r"unsupported SVA feature", re.I)
_ERR_LINE = re.compile(r"^.*: error: .*$", re.M)

#: Macros designs conventionally use to guard simulation-only assertion code.
#: Escalated in order, and only when SVA actually blocks elaboration.
ASSERT_GUARDS = ("SYNTHESIS", "VERILATOR", "YOSYS")


@dataclass
class Elaboration:
    """A source list that elaborates, and the record of how it was reached."""

    sources: list[Path] = field(default_factory=list)
    includes: list[Path] = field(default_factory=list)
    defines: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.sources)

    #: Extra front-end flags that are part of HOW the design is elaborated,
    #: e.g. `--keep-hierarchy`. Carried here rather than bolted on by callers
    #: so that resolution and the miter config cannot disagree about them,
    #: elaborating the resolver's probe differently from the actual check is
    #: how a design "resolves" and then fails to build.
    front_end_flags: list[str] = field(default_factory=list)

    def read_cmd(self, base: str = "read_slang") -> str:
        """The front-end command line these settings imply."""
        parts = [base] + list(self.front_end_flags)
        parts += [f"-I {p}" for p in self.includes]
        parts += [f"-D {d}" for d in self.defines]
        return " ".join(parts)


def _run(cmd: list[str], cwd: Path, timeout: int = 900):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout)


def _declares(tree: Path, kind: str, name: str, exclude: set[Path]) -> list[Path]:
    pat = re.compile(rf"^\s*{kind}\s+(?:automatic\s+)?{re.escape(name)}\b", re.M)
    hits = []
    for q in sorted(tree.rglob("*.sv")) + sorted(tree.rglob("*.v")):
        if q in exclude:
            continue
        try:
            if pat.search(q.read_text(errors="ignore")):
                hits.append(q)
        except OSError:
            continue
    return hits


def _pick(hits: list[Path], prefer: str | None = None) -> Path:
    """Deterministic choice among candidate definitions.

    *prefer* narrows to paths containing that substring, and exists for the
    case where the candidates are NOT interchangeable. CVA6 ships 21 files that
    all declare `package cva6_config_pkg`; they are mutually exclusive build
    configurations, RV32 or RV64, with or without an FPU, so picking one by
    sort order would silently decide which processor is being verified. That is
    a configuration decision and it should be stated, not inferred.

    Otherwise: prefer a simulation model, then the shallowest path, then
    lexicographic order. There the copies really are interchangeable, because
    both sides of a miter get the same one; what matters is that the choice is
    stable across runs and written down.
    """
    if prefer:
        narrowed = [q for q in hits if prefer in str(q)]
        if narrowed:
            hits = narrowed
    return sorted(hits, key=lambda q: (0 if "sim" in q.parts else 1,
                                       len(q.parts), str(q)))[0]


_VAR_RE = re.compile(r"\$\{(\w+)\}")


def seed_sources(tree: Path, filelist: str | None = None,
                 glob: str | None = None,
                 variables: dict[str, str] | None = None) -> list[Path]:
    """Where to start: the design's own filelist, else a glob of its RTL.

    A filelist states the design's cone, so it excludes siblings that break a
    blind glob, a tracing wrapper that needs a macro defined, a second
    register-file implementation. Where it is stale or absent, :func:`resolve`
    adds back whatever the front end reports missing, so neither seed has to be
    right on its own.

    Filelists in the wild are not plain path lists. They carry ``//`` comments,
    ``+incdir+`` directives, and ``${VAR}`` references to directories the build
    system sets, CVA6's list is 188 lines of exactly this. Handling:

    - ``${VAR}`` is expanded from *variables*; an unknown one falls back to the
      checkout root, which is what the common ``${<PROJECT>_REPO_DIR}`` spelling
      means. A path that does not exist afterwards is dropped, so a genuinely
      unresolvable variable costs that entry rather than the whole list.
    - ``+incdir+`` lines are skipped rather than parsed. :func:`resolve`
      discovers include directories from the front end's own missing-header
      errors, which stays correct when the filelist's paths are stale.
    """
    if filelist and (tree / filelist).exists():
        base = (tree / filelist).parent
        subs = dict(variables or {})
        out: list[Path] = []
        for line in (tree / filelist).read_text().splitlines():
            line = line.split("//")[0].strip()
            if not line or line.startswith(("+", "-", "#")):
                continue
            line = _VAR_RE.sub(lambda m: subs.get(m.group(1), str(tree)), line)
            cand = Path(line)
            cand = cand if cand.is_absolute() else (base / cand)
            try:
                cand = cand.resolve()
            except OSError:
                continue
            if cand.exists() and cand.is_file() and cand not in out:
                out.append(cand)
        if out:
            return out
    return sorted(tree.glob(glob)) if glob else []


def resolve(tree: Path, seed: list[Path], top: str, yosys: str, *,
            max_rounds: int = 16, fallback: Path | None = None,
            stash: Path | None = None,
            prefer: dict[str, str] | None = None,
            front_end_flags: list[str] | None = None) -> Elaboration:
    """Drive elaboration to success, resolving what the front end asks for.

    Args:
        tree: The checkout being elaborated.
        seed: Starting source list, e.g. from :func:`seed_sources`.
        top: Top module name; passed to the front end explicitly, because
            `read_slang` otherwise infers one and may pick the wrong candidate.
        yosys: Path to a slang-enabled yosys.
        fallback: A second checkout of the *same project* to take a definition
            from when the tree defines it nowhere. Older commits instantiate
            technology primitives such as `prim_clock_gating` without shipping
            one. Taking the project's own later definition is preferable to
            inventing a stub, which would mean choosing the semantics
            ourselves underneath a soundness claim.
        stash: Directory to copy a fallback definition into, so the resolved
            source list is self-contained and the instance reproduces.
        prefer: Map of module/package name to a path substring, for names whose
            candidate definitions are not interchangeable. Without it CVA6's
            21 mutually exclusive `cva6_config_pkg` files would be decided by
            sort order.

    Returns:
        Elaboration: ``ok`` when a source list elaborates; otherwise ``error``
        carries the front end's own diagnostics.
    """
    srcs, incs, defs = list(seed), [], []
    flags = list(front_end_flags or [])
    notes: list[str] = []
    seen: set[tuple[str, str]] = set()

    for _ in range(max_rounds):
        inc = " ".join(f"-I {i}" for i in incs)
        dfs = " ".join(f"-D {d}" for d in defs)
        files = " ".join(str(s) for s in srcs)
        fl = " ".join(flags)
        r = _run([yosys, "-p",
                  f"read_slang {fl} {inc} {dfs} {files} --top {top}"], cwd=tree)
        if r.returncode == 0:
            return Elaboration(srcs, incs, defs, notes, None, flags)

        blob = r.stdout + r.stderr
        progressed = False

        if _SVA_ERR.search(blob):
            nxt = [g for g in ASSERT_GUARDS if g not in defs]
            if nxt:
                defs.append(nxt[0])
                notes.append(f"-D {nxt[0]} (to exclude unsynthesisable SVA)")
                progressed = True

        for name in dict.fromkeys(_MISSING_INC.findall(blob)):
            if ("inc", name) in seen:
                continue
            seen.add(("inc", name))
            depth = name.count("/")
            hits = sorted(tree.rglob(Path(name).name))
            for h in hits:
                # For `common_cells/registers.svh` the include directory is the
                # one holding `common_cells`, so climb one level per separator.
                # Adding the file's own parent instead would leave the
                # qualified path still unresolvable.
                root = h.parent
                for _ in range(depth):
                    root = root.parent
                if (root / name).exists() and root not in incs:
                    incs.append(root)
                    notes.append(f"include {name} -> {root.relative_to(tree)}")
                    progressed = True
                    break

        for kind, rx in (("module", _UNKNOWN_MOD), ("package", _UNKNOWN_PKG)):
            for name in dict.fromkeys(rx.findall(blob)):
                if (kind, name) in seen:
                    continue
                seen.add((kind, name))
                want = (prefer or {}).get(name)
                hits = _declares(tree, kind, name, set(srcs))
                if hits:
                    pick = _pick(hits, want)
                    srcs.append(pick)
                    notes.append(
                        f"{kind} {name} -> {pick.relative_to(tree)}"
                        + (f" [{len(hits)} candidates"
                           + (f", prefer={want!r}" if want else "") + "]"
                           if len(hits) > 1 else ""))
                    progressed = True
                elif fallback is not None and stash is not None:
                    fb = _declares(fallback, kind, name, set())
                    if fb:
                        src = _pick(fb, want)
                        stash.mkdir(parents=True, exist_ok=True)
                        dst = stash / src.name
                        dst.write_text(src.read_text(errors="ignore"))
                        srcs.append(dst)
                        notes.append(f"{kind} {name} -> "
                                     f"{src.relative_to(fallback)} @fallback "
                                     f"(declared nowhere in this checkout)")
                        progressed = True

        if not progressed:
            errs = [e.strip() for e in _ERR_LINE.findall(blob)[:4]]
            return Elaboration(notes=notes,
                               error=" | ".join(errs) or "elaboration failed")

    return Elaboration(notes=notes, error="resolution did not converge")
