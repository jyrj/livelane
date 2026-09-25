"""Is a reported failing signal the SAME NET as a port of the module that was edited?

`compare_frontend.py` answers a coarser question, does the failing partition
sit inside the changed module, and on that question hierarchy wins 7/7 against
flat's 5/7. But "flat missed" is not the interesting part. The interesting claim
is WHERE flat lands instead, and that was traced by hand on two instances:

    #48  flat says ibex_if_stage.illegal_c_insn
         ibex_if_stage.sv:500  .illegal_instr_o (illegal_c_insn)
        , the parent's net, connected to the changed module's output port

    #157 flat says ibex_cs_registers.csr_save_cause_i
         ibex_core.sv:1550     .csr_save_cause_i (csr_save_cause)
         ibex_core.sv:753      .csr_save_cause_o (csr_save_cause)
         ibex_id_stage.sv:693  .csr_save_cause_o (csr_save_cause_o)
        , three hops of port connections back to the changed module's output

Two hand-traced cases are an anecdote. This makes it mechanical.

It builds the hierarchical net-merge from the RTL text, the same merge a
front end does, as a union-find over (module, identifier) nodes: for every
instantiation of M inside P with a connection `.p (net)`, `(M, p)` and
`(P, net)` are the same electrical node. Then it asks whether each reported
signal is in the same class as any port of the module the PR changed.

What that buys: "flat named a signal three modules away" becomes "flat named
THE SAME NET, observed three modules away", which is a much more precise
statement and a falsifiable one. If a flat failure turns out NOT to be on a net
touching the changed module, the downstream story is wrong for that instance and
it is reported as `no` rather than quietly dropped.

Limits, stated because they bound the result:
  * connections whose expression is not a bare identifier (concatenations,
    slices, constants) are not merged, and are counted and reported;
  * a module instantiated more than once collapses into one class, which can
    only ever make this MORE permissive, never less;
  * this is connectivity, not dataflow: it says same net, not "downstream".
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# `<module> [#(...)] <instance> (` at an indent, the way ibex writes it.
_INST = re.compile(
    r"^\s{2,}(\w+)\s*(?:#\s*\((?:[^()]|\([^()]*\))*\)\s*)?(\w+)\s*\(", re.M)
# `.port (expr)`, expr captured raw so the non-bare ones can be counted.
_CONN = re.compile(r"\.\s*(\w+)\s*\(\s*([^()]*?)\s*\)")
_MODULE = re.compile(r"^\s*module\s+(\w+)", re.M)
_BARE = re.compile(r"^\w+$")


def decomment(text: str) -> str:
    """Blank out comments and strings, preserving every byte offset.

    Not optional. ibex_decoder.sv:44 ends a comment with

        // replicated to ease fan-out)

    and that stray `)` closes the port-list paren counter 22 lines early, which
    surfaced as ibex_decoder having 15 ports instead of 67, and therefore as
    every flat failure in that instance classified "unknown" instead of being
    classified at all. Offsets are preserved because the callers index back
    into the original text.
    """
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            while i < n and not (text[i] == "*" and i + 1 < n
                                 and text[i + 1] == "/"):
                if text[i] != "\n":
                    out[i] = " "
                i += 1
            # An unterminated block comment runs to EOF: i == n here, and
            # out[i] would raise. Blanking to EOF is the right reading of an
            # unclosed comment anyway.
            if i < n:
                out[i] = " "
                if i + 1 < n:
                    out[i + 1] = " "
            i += 2
        elif c == '"':
            out[i] = " "
            i += 1
            while i < n and text[i] != '"':
                if text[i] == "\\":
                    out[i] = " "
                    i += 1
                if i < n and text[i] != "\n":
                    out[i] = " "
                i += 1
            if i < n:
                out[i] = " "
            i += 1
        else:
            i += 1
    return "".join(out)


def _match_paren(text: str, i: int) -> int:
    """Index just past the ')' closing the '(' at i, or len(text)."""
    depth = 0
    while i < len(text):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(text)


_CAND = re.compile(r"^[ \t]+(\w+)[ \t\n]", re.M)


def scan_instantiations(text: str):
    r"""(module, instance, body) for each instantiation, plus what was dropped.

    A scanner, not a regex, because the regex it replaces could only handle
    ONE level of parenthesis nesting in a parameter override. `#(.W($bits(t)))`
    nests two, so the whole instantiation failed to match and vanished from the
    net graph WITHOUT A DIAGNOSTIC, the same silent-drop shape as everything
    else in this file's history. Anything that looks like an instantiation but
    does not parse is now returned and counted.

    An instantiation is recognised structurally, `<name> [#(...)] <name> (`
    whose body holds at least one `.port(` connection, rather than by the
    module being declared under rtl/. That matters: the previous `declared`
    filter dropped every vendored `prim_*` whose source lives elsewhere, and
    the net merge then stopped dead at those boundaries.
    """
    out, dropped = [], []
    for m in _CAND.finditer(text):
        mod = m.group(1)
        if mod in _KEYWORDS:
            continue
        i = m.end() - 1
        while i < len(text) and text[i].isspace():
            i += 1
        if i < len(text) and text[i] == "#":
            j = text.find("(", i)
            if j < 0:
                continue
            i = _match_paren(text, j)
            while i < len(text) and text[i].isspace():
                i += 1
        im = re.match(r"(\w+)\s*\(", text[i:])
        if not im:
            continue
        inst = im.group(1)
        if inst in _KEYWORDS:
            continue
        open_paren = i + im.end() - 1
        end = _match_paren(text, open_paren)
        body = text[open_paren + 1:end - 1]
        if "." not in body:
            continue
        conns = scan_connections(body)
        if not conns:
            # looked like an instantiation, parsed to nothing usable
            if re.search(r"\.\s*\w+\s*\(", body):
                dropped.append((mod, inst))
            continue
        out.append((mod, inst, conns))
    return out, dropped


def scan_connections(body: str):
    r"""(port, expr) for each `.port(expr)`, expr possibly containing parens.

    `\.\s*(\w+)\s*\(\s*([^()]*?)\s*\)` cannot match `.a_i (foo[1] | bar(x))`
    at all, so such a connection was neither merged NOR counted as
    unresolved, it simply did not exist. Now every connection is found and
    the ones whose expression is not a bare identifier are reported.
    """
    out, i = [], 0
    while True:
        m = re.compile(r"\.\s*(\w+)\s*\(").search(body, i)
        if not m:
            return out
        end = _match_paren(body, m.end() - 1)
        out.append((m.group(1), body[m.end():end - 1].strip()))
        i = end


_KEYWORDS = {
    "if", "else", "for", "while", "case", "casez", "casex", "begin", "end",
    "always", "always_ff", "always_comb", "always_latch", "assign", "initial",
    "generate", "endgenerate", "function", "task", "return", "module",
    "endmodule", "input", "output", "inout", "logic", "wire", "reg", "parameter",
    "localparam", "typedef", "import", "package", "assert", "assume", "cover",
    "posedge", "negedge", "unique", "priority", "default", "do", "repeat",
}


class Nets:
    """Union-find over (module, identifier) nodes."""

    def __init__(self) -> None:
        self._p: dict[tuple[str, str], tuple[str, str]] = {}
        self.unresolved: list[tuple[str, str, str]] = []
        self.unparsed: list[tuple[str, str]] = []
        self.collisions: dict[str, set] = {}
        self.inst_module: dict[str, str] = {}

    def find(self, k):
        self._p.setdefault(k, k)
        while self._p[k] != k:
            self._p[k] = self._p[self._p[k]]
            k = self._p[k]
        return k

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._p[ra] = rb

    def same(self, a, b) -> bool:
        return self.find(a) == self.find(b)


class Tree:
    """The RTL as it stood at ONE commit.

    Not optional either, and this one nearly produced a wrong answer rather
    than an obvious error. Each corpus instance is checked out at its own PR's
    `base.sha`, so its partition names describe the hierarchy as it was THEN.
    Mining the net graph from the working tree instead compares those names
    against a different design: `ibex_core.cs_registers_i.pc_set_i` is a real
    partition from #157, and `ibex_cs_registers` has no `pc_set` port at all in
    the checkout sitting on disk today. Classified against the wrong tree it
    came out "elsewhere", which reads as a finding.

    So every instance gets the RTL at its own sha, cached per sha.
    """

    _cache: dict[str, dict[str, str]] = {}

    def __init__(self, repo: Path, sha: str | None, fallback: Path) -> None:
        self.repo, self.sha, self.fallback = repo, sha, fallback
        self.files = self._load()

    def _load(self) -> dict[str, str]:
        if self.sha is None:
            return {f.stem: f.read_text(errors="ignore")
                    for f in sorted(self.fallback.glob("*.sv"))}
        if self.sha in Tree._cache:
            return Tree._cache[self.sha]
        import subprocess
        names = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", self.sha, "rtl/"],
            cwd=self.repo, capture_output=True, text=True)
        out: dict[str, str] = {}
        for name in names.stdout.split():
            if not name.endswith(".sv"):
                continue
            blob = subprocess.run(["git", "show", f"{self.sha}:{name}"],
                                  cwd=self.repo, capture_output=True,
                                  text=True)
            if blob.returncode == 0:
                out[Path(name).stem] = blob.stdout
        Tree._cache[self.sha] = out
        return out

    def items(self):
        return sorted(self.files.items())

    def get(self, stem: str) -> str | None:
        return self.files.get(stem)


def build(rtl) -> Nets:
    n = Nets()
    srcs = [(stem, decomment(raw)) for stem, raw in
            (rtl.items() if isinstance(rtl, Tree)
             else [(f.stem, f.read_text(errors="ignore"))
                   for f in sorted(rtl.glob("*.sv"))])]
    # What counts as an instantiation is decided by the modules the tree
    # actually declares, not by a name prefix. The first version filtered on
    # `ibex_`/`prim_`, which is both a design-specific assumption and a silent
    # one: on any other tree every instantiation is skipped and every signal
    # classifies "elsewhere", a clean-looking answer built on nothing.
    for _stem, text in srcs:
        m = _MODULE.search(text)
        parent = m.group(1) if m else _stem
        insts, dropped = scan_instantiations(text)
        n.unparsed += dropped
        for mod, inst, conns in insts:
            # An instance NAME is not unique across a design. Keyed bare, two
            # different modules instantiated as e.g. `u_buf` in different
            # parents silently overwrite each other and classify() then
            # attributes a signal to the wrong module. Collisions are recorded
            # and the name is poisoned rather than resolved to a coin flip.
            prev = n.inst_module.get(inst)
            if prev is not None and prev != mod:
                n.collisions.setdefault(inst, {prev}).add(mod)
            n.inst_module[inst] = mod
            for port, expr in conns:
                if _BARE.match(expr):
                    n.union((mod, port), (parent, expr))
                else:
                    n.unresolved.append((inst, port, expr))
    return n


def ports_of(rtl, module: str) -> set[str]:
    """Identifiers this module declares as ports, from its header."""
    if isinstance(rtl, Tree):
        raw = rtl.get(module)
        if raw is None:
            return set()
    else:
        f = rtl / f"{module}.sv"
        if not f.is_file():
            return set()
        raw = f.read_text(errors="ignore")
    text = decomment(raw)
    m = _MODULE.search(text)
    if not m:
        return set()
    # Walk to the PORT list. Two things sit between the module name and it,
    # and both broke the first version of this:
    #
    #   module ibex_compressed_decoder import ibex_pkg::*; #( params ) ( ports );
    #                                  \_____ import _____/  \_ params _/
    #
    # Stopping at the first `(` grabs the parameters. Stopping at the first `;`
    #, there to stop a runaway scan, stops on the IMPORT. Both returned an
    # empty port set, which surfaced as every flat failure classified
    # "unknown" rather than as an error.
    i = m.end()

    def _skip_parens(j: int) -> int:
        depth = 0
        while j < len(text):
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
                if depth == 0:
                    return j + 1
            j += 1
        return j

    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if text.startswith("import", i):
            i = text.index(";", i) + 1
            continue
        if text[i] == "#":
            i = _skip_parens(text.index("(", i))
            continue
        break
    if i >= len(text) or text[i] != "(":
        return set()
    start, j = i, _skip_parens(i)
    i = j - 1
    head = text[start:i]
    # `(\w+)\s*(?:,|$)` misses an UNPACKED-ARRAY port, and ibex has them:
    #   ibex_alu.sv:23   input  logic [31:0] imd_val_q_i[2],
    # The trailing `]` sits between the name and the comma, so the port was
    # dropped from the module's port set, and a dropped port silently
    # weakens the same-net test, because a signal really connected to it
    # classifies "elsewhere".
    return set(re.findall(
        r"^\s*(?:input|output|inout)[^,;]*?(\w+)\s*(?:\[[^\]]*\]\s*)*(?:,|$)",
        head, re.M))


def classify(nets: Nets, rtl: Path, path: str, changed: str) -> str:
    """Where a reported failing signal sits relative to the changed module."""
    parts = path.split(".")
    sig = parts[-1]
    inst_path = parts[:-1]
    owner = None
    for p in reversed(inst_path):
        if p in nets.inst_module:
            owner = nets.inst_module[p]
            break
    if owner is None:
        owner = inst_path[0] if inst_path else ""
    if owner == changed:
        return "in-module"
    cps = ports_of(rtl, changed)
    if not cps:
        return "unknown (no port list for the changed module)"
    if any(nets.same((owner, sig), (changed, p)) for p in cps):
        return "same net as a changed-module port"
    return "elsewhere"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+", help="corpus results JSON")
    ap.add_argument("--repo", default="thirdparty/ibex",
                    help="git checkout to read each instance's own sha from")
    ap.add_argument("--rtl", default="thirdparty/ibex/rtl",
                    help="fallback when an instance carries no sha")
    ap.add_argument("--label", action="append", default=None)
    a = ap.parse_args()
    repo, rtl = Path(a.repo), Path(a.rtl)
    labels = a.label or [Path(p).stem for p in a.results]

    for label, res in zip(labels, a.results):
        rows = [r for r in json.loads(Path(res).read_text()).get("rows", [])
                if r.get("status") == "ok"]
        tally: dict[str, int] = {}
        no_sha, unresolved = 0, 0
        collisions: dict = {}
        unparsed: list = []
        repos = set()
        print(f"\n=== {label} ===")
        for r in rows:
            sha = r.get("sha")
            if not sha:
                no_sha += 1
            repos.add(r.get("repo"))
            tree = Tree(repo, sha, rtl)
            nets = build(tree)
            unresolved += len(nets.unresolved)
            for k, v in nets.collisions.items():
                collisions.setdefault(k, set()).update(v)
            unparsed += nets.unparsed
            changed = Path(r.get("file", "")).stem
            for p in r["measure"].get("failed") or []:
                k = classify(nets, tree, p, changed)
                tally[k] = tally.get(k, 0) + 1
                if k != "in-module":
                    print(f"  #{r['number']:<5} {p:<62} {k}")
        total = sum(tally.values()) or 1
        print(f"  --- {total} reported failing partitions over "
              f"{len(rows)} instances ---")
        for k in sorted(tally, key=lambda x: -tally[x]):
            print(f"  {tally[k]:>4} / {total}  ({100*tally[k]/total:5.1f}%)  {k}")
        print(f"  ({unresolved} port connections were not bare identifiers "
              f"and were not merged; a signal reachable only through one of "
              f"those is reported 'elsewhere', so that bucket is an UPPER "
              f"bound)")
        if no_sha:
            print(f"  !! {no_sha} instance(s) carried no sha and were read "
                  f"from the working tree, which is a DIFFERENT design")
        if unparsed:
            print(f"  !! {len(unparsed)} construct(s) looked like an "
                  f"instantiation and did not parse; their connections are "
                  f"MISSING from the net graph: "
                  f"{sorted(set(unparsed))[:6]}")
        if collisions:
            print(f"  !! {len(collisions)} instance NAME(s) map to more than "
                  f"one module in this tree, so any failing partition under "
                  f"one of them may be attributed to the wrong module: "
                  + ", ".join(f"{k}={sorted(v)}"
                              for k, v in sorted(collisions.items())[:4]))
        if len(repos - {None}) > 1 or (repos - {None}) and repo.name not in \
                {str(x).split("__")[-1] for x in repos if x}:
            print(f"  !! results carry repo(s) {sorted(x for x in repos if x)} "
                  f"but every sha was read from --repo {repo}; if those are "
                  f"different checkouts the net graph is the wrong design")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
