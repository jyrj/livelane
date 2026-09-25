# LiveLane

[![CI](https://github.com/jyrj/livelane/actions/workflows/ci.yml/badge.svg)](https://github.com/jyrj/livelane/actions/workflows/ci.yml)

**Formally verifying every edit of an RTL agent on a whole RISC-V core.**

A³ CHIA Hackathon · Track: *Demonstration of agentic formal verification on a
large RTL IP block* · Jayaraj Jayakumar · `jj8@ucsc.edu` · UC Santa Cruz

**[Paper (4 pages, PDF)](paper/livelane.pdf)**. Every number below comes from a
file in [`measurements/`](measurements/).

---

## Motivation

An agentic RTL optimisation loop scores its own edits. If the score comes from
simulation and synthesis quality of results (QoR), **the loop accepts edits that
change what the design computes**. This happens systematically, not
occasionally, because the objective rewards removed logic.

An example from a running loop (`gemini-3.1-pro` on `picorv32`, iteration 9):

```
propose   factor late signal to the top of the logic tree
simulate  PASS (1.31s)
score     10.31 ns vs 11.97 ns      -- 13.8% FASTER
prove     REFUTED (1.48s)           -- picorv32.mem_done
```

The edit passed simulation and scored 13.8% faster, so every inexpensive check
favoured accepting it. The equivalence check (a partition-based equivalence
checker, eqy) rejected it.

Only formal equivalence checking rejects such an edit.
CHIA's RTL-editing loops do not include one. LiveLane adds it and runs it on
**every iteration, on the whole `ibex_core`**, in 13.9 s.

## Contributions

LiveLane moves formal verification from sign-off into the loop, on the critical
path of every iteration, on a whole RISC-V core, using only open-source tools:

- **`LecGateNode`**: the equivalence check. An edit is admitted only if
  equivalence is proven. It checks 11,117 partitions of `ibex_core` per edit, in
  13.9 s per iteration.
- **`second_stage`**: a sequential equivalence check (PDR) that resolves each
  rejection of the equivalence check into a proof or a counterexample from
  reset, and reports the result to the agent. In the loop, 81 of 82 rejected
  edits were proven correct and the remaining one, a real bug, was refuted; all
  78 accepted edits were re-proven from reset, whatever the power-up state.
- **`DelayNode`**: makes evaluator latency an experimental variable, so any loop
  can measure how fast its evaluator needs to be.

## Results

**1. Synthesis scores reward non-equivalent rewrites, measured as a rate on a whole core.**
We applied 130 optimisation-style rewrites to `ibex_core`, of two kinds:

| | |
|---|---|
| equivalence-preserving rewrites of instantiated logic (commute `&&` `\|\|` `&` `==`) | **48 / 48 proven** |
| simplifying rewrites (drop a term, remove an inversion, fold a guard, swap a mux) | 75 |
| &nbsp;&nbsp;rejected by the equivalence check | **50** |
| &nbsp;&nbsp;&nbsp;&nbsp;confirmed by a counterexample trace (instance scope) | **49 / 50** |
| &nbsp;&nbsp;&nbsp;&nbsp;**scored better on QoR than the original** | **31 / 50 (62%)** |
| &nbsp;&nbsp;&nbsp;&nbsp;failing only inside the rewritten module | **50 / 50** (46 matched automatically) |

The largest timing gain comes from a rejected rewrite: replacing
`if (illegal_c_insn_i)` with `if (1'b1)` in `ibex_decoder.sv` shortens the
critical path by **19%**. A loop that optimises QoR would keep it. Of the 49
rejections confirmed by a counterexample, 30 (61%) still score better in
synthesis. Every counterexample starts from reset and is found at instance
scope: it shows that the edited module computes a different function, not that
the core exposes the difference. For example, three LSU rewrites that drop
`|| pmp_err_q` are correct in this core, whose PMP is disabled. On the
equivalence-preserving rewrites, the sequential check proves **all 48** from any
shared initial state and produces **zero** counterexamples.

```bash
python scripts/chia/rewrites.py            # generate, prove, synthesise -> var/final/rewrites.json
python scripts/chia/resettle.py --results measurements/data/rewrites.json --redo                 # rejections
python scripts/chia/resettle.py --results measurements/data/rewrites.json --rows controls --redo
```

`rewrites.py` is seeded, and its `var/rewrites/m*/` candidates are what
`resettle.py` re-proves. `resettle.py` updates the results file in place.

**2. The equivalence check rejects correct edits; a sequential check resolves the rejections.**
Five agent runs (`gemini-3.1-pro` ×3, `gemini-2.5-flash` ×2) optimised the
whole `ibex_core`: 85 iterations, $13.01 of model usage. Every rejected edit was
re-checked with an unbounded sequential equivalence proof (PDR) of the edited
instance, at the parameters the core gives it, with an explicit initial state:
both copies start in the same arbitrary state, and reset is asserted in the
first cycle.

| | flash | pro |
|---|---|---|
| edits that reached the equivalence check | 27 | 42 |
| rejected | 6 | 15 |
| &nbsp;&nbsp;real bugs (counterexample trace) | **4** | **0** |
| &nbsp;&nbsp;**correct edits, falsely rejected** | 2 | **15** |

**81% of the agent's rejected edits were correct, including all of the stronger
model's.** Its fetch-FIFO rewrites substitute the definition of `out_valid_o`
into `pop_fifo`; the partition-based checker places a cut point at
`out_valid_o`, treats it as a free input, and rejects `fifo_i.pop_fifo`.
**16 of the 17 are equivalent from any shared initial state, with or without
reset.** The sequential check decides all 71 rejections (agent runs and rewrite
campaign) in seconds, and produces **0** counterexamples on the
equivalence-preserving rewrites. Parameters matter: checked standalone, one
`ibex_counter` edit appears proven although it breaks the core's `minstret`
counter. The initial state matters too: starting both copies from the all-zero
state (the usual shortcut) accepts an edit that relies on un-reset flops
powering up to zero, and refutes a one-hot state machine whose all-zero state
is illegal.

**Edits to a shared package.** Some of the stronger model's edits change no
module: 12 edits, all in one run, re-encode enums in `ibex_pkg` (PC select,
exception-PC select, immediate-B select, the controller's state machine). The
equivalence check accepted none of them, and a package has no instance to
check. `second_stage` checks each edit at the scope its changed types reach:
one module at its instances, or the top level for a type that crosses module
ports. **All 12 are correct**; 10 are proven at the scope of the whole core
(median 3 s, at most 17 s) using register correspondence: every register is
matched, each matched pair is replaced by one shared free variable, and a
single cycle is proven, so unchanged logic merges away. The same
register-correspondence proof fails, as it should, on an edit that moves a CSR
address. The one-hot controller is proven from reset by PDR. Three of the 12
were faster; the best, **6.2% faster and the best edit of its run**, would take
that run from 4.1% to 10.1%.

**In the loop.** Across 9 runs of `gemini-3.1-pro` with the equivalence check
alone, **all 35 rejections were correct edits**, costing 26% of its iterations.
With the initial version of the sequential check (all-zero initial state,
instance scope; 9 runs), the agent was told 20 times that a rejected edit was
correct; none of those rewrites was faster, so the median barely changes (6.5%
vs 6.2%). With the final version (from reset, with package edits checked at the
scope their changed types reach; 18 runs), **81 of 82 rejections were proven
correct during the run, and the remaining one, a real bug, was refuted** with a
counterexample from reset. With `gemini-3.8-flash` and `gemini-2.5-pro` (3 runs
each), the same check decided all 31 of their rejected edits: 29 proven correct
and 2 real bugs refuted. **Every edit the loop accepted in any run (78 of 78)
is re-proven from reset, whatever the power-up state, and 76 of them also from
any shared initial state without reset.**

Every run improved the core's critical path (4.8 to 11.2%), with every accepted
edit proven, at **13.9 s** of equivalence checking per iteration (13% of the
median 126 s iteration).

```bash
python scripts/chia/agent_analysis.py      # parents, rates -> var/final/agent.json
python scripts/chia/resettle.py --results measurements/data/agent.json --redo   # sequential check
```

Re-running `agent_analysis.py` (and `gate_ab.py`, which calls it) requires the
raw run directories (`var/agent*`), which are not included. The sequential-check
verdicts can be recomputed from the included JSON and candidate files
(`measurements/data/agent/`) with `--redo`.

**3. The evaluator latency an agent tolerates depends on the model.**
We inject evaluator delay as a controlled variable, with the tool stack
otherwise identical:

| delay | iterations | improvement |
|---|---|---|
| 0 s | 10.7 | 18.2% |
| 30 s | 13.7 | 22.6% |
| 120 s | 8.0 | **28.0%** |
| **600 s** | **2.0** | **0.0%** |

(`gemini-3.1-pro`, `picorv32`, fixed 1800 s budget.) The 0, 30 and 120 s
conditions show no significant difference at 2 to 3 seeds; at 600 s the agent
finds no improvement. Given the same 8 iterations, the 600 s condition still
reaches 18.5%, with no significant difference across the four delays
(p = 0.11): latency costs the agent iterations, not per-iteration quality. A
faster model (`gemini-2.5-flash`, 33 s turns) degrades steadily from 30 s.
LiveSim targets 2 s for human designers; an agent needs an evaluator faster
than its own turn.

**4. At core scale, the cost of formal checking is partitioning, not proving.**

| phase | cold | cached |
|---|---|---|
| `eqy -m` (elaborate + partition) | 100 s | **148 s** |
| prove | 228 s | 25 s |
| **wall-clock time, 12 instances** | **388 s** | **227 s** |

Proving is 11% of the cached run; `eqy -m` is 65.5%, of which partitioning is
92%. **~60% of the equivalence check's run time is a single-threaded C++
pass**, so a proof cache, which removes only solver work, is bounded at
388/(227−25) = 1.93×; we measure 388/227 = 1.71×.

**5. Two tool defects, not design limits, prevented the partitioner from completing on large blocks.**
`eqy_partition` emits an O(n²) debug matrix (one `log()` per partition×fragment
cell) and hits a 255-byte filename limit. CVA6's whole core produced **zero**
partitions in 17 min while writing a 2.2 GB log. With both defects fixed:

| design | partitions |
|---|---|
| picorv32 | 590 |
| `ibex_core` (whole core) | 5,261 to 15,247 |
| XiangShan DivUnit | 16,065 |
| **CVA6 (whole core)** | **280,764** in 790 s |

**6. The equivalence check detects real bugs, and keeping the hierarchy localises them.**
On 12 merged `lowRISC/ibex` bug-fix pull requests, each checked at its own base
commit:

| | flattened | `--keep-hierarchy` |
|---|---|---|
| verdict agreement | **12 / 12** | |
| **failing partitions in the changed module** | 47.2% | **100%** |
| distinct modules implicated (mean) | 2.18 | **1.00** |
| partitions | 9,110 | **6,022** |
| iteration time, including the check | 20.6 s | **15.2 s** |

11 of 12 are refuted, with every failing partition localised to the changed
module; the twelfth (#1780) fixes PMP logic that this configuration does not
instantiate.

## CHIA nodes

Each node declares its own resources and can be used without the others.

| node | resource | role |
|---|---|---|
| `LecGateNode` | `eqy` | equivalence check: prove or refute an edit |
| `second_stage` | `eqy` | sequential check of a rejected edit: PDR of the edited instance from reset (`init="reset"`, the default) |
| `YosysStaNode` | `yosys_sta` | synthesise and score |
| `VerilatorSimNode` | `verilator_run` | simulate |
| `DelayNode` | none | add a controlled delay to any evaluator |

`LecGateNode` is the most directly reusable node: any CHIA loop that edits RTL
can use it to accept or reject edits. Its verdicts are `proven`, `refuted`,
`undecided`, `error`, `timeout` and `skipped`, and **only `proven` admits an
edit**, so a timeout or a solver crash is never treated as equivalence.

`DelayNode` lets any loop measure, on its own toolchain, how much evaluator
latency its agent tolerates before it stops finding improvements.

## Continuous integration

Every push runs two jobs ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)):

- **CHIA nodes and harness**: a clean install from PyPI, including CHIA
  (`chialoops`), and the full test suite; tests that need EDA tools are skipped.
- **Equivalence and sequential checks, whole `ibex_core`**: the unmodified OSS
  CAD Suite (pinned release) and `ibex` at its pinned commit. The equivalence
  check proves and refutes real edits, the sequential check decides rejected
  edits from reset, and the whole-core proofs of the enum re-encodings run on
  every push.

## Reproduce

```bash
scripts/setup/10_clone_thirdparty.sh      # pinned in configs/pins.env
scripts/build/toolchain.sh                # yosys, eqy (+ patches/), verilator, OpenSTA
scripts/build/kepler.sh                   # optional: kepler-formal; its tests skip without it
scripts/build/livehd.sh                   # optional: LiveHD (lhd), an alternative synthesis flow
python3.12 -m venv .venv && source .venv/bin/activate   # any Python 3.10-3.13
pip install -e ".[test,analysis,agent]"   # agent: the Gemini client (Vertex AI)
scripts/setup/40_pdk.sh                   # sky130 Liberty via ciel, sha256-checked
source env.sh
python -m pytest                          # 448 tests, real tools on PATH
```

The agent runs need a Vertex AI project: `export GOOGLE_CLOUD_PROJECT=<id>`
(or put it in `configs/gcp.local.env`, which is gitignored); Gemini 3.x models
are available only in the `global` location, which `env.sh` sets by default.

Then run any of the following:

```bash
# the loop would accept an incorrect design; only the equivalence check rejects it
python scripts/chia/cascade.py --design picorv32 --inject-bge-bug

# the ibex bug-fix corpus, hierarchical
python scripts/chia/hwebench.py --repo lowRISC__ibex --ladder sat-first \
       --keep-hierarchy --out var/final/ibex-hier.json

# flat vs hierarchical: require matching verdicts, report the other metrics
python scripts/chia/compare_frontend.py \
       measurements/data/ibex-satfirst2.json measurements/data/ibex-hier2.json \
       --name-a flat --name-b hier
```

## Measurement methodology

- Every speedup states its denominator and phase; prove-phase and end-to-end
  times are never combined.
- The harness discards its own timings when machine load changes during a run,
  and reports how many instances it dropped.
- Percentages are reported with raw counts; no-op edits are rejected rather
  than measured.
- Proof-cache keys encode the full query (strategy definition, strategy order,
  tool versions), so a shallower proof is never returned for a deeper query.
- Every corpus instance is also proven against itself, and verdicts can be
  compared per partition against a run without the cache.

## Repository layout

| path | contents |
|---|---|
| `src/chia_livelane/` | the five CHIA nodes, with their tests |
| `src/livelane/` | the latency-study harness, agent clients, fixed QoR scoring flow |
| `scripts/chia/` | the ibex loop, rewrite campaign, sequential re-check, A/B comparison, bug-fix corpus harness |
| `measurements/data/` | every result cited in the paper, with the agents' candidate files |
| `configs/`, `scripts/setup/`, `scripts/build/` | pinned sources, PDK and toolchain |
| `patches/` | the `eqy_partition` fixes, as a patch file |
| `designs/` | `picorv32`, the design used in the latency study |
| `tests/` | unit tests for the harness |
| `paper/` | the 4-page paper, its LaTeX source and the script that generates its numbers |

## Acknowledgement of AI assistance

Claude, Gemini was used for writing and refactoring code, running and analysing the measurement campaign, and drafting the paper. All experimental design decisions, all measurements, and the final content are by human.
