# chia-livelane

Three [CHIA](https://github.com/ucb-bar/chia) nodes for RTL design loops:

- **an equivalence gate**, so a loop can refute an edit instead of trusting it;
- **open-source synthesis and timing**, so a QoR number needs no commercial licence;
- **controlled evaluator latency**, so a loop can be run at a chosen feedback delay.

Each is a plain `@ChiaFunction`. Call it directly to run in process, or
`.chia_remote(...)` to dispatch it onto a worker whose cluster entry declares its
resource token.

| Node | Module | Token |
|---|---|---|
| `lec_gate` | `chia_livelane.formal.lec_gate` | `eqy: 1` |
| `yosys_sta_qor` | `chia_livelane.vlsi.yosys_sta` | `yosys_sta: 1` |
| `delay_seconds`, `DelayNode` | `chia_livelane.base.delay` | none (`num_cpus=0`) |

## Install

```bash
pip install -e .
pytest                       # live tests skip when the EDA tools are absent
```

The nodes shell out to `yosys`, `eqy`, `sby`, a solver, and `sta`. To build them
from the commits pinned in `configs/pins.env` into a local prefix, with nothing
installed system-wide:

```bash
./scripts/build/toolchain.sh     # -> tools/bin
source env.sh                    # puts tools/bin on PATH
pytest                           # now runs the live tests too
```

## Equivalence gate

Only a positive proof admits an edit. A crash, a timeout, a missing binary, an
unparsable log and an undecided partition are all non-admitting, and `refuted` is
reported separately from `undecided` because only the first is evidence that the
designs differ.

```python
from chia_livelane.formal.lec_gate import lec_gate, auto_jobs

r = lec_gate(
    gold_srcs=["parent.v"],       # the known-good parent
    gate_srcs=["candidate.v"],    # the proposed edit
    top="picorv32",
    read_cmd="read_slang",        # SystemVerilog needs this, not read_verilog
    jobs=auto_jobs(),
)

if r["equivalent"]:
    accept(r["evidence_strength"])          # e.g. "proof[undef-init=zero]"
elif r["refuted"]:
    reject(r["message"], r["counterexample_path"])
else:
    reject(r["verdict"])                    # undecided / error / timeout
```

**Set `jobs`.** `eqy` writes one make target chain per partition and runs them
serially unless `-j` is passed — picorv32 partitions into 590 chains, so on a
many-core host this is the difference between proving them one at a time and
proving them concurrently, for an identical verdict. `auto_jobs()` reads the CPU
share CHIA assigned the task when called inside a worker, and the machine's CPU
count otherwise, so a worker sharing a node cannot oversubscribe it.

**Uninitialised state.** `eqy` reads the gold side with x-propagating semantics
and the gate side with each `x` replaced by an arbitrary value, so a design whose
flip-flops have no initial value is not provably equivalent even to itself. By
default the gate resolves undefined initial values to zero on both sides and says
so: such a proof reports `evidence_strength` of `proof[undef-init=zero]`, never a
bare `proof`. Pass `undef_init=None` for the unqualified claim, accepting that
designs with uninitialised state may not prove at all.

### A second opinion

`eqy` does partition-based gate-level checking, which requires the sequential
boundaries to be unchanged — so a retimed pipeline or a merged register is an
edit it structurally cannot prove. `kepler-formal` additionally does RTL-level
sequential equivalence checking, which covers that class. Build it with
`./scripts/build/kepler.sh`, then compare the two:

```python
from chia_livelane.formal.lec_gate import KeplerBackend, cross_check

verdict = cross_check(primary, KeplerBackend().check(gold, gate, top))
verdict["agree"]      # False is a finding, not noise: it is reported, never averaged
```

## Synthesis and timing

```python
from chia_livelane.vlsi.yosys_sta import yosys_sta_qor

r = yosys_sta_qor(
    sources=["picorv32.v"], top="picorv32",
    liberty="sky130_fd_sc_hd__tt_025C_1v80.lib",
    clock_port="clk", clock_period_ns=10.0,
    read_cmd="read_slang", script="baseline-flat",
)
r["cells"], r["area_um2"], r["max_delay_ns"], r["slack_ns"]
```

`liberty` is required and never defaulted: two area or delay numbers are
comparable only if they came from the same Liberty. Every report records the
recipe, the front-end command and the clock, and says whether the run was
constrained — unconstrained, `report_checks` returns the longest combinational
I/O path, which is a different quantity, and the node says so rather than quietly
returning the smaller number. Two recipes ship: `baseline-flat` (full flatten)
and `tuned-hier` (preserves hierarchy, `abc -fast`).

## Controlled latency

```python
from chia_livelane.base.delay import DelayNode, delay_seconds

delay_seconds(120.0)                    # as a step between two graph nodes

node = DelayNode(seconds=120.0)         # or around one evaluator call,
with node.around(lambda: True):         # with a record of what it did
    report = evaluate(candidate)
```

`observed = real + seconds`, so a loop can be re-run at a chosen feedback latency
with everything else held fixed.

## On a cluster

```yaml
available_node_types:
  eqy_worker:
    resources: {eqy: 1}
    docker: {image: chia-eqy:latest}
  synth_worker:
    resources: {yosys_sta: 2}
    docker: {image: chia-yosys-sta:latest}
```

```python
from chia_livelane.formal.lec_gate import lec_gate
ref = lec_gate.chia_remote(gold_srcs, gate_srcs, "picorv32", read_cmd="read_slang")
```

Images are in `dockerfiles/`. Both run a real proof and a real refutation at build
time and fail the build if either comes out wrong: a worker that cannot run its
tool returns `ERROR`, the gate fails closed, and a broken image would otherwise
look exactly like a correctly cautious gate.

## Layout

```
src/chia_livelane/base/    delay node
src/chia_livelane/formal/  equivalence gate
src/chia_livelane/vlsi/    Yosys + ABC + OpenSTA QoR, instrumented tool runner
dockerfiles/               worker images and their build workflows
scripts/build/             from-source toolchain build
```

BSD-3-Clause.
