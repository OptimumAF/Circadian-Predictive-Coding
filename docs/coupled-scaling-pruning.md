# Raw coupled PC/CPC pruning

This opt-in operation connects the [raw scaling state](coupled-pc-scaling.md) to
structural pruning. It removes one explicit unit from each hidden layer, keeping
the uniform-width configuration supported by that state. It returns a complete
candidate; a separate application gate checks observed train/validation damage.
Neither operation publishes a model.

## Run and test

```powershell
.venv/Scripts/python.exe -m scripts.run_coupled_scaling_pruning --describe
.venv/Scripts/python.exe -m scripts.run_coupled_scaling_pruning
.venv/Scripts/python.exe -m pytest tests/test_coupled_scaling_pruning.py tests/test_coupled_scaling_pruning_gate.py tests/test_coupled_scaling_pruning_cli.py -q -o addopts= -p no:cacheprovider
.venv/Scripts/python.exe -m ruff check src/core/coupled_scaling_pruning.py src/app/coupled_scaling_pruning_gate.py scripts/run_coupled_scaling_pruning.py
.venv/Scripts/python.exe -m ruff format --check src/core/coupled_scaling_pruning.py src/app/coupled_scaling_pruning_gate.py scripts/run_coupled_scaling_pruning.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/core/coupled_scaling_pruning.py src/app/coupled_scaling_pruning_gate.py scripts/run_coupled_scaling_pruning.py
```

The fixed engineering example uses mean-field N3/H2, different removed coordinates
`(0, 1)`, PC/neutral/active CPC, and three scenarios: disconnected accepted,
prediction damage rejected and exhausted budget failed. It prints complete raw
states, factors, chemistry, coordinate records, probe diagnostics and owned fork
continuation. Its nonzero starting chemistry/count is a synthetic valid fixture,
not a measured training history. There is no scientific comparison or final data.

## Responsibilities and boundaries

```text
src/core/coupled_scaling_pruning.py        selectors, raw rebinding, full owned state
src/app/coupled_scaling_pruning_gate.py    probe admission, damage decision, fallback
scripts/run_coupled_scaling_pruning.py     fixed fixtures, continuation and JSON
tests/test_coupled_scaling_pruning*.py     independent core/gate and CLI controls
```

`prune_coupled_scaling_width(state, removed_units)` returns
`CoupledScalingPruneChange(state, lineage, max_effective_relative_error)`.
The format1 lineage owns a copied source config and one native integer coordinate
per hidden layer. Its target config changes width from N to N-1; retained indices
describe each layer's immediate coordinate mapping. Metadata grants no authority.
Source width must be at least 2. Existing 16-layer/1M-parameter caps are checked
before target allocation, and complete raw/native/wake state is revalidated.

Why this: pruning a single layer would produce a nonuniform geometry unsupported
by the current raw config. One explicit coordinate per layer is an atomic
operation within that domain, without publishing intermediate geometries.

## Effective selection and raw factors

For column selectors S_l, keep effective weights A0 S0, S_(l-1)^T A_l S_l and
S_last^T A_out; keep hidden biases b_l S_l and the whole output bias. Then encode
each retained effective weight under the NEW forward factor and each bias under
the NEW bias factor. Direct division avoids forming a nonrepresentable reciprocal.
Copying old raw values would change the effective model when width factors change.

Every reconstructed retained effective entry must satisfy relative error <=
8*float64 epsilon, absolute tolerance 0. Exact zeros must remain zero; nonzero
values cannot become zero or change sign. Overflow or unrepresentable raw/target
state refuses. Signed-zero signs are not preserved. The returned maximum relative
error describes representation only. This is not a bound on pruning damage.

All precisions, scheme, gamma and CPC policy/count remain unchanged. Last-layer
chemistry/gates are selected with the same retained coordinates, without reset.
Native SGD has no momentum or adaptive optimizer state. Future raw/effective rate
factors change with width; neither predictions, full energy nor future trajectories
are generally preserved. Disconnected fixtures are verified separately.

## Observed damage gate

`gate_coupled_scaling_prune` accepts exact `ScalingBatch` train/validation probes,
`CoupledPruneLimits`, a `ToyBudgetSession` and fixed inference settings. At least
one validation batch is required; final_test is refused. IDs and all array storage
must be disjoint. Existing finite float64/binary-target, row and activation bounds
apply. Input state, probes and limits are copied before execution callbacks.

The gate uses separate native copies for source/candidate prediction and inference,
then owns returned scores before subsequent callbacks. It checks all probes before
deciding prediction change, common clipped BCE increase and fixed-step guarded
relaxed-energy increase. Probe energies are not equilibria or global guarantees.
There is no training or chemistry update in the gate.

Invalid preflight raises. Accepted results contain the complete new-width raw state,
lineage and encoding error. Rejection/failure returns an independent full source
copy, completed diagnostics and no accepted lineage/error claim. Candidate and
raw fallback remain isolated from scorer mutations and budget faults.

Construct a new `CoupledScalingLearner` from an accepted state to continue learning.
Existing learners still refuse restores with a different raw config/width. Raw
topology-aware runtime publication/rollback and split/merge remain separate open
work; extending those paths must carry configuration, chemistry and lineage together.
