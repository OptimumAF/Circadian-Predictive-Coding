# Isometric width changes for linear PC

Run the bounded engineering example:

```powershell
python -m scripts.run_linear_width_structure --describe
python -m scripts.run_linear_width_structure
```

The first command performs no SGD. The second prints six controls (MLP/residual,
legacy Gaussian/fan-in/mean-field), exact source/expanded/merged tensors and
equilibrium diagnostics. It splits common width2 into3, merges back, demonstrates
an incompatible merge refusal, and runs four actual SGD steps to show continuation
at expanded/original width. Seed23, input2, hiddenlayers2, fixed two-row training
batch and rate.01. No heldout data, threshold tuning or scientific experiment.

## API and boundaries

`core/linear_width_structure.py` takes existing `LinearScalingState` or
`LinearResidualState`. `split_linear_width(state, index)` returns a detached
`LinearWidthChange(state, lineage)`. `merge_linear_width(state, lineage)` returns
a detached projected state or raises before changing any borrowed source.
Lineage format1 binds family, original config and split coordinate. It records
coordinate ancestry only; it grants no runtime/replay/consent authority.

The public index is a native integer in `[0,N)`, selecting a coordinate in every
hidden layer. All hidden layers retain the configured common width. New configs
retain scheme/input/depth/gamma, changing width only. Exact family/config/version/
shape/finite/allocation guards are rechecked. No new dependencies or persistent
environment settings; the CLI sets OMP/MKL/OpenBLAS threads1 before NumPy imports.

Why this: raw weights, forward scales and SGD factors all depend on configuration.
Duplicating a raw tensor without rebinding width would implement another model.
Existing learner/default/state modules are unchanged; this is opt-in pure core
functionality, with no file IO, optimizer, chemistry or structural sleep scheduler.

## Split map and equilibrium

Let `S` have shape `(N,N+1)`: every old coordinate keeps its column, except the
selected coordinate is divided equally between its old column and new last
column using `1/sqrt(2)`. Then `S Sᵀ=I`. Complete effective MLP matrices map as:

```text
input:    A0' = A0 S
middle:   Al' = Sᵀ Al S
output:   Ao' = Sᵀ Ao
```

Residual middle branches `B=Al-I` map as `Al'=I_new+Sᵀ B S`. The orthogonal
complement retains identity. For embedded hidden activity `z'=zS`, predictions
and scalar-output downstream squared norms are preserved. Thus the full clamped
linear equilibrium denominator, energy, hidden latents/residuals and output
residual preserve their mathematical values. Tests compare these at declared
numerical tolerances; bitwise equality is not promised.

Raw branch tensors receive the same coordinate map and old/new forward-factor
conversion. Axis operations take O(raw entries) storage; no dense `S` or identity
is allocated by the structural module. Binary-scaled arithmetic avoids avoidable
intermediate overflow/underflow. Existing raw allocation caps apply before new
state construction. Unrepresentable candidates are refused atomically.
Structural validation checks raw/effective tensors without a data batch. A
particular batch's equilibrium/gradient can still overflow; the existing
equilibrium/SGD entry points validate that computation separately.

## Merge and limits

Merge projects through the recorded inverse, then re-expands and checks **scaled
branch coordinates**, with fixed `rtol=1e-10`, `atol=1e-12`. An off-image candidate
outside these bounds raises; no arbitrary neuron merge is implemented. Residual
compatibility checks the trainable branch, retaining complementary identity.
Exact split images round-trip mathematically; numerical round trips are tested.

**A tolerance-compatible projection has no global function-error guarantee.**
Small discarded components can be amplified by large downstream weights. An
accepted perturbation counterexample is a required test. Measured preservation
on the example does not establish bounded damage for arbitrary finite states.
Pruning, non-preserving damage validation/transactional recovery and nonlinear/
CPC optimizer/trace integration remain open under parent R1.6.

Mean-field effective rates change with width (first proportional to N, middle
unchanged at fixed depth, output proportional to1/N). The operation preserves the
current function/objective; future SGD trajectories need not match. New-width
initialization distributions also are not sampled anew by a split. Coordinate
lineage/functional equivalence do not imply parameter identity or new consent.

## Checks and extension

```powershell
python -m pytest tests/test_linear_width_structure.py tests/test_linear_width_structure_cli.py -q -o addopts= -p no:cacheprovider
python -m ruff check src/core/linear_width_structure.py scripts/run_linear_width_structure.py tests/test_linear_width_structure.py tests/test_linear_width_structure_cli.py
python -m ruff format --check src/core/linear_width_structure.py scripts/run_linear_width_structure.py tests/test_linear_width_structure.py tests/test_linear_width_structure_cli.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/core/linear_width_structure.py scripts/run_linear_width_structure.py tests/test_linear_width_structure.py tests/test_linear_width_structure_cli.py
```

Repeat scoped mypy with `--platform linux`. Native controls require the recorded
local engineering budget. Add non-preserving operations through a separate
candidate/probe/recovery boundary with explicit accepted damage and rollback,
keeping evaluation roles separate. Do not silently generalize linear isometries
to nonlinear activations or CPC chemistry.

R1.6a accepted 2026-10-15:73distinctfocusedcases, scoped Ruff/format/Windows+Linux
mypy and independent math/state/actual-outcome review pass. The saved six-control
CLI has maximum prediction change1.12e-16 on its fixed example and four real SGD
continuations. Evidence and commands:
`artifacts/runs/linear-width-structure-20261015/{RESULTS.md,example.json,worker-tests.xml,supplement-tests.xml,cli-tests.xml,extreme-gamma-tests.xml,statics.json}`.
No scientific study or general merge damage bound follows. Parent R1.6 remains
open for pruning, non-preserving recovery and nonlinear/CPC state integration.
