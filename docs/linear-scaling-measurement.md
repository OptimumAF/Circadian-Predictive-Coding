# Fixed linear scaling measurements

Inspect the fixed protocol without training:

```powershell
python -m scripts.run_linear_scaling_pilot --describe
```

Omit `--describe` to run the complete local protocol and print JSON. A failed or
unreached case remains in the output; incomplete execution exits 2. The first
session invocation is separately limited to one grid in the prospective
`artifacts/runs/linear-scaling-measurement-20261014/PROTOCOL.md`.

## Scope and controls

This measures the [linear scaling implementation](linear-pc-scaling.md): identity
activation, scalar output, unit precision, no bias or skip. It compares explicit
legacy Gaussian, Gaussian fan-in and mean-field parameterizations. It implements
no residual depth parameterization and makes no claim about nonlinear PC/CPC.

The ordered grid is seeds `(23,41,59)`, widths `(2,8,32)`, hidden layers `(1,2,4)`,
then the three schemes: 81 cases, six raw SGD updates each, rate `.01`, gamma0 `1`.
Same shape/seed uses identical underlying standard-normal draws. Each result
contains exact raw/effective initial weights and initialization/rate factors.

Training uses 16 uniform `[-1,1]` two-dimensional rows (RNG 7311); final scoring
uses 64 independent rows (RNG 9111). Both use `y=x0-.5*x1`, with finite real
regression targets. Both datasets are fixed across all cases. No old attribution
pilot data is reused and no result changes subsequent cases.

Why this: fixed geometry, shared draws and fixed data expose initialization and
update scaling without introducing data, seed or rate selection. Six steps are
an exploratory measurement window, not a convergence study.

## Isolation and failure behavior

`ScalingBatch` preflight validates exact float64 shapes, roles, native IDs,
global ID uniqueness and disjoint cross-role storage. State and batches are
copied before external budget or clock calls. All training and diagnostics for
one case finish before scoring its detached checkpoints 0 through 6. Prediction
receives weights and copied features only; heldout targets enter only the metric
`mean((prediction-target)^2)/2`.

Numeric, budget or clock failures retain attempted/committed updates and partial
training checkpoints. They discard all heldout scores, including scores computed
before a final budget failure. Preflight mistakes raise useful errors. The grid
preserves failed/unreached rows; it does not retry or replace them.

## Measurements and interpretation

Every training checkpoint records equilibrium energy, train half-MSE, denominator
rescaling, raw/effective weight and gradient Frobenius norms, feedforward layer
RMS (all learned outputs) and equilibrium hidden-state RMS. Scaled norms avoid
premature square overflow. Arithmetic beyond representable float64 is refused.

Each width/depth/scheme summary retains all three ordered seeds and initial/final
heldout half-MSE. Complete vectors have mean, range and nominal marginal 95% t
intervals (df 2, critical 4.303), using the existing seed statistics policy. Missing
seeds leave the complete mean/interval unavailable; observed-only descriptive
fields are explicitly separate. Zero variance has no estimated interval. These
are conditional on fixed data, with untested normal seed effects and no multiple
comparison coverage, significance or selected-winner claim.

Timing separates initialization, kernel steps, training diagnostics and offline
checkpoint scoring. Step time includes candidate validation. A complete case
performs 12 step equilibrium/gradient evaluations plus seven diagnostic ones,
seven heldout forward passes and 96 training row exposures. Counts for failed
internal work or discarded scoring are unknown. This is not optimized runtime,
FLOPs, equal compute or an isolated speed benchmark.

The script sets OMP/MKL/OpenBLAS threads to one before imports. Fixed ceilings
are 486 planned attempts (original ceiling 512), 60 execution seconds after
imports/before JSON encoding, sampled absolute process RSS 256 MiB, output 32 MiB,
CPU only, no download/replay/environment interaction. Checks occur at safe
boundaries; they cannot preempt a native operation or prove exhaustive peak RSS.

## Modules and verification

```text
src/app/linear_scaling_measurement.py     copied case inputs, train diagnostics, deferred scoring
src/app/linear_scaling_summary.py         complete ordered seed summaries
src/adapters/linear_scaling_pilot.py      fixed data/grid, factory, RSS and wall limits
scripts/run_linear_scaling_pilot.py       fixed CLI and JSON output
tests/test_linear_scaling_measurement.py independent spies and six-update native control
tests/test_linear_scaling_pilot.py        data, summary, failures and describe CLI controls
```

```powershell
python -m pytest -q tests/test_linear_scaling_measurement.py tests/test_linear_scaling_pilot.py
python -m ruff check src/app/linear_scaling_measurement.py src/app/linear_scaling_summary.py src/adapters/linear_scaling_pilot.py scripts/run_linear_scaling_pilot.py tests/test_linear_scaling_measurement.py tests/test_linear_scaling_pilot.py
python -m ruff format --check src/app/linear_scaling_measurement.py src/app/linear_scaling_summary.py src/adapters/linear_scaling_pilot.py scripts/run_linear_scaling_pilot.py tests/test_linear_scaling_measurement.py tests/test_linear_scaling_pilot.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL src/app/linear_scaling_measurement.py src/app/linear_scaling_summary.py src/adapters/linear_scaling_pilot.py scripts/run_linear_scaling_pilot.py tests/test_linear_scaling_measurement.py tests/test_linear_scaling_pilot.py
```

Extend via a separately declared protocol and exact mathematical controls;
preserve existing outcomes. Residual depth factors need their own implementation
and correctness tests before any residual scaling experiment.

The verified [residual implementation](residual-linear-pc.md) now uses the shared
native measurement ports. Its separate [measurement protocol](residual-scaling-measurement.md)
and [report](reports/residual-scaling-measurement-20261014.md) preserve this MLP
study's outputs and keep both topology scopes explicit.
