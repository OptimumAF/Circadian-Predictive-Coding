# Fixed residual scaling measurements

Inspect the fixed protocol without training:

```powershell
python -m scripts.run_residual_scaling_pilot --describe
```

Omit `--describe` to execute the fixed local grid and print JSON. Incomplete
execution exits 2, retaining every failed/unreached row. This session permits
one scientific invocation after correctness and review, under
`artifacts/runs/residual-scaling-measurement-20261014/PROTOCOL.md`.

## Composition and controls

The [residual core](residual-linear-pc.md) now runs through the same application
measurement engine as the [MLP study](linear-scaling-measurement.md). Typed
`LinearMeasurementPorts` supply native copied state, complete effective matrices
and one raw SGD step. Public wrappers choose the exact native family; arbitrary
model operations are not supplied through user input. The MLP public API and
its fixed protocol remain compatible. Shared outer case/grid functions own
resource checks, complete row retention, nominal work and seed summaries.

Why this: one loop keeps role validation, snapshots, diagnostics, timing and
failure-score suppression consistent across the two topologies. The native
residual port supplies fixed identities in every middle effective matrix and
advances only residual branch parameters.

The ordered grid is seeds `(23,41,59)`, widths `(2,8,32)`, hidden layers `(1,2,4)`
and schemes `legacy_gaussian`, `fan_in`, `mean_field`: 81 cases, six updates each.
All three controls share fixed identity skips only in middle layers. Input width
is 2, gamma0 1, rate .01, scalar linear output, unit precision and no bias. `L`
counts all learned matrices (`H+1`); the mean-field middle branch is scaled by
`1/sqrt(N*L)`, raw SGD by `gamma0²*N`. Exact initial raw/effective arrays, factors,
configuration, depth and zero-based skip positions are included in JSON.

Data/rate/grid are held at the previous prospective values: uniform `[-1,1]`
train 16×2 from RNG 7311, final 64×2 from RNG 9111, `y=x0-.5*x1`. Fresh owned
buffers are constructed per case, with disjoint train/final IDs and storage.
No previous final score chooses settings or enters training/stopping. This
topology is derived from the inspected source, not selected for its outcomes.

## Isolation, metrics and costs

Both wrappers validate the exact native state family, roles, unique printable
IDs, float64 shapes, disjoint storage, controls, budget and clock. They clone
configuration, raw matrices and batches before external clock/budget callbacks.
All six training updates and diagnostics finish before any checkpoint scoring
for that case. Heldout prediction takes complete effective matrices and a fresh
feature copy; targets enter only `mean((prediction-target)^2)/2`.

Every checkpoint 0–6 records train energy/half-MSE/rescaling, raw/effective weight
and gradient Frobenius norms, feedforward layer RMS (all learned outputs) and
equilibrium hidden RMS. The fixed identity appears in effective norms and suffix
products; only the branch scale maps effective gradients to raw gradients.

Numeric, budget or clock failures retain partial training records, attempts and
commits, but discard every heldout score, including a failure after the last
prediction. All 81 rows remain. There is no retry or seed replacement. Existing
full-seed summaries retain all three seeds for each of 27 groups, with initial
and final mean/range/nominal marginal 95% t intervals (df 2, critical 4.303).
Incomplete vectors have no complete mean/interval; zero variance has no estimated
interval. Observed-only descriptive fields are separate. These intervals are
conditional on fixed data with untested normal seed effects and no simultaneous
coverage, significance or universal-winner claim.

Each complete case has six steps (12 equilibrium/gradient evaluations), seven
extra diagnostics, seven heldout forwards and 96 training row exposures. The
full grid has 1539 evaluations, 567 heldout forwards and 7776 row exposures.
Failed internal/discarded work is unknown. Timings separate initialization,
kernel/candidate validation, diagnostics, offline scoring, application elapsed,
full case elapsed and global execution. These are descriptive costs, not an
optimized or equal-compute benchmark.

Original ceilings remain 512 updates, 60 execution seconds after imports/before
JSON encoding, 256 MiB sampled absolute RSS, CPU threads one, output 32 MiB, and
no GPU/download/replay/environment interaction. Safe-boundary checks cannot
preempt native calls or establish exhaustive peak RSS. The script sets
OMP/MKL/OpenBLAS threads to one before NumPy imports. No dependency is added.

## Modules and checks

```text
src/app/linear_scaling_measurement.py   typed ports and shared isolation engine
src/adapters/linear_scaling_pilot.py    shared fixed-grid resource/report boundary
src/adapters/residual_scaling_pilot.py  residual composition and declared protocol
scripts/run_residual_scaling_pilot.py   fixed CLI / JSON / incomplete exit code
tests/test_residual_scaling_measurement.py independent native routing/isolation spies
tests/test_residual_scaling_pilot.py    fixed factory/grid/failures/CLI controls
```

```powershell
python -m pytest -q tests/test_linear_scaling_measurement.py tests/test_linear_scaling_pilot.py tests/test_residual_scaling_measurement.py tests/test_residual_scaling_pilot.py
python -m ruff check src/app/linear_scaling_measurement.py src/adapters/linear_scaling_pilot.py src/adapters/residual_scaling_pilot.py scripts/run_residual_scaling_pilot.py tests/test_residual_scaling_measurement.py tests/test_residual_scaling_pilot.py
python -m ruff format --check src/app/linear_scaling_measurement.py src/adapters/linear_scaling_pilot.py src/adapters/residual_scaling_pilot.py scripts/run_residual_scaling_pilot.py tests/test_residual_scaling_measurement.py tests/test_residual_scaling_pilot.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL src/app/linear_scaling_measurement.py src/adapters/linear_scaling_pilot.py src/adapters/residual_scaling_pilot.py scripts/run_residual_scaling_pilot.py tests/test_residual_scaling_measurement.py tests/test_residual_scaling_pilot.py
```

New model families need their own coherent objective, derivative/state controls
and native ports before measurement. Do not reinterpret linear PC as binary CPC
or infer asymptotic stability, convergence or comparative benefit from a tiny
six-step fixed-data grid. Preserve prior results and declare new studies before
using new data, rates, seeds or metrics.
