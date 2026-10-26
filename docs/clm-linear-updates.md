# Sequential wide categorical correction

`update_clm_linear_control(prior, batch, targets, report)` advances an existing
detached linear control on one arrived batch. The prior remains unchanged. This
is an explicit warm update; `fit_clm_linear_control` remains the original cold
fit. Neither operation trains the released CLM head or the frozen encoder.

Why this: the cold marker diagnostic fitted new markers but answered only2/4
old-task checks. It did not measure retention of an acquired model. A warm
correction provides the missing numerical step for a subsequent sequential
learning study without loading the encoder again.

For augmented features `Z=[X,1]`, prior coefficients `W`, and one-hot targets `Y`:

```text
residual = Y - Z @ W
delta = lstsq(Z, residual, rcond=1e-12)
Wnext = W + delta
```

The existing thin least-squares solver gives the minimum-norm correction under
its fixed rank cutoff. Prior components invisible to the retained design space
remain. Old predictions can change: there is no preservation guarantee without
old constraints. Rank-deficient contradictory labels remain imperfect fits.
The intercept is part of the coefficient-distance objective. This is neither
cross-entropy optimization nor an optimizer step. There is no regularizer,
rescaling, tuning parameter, RNG or feature selection.

Use the existing role-bound batch types and current explicit catalog:

```python
from src.adapters.clm_linear_fit_control import fit_clm_linear_control
from src.adapters.clm_linear_update import update_clm_linear_control

initial_report = {}
current = fit_clm_linear_control(first_batch, first_targets, initial_report)
update_report = {}
current = update_clm_linear_control(current, next_batch, next_targets, update_report)
```

Each update needs a fresh `linear_update` report key, the same encoder identity,
feature width, ordered candidate IDs and original fitting role. Exact TRAIN
controls accept exact `FrozenCatalogTrainingBatch`; disposable probe controls
accept exact `FrozenProbeCatalogBatch`. Role conversion, final/readout batches,
changed catalog order and mismatched labels are refused before the solve.
The existing readout function accepts the returned control without new APIs.
Probe controls remain nondeployable under the existing serving boundary.

CLM batches retain the original bounds:2..16 query rows,2..16 classes,1..4096
features, normalized detached owned-layout CPU float32 tensors. Core arithmetic
uses owned float64 arrays; its one-row lower bound remains available to pure
numerical callers. The core knows no data roles; use the adapter at role-bearing
boundaries. Roles describe trusted local declarations, not hostile-code
containment or consent authority.

The returned immutable fit stores only current coefficients and **last-batch**
row count/rank/spectrum/squared error. It retains neither a prior model nor
training history. Do not read those fields as lifetime work or old-task retention.
The caller owns experiment limits, lifetime solve counts and row exposure; every
repeated/replay row counts again. This API grants no retry, retained-bank access,
TRAIN conversion, promotion or scientific budget.

`linear_update` reports one attempted/returned residual solve and rows; source
coefficient, correction, one-hot, residual-target, input/design/output and guard
array sizes; elapsed time; and zero encoder/head-forward/optimizer work. These
are logical arrays, not all Python copies, LAPACK workspace, allocation peaks or
FLOPs. Source ownership/values and input features/labels are checked after return.
A failed/lost or unadmitted result remains incomplete/uncertain with its paid
prefix and error. Stop the owning procedure and preserve that report; a fresh
dict does not renew an external allowance.

Files and boundaries:

```text
src/core/wide_categorical_linear_fit.py       pure numerical fit/update/readout
src/adapters/clm_linear_update.py             existing role-bound batch adapter
tests/test_wide_categorical_linear_update.py  independent projection oracles
tests/test_clm_linear_update.py               sequential behavior and role/failure tests
```

No dependencies, environment variables, storage framework or stream framework
are added. The original CLI/hook/defaults remain unchanged. The next integration
should call this port from the existing temporal stream, with a separately fixed
TRAIN-only sequence, complete lifetime costs and isolated probes. Final data must
remain unavailable to learning, replay, promotion, tuning and selection.

Focused commands (observe the owning issue's allowance before execution):

```powershell
python -m pytest -q tests/test_wide_categorical_linear_update.py tests/test_clm_linear_update.py
python -m ruff check src/core/wide_categorical_linear_fit.py src/adapters/clm_linear_update.py
python -m ruff format --check src/core/wide_categorical_linear_fit.py src/adapters/clm_linear_update.py
python -m mypy --follow-imports=silent src/core/wide_categorical_linear_fit.py src/adapters/clm_linear_update.py
```

The [software report](reports/clm-linear-update-20261025.md) records actual checks
and costs. Literal fixtures prove arithmetic and boundary behavior, not native
task usefulness, general retention/recovery or a CPC benefit.

The [temporal stream option](clm-linear-stream.md) now connects this correction to
the original delayed-label runner, with explicit fixed-union tool identity and
separate solve-based work counters. It remains an opt-in conventional control.
