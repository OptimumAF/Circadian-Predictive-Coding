# Linear width pruning with observed damage limits

```powershell
python -m scripts.run_linear_width_pruning --describe
python -m scripts.run_linear_width_pruning
```

The first command performs no evaluation or SGD. The full engineering example
shows accepted, rejected and failed decisions for both linear MLP and residual
families, exact returned states and probe diagnostics. It uses four real SGD
continuations and twenty equilibrium evaluations, one CPU thread, no final-test
data and no scientific comparison. Separate continuation training labels never
come from validation-only probes.

## Core candidate

`core/linear_width_pruning.py` adds
`prune_linear_width(state, index) -> LinearPruneChange(state, lineage)`.
It removes the same coordinate from every hidden layer, changing common width
N to N−1. Width1 and invalid native indices/families/configurations refuse.
Source arrays are borrowed quiescent; returned arrays and config are owned.
Lineage format1 records family, source config and removed coordinate only.
It confers no runtime, consent or provenance authority.

With selector P, complete MLP matrices become `A0 P`, `Pᵀ Al P`, `Pᵀ Ao`.
Residual middles use `I_new + Pᵀ(Al−I_old)P`; the fixed identity remains implicit.
Raw branches rebind old/new forward scales using the stable structural arithmetic.
Axis gathers use O(raw entries), with target configuration validation before
allocation. Existing split/merge and learner defaults are unchanged.

General pruning changes predictions and the full scalar equilibrium denominator.
A completely disconnected coordinate has zero relevant raw rows/columns and
output row; removing it preserves the current prediction/equilibrium mathematically,
including the residual identity case. Numerical tests declare tolerances.
Width-dependent SGD rates change; future trajectories are not preserved.

## Application gate

`app/linear_pruning_gate.py` adds:

```python
limits = PruneDamageLimits(
    max_prediction_change=0.01,
    max_half_mse_increase=0.001,
    max_energy_increase=0.001,
)
decision = gate_linear_prune(
    source, index, probes, limits=limits, budget=budget,
)
```

The example limits above are illustrative API values, not tuned defaults.
All three are required finite native nonnegative values. Every declared probe
must satisfy every inclusive limit:

| Diagnostic | Meaning |
|---|---|
| prediction change | Maximum absolute source/candidate prediction difference over probe examples |
| half-MSE increase | `max(0, candidate half-MSE − source half-MSE)`, a per-probe aggregate |
| energy increase | `max(0, candidate full equilibrium energy − source energy)`, a per-probe aggregate |

Why three limits: identical predictions can coexist with increased equilibrium
energy because pruning changes downstream norms. The fixed CLI control has
prediction0 and half-MSE.5 both before/after, but energy.25→.5. A .24 energy limit
rejects; the independent inclusive-boundary test accepts at .25.

Probes are exact `ScalingBatch` records from `app/linear_scaling_measurement.py`.
Only declared `train` and `validation` roles are allowed, with at least one
validation probe. Final-test roles are refused. Inputs require 1..8 probes,
each 1..1024 rows, finite exact float64 shapes, unique globally disjoint printable
IDs and no shared probe buffers. Per-probe input/target plus hidden activation
entries must not exceed1,000,000, checked before copies. Roles are caller
declarations; relabeling final data does not establish independent validation.

The gate copies the source, limits and probes before budget callbacks. It retains
a private fallback, builds a separate candidate, and uses fresh detached states,
weights and probe arrays for **each** source/candidate evaluation. Relevant score
outputs are copied before post-evaluation callbacks. Faults cannot contaminate
the retained source, another probe or the comparison baseline.

An exact `ToyBudgetSession` supplies wall/RSS checks before/after evaluations and
before return; gate evaluation records no training updates. Every probe completes
before observed-limit acceptance/rejection. Invalid public preflight raises.

| Status | Returned state | Lineage/diagnostics |
|---|---|---|
| accepted | Complete owned candidate | Accepted coordinate lineage; all probes completed |
| rejected | Owned retained source | No accepted lineage; all observed damage diagnostics |
| failed | Owned retained source | No accepted lineage; completed diagnostics retained alongside planned count and failure reason |

Candidate, numerical, evaluator and budget errors produce `failed`, without retry
or partially publishing a candidate. The caller uses `decision.state` as the
complete selected replacement; this functional API does not mutate a shared
runtime holder or implement cross-thread/durable publication.

Acceptance bounds only the declared observations. It is not a global damage bound,
future-SGD guarantee, generalization result or authorization to tune on final data.
Linear utilities have no adaptive optimizer or chemistry traces; nonlinear/CPC
state transfer and parent R1.6 remain open.

## Checks and extension

```powershell
python -m pytest tests/test_linear_width_pruning.py tests/test_linear_pruning_gate.py tests/test_linear_width_pruning_cli.py -q -o addopts= -p no:cacheprovider
python -m ruff check src/core/linear_width_pruning.py src/app/linear_pruning_gate.py scripts/run_linear_width_pruning.py tests/test_linear_width_pruning.py tests/test_linear_pruning_gate.py tests/test_linear_width_pruning_cli.py
python -m ruff format --check src/core/linear_width_pruning.py src/app/linear_pruning_gate.py scripts/run_linear_width_pruning.py tests/test_linear_width_pruning.py tests/test_linear_pruning_gate.py tests/test_linear_width_pruning_cli.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/core/linear_width_pruning.py src/app/linear_pruning_gate.py scripts/run_linear_width_pruning.py tests/test_linear_width_pruning.py tests/test_linear_pruning_gate.py tests/test_linear_width_pruning_cli.py
```

Repeat mypy with `--platform linux`. Native controls need their recorded engineering
budget. No dependencies or persisted environment settings were added; the CLI
sets OMP/MKL/OpenBLAS threads1 before NumPy imports. New structural proposals
should pass through the same ownership/probe-role/damage/recovery principles.
Do not replace the final-test exclusion with a caller-selected metric shortcut.

R1.6b accepted 2026-10-15:140 focused cases, six-file scoped statics and independent
math/state/source/actual-outcome review pass. The retained six-control example
uses12/24 joint SGD calls and108/512 equilibrium evaluations across tests and saved
CLI; unused allowance closed. Detailed commands and evidence:
`artifacts/runs/linear-width-pruning-20261015/RESULTS.md`. Parent R1.6 remains open
for nonlinear/CPC state transfer and publication/rollback integration.
