# Native coupled PC/CPC prune transfer

```powershell
python -m scripts.run_coupled_width_pruning --describe
python -m scripts.run_coupled_width_pruning
```

Describe-only performs no training or inference. The full engineering example
shows PC, neutral CPC and active CPC after real wake history, then accepted,
rejected and failed pruning. It uses12 training calls,24 guarded inferences and
120 coupled-energy evaluations, one CPU thread and no final-test data. JSON
contains exact model state, coordinate lineage, observations and continuation.
It is an engineering record, not a checkpoint or scientific learning comparison.

## Transfer contract

`core/coupled_width_pruning.py` provides:

```python
owned = copy_coupled_model(model)
change = prune_coupled_width(model, layer_index, unit_index)
candidate = change.model
```

Only exact native `CoupledPredictiveCodingNetwork` and
`CoupledCircadianPredictiveCodingNetwork` families are accepted. The new structural
utility checks1..16 hidden layers and at most1M parameter entries, including all
biases/output, before snapshots or native allocation. Existing constructors,
adapters and their domains/defaults are unchanged.

One selected layer loses one unit (source width>=2):

| State | Transfer |
|---|---|
| Incoming hidden weight | Remove selected column |
| That hidden bias | Remove selected entry |
| Next hidden weight, or output weight | Remove corresponding row |
| Other parameters and output bias | Preserve copied values |
| Layer precisions | Preserve all values |
| CPC last-hidden chemistry/gates | Select matching entries only when pruning the last hidden layer |
| Earlier-layer pruning | Preserve complete chemistry/gate vectors |
| CPC chemistry policy/wake count | Preserve values, with an independently owned policy record |
| Optimizer/momentum | None exists in these native SGD learners |

The source and target pass complete format1 snapshot restore before use. This
checks all parameters, policy binding, gate/chemical consistency and zero-wake
rules. Source arrays are borrowed quiescent; returned model/state/metadata own
their copies. Lineage format1 binds family, input/hidden geometry, precisions and
removed coordinate. It records coordinate ancestry only, not runtime authority.

Why this: parameter-only restore would lose or mismatch adaptive state. A retained
gate must continue to refer to its retained neuron. No wake is added by transfer.
General predictions, energy or future training trajectories are not preserved.
These native models store effective parameters; no new raw scaling is introduced,
and open R1.8 is not satisfied by the transfer.

## Observed damage gate

`app/coupled_pruning_gate.py` takes required `CoupledPruneLimits`:

- Maximum probability change over each probe's examples.
- Aggregate probability-BCE increase, clipped at1e-12 using the common metric.
- Aggregate fixed-step guarded relaxed-energy increase.

All limits are explicit finite native nonnegative values, inclusive. Increases
clamp at0; all probes complete before damage acceptance/rejection. Probes are exact
`ScalingBatch` train/validation records with at least one validation record;
final-test roles are refused. Require1..8 probes, each1..1024 rows, finite float64
shapes/targets in[0,1], globally unique printable IDs and disjoint buffers. Per-probe
input/target+hidden activation entries must be<=1M. Caller role declarations do
not authenticate relabeled final data.

The gate accepts explicit `inference_steps`1..64 and positive finite inference
rate. It uses the accepted pure inference kernel with minimum=cap and both
convergence tolerances disabled. Both source/candidate must terminate `max_steps`
with attempted=accepted=cap and no fallback; otherwise the gate fails and retains
the source. **The energy is fixed-step relaxed energy, not equilibrium/minimum.**
Equal step counts do not imply equal computation after topology changes.

Source, limits and probes are copied before budget callbacks. Prediction and
inference use independently copied models/snapshots/parameters/data for every
source/candidate score. Relevant probability/scalar/diagnostic outputs are copied
before later callbacks. The candidate's family/geometry/precision/lineage and CPC
policy/count/selected chemicals/gates are checked against the retained source.
No gate scoring trains parameters or updates chemistry.

An exact `ToyBudgetSession` checks around each bounded scoring procedure and
before selection. Native calls are not preempted; step/workspace caps bound them.
Invalid preflight raises. Damage produces `rejected`; candidate/prediction/
inference/numerical/budget faults produce `failed`. Both return the complete owned
source, no accepted lineage, and completed diagnostics with planned probe count.
Only `accepted` returns the candidate and accepted lineage. No retry or threshold
tuning occurs.

This is a functional replacement boundary. It does not change an existing
`CoupledLearner`, `ActorShadowRuntime`, shared holder or durable checkpoint. Native
restore into an old geometry remains strict. Runtime publication/rollback needs
its own integration; an accepted local probe does not authorize promotion.
Limits cover the declared observations, not global damage, generalization, future
wakes, equal compute or CPC benefit.

## Fixed engineering example

All three variants start with seed23/D2/hidden(3,3)/precision(1,1), two wakes at
parameter rate.03, latent cap2/rate.05. The fixture then disconnects last unit2 by
parameter-only restore, deliberately retaining real nonzero CPC chemistry/count2.
Removing it accepts within1e-12 limits; removing connected unit0 rejects at zero
damage limits. Exhausted wall budget fails before scoring. Accepted/retained
models each receive one continuation using the declared training batch only.
PC/neutral parameters match exactly through transfer and continuation; active
gates remain associated with their selected chemical traces.

## Checks and extension

```powershell
python -m pytest tests/test_coupled_width_pruning.py tests/test_coupled_pruning_gate.py tests/test_coupled_width_pruning_cli.py -q -o addopts= -p no:cacheprovider
python -m ruff check src/core/coupled_width_pruning.py src/app/coupled_pruning_gate.py scripts/run_coupled_width_pruning.py tests/test_coupled_width_pruning.py tests/test_coupled_pruning_gate.py tests/test_coupled_width_pruning_cli.py
python -m ruff format --check src/core/coupled_width_pruning.py src/app/coupled_pruning_gate.py scripts/run_coupled_width_pruning.py tests/test_coupled_width_pruning.py tests/test_coupled_pruning_gate.py tests/test_coupled_width_pruning_cli.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/core/coupled_width_pruning.py src/app/coupled_pruning_gate.py scripts/run_coupled_width_pruning.py tests/test_coupled_width_pruning.py tests/test_coupled_pruning_gate.py tests/test_coupled_width_pruning_cli.py
```

Repeat scoped mypy with `--platform linux`; native fixtures require their recorded
engineering budget. No dependencies/persistent environment settings were added;
the CLI sets OMP/MKL/OpenBLAS threads1 before NumPy imports. Future structural
operations must define parameter/precision/adaptive-state mappings, validate full
restore and neutral continuation, then integrate publication with rollback before
claiming parent R1.6 or nonlinear scaling complete.

R1.6c accepted 2026-10-15:111 focused cases, six-file scoped statics and independent
math/state/source/actual-test+CLI review pass. Saved nine-row example78,882B has
all five PC/neutral parameter-parity flags true and retained CPC count/chemistry.
Joint38train/62guardedinfer/344E within frozen scope; unused allowance closed.
Detailed commands, evidence and limitations:
`artifacts/runs/coupled-pruning-transfer-20261015/RESULTS.md`. Parent R1.6/R1.8 and
topology-aware runtime publication/rollback remain open.
