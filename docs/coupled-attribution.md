# Fixed stationary and recurring attribution pilot

Describe the protocol without training:

```powershell
python -m scripts.run_coupled_attribution --describe
```

Run the fixed local exploratory pilot:

```powershell
python -m scripts.run_coupled_attribution
```

There are no seed, metric or tuning flags. The command prints JSON and exits 2
if any planned run or resource gate is incomplete. Redirect stdout to preserve
all outcomes. A new invocation repeats work; it is not a resume or a new seed.

## Protocol

Protocol `coupled_stationary_recurring_pilot_v1` was specified before comparative
outcomes. The prospective record is
`artifacts/runs/coupled-attribution-20261014/PROTOCOL.md`.

| Item | Fixed value |
| --- | --- |
| Initialization seeds | 23, 41, 59, in that order |
| Methods | matched BP, PC, neutral CPC, gated CPC |
| Topology | input 2, hidden `(8,8)`, scalar binary output |
| Initialization | exact shared fan-in Gaussian weights and zero biases per seed |
| Training streams | stationary A→A→A; recurring A→B→A |
| Phases | 6 updates each, batch size 8, 18 updates per learner run |
| Training data | uniform `[-1,1]`, generator seed 7301, same input order for all |
| Context labels | A: `x0 >= 0`; B: `x1 >= 0` |
| Held-out data | 64 fresh rows per context, generator seed 9101 |
| Update rate | 0.03 for every method |
| PC inference | 4 simultaneous steps at rate 0.2, precisions `(1,1)` |
| Neutral CPC | real chemistry, minimum plasticity 1 |
| Gated CPC | existing default decay .995, buildup .02, sensitivity .7, floor .2 |
| Label delay | 2 logical ticks |
| Checkpoints | 0, 6, 12, 18 committed updates |
| Evaluation | both contexts at every checkpoint, deferred until training finishes |

The fixed data are shared across initialization seeds. Intervals therefore
describe initialization variability conditional on these synthetic data, rather
than sampling uncertainty across new datasets. There is no replay, sleep,
structural adaptation, promotion, hyperparameter search or winner selection.

## Isolation and scoring

Application validation requires train and `final_test` roles, unique row IDs,
compatible float64 shapes, binary target domains and disjoint buffers across
roles. These declarations and array checks do not establish physical provenance
for arbitrary caller data. The supplied generator creates separate datasets.

Only training batches enter the existing delayed-label actor/shadow runtime.
The runner saves detached candidate states at the fixed checkpoints. Each
learner/stream run completes **all** training before a detached scorer restores
those states and receives held-out features. Held-out labels enter application
metrics only; they never enter `train_batch`, promotion, stopping or configuration.
Outputs from one fixed run do not alter subsequent run settings or execution order.

Primary metric is the same probability BCE for all methods, clipping probabilities
to `[1e-12, 1-1e-12]`; lower is better. Secondary metric is accuracy at `p >= 0.5`;
higher is better. The generator uses hard labels; the generic runner thresholds
soft targets at .5 for accuracy. Native BP loss and relaxed PC energies retain
their different IDs and are not comparison metrics.

Both contexts are reported throughout: A after learning B describes old-context
retention, B describes the new context, and the final A phase describes recurrence.
Snapshot scoring is offline, with no prediction-state update to the training model.

## Summaries and failures

All 24 seed×stream×method rows are retained. Failed and unreached rows have
explicit reasons and committed work counts. A failed stream is not scored as a
partial trajectory. Missing or duplicate records are refused by aggregation.
Any missing member of a three-seed vector suppresses its complete mean/interval;
the observed subset remains visible as descriptive data.

The existing `seed_statistics` implementation reports nominal marginal 95%
Student-t mean intervals with 2 degrees of freedom and critical value 4.303,
from the [.975 column of the NIST table](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3672.htm).
Three seeds cannot validate the approximate normality assumption. These are
exploratory intervals with no simultaneous coverage or significance claim.
Zero observed variance yields an explicit unavailable interval.
Intervals are untruncated; accuracy endpoints can exceed `[0,1]` and should not
be read as probability ranges.

The fixed paired differences are `pc - bp`, `neutral_cpc - pc`, and
`gated_cpc - neutral_cpc`, preserving these signs for both metrics. Negative BCE
differences favor the left method; positive accuracy differences favor it.
No direction is changed after seeing outcomes. Neutral parity is a correctness
control; it is not evidence of learning benefit.

## Work and resources

The whole pilot plans 432 committed updates, 3,456 training row exposures and
1,296 latent steps. This fits the existing first-pilot ceiling of 512 updates.
The script sets only its process's OMP/MKL/OpenBLAS thread settings to 1 before
importing NumPy. No GPU, download, environment rollout or paid service is used.

The 60-second wall limit covers orchestration after imports. Each learner run
receives the remaining wall allowance and an 18-update cap. The existing RSS
sampler and execution budget check a 256 MiB observed process limit. Checks occur
at safe boundaries; they do not preempt native calls or measure every transient
allocation. Final wall/RSS excess makes the report incomplete. JSON output is
limited to 32 MiB; no raw training data are written by the runner.

Reported native work is nominal completed kernel work: BP performs two full
feedforward gradient evaluations per update (including candidate validation),
whereas PC/CPC performs six coupled energy/gradient evaluations and four latent
steps. Copies, chemistry, failed attempts, schema-check overhead and scoring are
separate costs. Score forward counts and row exposures are reported; nominal
kernel counts are not FLOPs, matched compute or comparative timing measurements.

## Structure, checks and extension

```text
src/app/coupled_attribution.py                validated delayed training/deferred scoring
src/app/coupled_attribution_summary.py        complete seed vectors and paired summaries
src/adapters/coupled_attribution_pilot.py     fixed synthetic data/model/resource composition
scripts/run_coupled_attribution.py            fixed CLI and process thread settings
tests/test_coupled_attribution.py             independent isolation spies and native control
tests/test_coupled_attribution_pilot.py       data/factory/aggregation/failure/describe controls
```

Dependency direction stays inward: script → adapter → app/core. No new dependency
or environment variable is required from the user. Model snapshots remain
family-specific local state; the JSON is a result report, not a checkpoint.

```powershell
python -m pytest -q tests/test_coupled_attribution.py tests/test_coupled_attribution_pilot.py
python -m ruff check src/app/coupled_attribution.py src/app/coupled_attribution_summary.py src/adapters/coupled_attribution_pilot.py scripts/run_coupled_attribution.py tests/test_coupled_attribution.py tests/test_coupled_attribution_pilot.py
python -m ruff format --check src/app/coupled_attribution.py src/app/coupled_attribution_summary.py src/adapters/coupled_attribution_pilot.py scripts/run_coupled_attribution.py tests/test_coupled_attribution.py tests/test_coupled_attribution_pilot.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL src/app/coupled_attribution.py src/app/coupled_attribution_summary.py src/adapters/coupled_attribution_pilot.py scripts/run_coupled_attribution.py tests/test_coupled_attribution.py tests/test_coupled_attribution_pilot.py
```

Tests run only small controls; they do not launch the full comparison. Repeating
the native control consumes another 8 updates and must fit the declared local
engineering allowance. Scientific changes to data, seeds, metrics, gates or work
require a new prospective protocol. Keep negative and incomplete outcomes intact;
do not tune the baseline or reuse final scores as selection data.
