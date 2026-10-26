# Bounded coupled unit renewal

## Fixed adaptation and fresh-task scenario

Describe the opt-in scenario with:

```powershell
python -m scripts.run_coupled_attribution --scenario renewal-learnability --describe
```

Omit `--describe` to execute its single fixed local invocation. The original
scenario remains the default. This scenario uses the existing A→B→A stream and
isolated sixteen-update C probe. It executes only the three renewable PC-family
arms; accepted BP/PC/neutral/active no-renewal controls are named for post-run
comparison and are never opened by training. The protocol fixes age6, gradient
EMA threshold.001, at most1 replacement every6 updates and3 per arm. Zero eligible
units remain a result. These are declared exploratory settings, not recommended
or tuned values. Final and probe outcomes never select eligibility or checkpoints.

Renewable snapshot3 now has an explicit read-only diagnostic path before feature
work. It reports real heterogeneous ages and stable IDs using native nextID;
unknown ancestry stays unknown. Probe forks restore complete state but run no
renewal helper and spend none of the benchmark replacement allowance. The old
tracked reader, original descriptors and probe horizon retain their meanings.

New `src/adapters/unit_renewal_pilot.py` owns only fixed scenario composition,
resource/failure reporting and native bindings. The existing stream/runtime/
fresh-probe applications still own execution and evaluation isolation. Tests:
`tests/test_renewal_learnability.py` and `tests/test_unit_renewal_pilot.py`.
Prospective protocol and executed evidence:
`artifacts/runs/r22-renewal-adaptation-20261016/`.

Opt-in renewable PC and CPC models refresh mature last-hidden units whose
outgoing gradient EMA is at or below a fixed threshold. The statistic is the
absolute batch-aggregate outgoing coupled-energy gradient before CPC gates.
It is a proxy for selection, not a causal measure of usefulness.

## Use through the existing stream

```python
from src.adapters.coupled_learners import CoupledLearner
from src.app.stream_unit_renewal import StreamUnitRenewal
from src.core.coupled_unit_renewal import RenewalPolicy
from src.core.renewable_coupled_predictive_coding import (
    RenewableCoupledPredictiveCodingNetwork,
)

model = RenewableCoupledPredictiveCodingNetwork(
    2, (8,), seed=17,
    renewal_policy=RenewalPolicy(
        min_age_updates=20, max_importance=0.001, max_per_event=1, seed=29,
    ),
)
learner = CoupledLearner(
    model, learning_rate=0.02, inference_steps=2, inference_learning_rate=0.04,
)
# budget is the original, fresh ToyBudgetSession for this invocation.
renewal = StreamUnitRenewal(
    learner, budget, every_updates=20, total_updates=60, max_replacements=3,
)
# Pass renewal=renewal to run_attribution_stream with declared train/final-test
# batches and the same learner/budget/count. Final scoring follows all updates.
```

These example thresholds are configuration syntax, not recommended or measured
learning settings. Freeze eligibility, schedules and comparisons before a study;
never choose them from final evaluation or probe outcomes.

## Behavior and state

- Eligibility requires both minimum native-update age and maximum importance.
  Eligible units sort by importance, stable unit ID, then index. Selection is
  deterministic and bounded by policy and remaining session allowance.
- Only the runtime candidate changes. A private owned fork prepares the reset
  from its current complete snapshot through the existing consolidation boundary.
- Reset changes the selected incoming column, selected bias and outgoing row.
  Incoming weights use fan-in scaling and a local seed derived from policy seed
  and new unit ID; bias and outgoing weights become zero. Selected CPC chemistry
  becomes zero and gate one. Selected history becomes zero age/importance with
  a new ID. Earlier layers, output bias, other units and native clocks persist.
- Every successful native training call advances age, including replay if used
  directly. Renewal and prediction do not advance native-update clocks.
- Renewable history format2 supports heterogeneous ages and IDs. Complete native
  snapshot format3 binds history, policy and CPC state. Existing tracked snapshot
  format2 and default models retain their meanings. Unsupported disk codecs,
  managed capture and old diagnostic readers do not silently drop new state.
- Charges occur before private preparation and survive failures and older native
  state restores. Reports separate charged, attempted, returned and committed
  units. A failure stops the helper; it cannot be rebound. The helper allowance
  is not a resumable model field. No new invocation is authorized by restoration.
- Storage is fixed-width history plus at most512 event slots and128 replacement
  IDs per helper. No retired-unit or unlimited success-record chain is retained.

The initial stream integration refuses simultaneous shared rehearsal and renewal.
It supplies no promotion, checkpoint persistence or automatic eligibility tuning.
Preserve those unsupported cases when extending it: composition must retain both
permanent budgets, current-candidate selection and post-training-only evaluation.

Observation-only capacity can now accompany renewal through the existing runner.
See [capacity around renewal](renewal-capacity.md) for boundary ordering,
per-event time, unchanged permanent charges and late-error readback. Scheduled
split/renewal and replay/helper pairs remain unsupported; final evaluation stays
unavailable to selection/training. Native formats remain distinct.

## Modules and checks

```text
src/core/coupled_unit_renewal.py                 policy/history/selection
src/core/renewable_coupled_predictive_coding.py  atomic native reset/full state
src/app/stream_unit_renewal.py                  schedule/permanent allowance
src/adapters/coupled_learners.py                owned exact native model copies
src/app/coupled_attribution.py                  existing stream integration
tests/test_coupled_unit_renewal.py               native behavior/reset/refusal
tests/test_stream_unit_renewal.py                runtime/isolation/budget failures
```

Run focused tests with `python -m pytest -q tests/test_coupled_unit_renewal.py
tests/test_stream_unit_renewal.py`. Scoped Ruff/format and Win/Linux mypy commands,
bounded worker receipts and native engineering evidence are recorded in
`artifacts/runs/r22-bounded-renewal-20261016/RESULTS.md`.
