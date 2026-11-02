# Continuous physical affine TRAIN control

The separate affine control fits one fixed map from the original 12D cube
task-space observations to four normalized physical actions. It uses the
existing validated least-squares solver with an intercept and fixed
`RCOND=1e-12`. Positive row weights preserve every row. It reports raw actions
and then clips predictions to the existing `[-1,1]` controller bounds.

TS12 already contains observed grasp, two displacement vectors divided by0.1m,
previous executed actions and closing history. “Raw” means this existing view
without fitted observation standardization. No task names or future states
enter the model.

```python
from src.core.continuous_affine_fit import (
    fit_continuous_affine, predict_continuous_affine,
)
from src.core.cube_taskspace_state import cube_taskspace_spec

# Exact float64 arrays: X(B,12), actions(B,4), positive weights(B); B1..198.
# Caller must hold original TRAIN release authority and a frozen work budget.
model = fit_continuous_affine(X, actions, weights, cube_taskspace_spec())
predictions = predict_continuous_affine(model, X)
# predictions.raw, predictions.bounded, predictions.clipped_per_axis
```

The fit minimizes weighted **unclipped** physical squared error. Clipping is a
prediction rule; this is not clipped-loss optimization, gripper BCE, or the
existing frozen-hidden/inverse-tanh readout objective. Existing categorical and
hidden-readout128-row APIs and BP/PC/neutral/active defaults remain unchanged.
The immutable model retains only52coefficients,13or fewer singular values and
small metadata, no training examples, optimizer, RNG or success-record chain.

The development CLI composes the existing qualified six-group reader and frozen
weights without rebuilding a historical model:

```text
python -m scripts.fit_aggregated_affine_control FRESH_OUTPUT.json
```

This performs data reads and one fit; use only a separately authorized prospective
allowance. The recorded398case is spent. Original qualified artifacts must exist;
this is not a bundled/clean-clone or installed-wheel workflow. An exclusive output
marker precedes reading, and its single attempt slot is permanently charged
before solving. Source/role/recipe/clock/budget changes and lost returns cannot
qualify a model or renew the same attempt.

The actual198-row control is accepted as a TRAIN-fit result. Weighted half-MSE
is0.02534325raw/0.02469080bounded; rank13/condition26.58. Compared with the
accepted historical neural fit, reported errors improve for Push/correction
groups but worsen for all three Pick groups. The original fit-only case performed
no physical rollout. Neither retention nor native success follows from residuals.

`FITTED-MODEL.json` in the local feasibility stage preserves this **new** map
from the sole lead result read. It is unqualified for robot/deployed use. The
existing frozen-rollout entry point now also admits an explicit
`ReadonlyContinuousAffine` reader, with a distinct immutable policy/snapshot and
no optimizer, update or restore methods. It owns a detached fitted map and
refuses changed or equal-replaced native state. Pick forks a separate readonly
inference owner; both owners are checked after the last history observer before
control and final release. Push retains its existing direct frozen-owner checks.

```python
from src.adapters.readonly_continuous_affine import ReadonlyContinuousAffine
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from scripts.maniskill_continuous_smoke import run_frozen_rollout

reader = ReadonlyContinuousAffine(model)  # already admitted TRAIN fit
result = run_frozen_rollout(
    fresh_output, learner=reader, train_seed=17,
    action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
)
```

Pick admission requires the exact executed task-space recipe and ordinary TRAIN
reset. Reserved scoring, teacher/correction queries and optional interventions
are refused. The existing physical feature/history encoder supplies inputs;
outputs are bounded actions with explicit affine metadata and zero optimizer
updates. BC/PC/nearest and constant-zero defaults retain their existing branches.
The existing `scripts.push_cube_continuation_study.roll_push` also admits this
exact reader, with its existing TRAIN reset validation and 50-control limit.

Software checks use literal maps and fake environments. They do not qualify the
saved fit for robot or deployed use. The initial all-four native attempt stopped
before simulator creation because its report conversion forwarded the diagnostic
`coefficient_norm` to the core constructor. All four outcomes are unobserved;
that attempt and its cached-map read remain failed/spent.

The corrected pure conversion is available for an already admitted TRAIN report:

```python
from src.adapters.readonly_continuous_affine import readonly_affine_from_fit_record

reader = readonly_affine_from_fit_record(admitted_report["model"], role="train")
```

It validates the exact bounded JSON model section and its norm diagnostic,
without IO, fitting or prediction. Caller envelope/provenance/role and budget
admission remain separate. The literal production-shape round trip and malformed
record checks pass; the actual spent cached file was not reread. No native
allowance or scientific benefit follows from this repair.

[Initial native failure and conversion repair](reports/continuous-affine-native-20261101.md).

An explicit prospective engineering continuation retained that failure and all
cumulative costs while keeping the same map, four TRAIN cases and total physical
quota. The corrected run completed Push17/Pick17/Pick19/Pick20 with50controls
each and **zero successes in all four**. Model/history/closure/budget checks pass;
no refit, weight/seed selection or final/probe input occurred. The affine control
is a completed negative qualification, not a useful robot policy. Prefer the
existing working fixed-input memory or known neural Pick baseline; no further
launch is allowed under these spent protocols.

[Accepted outcomes, cumulative costs, review limits and next learner action](reports/continuous-affine-native-continuation-20261101.md).

[Readonly port checks, preserved failures and review scope](reports/continuous-affine-port-20261101.md).

[Exact execution, tests, costs, failures, review limits and remaining work](reports/continuous-affine-feasibility-20261101.md).
