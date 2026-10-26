# Causal executed-action history

The existing state learner and frozen robot rollout can use47 inputs: the42 state
coordinates, the previous actually executed four-axis command, and a clipped
consecutive negative-gripper-command count divided by6. Observationt uses commands
strictly beforet; nextobservation includes actual executiont. History resets for
every episode. A nonnegative gripper command resets the count.

Why this: the existing teacher has explicit phase/closing memory while the student
was feedforward on current state alone. The counter is a causal proxy: it can
saturate before reaching the cube and does not observe contact or lost readiness.
This representation does not establish the cause of earlier failures or promise
successful acquisition. All earlier zero-success and conditioning results remain.

## Existing CLI and API

Add `--action-history executed` or `--action-history zero` to the existing
`python -m scripts.run_continuous_robot_learning` command, with qualified TRAIN
teacher paths and its other declared options. `--canonical-state-quaternions`
selects sign-stable first42 coordinates within the same history recipe.
`--readiness-stage-residuals --readiness-teachers` retains recorded phase readouts.
The zero mode appends five zeros with the same47D architecture and seed. Both
modes fit only the first42 TRAIN coordinates, then fix history means0/scales1.
Optional42 scale floors affect only the prefix; history divisors remain1.

```python
from src.core.pick_cube_action_history import PickCubeActionHistoryRecipe
from scripts.run_continuous_robot_learning import run_learning_rollout

recipe = PickCubeActionHistoryRecipe(mode="executed", canonical_quaternions=True)
# This call spends training and simulator work; declare a local budget first.
report = run_learning_rollout(output, warm, new, action_history_recipe=recipe)
```

The generic stream accepts already derived complete TRAIN47D episodes plus the
explicit recipe. Use `action_history_training_episode` on each original raw TRAIN
episode independently. The fitter revalidates the complete causal tails before
fitting. Do not derive history from teacher targets, proposed actions or the next
row's command. The native port's actual records already contain float32 commands;
nonrepresentable records are refused rather than silently rounded.

Live serving advances only when a returned record matches the pending raw input,
original step and quantized intended command. It checks state/counters against
the existing successful native record prefix before more work. Uncertain/failed
controls stop; paid work remains in an incomplete report. Raw native traces stay
42D. Public metadata is never an authority for the next input. Source and private
inference owners remain frozen, including after final timing/telemetry callbacks.

History is state-only; actors, elapsed clocks and context interventions remain
unsupported. Explicit readiness corrections are supported as described below. Omitted
history preserves existing APIs/kwargs/default behavior. No new dependencies or
environment variables were introduced.

## Files and boundaries

```text
src/core/pick_cube_action_history.py       pure recipe/state/causal TRAIN views
src/app/action_history_encoding.py         one pending input and native-return binding
src/adapters/numpy_continuous_bc.py         TRAIN-only prefix fit and fixed tail
src/adapters/continuous_learnability_demo.py existing four-family stream and diagnostics
scripts/run_continuous_robot_learning.py   raw/model binding and CLI
scripts/maniskill_continuous_smoke.py       existing frozen native collection
```

Live history retains O(1) state and one pending raw input. Validation folds the
already bounded native prefix (at most50 records), adding O(n) work per boundary;
it is not constant computation. A single bounded completed raw trace is retained
privately through final callbacks. Arrays, snapshots, JSON copies, validations and
struct conversions add costs beyond reported dense MACs.47D models have3332
parameters versus3012 for42D at H64; velocities and preprocessing add memory.

## Focused verification

```powershell
python -m pytest -q tests/test_pick_cube_action_history.py tests/test_action_history_integration.py
```

Exact affected legacy selectors, eight-file lint/format/type command arguments,
paid failed prefixes and accepted results are recorded under
[implementation evidence](../artifacts/runs/causal-action-history-20261020/).
Check scoped evidence there before asserting acceptance. Tiny fake rollouts are
not simulator physics, successful task acquisition or a scientific comparison.

Extend this recipe through existing components; test new temporal/reset/native
failure boundaries and keep an equal-width control before a separate experiment.
Final evaluation remains unavailable to fitting, tuning, promotion and selection.


## Readiness corrections

Use the existing API with a47D source whose first42 statistics were fitted on
TRAIN and whose history tail has zero means/unit effective scales:

```python
from scripts.maniskill_continuous_smoke import run_corrective_adaptation
from src.core.pick_cube_readiness_roll_in import ReadinessRollInPlan

# Spends queries, controls and8privateupdates; freeze a local protocol first.
result = run_corrective_adaptation(
    source, output_directory=output_directory,
    visual_observation=False, readiness_corrections=True, acquisition_only=True,
    action_history_recipe=recipe, readiness_roll_in=ReadinessRollInPlan(1),
)
```

Omit readiness_roll_in for fully student collection. Acquisition-only collects
TRAIN17, updates a detached private learner, then serves pure student17; default
collection19/returns17,19 remains available. The recipe carries canonicalization;
keep observation_projection omitted. BP, PC, neutral CPC and active CPC share this
route; nearest policies do not admit history. No additional CLI or environment
variable is introduced by this composition.

The core pick_cube_action_history_corrections module owns raw42/model47 readiness
batches and work counts. Targets/stages/arrivals remain original; actual executed
mixed commands determine the causal tail. It does not release labels, fit stats,
run a learner or touch IO. Adapter continuous_correction_demo checks raw arrival
first, then uses the existing eight-update runtime and native owner/receipt/clock/
budget checks. The smoke adapter finishes queries while the SDK is live, seals
the batch acrossclose, and binds requested resets and the original recipe before
later work. Pure return resets history and carries no teacher-prefix success claim.

Projection adds2n42→47 transforms/nlabelviews/nadvances and optional4nquaternion
unitchecks; retained prefix validation, arrays, copies and serialization add costs.
See [62-case scoped acceptance and exact commands](../artifacts/runs/action-history-corrections-20261020/RESULTS.md).
This is implementation correctness. The native correction trial subsequently completed: teacher-assisted collection
succeeded after20controls, but the updated pure student failed after50. See
[accepted negative result](../artifacts/runs/action-history-correction-acquisition-20261020/RESULTS.md).
All original zero-success and incomplete-comparison results remain unchanged.
