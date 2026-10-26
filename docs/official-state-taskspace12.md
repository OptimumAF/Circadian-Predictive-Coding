# Official-state taskspace12

The optional taskspace recipe selects twelve inputs through the existing history,
learner and robot paths. It is intended for the admitted default ManiSkill Panda
PickCube state setup, not RGB-only or arbitrary robot deployments.

| Coordinates | Meaning | Scaling |
|---|---|---|
|0|Official observed is_grasped raw18|Unit/binary|
|1–3|WORLD cube−TCP raw36:39|Divide by0.1 metres|
|4–6|WORLD goal−cube raw39:42|Divide by0.1 metres|
|7–10|Previous actually executed native32 command|Unit normalized control|
|11|Consecutive negative-gripper-command count, capped6|Divide by6|

Why this: accepted47D runs remain autonomous failures despite successful assisted
coverage. Savedpure displacements reached over400 empiricalstandarddeviations;
quaternion nuisance removal alone cannot address that. A fixedphysical view is a
hypothesis about conditioning/information, not proof of failure causality or benefit.

All42raw values must be finite; flag must be0/1 and physicaldivisionfinite. Outputs
are not clipped. WORLD vectors are named truthfully. The nativefactory pins the
default Panda/table/controller whose world/root axes align; no runtime rotation
inference or guarantee for arbitraryroots is added. Contextinterventions/actor/
clock/visual routes are refused. The official flag is observed simulatorstate,
not an oracleteacher phase or a physicalrobot/RGB sensing claim.

## Existing CLI and API

Add `--taskspace-history executed` to the existing
`python -m scripts.run_continuous_robot_learning` command with qualified TRAIN
teacher17/19paths and declared localbudget. `--taskspace-history zero` uses equal
width with5zero historyvalues. These flags are incompatible with `--action-history`,
`--canonical-state-quaternions`, scale floors and visual mode. Readinessteacher/
stage diagnostics flags continue to work. Default and47D selection are unchanged.

```python
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from scripts.run_continuous_robot_learning import run_learning_rollout

recipe = PickCubeTaskSpaceHistoryRecipe(mode="executed")
# Spends declared training and simulator work; freeze/review a budget first.
report = run_learning_rollout(output, warm, new, action_history_recipe=recipe)
```

The stream accepts complete derived TRAIN12 episodes plus explicitrecipe. Derive
each originalraw episode separately with action_history_training_episode. Beforet
uses commands beforet; nextinput includes actualt, resetper episode. Encoded12
validation checks prefix/causal history but cannot authenticate droppedraw42;
outerrobot/readinessview binds originalraw evidence. Never use teacher labels or
unsent studentproposals as history. Actualnative32 provenance remains mandatory.

Fixedscaling occurs in the projection. Modelpolicy standardization isNone; selected
fit_training_standardization returnsNone aftervalidation and does no statistical
fit. Work reports zero fits/fitrows/fitcoords/statsbytes, plus explicit fixedinput
metadata. Existing normalizer identitycopies still cost calls/rows/coordinates.
Floors/RGB/expectedempiricalstats are refused; nofakefitted_rows are created.

Readiness corrections use the same `run_corrective_adaptation` API with
`action_history_recipe=recipe`, `visual_observation=False`, readinessselection and
optional existingfixedteacherprefix. Arrivedraw42labels retaintargets/stages/ticks,
model12views use actualmixedcommands,8updates stayprivate/sourcefrozen. Student-
only return resets history and excludesmixedsuccess; no automaticpromotion.

## Files and extension boundaries

```text
src/core/pick_cube_taskspace_history.py      fixed recipe and observed7D prefix
src/core/pick_cube_action_history.py         shared causal TRAIN episode views
src/app/action_history_encoding.py           pending input/native-return binding
src/adapters/numpy_continuous_bc.py           selected validation/no empiricalfit
src/adapters/continuous_learnability_demo.py  existing fourfamily stream/work
src/core/pick_cube_action_history_corrections.py raw/model label pairing
scripts/run_continuous_robot_learning.py     existing CLI/candidate admission
scripts/maniskill_continuous_smoke.py         existing frozen serving/corrections
```

No new dependencies or environmentvariables. To extend, specify a separateexact
recipe and units/setup contract, retainraw/causal authority and invalidmetadata/
arrival/frozenowner tests; do not silently reinterpret WORLD axes or reuse another
setup's qualification. Count physicaldivisions/copies/prefixvalidation as costs.

## Checks and claims

[Final197cases/platform, exactcommands, statics, independentreview and paidfailedprefix](../artifacts/runs/official-state-taskspace12-20261020/RESULTS.md).
These are scoped software checks. Native usefulacquisition and fullrepository/
release readiness require their own evidence; all previous negatives remain.


The first separatelybudgeted nativepureBP12 case completed50controls with zero
success/nograsp after192sourceupdates; [accepted negative evidence](../artifacts/runs/official-state-taskspace12-acquisition-20261020/RESULTS.md).
A correct fixedtransform and lower imitationerror do not establish usefulcontrol.
All earlier47/42failed/negative runs and incompletecomparisons remain preserved.


A subsequent fully student correction trial collected50ALIGN labels and applied
8privateupdates; both native episodes failed. Original A TRAIN reference half-MSE
worsened .0254→.0738 while correction-batch loss fell. This does not establish
native retention or successful adaptation. [Accepted case and costs](../artifacts/runs/taskspace12-failure-state-corrections-20261020/RESULTS.md).
Optional bounded reference rehearsal is now composed through the existing stream
helper and correction runner; [API and scoped checks](correction-reference-rehearsal.md).
A separate four-reference-update/80-row trial still failed autonomous return and
worsened source A fit, despite slightly less aggregate damage than correction-only
with extra work. [Accepted negative evidence](../artifacts/runs/taskspace12-reference-rehearsal-acquisition-20261020/RESULTS.md).
Next address conventional action fit through the existing BP binary-gripper option;
no further replay-ratio/dose or capture variant is implied.
