# Saved camera TRAIN teachers in the native stream CLI

The existing stream CLI accepts the one declared TRAIN17 camera displacement.
It reads a qualified saved teacher, preserves acquisition costs, and feeds owned
TRAIN pairs to the existing nearest or continuous learnability implementation.
It creates no simulator and does not turn saved data into a new native result.

```powershell
.venv/Scripts/python.exe -m scripts.run_native_stream_demo --scenario continuous-nearest --warm-teacher artifacts/runs/pick-cube-rgb-features-20261019/teacher-visual.json --new-teacher artifacts/runs/camera-context-return-20261019/teacher-b.json --visual-observation --camera-drift
```

## Data and evaluation boundary

Warm A remains default TRAIN17. B is the declared `pick-cube-camera-y005-train-17`
teacher, with its own complete recipe: camera eye moves laterally by5cm, and the
789D public qpos/qvel/goal/RGB feature order stays fixed. It is not a seed19
episode. Both `input_observation.warm_recipe` and `.new_recipe` are reported;
the CLI never labels both inputs as using the default camera.

`_read_camera_teacher` reuses `TaskTeacherResult.training_episode` qualification,
the existing bounded episode decoder and RGB lifecycle validator. The camera
qualification also requires retained paid acquisition fields to match: one
creation, one geometry query/action per transition, controller physics5per action,
the fixed teacher stage sequence, one encoding per observed state and no frame
exports. These are historical acquisition costs, not new simulator work.

Why this: existing native camera and tuple-projection support lacked the saved
input route used by ordinary stream commands. This adds that boundary to the
existing script, without a source codec, runtime, runner or qualification system.

Before fitting, admission rejects heldout/probe/final/guard teacher roles,
guard-arrival markers, mixed mass/lighting declarations, changed or incomplete
camera contexts/recipes, duplicate/nonfinite JSON, inconsistent paid work and
receipts above4MiB. Default, mass and lighting routes retain their original
semantics. Separate reserved probe inputs still use the existing isolated
ContinuousProbePlan rules; they are not relabeled camera teachers or final data.

## Public-vector and model options

For the existing MLP learnability scenario, append `--public-vector-only` to use
the existing public21 projection. It validates the full raw A/base and B/camera
records before discarding pixels. `input_projection.source_recipe` identifies
warm A and `.new_source_recipe` identifies B. No new projection is introduced.

Existing learner-family, phase-update, A-only, gripper and range-normalization
options keep their original scope. Camera drift requires a visual continuous
scenario and excludes simultaneous mass/light drift selection. A saved-stream
run returns delayed-update correctness/readouts; it is not a native camera return
or evidence of generalization, physical safety, or a circadian advantage.

## Files and checks

Only the existing script changes; no new production module or dependency:

```text
scripts/run_native_stream_demo.py     camera admission, CLI routing, recipe metadata
tests/test_camera_stream_input.py     bounded input, isolation, projection and CLI tests
```

```powershell
.venv/Scripts/python.exe -m pytest tests/test_camera_stream_input.py -q
.venv/Scripts/python.exe -m ruff check scripts/run_native_stream_demo.py tests/test_camera_stream_input.py
.venv/Scripts/python.exe -m ruff format --check scripts/run_native_stream_demo.py tests/test_camera_stream_input.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent scripts/run_native_stream_demo.py tests/test_camera_stream_input.py
```

Additional contexts require their own declared qualification and recipe; do not
generalize this flag to arbitrary camera metadata. Existing callers can use the
owned episode with the bounded sequence API. Native studies need a separate small
prospective protocol, preserving the accepted cumulative mass/lighting evidence.

[Protocol, exact selected checks and allowances](../artifacts/runs/camera-stream-input-20261020/PROTOCOL.md)
and [independent review](../artifacts/runs/camera-stream-input-20261020/REVIEW.md).

## Accepted cumulative camera case

The separate frozen native case reuses four saved29-row TRAIN teachers and the
existing sequence API. Camera before release fails50controls/return0; after its
arrived29 labels, camera succeeds29/1, mass30/1, lighting29/1, A29/1. Three private
updates at2/5/8 consume one birthbudget3; source/actor remain unchanged, candidate
has116rows/4publications, nexttick9. No final/probe labels or serving promotion.

167controls/835knownphysics,172sampledjointstates/zero violations; prior lighting
failure's eight sampled violations and mass no-gain remain historical evidence.
This and the earlier mass/light curve meet original R4.5 bounded known-TRAIN
acceptance, not unseen-context/neural/physical-safety/CPC benefit. They are separate
paid cases, not one uninterrupted13-scene invocation. Stage-only composition and
metadata tests add no new production layer:

```text
artifacts/runs/camera-stream-input-20261020/camera_case.py   frozen native composition
tests/test_cumulative_camera_case.py                       metadata/stub integration
```

Final scoped checks also cover both files above; exact platform argv/receipts,
source snapshots, budgets, failures, resource accounting and original criterion
map are in [results](../artifacts/runs/camera-stream-input-20261020/RESULTS.md) and
[review](../artifacts/runs/camera-stream-input-20261020/REVIEW.md). One-use offline
and native allowances are spent. No new run is implied by these instructions.

Next extend the existing R4.3 recurrent quota route with a paired trained/frozen
comparison, preserving each policy's own trajectories and deferred final access.
Freeze and review a small repeated-instance protocol before numerical execution.
