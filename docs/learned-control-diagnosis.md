# Diagnose saved robot control

## Explicit saved Push D12 route

`--push-taskspace` compares the accepted balanced120 Push rollout with its
original18-row expert and released50-row provisional correction packet. It uses
all12 input coordinates with identity physical scaling, then reports physical7
and history5 range departures separately. Exact-input action error requires
all12 raw coordinates to match; nearest/time-aligned actions are not labels.
Correction observations retain the original student's executed history even
when their target commands differ. No new model, teacher or simulator calls.

```powershell
python -X utf8 -B -m scripts.diagnose_learned_robot_control --teacher artifacts/runs/push-on-policy-corrections-20261021/run.json --learned artifacts/runs/balanced-correction-study-20261021/run.json --push-taskspace
python -m pytest -q tests/test_diagnose_learned_robot_control.py tests/test_state_control_diagnosis.py tests/test_push_control_diagnosis.py
```

Use this saved-data command only within its reviewed local diagnostic budget.
The accepted study bytes, original released packet/receipt, TRAIN/task/seed/spec,
prior counts, actual native history, prediction captures, scores and paid work
are checked before arrays. Only equivalent spellings of the same receipt path
are resolved across platforms; all other values stay exact. The Push flag and
state flag are mutually exclusive. Defaults stay unchanged.

Why this: better first-action/expert fit did not acquire Push. The diagnostic
tests the remaining coverage/control discrepancy without another learning dose.
Distances are descriptive; this is not a corrected policy or a causal explanation.
The helper reuses the existing arithmetic in the same CLI module; new
tests/test_push_control_diagnosis.py covers Push behavior and refusals. No new
dependency, configuration or orchestration framework.
[Prospective protocol and bounded work](../artifacts/runs/push-control-gap-20261021/PROTOCOL.md).

Executed and independently accepted:122 checks/platform and scoped statics pass.
Both Push references match only reset, where maximum action error is.3428077102;
full12 support departs at step1. No action error is assigned later. One saved-data
diagnosis used no model/teacher/SDK work; original native failure remains.
[All results, commands, costs and limits](../artifacts/runs/push-control-gap-20261021/RESULTS.md).

## Existing Pick routes

The diagnostic reads an accepted qualified TRAIN17 RGB teacher and the saved
BP training/rollout result. It uses the original frozen normalization and creates
no learner, fit, gradient, RNG state or simulator episode. Inputs remain unchanged.
An explicit `--state-observation` mode accepts the original official state42 route
and uses all42 coordinates. The default RGB789 route is unchanged.

```text
scripts/diagnose_learned_robot_control.py       bounded saved-data CLI and arithmetic
tests/test_diagnose_learned_robot_control.py   synthetic behavior and admission checks
tests/test_state_control_diagnosis.py          explicit full42 behavior and refusals
```

Run from the repository root with the existing Python environment:

```powershell
python -X utf8 -B -m scripts.diagnose_learned_robot_control --teacher artifacts/runs/pick-cube-rgb-features-20261019/teacher-visual.json --learned artifacts/runs/continuous-learned-robot-20261020/native-bp-combined.json
```

This command performs descriptive array arithmetic on existing TRAIN data;
it does not repeat the spent training/native cohort. Each input is bounded to
8 MiB, each trajectory to 50 rows, selected array payload to 4 MiB and stdout to
64 KiB. Validation copies, JSON/Python containers and process RSS are additional.
There is no new dependency, configuration, environment variable or output file.

## Interpretation

Action errors require an **exact match of all 789 raw coordinates**. Conflicting
teacher labels for an identical input are reported without scoring. A nearest
public21 input or corresponding time index never supplies an expert label for
an off-trajectory state. Why this: after two trajectories diverge, the teacher's
recorded action is generally not the correct action for the learner's state.

Nearest public21 RMS distances use the original frozen TRAIN statistics. Raw
coordinate range departures and normalized magnitudes describe support; they
are neither calibrated OOD thresholds nor causal explanations. Translation
command error times the accepted controller gain of 0.1 m is a **desired delta
error**, not observed physical displacement. Invalid/nonfinite distance arithmetic
is refused before nearest-row selection. Saved metadata is bound internally;
the command does not authenticate arbitrary third-party models or teachers.

## Executed evidence

The single saved-data case found one exact match, at reset. BP's maximum action
error there was 0.418341815; desired translation error was approximately
(-0.04183418, +0.01316220, +0.00405491) m. Step 1 already leaves the stored teacher
points. Original return 0/no success remains unchanged. This establishes a
known-input prediction error before drift, without identifying its sole cause.

Both Windows and pinned Linux passed the same 44 ordered cases, with no skips.
Scoped Ruff, format and mypy checks and independent review passed. One initial
two-error type-check failure was repaired and preserved. No model or native
work ran. [Results, commands and costs](../artifacts/runs/learned-control-diagnosis-20261020/RESULTS.md)
and [independent review](../artifacts/runs/learned-control-diagnosis-20261020/REVIEW.md).

```powershell
python -m pytest -vv tests/test_diagnose_learned_robot_control.py
python -m ruff check scripts/diagnose_learned_robot_control.py tests/test_diagnose_learned_robot_control.py
python -m ruff format --check scripts/diagnose_learned_robot_control.py tests/test_diagnose_learned_robot_control.py
python -m mypy --follow-imports=silent scripts/diagnose_learned_robot_control.py tests/test_diagnose_learned_robot_control.py
```

Future analyses must keep off-trajectory labels unavailable unless separately
observed and released through the existing TRAIN correction path. This diagnostic
does not choose a probe, training recipe, promotion or final evaluation.

## Official state42 mode

```powershell
python -X utf8 -B -m scripts.diagnose_learned_robot_control --teacher artifacts/runs/state-readiness-acquisition-20261020/teacher-17.json --learned artifacts/runs/state42-fit-dose-20261020/learner-bp.json --state-observation
```

Python callers select `state_observation=True` (native bool required). Teacher and
rollout must explicitly declare no observation/render recipe or visual work, with
matching42D TRAIN17/PickCube/source/frozen-policy metadata. RGB data cannot be sliced
into this mode. Output uses state42 names and one full42 range block; public21/RGB
names remain with the default mode. The input-role/resource/finite/ownership rules
remain, and exact-label matches require all42 raw coordinates, not nearest or time.

Final84 focused cases pass on Windows and pinned Linux, scoped lint/format/types
and independent review accepted. The two separately budgeted saved analyses ran
once each with zero new model/native work. Both24- and192-dose returns remain
zero-success. Their first stored-point departures are step1; maximum frozen
normalized magnitudes282.40/4207.98. These are descriptive, not calibrated OOD or
causal explanations. Source inspection binds a TCP quaternion sign change near
the larger-dose spike, motivating a separate sign-stability representation test.
Original official42 data, metrics, receipts and34sampled larger-dose joint
violations are preserved. No new training recipe or native permission follows.
[Commands, fixture costs, findings and tool failures](../artifacts/runs/state42-control-diagnosis-20261020/RESULTS.md)
· [Independent acceptance](../artifacts/runs/state42-control-diagnosis-20261020/REVIEW.md).

```powershell
python -m pytest -vv tests/test_diagnose_learned_robot_control.py tests/test_state_control_diagnosis.py
python -m ruff check scripts/diagnose_learned_robot_control.py tests/test_state_control_diagnosis.py
python -m ruff format --check scripts/diagnose_learned_robot_control.py tests/test_state_control_diagnosis.py
python -m mypy --platform linux --python-version 3.12 --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL scripts/diagnose_learned_robot_control.py tests/test_state_control_diagnosis.py
```

Extend through explicit observation contracts and keep default outputs stable.
Do not reinterpret the state distance as physical motion or a corrective oracle.
