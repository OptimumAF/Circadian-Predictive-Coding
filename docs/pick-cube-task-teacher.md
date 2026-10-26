# Fixed PickCube task-space TRAIN teacher (R4.4)

The teacher provides an explicit alternative to the official joint planner,
whose fixed50-step source qualification earned no sparse success. That negative
and all earlier SDK failures remain preserved. The task, four-axis controller,
action bounds, seed17 and50-step limit are unchanged. There is no new runner,
dependency, learned policy, BC update or comparative experiment here.

The shared collector now also has an explicit [guard-only path](guard-only-teacher.md)
with fixed31/37 allocation and arrival-gated records. This original TRAIN API and
policy remain unchanged; guard data cannot be extracted for BC training.

```text
src/core/pick_cube_teacher.py          Pure root-frame geometry and fixed policy
src/infra/maniskill_task_teacher.py    TRAIN geometry/readback and bounded collection
scripts/maniskill_task_teacher_smoke.py Exclusive-output fixed qualification CLI
tests/test_pick_cube_teacher.py       Pure policy geometry/schedule tests
tests/test_maniskill_task_teacher.py   Boundary/authority/failure tests
```

## Frozen policy

Positions are controller-root coordinates in metres. Translation commands are
`clamp(position_error / 0.1m, -1, 1)` independently per axis. This clipping is an
explicit teacher policy; the existing port does not clip or relabel commands and
still records the actual float32 action sent. Orientation remains fixed.

| Steps | Stage | Position error | Gripper |
| --- | --- | --- | --- |
| 0–7 | Align | cube center +.08m root Z minus TCP | +1 open |
| 8–15 | Descend | cube center minus TCP | +1 open |
| 16–21 | Grasp | zero | -1 closed |
| 22–39 | Lift | goal minus actual cube center | -1 closed |
| 40–49 | Settle | zero | -1 closed |

Why this: a simple controller-aligned teacher is the missing data prerequisite
after runtime compatibility passed and the official joint-planning teacher was
unsuccessful. The settle period addresses the task's static-robot success check.
The schedule/offsets were frozen before native execution, with no reward/metric
feedback in the policy and no seed selection. It is an engineering candidate,
not evidence of skill until the real qualification succeeds.

The pure policy stores only its call count. Invalid geometry does not advance
it; calls stop at50. Geometry uses exact finite three-coordinate native tuples,
revalidates owned inputs, and refuses unrepresentable position error. No physical
robot or arbitrary task/controller compatibility is implied.

## Geometry, data and work authority

The infra collector creates exactly one TRAIN seed17 episode using the existing
port. It transforms TCP, cube and goal poses through `robot.pose.inv()` before
extracting positions. No object teleport, alternate dynamics, native step bypass,
MPlib planner, replay, probe or final-data access is used. Privileged actor points
are never a deployed learner interface; the learner still receives the existing
42-coordinate state observation when actual data is released separately.

Each geometry attempt is one grouped read of three actor points, with one root
inverse and three pose compositions. Attempted/returned groups are separate
from teacher actions and actual native control calls. Storage is bounded to50
stage labels plus the existing50-transition episode. Constructor/reset physics
remain unknown; completed/possibly failed control physics retain port accounting.
The constructor seed2022 reset and explicitseed17 reset are unchanged.

```python
from src.infra.maniskill_task_teacher import collect_task_teacher

result = collect_task_teacher()  # only under an admitted local native budget
if result.qualified:
    episode = result.training_episode()
    # A separate frozen BC protocol may release this episode to the existing
    # NumpyContinuousBCLearner.train_episode API; collection never trains it.
```

Qualification requires complete valid target data, final sparse reward1, and no
collection/validation/cleanup error. Any-step success is reported separately.
`candidate_episode` can preserve a complete unsuccessful trace; it is not
certified expert data. `training_episode()` refuses unqualified data and returns
a revalidated owned immutable episode. No role/seed/tuning option is exposed.
Exceptions preserve counters/trace; malformed outputs poison the existing port.

## Running and extending

Use the accepted isolated WSL CPU profile/software Vulkan configuration in
[the port guide](maniskill-continuous.md). The new teacher does not depend on the
MPlib constructor, but the fixed qualification uses the existing compatible
NumPy1 profile so later BC validation can use the same numerical stack.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.maniskill_task_teacher_smoke --help
.venv/Scripts/python.exe -m pytest -q tests/test_pick_cube_teacher.py tests/test_maniskill_task_teacher.py tests/test_maniskill_continuous.py tests/test_continuous_demonstration.py
.venv/Scripts/python.exe -m ruff check src/core/pick_cube_teacher.py src/infra/maniskill_task_teacher.py scripts/maniskill_task_teacher_smoke.py tests/test_pick_cube_teacher.py tests/test_maniskill_task_teacher.py
.venv/Scripts/python.exe -m ruff format --check src/core/pick_cube_teacher.py src/infra/maniskill_task_teacher.py scripts/maniskill_task_teacher_smoke.py tests/test_pick_cube_teacher.py tests/test_maniskill_task_teacher.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --python-version 3.12 --platform linux src/core/pick_cube_teacher.py src/infra/maniskill_task_teacher.py scripts/maniskill_task_teacher_smoke.py
```

The CLI reserves fresh output before simulator work; flushed boundary markers
and faulthandler preserve future fatal location evidence when possible. Exit0
requires qualification. Simulator attempts require a frozen local protocol;
the [current protocol](../artifacts/runs/pick-cube-task-teacher-20261019/PROTOCOL.md)
admits one qualification only, after focused checks and independent review.
Do not extend horizons, choose favorable seeds, erase unsuccessful candidates or
tune baselines to manufacture a circadian advantage. Strong BC, retained skill
and useful adaptation remain separate requirements. Extend through the existing
ContinuousEpisode/NativeLearner/stream boundaries after real data qualification.

## Executed evidence — 2026-10-19

288focused cases and scoped statics pass. Independent native review accepts one
complete29-stepTRAINdemonstration ending sparse success/termination (145known
physics steps); the50-call ceiling and fixedstage policy remain unchanged.
The separate frozen BCpilot published64updates/1856TRAINrows and lowered TRAIN
normalized half-MSE.265632→.062273. Both untrained and trained frozenprobe_test
seed23 policies completed50steps with zero return/finalsuccessfalse. Exactinitial
observations/fullfrozenmodelstates checked; no taskskill/CPCbenefit established.
Allcurvepoints/updates/rollouts/costs retained; no retries/tuning.
[Exact commands, artifacts, limits and next learner diagnostic](../artifacts/runs/pick-cube-task-teacher-20261019/RESULTS.md).
