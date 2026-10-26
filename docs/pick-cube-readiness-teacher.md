# Geometry-readiness PickCube teacher

The explicit readiness teacher uses current privileged root-frame TCP/cube/goal
positions to choose its next command. It addresses the fixed teacher's inability
to wait for an off-reference student to approach before closing or lifting.
It is a conventional teaching policy, not a learned policy or a circadian feature.
The original fixed teacher and correction provenance remain available unchanged.

## Structure and boundaries

```text
src/core/pick_cube_readiness_teacher.py  pure decisions, bounded scalar state
src/infra/maniskill_task_teacher.py      existing collector/geometry/port lifecycle
scripts/maniskill_task_teacher_smoke.py  existing CLI, explicit selection
tests/test_pick_cube_readiness_teacher.py independent core and fakeIO checks
```

Core takes exact `PickCubeGeometry` and returns frozen `ReadinessDecision(stage,
action)`. The decision stage names the returned command, including when a
readiness condition changes phase in that call. At most50successful decisions;
failed validation/translation consumes no decision or close count. State stores
only total count, phase and bounded closing count. No RNG, learner, reward,
optimizer, simulator or IO in core. Existing `.1m` controller delta and per-axis
bounds are reused. Privileged geometry stays on the teacher side.

## Fixed behavior

- Align to cube position plus8cm vertical clearance until within1cm on all axes.
- Descend toward cube; lateral error over2cm restarts alignment with gripper open.
- Within1.2cm on all axes, issue six closing commands while staying within2cm.
  Losing readiness restarts alignment and resets closing count.
- Then attempt lift toward the actual cube-to-goal displacement. More than6cm
  TCP/cube separation restarts approach/open. Proximity does not prove grasp.
- Within2cm of goal, hold closed; beyond3cm, resume lift. Loss of proximity still
  restarts approach. Fixed hysteresis avoids a single shared boundary chattering.

Why this: centimetre readiness checks represent geometric approach rather than
elapsed student steps; six closing commands retain the original attempt duration.
These are fixed engineering choices, not parameters fitted to successful runs.
No contact-force sensing, collision/safety assurance or arbitrary-state expert
qualification follows. Geometry alone can miss a failed grasp.

## Use and provenance

```python
from src.core.pick_cube_teacher import PickCubeGeometry
from src.core.pick_cube_readiness_teacher import ReadinessPickCubeTeacher
from src.infra.maniskill_task_teacher import collect_readiness_teacher

teacher = ReadinessPickCubeTeacher()
decision = teacher.decide(PickCubeGeometry(
    tcp=(0.0, 0.0, 0.1), cube=(0.0, 0.0, 0.02), goal=(0.1, 0.0, 0.1)))
# decision.stage == 'descend'; action is expressed in existing root controls.

# This second example runs actual physics: requires a separate local budget.
result = collect_readiness_teacher(train_seed=17, visual_observation=True)
# Only successful final reward and complete cleanup permit training_episode().
```

```powershell
.venv/Scripts/python.exe -m scripts.maniskill_task_teacher_smoke --output fresh-readiness.json --train-seed 17 --visual-observation --geometry-readiness
```

Only default TRAIN17/19 contexts are admitted. Guard labels and mass/lighting/
camera interventions cannot be combined with this new mode. CLI refuses those
combinations before output/native construction. Reports declare
`teacher_definition=pick_cube_geometry_readiness_v1` and distinct
`pick-cube-readiness-teacher-{seed}` episode IDs. They cannot masquerade as the
old fixed schedule or be read by its original saved-teacher/correction schema.
Original CLI JSON omits the optional definition when unused; default public
collector signatures remain unchanged. Failed controls/geometry/cleanup keep
paid counters and do not qualify a training episode. There is no retry.

## Verification and remaining evidence

Scoped core/collector/CLI tests, statics and independent review are required.
[Current protocol, commands and outcomes](../artifacts/runs/readiness-teacher-20261020/RESULTS.md).
Tests with fake environments establish decisions and boundary accounting only.
New native qualification is separately frozen after implementation acceptance;
no student optimization, heldout/probe/final access or baseline comparison is
included. Existing zero-success, normalization and incomplete four-arm results
remain preserved. Neural usefulness, off-reference expert quality, adaptation,
constraints, generalization and circadian benefit remain separate requirements.

```powershell
.venv/Scripts/python.exe -m pytest -qtests/test_pick_cube_readiness_teacher.py tests/test_pick_cube_teacher.py tests/test_maniskill_task_teacher.py
.venv/Scripts/python.exe -m ruff check src/core/pick_cube_readiness_teacher.py src/infra/maniskill_task_teacher.py scripts/maniskill_task_teacher_smoke.py tests/test_pick_cube_readiness_teacher.py
.venv/Scripts/python.exe -m ruff format --check src/core/pick_cube_readiness_teacher.py src/infra/maniskill_task_teacher.py scripts/maniskill_task_teacher_smoke.py tests/test_pick_cube_readiness_teacher.py
.venv/Scripts/python.exe -m mypy --platform linux --python-version 3.12 --follow-imports=silent src/core/pick_cube_readiness_teacher.py src/infra/maniskill_task_teacher.py scripts/maniskill_task_teacher_smoke.py tests/test_pick_cube_readiness_teacher.py
```

Extend teaching behavior only through an explicit new definition and qualification
case. Do not relabel original negative results or inject new labels into old
batches. The existing learner/runtime remains the future integration boundary.

## Accepted scoped evidence (2026-10-20)

138focused tests pass on Windows and pinnedLinux; scoped lint/format/types and
independent combined implementation review accepted. Separately frozen native
TRAIN17 succeeded after20controls and TRAIN19 after22; both final sparse return1,
42controls/210knowncontrolphysics total. Independent rawtrace/resource review
accepted. This qualifies two defaultreset teacher trajectories, with no new
student training, arbitraryoff-reference/safety/learnedadaptation/CPC claim.
The existing renderer glvnd warning was retained; explicitsoftwareVulkan worked.
Native200s one-use allocation is consumed as declared, not available for retries.

Next integrate distinct provisional readiness correction records into the existing
collection/private runtime; preserve old fixed schemas and actual student actions.

## Query feedback extension (2026-10-20)

The optional `decide(..., executed_gripper=previous_student_command)` counts the
previous actual closing intention only following a grasp phase. Opencommands or
lostreadiness reset the count. Query labels are not executed commands. Default
None keeps the original selfexecuting behavior, tested again by the unchanged
50-case module; accepted two-reset native evidence reused without rerun.
[Distinct provisional correction integration](readiness-corrections.md).

## Explicit learner input route

Qualified readiness demonstrations now have an explicit route through the existing
stream and learned-robot CLIs. Use `--readiness-teachers` and retain their original
definition/IDs; do not relabel them as fixed teachers. [Input validation, commands
and learning-to-control results](readiness-demonstration-learning.md).
