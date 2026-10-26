# Bounded readiness teacher roll-in

The existing TRAIN query rollout accepts an explicit fixed teacher prefix, then
returns control to the frozen student. This addresses the observed ALIGN-only
student data; it does not promise a successful learned policy. Mixed behavior
success is explicitly unavailable as a student-only success claim.

```text
src/core/pick_cube_readiness_roll_in.py   fixed prefix and native command choice
src/app/readiness_roll_in.py             bounded attempted/returned control receipts
src/infra/maniskill_corrections.py        owned latest readiness label, no new query
scripts/maniskill_continuous_smoke.py     existing query/update/frozen API composition
 tests/test_readiness_roll_in.py          independent behavior and failure tests
```

## Fixed student prefix followed by teacher completion

The existing API accepts a distinct plan alongside observed-v2 queries:

```python
from pathlib import Path
from scripts.maniskill_continuous_smoke import run_frozen_rollout
from src.core.pick_cube_readiness_roll_in import ReadinessTeacherCompletionPlan
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe

# A separately budgeted run; source is an already prepared exact BP learner.
mixed = run_frozen_rollout(
    Path("fresh-mixed.json"), learner=source, train_seed=17,
    corrective_queries=True, readiness_corrections=True,
    observed_grasp_corrections=True, goal_consistent_corrections=True,
    action_history_recipe=PickCubeTaskSpaceHistoryRecipe(mode="executed"),
    readiness_roll_in=ReadinessTeacherCompletionPlan(student_steps=12),
)
```

The prefix is a native integer1..49 and switches by declared step, never by
success, contact or loss. The selected composition requires BP, nonvisual
executed taskspace12 and observed-v2 readiness; old teacher-prefix behavior is
preserved. The convenience CLI flag is
`--readiness-teacher-completion-student-steps`, mutually exclusive with
`--readiness-roll-in-steps`; main still requires the caller's prepared learner.
No model loader or new environment variable was introduced.

Every step computes and pays for the student proposal and teacher query. Both
original commands remain separate from the exact float32 action actually sent.
Only actual commands advance history. `teacher_takeover_returned` becomes true
only after a teacher control returns; this mode has no `handed_back` field.
Mixed policy metadata sets `student_only_success_available=False`. Early end,
failed teacher calls and returned controls followed by observer failures retain
actual attempted/returned costs. Changed original requests or prior full receipts
refuse more work; a failed receipt sequence cannot resume with a rewritten report.

The same plan may be passed to `run_corrective_adaptation`; it is forwarded only
to collection. The existing private update consumes the original provisional raw
query batch, and the subsequent frozen student return receives neither completion
nor query options. Existing independently budgeted reference replay remains
available; this increment adds no new replay-execution claim. A mixed episode is
not a qualified teacher artifact and must not enter a warm-teacher reader.

Why this: earlier student-only data had no SETTLE examples, and reweighting the
reset demonstration did not solve the task. The fixed native case completed with
12 student +4 teacher controls and one SETTLE label, but eight private updates
still produced a failed pure return and worsened both original reference fits.
Teacher-assisted recovery is demonstrated at that known reset; autonomous learning
is not. All100 focused cases/platform and scoped statics passed with independent
acceptance. [Software checks/costs](../artifacts/runs/student-prefix-teacher-completion-20261020/RESULTS.md) ·
[Native outcomes/costs](../artifacts/runs/student-prefix-teacher-completion-native-20261020/RESULTS.md).

## Original teacher-prefix contract and use

```python
from pathlib import Path
from scripts.maniskill_continuous_smoke import run_frozen_rollout, run_corrective_adaptation
from src.core.pick_cube_state_quaternions import state_quaternion_projection
from src.core.pick_cube_readiness_roll_in import ReadinessRollInPlan

# Declare a fresh local model/native allowance before actually running either.
result = run_frozen_rollout(Path("fresh-mixed.json"), learner=source,
    corrective_queries=True, readiness_corrections=True,
    observation_projection=state_quaternion_projection(),
    readiness_roll_in=ReadinessRollInPlan(12))
# Or the existing private-update composition; ONLY collection receives the prefix.
summary = run_corrective_adaptation(source, output_directory=Path("fresh-mixed-learning"),
    visual_observation=False, readiness_corrections=True, acquisition_only=True,
    observation_projection=state_quaternion_projection(),
    readiness_roll_in=ReadinessRollInPlan(12))
```

Plan steps must be a native integer1..49. The original teacher-prefix mode
requires nonvisual readiness queries with an explicit canonical42 projection or
an already supported action-history view, without actor, clock or interventions.
Its older geometry-only supervision remains separate from the new observed-v2
completion mode.
Omitted/None adds no default kwargs or output fields. CLI convenience:
`--readiness-roll-in-steps 12` alongside the programmatically supplied learner and
existing readiness/canonical options; main still refuses missing prerequisites.
No environment variable/dependency/framework was added.

Every control still pays for a student prediction, including teacher-prefix
controls. Original float64 student proposal/teacher target remain in each receipt;
executed_action is the exact selected IEEEfloat32 command sent/stored by the
existing native port. Conversion validates BOTH commands/bounds first, without
clipping or a tolerance. Core chooses once and ledger independently revalidates:
eight scalar float32 conversions per controlattempt, not four total. Extra copies/
validation/serialization are not model inference. The label accessor re-owns the
alreadyobserved label, with no new geometry/decision/counter work. Any accessor
refusal poisons its sequence; prior paid evidence survives.

Per-control receipts bind raw observations/actions/targets and teacher/student
source, attempted and returned status. At most50 records. handed_back means at
least one STUDENT control actually RETURNED; crossing the prefix, an attempted
failed student control, or termination during teacherprefix is insufficient.
All later phase/closing progress uses preceding EXECUTED gripper commands.

Native policy/success metadata names mixed behavior and sets
student_only_success_available=False. The driver binds the requested prefix and
mixed collection metadata before updates. Candidate return receives NO roll-in or
query option, requires family-correct frozen policy and refuses mixed metadata.
Its pure-student result is separate from collection success. Raw roles, arrivals,
source/actor/statistics/frozen state, budgets/clocks/receipts/charges remain.
No replay/probe/final/guard/promotion/deployment occurs.

## Checks and evidence

Final72focusedcases/platform pass. Exact selected pytest argv/costs/failures are in
[TEST_COMMANDS](../artifacts/runs/readiness-roll-in-20261020/TEST_COMMANDS.json) and
[budget](../artifacts/runs/readiness-roll-in-20261020/FIXTURE_BUDGET.md). Final scoped
commands (same five paths for Ruffformat--check and mypy):

```powershell
.venv/Scripts/python.exe -m ruff check src/core/pick_cube_readiness_roll_in.py src/app/readiness_roll_in.py src/infra/maniskill_corrections.py scripts/maniskill_continuous_smoke.py tests/test_readiness_roll_in.py
.venv/Scripts/python.exe -m ruff format --check src/core/pick_cube_readiness_roll_in.py src/app/readiness_roll_in.py src/infra/maniskill_corrections.py scripts/maniskill_continuous_smoke.py tests/test_readiness_roll_in.py
.venv/Scripts/python.exe -m mypy --platform linux --python-version 3.12 --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL src/core/pick_cube_readiness_roll_in.py src/app/readiness_roll_in.py src/infra/maniskill_corrections.py scripts/maniskill_continuous_smoke.py tests/test_readiness_roll_in.py
```

Extend the existing selector and collector for further bounded behavior rules;
keep control provenance/actual native precision explicit and test handoff/failure
before declaring another local native protocol. Generalization, reliability,
safety, autonomous acquisition and CPC benefit remain separate open requirements.

## Accepted native outcome

One frozen TRAIN17 case returned12teacher/38student controls with32ALIGN/4DESCEND/
6GRASP/8LIFT query labels; pure updated student still scored zero after8updates.
Mixed behavior is never autonomous success. [Exact commands, costs, negative and
next dependency](../artifacts/runs/readiness-roll-in-acquisition-20261020/RESULTS.md).
