# Guard-only PickCube teacher collection

The existing native teacher can now collect explicitly allocated `inner_guard`
episodes. The TRAIN collector and default CLI behavior remain unchanged. Guard
data has an owned, arrival-gated getter and cannot enter episode-based training,
standardization or the delayed TRAIN stream.

## Files and boundaries

```text
src/core/pick_cube_guard_context.py       Pure fixed guard allocation
src/core/continuous_demonstration.py      Inner_guard records; training refusal
src/infra/maniskill_continuous.py         Role/seed/context preflight, native IO
src/infra/maniskill_task_teacher.py       Shared teacher loop and owned getters
scripts/maniskill_task_teacher_smoke.py   Existing CLI, exclusive role selection
tests/test_guard_only_teacher.py         Fake IO and isolation checks
```

Dependency direction remains CLI→infra→core. App/controller/runtime/learners are
unchanged. No new runner, model, dependency, environment variable or publication
mechanism is added. Why this: actual promotable serving is implemented, but the
existing controller requires arrived disjoint guard data, which the TRAIN-only
teacher and continuous records previously could not produce.

## Allocation and API

Before observing any outcomes, this increment allocates seed31 and37 to guard
contexts `pick-cube-inner-guard-31` and `pick-cube-inner-guard-37`, with teacher
episode IDs `pick-cube-guard-teacher-31` and `pick-cube-guard-teacher-37`.
They are different same-task resets; no A/B equivalence, domain change or useful
generalization is assumed. Seeds are not selected by success. TRAIN17/19 and
reserved probe23/final29 are unchanged. The low-level port refuses31/37 under
any other role and requires the matching context for guard collection.

```python
from src.infra.maniskill_task_teacher import collect_guard_teacher, collect_task_teacher

# Existing TRAIN signature and defaults unchanged.
training = collect_task_teacher(train_seed=17, visual_observation=False)

# Requires a separately frozen native experiment budget before actual execution.
guard = collect_guard_teacher(
    guard_seed=31, labels_arrived_at=2, visual_observation=True
)
arrived = guard.guard_episode(now=2)
# guard.guard_episode(now=1) refuses before copying transition payload.
# guard.training_episode() and continuous_training_pairs(arrived) always refuse.
```

Arrival is an explicit nonnegative native logical tick, separate from simulator
or wall time. Collection stores local owner data; the getter validates `now` and
arrival before accessing/copying transition payload. It requires native success
flags, no collection/cleanup error, exact allocated seed/context/identity/role,
validated complete contiguous episode, sparse0/1 rewards ending in1, horizon≤50,
declared controller units/bounds, and any visual recipe/pixel range. Each getter
returns a fresh owned core record, revalidating trusted record mutations.
Visual acquisition keeps the existing frozen789D public RGB recipe; state mode
is unchanged. Privileged geometry supplies teacher actions only.

`TaskTeacherResult` has new default `guard_seed=None`, `labels_arrived_at=None`
fields. Guard receipts retain these metadata even on collection failure.
`qualified` retains its existing collection-success meaning; it does not mean
labels have arrived, guards are disjoint, or promotion is allowed. Existing
`training_episode()` additionally requires TRAIN and absent guard metadata.

Core records/NumPy raw arrays are trusted local owner data. These checks do not
certify source authenticity against arbitrary Python mutation, hide data from
the collecting process, or prevent an authorized caller from misusing raw arrays.
There is no automatic release to training or automatic promotion.

## Existing CLI

```powershell
python -m scripts.maniskill_task_teacher_smoke --guard-seed 31 --labels-arrived-at 2 --visual-observation --output fresh-guard.json
python -m scripts.maniskill_task_teacher_smoke --train-seed 19 --output fresh-train.json
```

These are usage examples, not authority to run an unbudgeted native experiment.
`--guard-seed` and `--train-seed` are exclusive. Arrival is required for guards
and forbidden for TRAIN. Invalid combinations fail before output reservation or
collection. Valid outputs remain exclusive, record role/seed/context/arrival,
preserve failed collection/work, and never overwrite prior evidence. The default
TRAIN17/no-argument mock call remains unchanged. No new receipt reader is added:
the existing TRAIN stream reader rejects guard role/seed/context/identity.

## Verification and next action

```powershell
python -m pytest -q -p no:cacheprovider tests/test_guard_only_teacher.py tests/test_continuous_demonstration.py tests/test_maniskill_task_teacher.py tests/test_pick_cube_train_context.py tests/test_maniskill_rgb_observations.py
python -m ruff check src/core/pick_cube_guard_context.py src/core/continuous_demonstration.py src/infra/maniskill_continuous.py src/infra/maniskill_task_teacher.py scripts/maniskill_task_teacher_smoke.py tests/test_guard_only_teacher.py
python -m ruff format --check src/core/pick_cube_guard_context.py src/core/continuous_demonstration.py src/infra/maniskill_continuous.py src/infra/maniskill_task_teacher.py scripts/maniskill_task_teacher_smoke.py tests/test_guard_only_teacher.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --python-version 3.12 --platform linux src/core/pick_cube_guard_context.py src/core/continuous_demonstration.py src/infra/maniskill_continuous.py src/infra/maniskill_task_teacher.py scripts/maniskill_task_teacher_smoke.py
```

[Executed scoped checks and independent review](../artifacts/runs/guard-only-teacher-20261019/RESULTS.md)
use fake robot IO. A subsequent [frozen native case](native-guard-promotion.md)
collected31/37 once each (28 controls/final success each); the actual controller
rejected the candidate on all three performance gates. No native committed
serving was executed. Existing evaluator/controller still require old/new guard identities
disjoint from ALL warm and actually applied training identities, arrived labels,
and the declared performance/resource policy before publication. Final/probe
data and accepted TRAIN trajectories cannot be converted into these guards.

That case is complete as a negative result; do not retry or retune it. Next address
TRAIN-only learner conditioning/learnability while keeping these guards outside
fitting/training/tuning. A rejected candidate is valid evidence; no circadian or
generalization claim follows from implementation tests or teacher qualification.
