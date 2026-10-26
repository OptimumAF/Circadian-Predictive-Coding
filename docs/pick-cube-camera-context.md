# Explicit TRAIN camera context

This R4.5 precursor adds one known camera displacement to the existing PickCube
RGB port, TRAIN teacher and frozen rollout. Its implementation increment used
fake-IO integration tests. A separately frozen follow-up now executes one native
camera acquisition-and-return case; see [results and limits](camera-context-acquisition.md).
Full domain-drift and generalization requirements remain open.

## Contract

| Property | Default | Explicit camera context |
| --- | --- | --- |
| Physical reset seed |17 (or declared TRAIN19) |17 only |
| Context |`pick-cube` for17 |`pick-cube-camera-y005-train-17` |
| Camera eye |`(.3,0,.6)` |`(.3,.05,.6)` |
| Target |`(-.1,0,.1)` |same |
| Recipe ID |`pick_cube_rgb_pool16_qpos_qvel_goal_v1` |`pick_cube_rgb_pool16_qpos_qvel_goal_camera_y005_v1` |
| Role |existing role allocation |TRAIN only |
| Features/controller |public789D, four normalized axes |same |

The single5cm lateral displacement and physical17 were frozen before frames or
outcomes in [PROTOCOL.md](../artifacts/runs/camera-drift-context-20261019/PROTOCOL.md).
The exact default recipe validator remains unchanged and rejects the new recipe.
`validate_camera_recipe(recipe, camera_context)` explicitly binds the alternate;
it does not accept arbitrary changed camera fields. Existing range-normalization
and old teacher-file loaders retain their default recipe checks. Do not relabel
their old records or silently widen those contracts to consume drift data.

## Structure and responsibilities

```text
src/core/pick_cube_camera_context.py    immutable known context + pure validation
src/infra/maniskill_continuous.py      camera factory, public frames, owned records
src/infra/maniskill_task_teacher.py    label-side geometry + qualified TRAIN data
scripts/maniskill_continuous_smoke.py  existing exclusive frozen rollout + CLI
scripts/maniskill_task_teacher_smoke.py existing teacher CLI
tests/test_pick_cube_camera_context.py  90 zero-SGD fake-IO cases
```

Core has no simulator, pixel rendering, learner, RNG or IO. Native imports remain
lazy. Context validators return fresh immutable records. The port owns its
binding and exports a fresh record; the teacher owns its result metadata and
extracts copied training data. The context/seed/role/visual mode must agree before
creation; nonvisual, TRAIN19, guard/probe/final and malformed requests refuse.

Port guards validate the binding before control, after returned native steps
before admitting RGB/records, and before episode extraction. Corruption poisons
the interaction: restoring a field cannot resume it. Completed physics remains
charged even when post-return validation fails. Teacher and frozen rollout check
exact native metadata/recipe at control boundaries and cleanup. Teacher success
qualification also binds the unique teacher ID, context, role,789D spec, action
units/bounds, sparse success and RGB range. Failed or replaced evidence cannot
release usable camera training data while any known camera marker remains.
Dropping the context binding while relabeling IDs does not bypass a retained
recipe marker. These are consistency checks, not provenance attestation against
an author who rewrites every marker and the records. Geometry remains confined
to teacher labels.

Defaults preserve old calls and CLI JSON keys. Explicit camera reports add
`camera_context` and the distinct observation recipe. Corrective queries are
refused with this context: current correction batches bind the old contexts.
No optimizer, normalization fit, budget renewal, promotion or serving swap is
introduced by camera selection.

## API

```python
from pathlib import Path
from scripts.maniskill_continuous_smoke import run_frozen_rollout
from src.core.pick_cube_camera_context import pick_cube_camera_context
from src.infra.maniskill_task_teacher import collect_task_teacher

camera = pick_cube_camera_context()
# Native calls require the separately installed pinned CPU simulator and a new
# prospective local protocol/budget. These examples were not executed here.
teacher = collect_task_teacher(visual_observation=True, camera_context=camera)
episode = teacher.training_episode()  # refuses incomplete/non-success/bad metadata
rollout = run_frozen_rollout(
    Path("fresh-camera-trace.json"), visual_observation=True, camera_context=camera
)
```

`run_frozen_rollout` also accepts the existing owned learner/actor and clock
options. Its no-learner default sends zero actions; it is not an imitation model.
For adaptation, use qualified camera episodes programmatically with the existing
stream runtime. Probe updates and deployed/benchmark learners must remain separate.

Equivalent native CLI options (not run in this increment):

```powershell
python -m scripts.maniskill_task_teacher_smoke --output fresh-camera-teacher.json --visual-observation --camera-drift
python -m scripts.maniskill_continuous_smoke --output fresh-camera-trace.json --visual-observation --camera-drift
```

The teacher CLI rejects invalid selection before opening output. Frozen rollout
reserves an exclusive output first and records invalid requests as failed reports
before model forks or native creation. Both preserve existing output files.
No new environment variable or dependency is required; existing optional pinned
Linux CPU stack requirements still apply.

## Verification and next extension

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_pick_cube_camera_context.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check src/core/pick_cube_camera_context.py src/infra/maniskill_continuous.py src/infra/maniskill_task_teacher.py scripts/maniskill_continuous_smoke.py scripts/maniskill_task_teacher_smoke.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check src/core/pick_cube_camera_context.py src/infra/maniskill_continuous.py src/infra/maniskill_task_teacher.py scripts/maniskill_continuous_smoke.py scripts/maniskill_task_teacher_smoke.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --python-version 3.12 --platform linux src/core/pick_cube_camera_context.py src/infra/maniskill_continuous.py src/infra/maniskill_task_teacher.py scripts/maniskill_continuous_smoke.py scripts/maniskill_task_teacher_smoke.py
```

[RESULTS.md](../artifacts/runs/camera-drift-context-20261019/RESULTS.md) records
the exact broader executed scope, XML/logs, one repaired initial type-check
failure, independent review, costs and skipped checks. Fake frames prove routing,
encoding and ownership; a fake SDK proves the configured pose. Neither proves
an actual rendered camera effect or task benefit.

The separate native sequence completed with a negative warm baseline and successful
private candidate/old-camera return; its bounded observations and extra memory do
not prove visual adaptation was necessary. Next implement an explicit public21
projection for a matched simpler control while preserving this RGB path. Strong/
generalizing neural R4.4 and full R4.5 lighting, object, dynamics, constraints and
recurrence remain open. See ADR-0469 and the linked native follow-up.
