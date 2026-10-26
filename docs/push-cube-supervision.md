# PushCube TRAIN supervision

Software and one frozen native TRAIN-17 supervision trial are independently
accepted: 48 focused checks pass on each Windows/Linux runtime; scoped lint,
format and type checks pass. The teacher succeeded in 18 controls and released
18 qualified training rows (90 known control-physics steps, 19 grasp queries,
38 source-derived force calls). Original failed fixtures and repairs remain
recorded in the [results](../artifacts/runs/push-cube-supervision-20261021/RESULTS.md).
The fixed `push_cube_closed_fist_xy_v1` policy uses the qualified PushCube port's
named physical state and actual executed action history. It is a hand-written
supervisor, not a learned policy or evidence of adaptation.

The teacher approaches six centimetres behind the cube along the current XY goal
direction, waits at a hover until six actual closing commands have returned,
descends and pushes horizontally. Commands are bounded at half the normalized
translation range. Lost alignment or closing history triggers recovery. A hold
phase within eight centimetres of the XY goal does not certify native success.
The full fixed recipe and its arithmetic are in the source and protocol; no tuning
parameters are exposed.

```python
from src.infra.maniskill_push_teacher import collect_push_cube_teacher

result = collect_push_cube_teacher(seed=17)  # constructs a real TRAIN scene
if result.error is None and result.final_success:
    training = result.training_episode()   # owned existing ContinuousEpisode
```

The example spends native work and needs a separately budgeted collection.
Only TRAIN seeds 17, 19 and 20 are admitted; guard/probe/final roles are refused
before scene construction. Every returned command is retained at its actual
float32 precision. Queried labels never advance the executed closing history.

The result retains bounded physical teacher inputs, decisions, valid returned
transitions, attempted/observed returned commands and permanent work, including
partial failures. Complete negative episodes remain available as evidence but
`training_episode()` refuses them. Export requires error-free completion, final
native TRAIN success, matching task/role/spec/labels/history and exact native work.
It does not re-query the teacher or convert evaluation roles.

```text
src/core/push_cube_teacher.py            pure fixed geometry and bounded decisions
src/infra/maniskill_push_teacher.py      existing port → supervision result
tests/test_push_cube_teacher.py         pure behavior and fake acquisition checks
```

```powershell
python -m pytest -q tests/test_push_cube_teacher.py
python -m ruff check src/core/push_cube_teacher.py src/infra/maniskill_push_teacher.py tests/test_push_cube_teacher.py
python -m ruff format --check src/core/push_cube_teacher.py src/infra/maniskill_push_teacher.py tests/test_push_cube_teacher.py
python -m mypy --platform linux --follow-imports silent src/core/push_cube_teacher.py src/infra/maniskill_push_teacher.py tests/test_push_cube_teacher.py
```

Why this: the accepted continuation adapter already consumes complete same-spec
TRAIN records. A distinct physical task needs its own supervision rather than
relabeled PickCube data. Use the existing continuation and stream runner with old
references for subsequent learning/retention/recovery studies. No new runner,
learner, initializer, optimizer, probe or promotion is introduced here.

[Protocol, budgets and validation artifacts](../artifacts/runs/push-cube-supervision-20261021/PROTOCOL.md).
