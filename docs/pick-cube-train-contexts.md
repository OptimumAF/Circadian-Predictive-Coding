# Explicit PickCube TRAIN reset contexts

The existing teacher and frozen policy smoke accept two declared TRAIN reset
contexts: seed17 (`pick-cube`, the unchanged default) and seed19
(`pick-cube-train-19`). They are different initial conditions of the same task.
A context return means a fresh reset, not a continuous physical return.

## Boundaries

```text
src/core/pick_cube_train_context.py      Pure exact seed-to-context mapping
src/infra/maniskill_task_teacher.py      Existing privileged TRAIN teacher
scripts/maniskill_task_teacher_smoke.py  Existing qualification CLI
scripts/maniskill_continuous_smoke.py    Existing frozen zero/MLP/nearest actor CLI
tests/test_pick_cube_train_context.py    Seed, identity, role and failure checks
```

The pure mapper accepts only exact native integers17/19. Collectors and smoke
bind that seed to the actual port reset, episode identity, context and TRAIN
role, and report the same binding. Probe23/final29 and arbitrary seeds are
refused before scene creation. Context labels are metadata; they are not
additional learner features. The low-level port's existing interface is unchanged.

A separate [guard-only collection path](guard-only-teacher.md) now allocates31/37
under inner_guard authority. It does not expand TRAIN seeds or reuse reserved
probe/final contexts. The port reserves those two new seeds against other roles.

Why this: a second declared TRAIN context is needed to measure acquisition and
retention without relabeling a held-out probe or choosing a favorable observed
seed. Default17 teacher controls, qualification, horizon and actor ownership
guards remain. This is a narrow extension of existing collectors, not a new
stream runner or evaluation authority.

## Usage

With the separately installed optional pinned CPU simulator stack and an
appropriate prospectively frozen local budget:

```powershell
python -m scripts.maniskill_task_teacher_smoke --train-seed 19 --output fresh-teacher.json
python -m scripts.maniskill_continuous_smoke --train-seed 19 --output fresh-zero.json
```

The second command uses zero actions unless a Python caller supplies an existing
owned TRAIN learner. These commands create exclusive output files. They do not
train a learner or authorize an experiment sweep.

```python
from src.infra.maniskill_task_teacher import collect_task_teacher
from scripts.maniskill_continuous_smoke import main

result = collect_task_teacher(train_seed=19)
# main(learner=owned_train_model, train_seed=19) uses the CLI --output argument.
```

The TRAIN collector keeps keyword-only `train_seed=17` and `visual_observation=False`.
Guard collection has a separate explicit API. Invalid native API
seeds raise before collector construction; the smoke writes a failed reserved
receipt with no scene creation. The smoke validates the original keyword value
before argparse's string-default conversion can grant it authority. An explicit
CLI seed overrides a valid keyword default.

## Focused checks

```powershell
python -m pytest -q -p no:cacheprovider tests/test_pick_cube_train_context.py tests/test_maniskill_task_teacher.py tests/test_pick_cube_teacher.py
python -m ruff check src/core/pick_cube_train_context.py src/infra/maniskill_task_teacher.py scripts/maniskill_task_teacher_smoke.py scripts/maniskill_continuous_smoke.py tests/test_pick_cube_train_context.py tests/test_maniskill_task_teacher.py
python -m ruff format --check src/core/pick_cube_train_context.py src/infra/maniskill_task_teacher.py scripts/maniskill_task_teacher_smoke.py scripts/maniskill_continuous_smoke.py tests/test_pick_cube_train_context.py tests/test_maniskill_task_teacher.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --python-version 3.12 --platform linux src/core/pick_cube_train_context.py src/infra/maniskill_task_teacher.py scripts/maniskill_task_teacher_smoke.py scripts/maniskill_continuous_smoke.py
```

The fixed A17→B19→fresh A17 protocol and all executed evidence are kept in
[`continuous-context-return-20261019`](../artifacts/runs/continuous-context-return-20261019/PROTOCOL.md).
Successful lookup reconstruction alone cannot establish new-task learning,
neural plasticity, generalization or a circadian benefit. Preserve all failures
and keep probe/final data separate from training and policy selection.
