# Reserved robot probe inputs

The existing native teacher collector supports a fixed held-out reset split:

| Role | Reset | Episode ID |
| --- | ---: | --- |
| probe_train |23|pick-cube-probe-train-teacher-23|
| probe_test (score-only) |24|pick-cube-probe-test-teacher-24|

Both use context `pick-cube-reserved-probe-23-24`. Reset23 was already reserved;
adjacent24 was added prospectively before observation to avoid scoring on the
same deterministic trajectory used for probe updates. Final reset29 is blocked
under every native role. Reset24 is blocked under every role except probe_test.
Generic caller-declared probe17 and probe_test23 APIs remain available; they do
not qualify as this fixed reserved split. This is reset adaptation within PickCube,
not a new physical task or a demonstration that a learner succeeds.

## Collection and extraction

The existing teacher loop runs the goal-consistent observed-grasp-v2 policy with
raw42 observations and at most50controls. It preserves native32 actual actions,
original role/seed/context/identity, current flags/stages and all paid work. Its
distinct `ReservedProbeTaskTeacherResult` can release only its original probe
episode after complete final success. TRAIN/guard extraction and erased family
markers refuse. Failed collections retain their outcomes and costs; never replace
a failed reset with a favorable one.

These commands spend simulator work and require a prospective local protocol:

```powershell
./.venv/Scripts/python.exe -m scripts.maniskill_task_teacher_smoke --output fresh-probe-train.json --probe-role probe_train --geometry-readiness --observed-grasp --goal-consistent
./.venv/Scripts/python.exe -m scripts.maniskill_task_teacher_smoke --output fresh-probe-test.json --probe-role probe_test --geometry-readiness --observed-grasp --goal-consistent
```

Python composition uses `collect_probe_teacher(role=...)` from
`src.infra.maniskill_task_teacher`; `result.probe_episode()` revalidates the
qualified role-preserving result. An ordinary unsuccessful result stays
unqualified without being mislabeled as a substituted request. Existing TRAIN,
guard and teacher-prefix paths keep their schemas and admission rules.

## Existing probe runner

For a taskspace12 benchmark, use explicit `action_history_probe_episode(raw,
PickCubeTaskSpaceHistoryRecipe(mode="executed"))`. The encoder preserves role,
ID, context, rewards/end flags and actual commands; history resets per episode.
`validate_action_history_probe_episode` checks encoded command history. It cannot
recover dropped raw42 values, so keep the qualified raw/model binding. TRAIN17/19
encoding remains separately strict. Existing47D history recipes also work.

Pass the two model-space episodes to existing `ContinuousProbePlan`, then the
existing stream's `probe_plan` argument. The existing delayed-label runtime uses
disposable probe updates and score-only probe_test data, separate from benchmark
and deployed learners. Constructing a plan performs no learning or scoring.
Qualified episode-asdict JSON can use the existing generic probe reader; reading
arbitrary role-labeled JSON by itself does not prove reserved collection.

No new learner, loader, runner or dependency. `core/pick_cube_probe_context.py`
owns literal allocation; existing core history code owns projection; native
collector owns IO/qualification; existing stream/probe adapters own learning.
Why this: provide genuinely separated inputs to the already implemented probe
without weakening TRAIN/final boundaries or relabeling benchmark data.

## Checks and evidence

```powershell
./.venv/Scripts/python.exe -m pytest -vv --maxfail=1 tests/test_reserved_robot_probe.py tests/test_maniskill_reserved_resets.py
./.venv/Scripts/python.exe -m ruff check src/core/pick_cube_probe_context.py src/core/pick_cube_action_history.py src/infra/maniskill_continuous.py src/infra/maniskill_task_teacher.py scripts/maniskill_task_teacher_smoke.py tests/test_reserved_robot_probe.py
./.venv/Scripts/python.exe -m ruff format --check src/core/pick_cube_probe_context.py src/core/pick_cube_action_history.py src/infra/maniskill_continuous.py src/infra/maniskill_task_teacher.py scripts/maniskill_task_teacher_smoke.py tests/test_reserved_robot_probe.py
./.venv/Scripts/python.exe -m mypy --follow-imports silent src/core/pick_cube_probe_context.py src/core/pick_cube_action_history.py src/infra/maniskill_continuous.py src/infra/maniskill_task_teacher.py scripts/maniskill_task_teacher_smoke.py tests/test_reserved_robot_probe.py
```

[Scoped software evidence and failures](../artifacts/runs/reserved-robot-probe-input-20261020/RESULTS.md)
and [separately budgeted native collection](../artifacts/runs/reserved-robot-probe-input-native-20261020/RESULTS.md)
distinguish implementation, executed behavior, independent acceptance and remaining
learning claims. Extending the allocation requires a new prospective split;
never select it using probe-test or final outcomes.
