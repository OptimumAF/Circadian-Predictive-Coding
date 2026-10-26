# Explicit gripper execution

Frozen neural serving can optionally map the gripper prediction to an endpoint:
negative values select `-1`; nonnegative values, including either signed zero,
select `+1`. XYZ stays unchanged until the existing float32 native conversion.
The rule acts on the original float64 prediction, before rounding.

Soft gripper targets are valid for the SDK. This option is an explicit action
representation hypothesis. It is separate from BP's `binary_gripper_axis` loss;
PC/CPC continue to reject that mixed loss.

## Use the existing rollout

```python
from pathlib import Path
from scripts.maniskill_continuous_smoke import run_frozen_rollout
from src.core.pick_cube_action_decoding import PickCubeBinaryGripperRecipe
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe

outcome = run_frozen_rollout(
    Path("new-decoded-trace.json"),
    learner=owned_train_learner,
    action_history_recipe=PickCubeTaskSpaceHistoryRecipe(mode="executed"),
    action_decoding_recipe=PickCubeBinaryGripperRecipe(),
    monitor_joint_positions=True,
)
```

Use an exact common-MSE BP, PC, neutral CPC or active CPC learner trained on the
same executed taskspace12 recipe. Selected serving supports the default Panda
TRAIN17/19 contexts without actors, queries, roll-in, clock, visual, projection
or context interventions. Other combinations refuse before simulator creation.
This is a programmatic option; the CLI has no new flag. Omitting the recipe keeps
the old soft commands, report fields and call shape.

`action_decoding` records the prediction, selected command and returned native
action separately for at most 50 controls. History advances from actual returned
actions. A returned control remains paid if later validation fails; corruption
prevents more work. `action_decoding_exact` confirms the completed execution
checks, not task success. The existing learner-to-rollout helper forwards the
same optional recipe and preserves source ownership.

## Structure and checks

```text
src/core/pick_cube_action_decoding.py   fixed rule and input validation
src/app/action_decoding.py             bounded pending/paid receipts
scripts/maniskill_continuous_smoke.py   existing frozen serving boundary
scripts/run_continuous_robot_learning.py  existing source-to-rollout forwarding
tests/test_pick_cube_action_decoding.py   pure mapping and receipt behavior
tests/test_gripper_decoding_integration.py  four-family fake native boundary
```

```powershell
python -m pytest -q tests/test_pick_cube_action_decoding.py tests/test_gripper_decoding_integration.py
python -m ruff check src/core/pick_cube_action_decoding.py src/app/action_decoding.py scripts/maniskill_continuous_smoke.py scripts/run_continuous_robot_learning.py tests/test_pick_cube_action_decoding.py tests/test_gripper_decoding_integration.py
python -m ruff format --check src/core/pick_cube_action_decoding.py src/app/action_decoding.py scripts/maniskill_continuous_smoke.py scripts/run_continuous_robot_learning.py tests/test_pick_cube_action_decoding.py tests/test_gripper_decoding_integration.py
python -m mypy --platform linux --follow-imports silent src/core/pick_cube_action_decoding.py src/app/action_decoding.py scripts/maniskill_continuous_smoke.py scripts/run_continuous_robot_learning.py tests/test_pick_cube_action_decoding.py tests/test_gripper_decoding_integration.py
```

The frozen software gate passed 39 checks on each platform with independent
acceptance. [Commands, failures, costs and scope](../artifacts/runs/gripper-action-decoding-20261020/RESULTS.md).

One separately frozen native BP check remained unsuccessful in 50 controls.
Every gripper prediction was positive, so all commands were open. The complete
negative result preserves the earlier soft control and establishes no CPC or
general control benefit. [Native evidence](../artifacts/runs/gripper-decoded-control-check-20261020/RESULTS.md).

Add future execution modes as explicit recipes with their own ownership,
history, paid-failure and finite-cost tests. Freeze a separate protocol before
model comparisons; never silently alter a saved baseline or tune this rule from
probe-test or final results.
