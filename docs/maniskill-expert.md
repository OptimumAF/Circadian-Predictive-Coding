# Official PickCube expert conversion (R4.4)

## Verified compatibility profile and actual negative — 2026-10-19

The MPlib0.1.1/NumPy2 profile can terminate in native ArticulatedModel creation.
The collector now refuses that combination before creating a scene. An executed
production CLI refusal records0creations/resets/controlcalls and actionableerror.
The original fatal attempt remains UNKNOWN in progress; a separate diagnostic
locates its own fatal after successful source creation/reset at native constructor.

The separate verified overlay uses Python3.12, NumPy1.26.4, SciPy1.15.3,
OpenCV4.11.0.86 and contourpy1.3.3; unchanged ManiSkill3.0.1/SAPIEN3.0.3/
MPlib0.1.1/Torch2.14.0+cpu come from the accepted prior Linux site via .pth.
Original NumPy2 prefix remains intact. NumPy alone could not be replaced because
the prior SciPy/OpenCV/contourpy metadata requires NumPy2. Exact four wheels and
SHA256s are in stage `numpy1-wheels.txt`; use --ignore-installed --no-deps
--require-hashes only in the separate owned prefix, never uninstall shared packages.
The source setup script and actual origins/pipcheck receipts record the full setup.
This local read-only reuse convention is not a clean-clone or OS isolation claim.

Actual constructor cut PASS5.637896318s. Full expert qualification then returned
an honest negative:50source steps,250control physics, no sparse success; replay
and target construction refused, no expert BC data. No horizon/seed/planner tuning.
The compatible four-package profile passes; unique causal attribution to NumPy
alone is not established. A matching primary report motivates the repair:
[ManiSkill issue1100](https://github.com/mani-skill/ManiSkill/issues/1100),
[NumPy compatibility guidance](https://numpy.org/doc/stable/user/troubleshooting-importerror.html).

For a separately budgeted future run, apply the same process-local CPU software
Vulkan/thread settings in the port guide, then use the verified interpreter:

```bash
/home/avery/.cache/circadian/maniskill-expert-numpy1-20261019/env/bin/python -X utf8 -B -m scripts.maniskill_expert_smoke --output fresh-expert.json
```

305focused cases and scoped statics pass. Full-lifetime tree memory remains
UNVERIFIED. This is runtime/boundary qualification, not successful teacher/BC
training or robot adaptation. Preserve unsuccessful data and all original failures.
[Exact commands, native costs and independent review](../artifacts/runs/maniskill-expert-diagnosis-20261019/RESULTS.md).

**Next implementation action:** Implement a fixed bounded TRAIN-only task-space PickCube teacher for the existing four-axis end-effector controller, using privileged training geometry and explicit align/descend/grasp/lift stages. Keep the same task, bounds, seed17 and50-step acceptance; test geometry/role/terminal/work boundaries and freeze/review one qualification before simulator execution. Preserve the official planner's failed qualification; do not tune seeds, extend the horizon, or train BC on unsuccessful data.

This training-only collector connects the bundled official Panda expert to the
existing four-axis continuous port. It adds no learner or experiment runner.
Native qualification and scoped checks are recorded in the
[local protocol/evidence](../artifacts/runs/maniskill-expert-bridge-20261019/PROTOCOL.md).
R4.4 strong BC, useful adaptation and release remain unfinished.

```text
src/infra/maniskill_expert.py       Bounded reference/conversion and qualification
scripts/maniskill_expert_smoke.py  One fixed TRAIN qualification; exclusive output
tests/test_maniskill_expert.py     Independent doubles for costs/failures/authority
```

## Why two controllers

The official expert plans Panda joint positions. These seven radian coordinates
plus normalized gripper are not valid labels for normalized root-frame dx/dy/dz
and gripper. A separate reference scene collects at most50actual joint commands;
the official converter replays a same-seed prefix and computes commands for the
existing `pd_ee_delta_pos` target. Every target step passes through the existing
port and its validation/quantization/termination checks. Only those actual
four-axis commands can become continuous BC labels. The deployed controller,
fixed orientation, action bounds and50-step horizon remain unchanged.

The source wrapper stops official loops that ignore terminal flags. If the
official solver finishes before the horizon, its last actual joint target is
held with real simulator steps until the source completes. This is an explicit
teacher hold policy, not fabricated transitions or zero-action padding. Planning
failure, no commands, failed success or missing terminal flags refuse admission.

Upstream conversion can clip a teacher command and spend up to four target
steps per reference step. The adapter does not clip again or substitute joint
labels. Target completion stops conversion immediately after the complete row
is recorded. Reference replay may therefore be a prefix: its cost and observed
success are reported separately, without asserting full replay identity.

## Qualification and data authority

```python
from src.infra.maniskill_expert import collect_pick_cube_expert

result = collect_pick_cube_expert()  # only under a separately frozen local budget
if result.qualified:
    episode = result.training_episode()
    # A separate BC protocol may release episode to the existing
    # NumpyContinuousBCLearner.train_episode API with this identical action spec.
```

Qualification requires all of:

- A complete collected source and observed sparse task success.
- Observed sparse success in the actual replayed reference prefix.
- A complete valid target episode and observed target sparse success.
- No collection, conversion, validation or cleanup failure.

`training_episode()` refuses an unqualified result. `candidate_episode` preserves
a valid complete unsuccessful target for diagnosis; it does not certify expert
quality or authorize BC training. No role/seed selection, final evaluation,
probe selection, model update or promotion API is exposed. Seed17 is fixed.

## Costs, failures and runtime

Source collection<=50, source replay<=50, target<=50control calls: total<=150.
The source has one constructor seed2022 reset plus two explicit seed17 resets;
the target has one constructor seed2022 reset plus its port's explicit seed17
reset if construction reaches it. Reset physics remain unknown, including failed
creations. Work distinguishes attempted calls, returned calls, completed control
physics and possible failed-call physics. Internal planner calls/FK/clipping and
MPLib RNG are unmeasured; environment seed alone does not qualify full planner
reproducibility or equal compute. `solver_completed=False` retains interruption
at a valid source horizon rather than claiming the planner script completed.

Factory/native/malformed-output/conversion/cleanup failures remain visible in
the result. Returned scenes are closed even after failure. An incomplete target
cannot export an episode. There are no native retries or state teleports.
Optional dependencies, software Vulkan settings and licensing are the same as
the [accepted CPU port](maniskill-continuous.md). No additional dependency.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.maniskill_expert_smoke --help
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_maniskill_expert.py tests/test_maniskill_continuous.py tests/test_continuous_demonstration.py
.venv/Scripts/python.exe -m ruff check src/infra/maniskill_expert.py scripts/maniskill_expert_smoke.py tests/test_maniskill_expert.py
.venv/Scripts/python.exe -m ruff format --check src/infra/maniskill_expert.py scripts/maniskill_expert_smoke.py tests/test_maniskill_expert.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --python-version 3.12 --platform linux src/infra/maniskill_expert.py scripts/maniskill_expert_smoke.py
```

Actual simulator commands use the isolated WSL interpreter and a prospective
native budget. The CLI reserves a fresh output before simulator work and returns
zero only for qualified data. Failed/negative JSON is permanent evidence.
Do not train on failed candidates, extend horizons, choose favorable seeds or
modify the teacher to manufacture success. Strong BC needs a separate frozen
training/quality protocol after real expert/controller qualification succeeds.

## Executed status — 2026-10-19

238focused cases and scoped statics pass. The one native expert qualification
FAILED SIGSEGV with empty result files; crash progress/work remain unknown.
Independent review accepts the boundary implementation and failure evidence,
not an expert dataset. Standalone imports pass; native qualification/BC remain
open. New phase/fatal markers are prepared, no native retry performed.
[Exact commands, costs, failures and next action](../artifacts/runs/maniskill-expert-bridge-20261019/RESULTS.md).
