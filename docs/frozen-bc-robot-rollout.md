# Frozen BC robot rollout through the existing smoke entry point

`scripts.maniskill_continuous_smoke.main(learner=model)` now runs one frozen BC
policy through the existing CPU PickCube port. The ordinary CLI still uses zero
actions and seed17. Both paths retain the same task, controller, 50-step bound,
exclusive output and TRAIN role. No model update or promotion is performed.

## Files and responsibilities

```text
scripts/maniskill_continuous_smoke.py  Existing entry point, frozen policy option
tests/test_maniskill_bc_smoke.py       CPU learner plus fake native boundary
src/infra/maniskill_continuous.py      Unchanged actual simulator/record port
src/adapters/numpy_continuous_bc.py    Unchanged owned learner and snapshot API
```

The script composes the existing learner and port. It adds no runner, checkpoint
codec, simulator abstraction, new training interface or dependency. A Python
caller restores an owned `ContinuousBCSnapshot` using the existing adapter API
before calling the entry point. This in-memory interface grants no evaluation
selection or data-release authority.

## Usage

The default engineering smoke remains:

```bash
python -X utf8 -B -m scripts.maniskill_continuous_smoke --output fresh-trace.json
```

For an already restored TRAIN learner in the same Python process:

```python
import sys
from scripts import maniskill_continuous_smoke

# model is an owned NumpyContinuousBCLearner restored through its existing API.
sys.argv = ["maniskill_continuous_smoke", "--output", "fresh-bc-trace.json"]
exit_code = maniskill_continuous_smoke.main(learner=model)
```

These commands create a real simulator episode. Use the pinned optional runtime
and a separately budgeted protocol before running another case. The accepted
local WSL configuration uses NumPy1.26.4, ManiSkill3.0.1, SAPIEN3.0.3, CPU physics,
software Vulkan and one CPU thread. See the linked exact executed command below;
this is local runtime evidence, not a tested clean-clone installation.

## Admission, state and failure behavior

The argument must be an exact `NumpyContinuousBCLearner` with TRAIN policy and
the declared four normalized root dx/dy/dz/gripper axes, each bounded `[-1, 1]`.
These checks occur before scene creation. The native observation/action spec must
match before prediction. The caller's model is forked; the private copy performs
inference only. Full policy, parameters, momentum and update/example counters
are checked after each prediction before control, after each control, and after
cleanup. Unexpected state changes refuse further work and remain visible.

State flags describe the latest observation. If a corrupted snapshot cannot be
observed, flags become null with an error; an earlier true does not survive as
current proof. No rollback hides a mutation. The caller retains the original
learner; successful native execution requires both original and private state
to remain exact.

Prediction attempts and returns are separate from validated rows and payload
bytes. A malformed returned prediction can count a return while admitting no
control. Inputs use the current actual state; output actions pass existing
float64 shape/finiteness/bounds validation before the port's float32 conversion.
The trace stores actual quantized commands. No clipping or controller change.

Creation attempts/returns and native work remain separate. Native or prediction
failures preserve charges; cleanup errors make completion false without erasing
the primary error. `complete` means a valid ordinary episode, not task success.
Sparse return, any-step success and final-step success are reported separately.
Reset physics remain unmeasured. Output reservation precedes model/scene work.

## Executed evidence and limitations

358 combined cases pass: 42 new frozen-policy cases, 90 existing port, 64 BC,
52 TRAIN diagnostics and 110 record cases. They use an actual CPU learner and
explicit fake native boundary, with no simulator scene. Scoped lint/format and
Linux-target types pass; the initial tuple-inference type failure and guard
formatting failure remain preserved. Independent review accepted source, tests,
checks and the one native result.

The fixed final256 policy was executed once in TRAIN seed17. Initial42D state
matched the successful teacher's first state exactly. Result: **50 steps, zero
sparse reward, no any/final success, terminated=false, truncated=true**. Full
model states stayed exact. Work was50 predictions/rows,1600 validated raw bytes,
50 controls/250 known physics, zero failed-possible physics and zero optimizer
updates. Trace150784B/audit292B; native supervisor6.156708265s; observed tree
high-water sum535314432B/final-self516321280B. Full-lifetime tree memory remains
unverified. Owned shader cache grew0→1313379B, within64MiB.

This is a working inference integration with a negative learned-policy result.
Lower offline loss has not produced useful closed-loop behavior, even in this
TRAIN context. No strong BC, generalization, retention/recovery, adaptation or
circadian advantage is established. Prior probe/teacher/runtime negatives stay.

## Next learner question

A root-only stdlib read of all42 TRAIN input dimensions found39 varying/3constant
features and positive standard deviations0.000360913 to0.659018 (ratio1825.98).
This motivates testing input conditioning; it does not prove a cause of failure.
The complete feature report is retained, with no model/simulator call.

Optional TRAIN-fitted frozen standardization is now implemented and tested,
keeping raw inputs as the default and full transform state owned. A fixed matched
TRAIN comparison improves standardized offline fit, but a separately reviewed
standardized256 TRAIN rollout also fails. [Current API, costs, complete results
and next optimization check](continuous-bc-standardization.md). No probe/final
observations fitted or selected the transform; useful robot benefit remains open.

[Protocol, exact commands, costs, failures and independent review](../artifacts/runs/frozen-bc-train-rollout-20261019/RESULTS.md).

## Exact continuous PC/CPC inference route

The same `run_frozen_rollout(output, learner=candidate, train_seed=17,
monitor_joint_positions=True)` Python API also supports exact
`NumpyContinuousPCLearner`, including neutral and active CPC policies. Existing
BC/nearest/actor/zero-policy defaults and CLI remain unchanged. Model creation or
a returned continuous-stream candidate grants no native experiment authority;
use a separate small frozen protocol before calling the real simulator.

Admission owns and validates the complete PC snapshot before fork/SDK, unwraps
only `policy.base` for TRAIN/controller/input geometry, and rejects alias or
foreign/subclass forks. It preserves native PC config, parameters, velocity,
row/update counters and complete coherent chemical/plasticity state. Prediction
is ordinary feedforward, with no latent inference, gradient, fit, wake advance or
update. Exact PC/CPC corrective-query collection now feeds the existing private eight-
update pipeline; see [continuation and costs](continuous-corrective-adaptation.md).
Unsupported actor/clock/context combinations still refuse before SDK. Serving actor routes
remain exact nearest backends.

Full source/private states are checked before prediction, after returned actions,
after the last observation/recipe/clock/context callback before control, after
steps and after cleanup/final telemetry. Public snapshot callbacks can mutate the
other owner; the PC branch therefore finishes by validating exact native-owned
state/policy fields without starting another callback loop. Flags stay unknown
until these checks finish, and any bad public return remains a failure. No
rollback hides changes or releases a successful trace/correction batch.

PC-specific report fields identify `frozen_continuous_pc`, retain full nested
`learner_policy`, PC energy objective, source wake count, parameter/momentum and
wake numeric bytes per owner, and feedforward-only prediction math. Returned
one-row forwards give a separately derived dense-MAC estimate; validation/gate
recomputation/copies/elementwise/temporary/process memory are not measured by it.
Source historical counters are distinct from rollout `optimizer_updates=0`.
Joint observations remain sampled control-time bounds, not continuous physical
safety. Return/success/violations are outcomes and cannot select a policy or seed.

```text
scripts/maniskill_continuous_smoke.py     existing integration, exact PC branch
src/adapters/numpy_continuous_pc.py       unchanged native learner/state API
src/adapters/continuous_learnability_modes.py reused full adaptive-state equality
tests/test_continuous_pc_rollout.py       independent fake-port PC/CPC checks
```

```powershell
python -m pytest tests/test_continuous_pc_rollout.py tests/test_maniskill_bc_smoke.py tests/test_joint_rollout_integration.py
python -m ruff check scripts/maniskill_continuous_smoke.py tests/test_continuous_pc_rollout.py
python -m ruff format --check scripts/maniskill_continuous_smoke.py tests/test_continuous_pc_rollout.py
python -m mypy --follow-imports=silent scripts/maniskill_continuous_smoke.py tests/test_continuous_pc_rollout.py
```

[Prospective integration protocol](../artifacts/runs/continuous-pc-rollout-20261020/PROTOCOL.md) ·
[Separate conditional one-use native protocol](../artifacts/runs/continuous-pc-rollout-20261020/NATIVE_PROTOCOL.md).

Scoped acceptance:156sameordered Windows/Linux cases,0skip,4file statics and
independent source+actual review. Sole untrainedactiveCPC nativeTRAIN17 smoke:
50controls/0updates/fullstateexact,51joint samples0violations,0return/no success.
That allocation is spent; use saved evidence. No learnedadaptation/safety/CPC
benefit inferred. [Commands, all failures/costs and next action](../artifacts/runs/continuous-pc-rollout-20261020/RESULTS.md) ·
[independent acceptance](../artifacts/runs/continuous-pc-rollout-20261020/REVIEW.md).
