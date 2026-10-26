# Explicit PushCube TRAIN port

Software implementation is independently accepted:69 focused checks pass on each
Windows/Linux runtime, with scoped lint/format/type checks. One native TRAIN17
qualification is also accepted:50controls, valid truncation and zero task success.
This is port functionality, not learning evidence. The port targets the installed
ManiSkill 3.0.1 `PushCube-v1` task with one CPU Panda and the normalized four-axis
`pd_ee_delta_pos` controller. It leaves the existing PickCube port unchanged.

```python
from src.infra.maniskill_push_cube import ManiSkillPushCubeEpisode

with ManiSkillPushCubeEpisode("push-train-17", seed=17) as robot:
    before = robot.state                 # named physical state, not flat offsets
    transition = robot.step((0.0, 0.0, 0.0, 1.0))
    paid_work = robot.work
```

This call constructs and resets a simulator and spends a native control. The
software fixture budget does not authorize running the example. A separately
frozen native qualification protocol is required before executing it.

## Observation and task contract

The input model view is exactly 12 coordinates:

1. Observed grasp flag from `Panda.is_grasping(env.obj)` with the same defaults
   used by the official PickCube task.
2. Object minus TCP position, in world metres divided by 0.1 (three coordinates).
3. Goal minus object position in the same units/frame (three coordinates).
4. Previous actually sent native float32 command (four coordinates).
5. Consecutive negative-gripper command count, capped at six and divided by six.

**This augments PushCube's standard observation with an explicit grasp sensor.**
The stock PushCube state does not contain that flag. The port queries the
existing Panda contact-based API and reports its costs; it never fills a missing
flag with zero. All future compared learners must receive the same declared
observation information. The count is a command proxy, not contact evidence.

Structured public `tcp_pose`, `obj_pose` and `goal_pos` supply positions. No
flattened PushCube shape/order or PickCube raw42 offset is assumed. Position
subtraction uses Python floats, so semantic correspondence with PickCube TS12
does not imply bitwise equality with its already rounded native displacement
vectors. The native port checks the actual root orientation and controller
configuration before equating world and root translation axes.

Only TRAIN seeds 17, 19 and 20 are admitted, with derived contexts
`push-cube-train-{seed}`. Probe, guard and final roles/seeds are unavailable.
The task keeps its official sparse success rule: XY object-to-goal distance
strictly below 0.1 metres and object height strictly below 0.025 metres.
Grasp and staticness are not additional success criteria. Reward and both end
flags remain independent native outputs; complete failure is a valid outcome.

## Ownership, costs and extension

The port owns one environment and one explicit reset. Native construction may
also reset internally; constructor/reset physics remain unknown. It records
control attempts/returns and known controller physics separately from state
encodings, grasp queries and cleanup. The two contact-force calls per grasp
query are source-derived accounting, not instrumented native arithmetic.
Malformed output or changed owners/frame permanently stops the episode.

`episode()` exports only a valid complete `ContinuousEpisode` with the actual
commands and causal feature history. Storage is bounded by the official
50-control horizon. No teacher, learner, model fitting, IO or deployment lives
in this port. Distinct TRAIN push supervision needs its own versioned protocol;
PickCube teachers must not be relabeled as PushCube data.

```text
src/core/cube_taskspace_state.py       named physical state and pure 12D encoding
src/infra/maniskill_push_cube.py       optional native SDK boundary and paid work
tests/test_push_cube_continuous.py    pure geometry and fake boundary behavior
```

[Contract, budgets and current verification](../artifacts/runs/push-cube-train-port-20261021/PROTOCOL.md).
After qualification, use the existing learned-source continuation API with new
TRAIN data and explicit old references. Native qualification alone will not
establish learning, retention, reliability, safety or a circadian advantage.


## Executed checks and native admission

```powershell
python -m pytest -q tests/test_push_cube_continuous.py
python -m ruff check src/core/cube_taskspace_state.py src/infra/maniskill_push_cube.py tests/test_push_cube_continuous.py
python -m ruff format --check src/core/cube_taskspace_state.py src/infra/maniskill_push_cube.py tests/test_push_cube_continuous.py
python -m mypy --platform linux --follow-imports silent src/core/cube_taskspace_state.py src/infra/maniskill_push_cube.py tests/test_push_cube_continuous.py
```

The metered actual argv/results are in [RESULTS.md](../artifacts/runs/push-cube-train-port-20261021/RESULTS.md).
The original native literal-qw refusal and constructor-only diagnosis stay failed.
The diagnosis observed qw=1+2^-23 with exactly zero vector components. The explicit
v2 admission allows abs(abs(qw)-1)<=2float32epsilon and requires qxyz exactlyzero;
robot/root poses must still agree exactly. No angular tolerance, state rewrite or
metric change is introduced. Final qualification costs50controls250knowncontrol
physics/51graspqueries102SOURCE-DERIVED force calls, plus unknown constructor/reset
physics and native arithmetic. All current execution allowances are closed.
[Decision and alternatives](adr/ADR-0502-explicit-physical-cube-task-state.md).
