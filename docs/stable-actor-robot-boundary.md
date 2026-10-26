# Stable actor at the robot control boundary

The existing PickCube smoke accepts `main(actor=runtime.actor)` for an exact
`StableActor` owning an exact nearest continuous BC learner. Each control uses
the supplied actor's public `predict` method and its versioned result. The smoke
does not restore a learner copy in place of serving, train the candidate or
promote it.

## Files and responsibilities

```text
scripts/maniskill_continuous_smoke.py       Existing command/control boundary
tests/test_stable_actor_robot_boundary.py  Actual runtime actor with fake native IO
```

The app runtime, continuous learner, pure core and native port are unchanged.
This script already depends on those inner layers. Its private binding retains
one admitted actor, backend identity, version and full owned snapshot. Backend
inspection is restricted to exact type and identity admission; predictions and
state observations go through public locked actor methods.

Why this: restoring a snapshot into another learner would exercise that copy
while bypassing actual runtime serving. A complete serving state and explicit
version are needed to attribute controls while a separate candidate changes.

## API and ownership

```python
# runtime is the existing ActorShadowRuntime over a warm TRAIN nearest learner.
# sys.argv supplies the existing --output fresh-trace.json argument.
from scripts.maniskill_continuous_smoke import main

exit_code = main(actor=runtime.actor, train_seed=17)
```

Use the separately installed optional simulator stack and a prospectively frozen
native budget before running real robot episodes. This increment is verified
with fake native IO. The route has since run in the fixed visual context case:
the actual A-only actor completed50 native B controls with return0/truncation
and unchanged full state. Successful candidate B/A readouts were separate;
no serving promotion occurred. [Actual evidence](../artifacts/runs/visual-context-return-20261019/RESULTS.md).

`actor` and `learner` are mutually exclusive. The original default zero policy,
MLP and nearest private-fork routes remain. Seed17 is default,19 is the declared
second TRAIN reset context; probe/final seeds remain unavailable. Existing
`observation_clock` works with the same explicit horizon50 recipe and preserves
raw native trace observations.

This original route supports the exact stable actor/nearest backend. A separate
explicit [promotable actor route](promotable-actor-robot-boundary.md) now binds the
complete serving bundle. Other actor subclasses, foreign backends, cold models, non-TRAIN
policies, invalid native snapshots, wrong controller units/bounds and invalid
versions are refused before scene creation. A valid source with an observation
dimension different from the actual port is refused after construction when that
dimension becomes observable, as in the existing learner route.

The script observes the actual actor version, backend identity and complete NN
state before/after prediction, immediately before/after native controls, and
after cleanup. Every snapshot is fully validated, including array dtype/shape,
statistics and native counters, before equality comparison. Replacing the backend
with an equal-state learner is still refused. Earlier successful flags are
cleared before new observations; failed observation cannot retain stale `True`.
No rollback hides an observed state or version change.

The returned prediction must be an exact `VersionedPrediction` with the admitted
native string version and a finite float64 action row in declared bounds. No
control is sent after a detected tag/action/state/configuration failure. A failure
after a completed control retains that work and prevents the next action.

## Receipts and work

The policy is `frozen_stable_nearest_continuous_bc`; `state_guard_scope` is
`actual_serving_actor_no_private_fork`. Here `frozen_state_exact` and
`original_owner_exact` both describe the actual serving owner. Actor/backend and
prediction-version flags are separate from each other and from state equality.
Actor attempts/returns, validated versions, source publications and retained
prototype numeric bytes are reported.

StableActor returns versioned actions, not `NearestPrediction`. Actor distance
candidate/coordinate upper bounds are recorded before each actual call. Completed
counts are explicitly **derived** from the exact unchanged typed NN, query shape
and a validated return; they are not returned native work measurements. An
unreturned/invalid call retains its upper bounds without pretending it completed
zero work. No selected indices or nearest reports are fabricated. Existing
nearest report fields continue to describe the concrete learner route.

Normalization/prediction outputs and snapshot-validation/copy overhead differ;
full-state copies are not distance coordinates. Existing output row/byte counts,
controller attempts/completions and known/possible failed physics accounting
remain. Whole-process peak memory is not established by these fake-boundary tests.

## Focused verification and extension

```powershell
python -m pytest -q -p no:cacheprovider tests/test_stable_actor_robot_boundary.py tests/test_nearest_continuous_bc.py tests/test_maniskill_bc_smoke.py tests/test_continuous_episode_clock.py tests/test_pick_cube_train_context.py
python -m ruff check scripts/maniskill_continuous_smoke.py tests/test_stable_actor_robot_boundary.py
python -m ruff format --check scripts/maniskill_continuous_smoke.py tests/test_stable_actor_robot_boundary.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --python-version 3.12 --platform linux scripts/maniskill_continuous_smoke.py
```

Preserve exact route admission and add another actor type only with explicit
version/state/ownership tests. Current candidate learning does not change serving
actions or grant promotion authority. This integration correctness result cannot
establish native skill, domain-drift recovery, generalization, neural plasticity
or CPC benefit. Accepted earlier native trajectories and all negative results
remain separate. [Prospective protocol and evidence](../artifacts/runs/stable-actor-robot-boundary-20261019/PROTOCOL.md).
