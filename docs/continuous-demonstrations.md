# Continuous-action demonstrations

This R4.4 prerequisite validates continuous controller records before a future
behavior-cloning learner or simulator adapter sees them. It executes no policy,
simulation, training, release decision, clipping or unit conversion. Existing
categorical demonstrations and generic `NativeLearner` remain unchanged.

## Structure

```text
src/core/continuous_demonstration.py     Units, bounds, transitions, complete episodes
tests/test_continuous_demonstration.py   Boundary behavior and rejection cases
```

`ContinuousActionSpec` requires an observation dimension and one declared unit,
finite lower bound and finite upper bound per action axis. Constant axes and
inclusive bounds are valid. There are1..1024 observation coordinates,1..16 action
axes and1..256 steps per episode. Physical observations/rewards are finite, without
the old toy record's artificial [-1,1] constraint. Native ints must convert to
float exactly; bools, NumPy scalars and mutable vector containers are refused.
Simulator adapters must explicitly convert their arrays/scalars into these native
tuples, with units/bounds taken from the actual controller configuration.

## Example

This is illustrative hand-authored data, not a collected robot demonstration:

```python
from src.core.continuous_demonstration import (
    ContinuousActionSpec, ContinuousTransition, ContinuousEpisode,
    continuous_training_pairs,
)

spec = ContinuousActionSpec(2, ("m",), (-0.01,), (0.01,))
step = ContinuousTransition(0, (0.0, 0.0), (0.005,), 0.0,
                            (0.005, 0.0), False, True)
episode = ContinuousEpisode("example", "reach", "train", spec, (step,))
features, actions = continuous_training_pairs(episode)
assert features == ((0.0, 0.0),) and actions == ((0.005,),)
```

Only the last transition may end an episode; it must terminate, truncate or both.
Both flags stay separate, including when both are true. Consecutive steps must
join exactly through their supplied observations. The module never fills missing
steps, guesses terminal observations or relabels a truncated episode as success.

Storage admits `train`, `probe_train`, `probe_test` and `final_test`. Training
extraction requires the exact matching `train` or explicit `probe_train` role;
evaluation records cannot be extracted for training. This is role validation,
not an authoritative runtime release, provenance or promotion mechanism. Re-copying
an episode validates it again and owns its spec, transitions and vector tuples.

## Checks and scope

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_continuous_demonstration.py tests/test_demonstration_episode.py
.venv/Scripts/python.exe -m ruff check src/core/continuous_demonstration.py tests/test_continuous_demonstration.py
.venv/Scripts/python.exe -m ruff format --check src/core/continuous_demonstration.py tests/test_continuous_demonstration.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --python-version 3.11 --platform win32 src/core/continuous_demonstration.py
# Same source check also passes with --platform linux.
```

165 focused cases pass:110 continuous and55 unchanged categorical cases. Scoped
statics and independent SOURCE/ACTUAL review pass. This establishes data-boundary
behavior only. ManiSkill was not installed in the inspected environment; no
simulator, BC learner, robot task, model update, reward benefit or release claim.
No dependency was installed. [Evidence and commands](../artifacts/runs/continuous-action-boundary-20261019/RESULTS.md).

Next add a continuous BC adapter through existing `NativeLearner`, with an explicit
loss/units convention, controller-bound predictions, owned snapshots/forks and
behavioral tests. Then bind the actual planned ManiSkill reach task/configuration
and strong BC baseline under a separately reviewed dependency/data/run protocol.
Do not replace the robot acceptance requirement with hand-authored examples.
