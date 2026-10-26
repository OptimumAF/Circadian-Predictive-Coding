# Experimental class-weighted recurrent demonstrations

The existing recurrent learner now has an explicit weighted episode update, and
the typed demonstration/reward owner accepts `ClassWeightedDemonstrationUpdate`.
Native array `train_batch`, `train_episode`, and typed `DemonstrationUpdate` keep the original
mean cross entropy. Reward updates keep their existing objective. No CLI schedule,
probe, comparative experiment or replay encoder is changed by this component.

```python
from src.adapters.torch_demonstration_reward_learner import (
    ClassWeightedDemonstrationUpdate,
    TorchDemonstrationRewardLearner,
)
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy

learner = TorchDemonstrationRewardLearner(TorchRecurrentPolicy(2, 3, 3, 17))
update = ClassWeightedDemonstrationUpdate(released_train_episode)
diagnostic = learner.train_batch(update, update.episode.episode_id)
```

The existing runtime accepts this packet as an opaque experience payload and its
exact episode ID as the released label. `ExperiencePermissions(training=True)`,
the train role, original budget and ordinary inbox release rules still apply.
An unweighted packet uses the same owner; no optimizer reset or separate model
is introduced. Future orchestration belongs in the existing runner.

## Weight rule and objective

The packet derives weights only from its own complete train episode. For N steps,
P present classes and target count n[c], `w[c] = N/(P*n[c])` for present classes.
Absent classes receive1.0 and contribute no target term. For labels `[0,0,1,2]`,
weights are `(2/3,4/3,4/3)`. No final, probe or evaluation records are accepted by
the derivation function or weighted episode method.

The native method uses:

```text
L = sum_t w[y_t] * (-log softmax(logits_t)[y_t]) / sum_t w[y_t]
dL/dlogits_t = w[y_t] / sum_t w[y_t] * (softmax(logits_t) - onehot(y_t))
```

Full ordered BPTT starts from zero hidden activation. Derived weights give equal
total target mass to each present class; steps retain their original order and
all still count as training rows. Common positive weight scaling leaves this
normalized objective unchanged. Uniform weights recover mean CE to the declared
float32 numerical tolerance. The diagnostic ID is
`torch_recurrent_episode_class_weighted_cross_entropy_v1`; it remains separate
from the original diagnostic and from reward returns.

Raw `TorchRecurrentLearner.train_weighted_episode(episode, class_weights)` accepts
an explicit exact K-tuple of native finite numbers in `[1e-6,1e6]`. The typed
packet is the normal orchestration boundary: its weights cannot be supplied at
construction, and dispatch revalidates their canonical float tuple against the
owned training labels. Input geometry remains K2–16/T1–16 with bound feature width.

## State, roles and costs

Weights belong to the update input. Native policy and snapshot version/fields
stay unchanged. Detached parameters, gradients and optimizer state are restored
before work; only a finite diagnostic and fully validated post-step state can
replace the original payload. Late exceptions and staged nonfinite state leave
the complete original owner unchanged. Full fork/restore carries optimizer
momentum across weighted, mean-CE and on-policy reward updates.

Each successful weighted update costs one actual SGD call and its original T
training rows. Deriving weights costs bounded O(T+K) work; tuple/Python metadata
and a temporary CPU float32 weight vector add memory. They are not model parameter
bytes. No new retained/replay payload byte accounting is claimed. Total process
RSS and invocation time include these costs in the scoped native checks.

## Structure and focused checks

```text
src/core/demonstration_class_weights.py         pure train-only derivation/bounds
src/adapters/torch_recurrent_weighted.py        native weighted CE and atomic commit
src/adapters/torch_recurrent_learner.py         explicit lazy public method
src/adapters/torch_demonstration_reward_learner.py typed derived packet/dispatch
tests/test_demonstration_class_weights.py       pure derivation/validation
tests/test_class_weighted_demonstration_update.py fake port/default/refusal checks
tests/test_torch_recurrent_weighted.py          native NumPy math/state/runtime checks
```

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_demonstration_class_weights.py tests/test_class_weighted_demonstration_update.py tests/test_torch_demonstration_reward_learner.py tests/test_torch_reward_update_learner.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check src/core/demonstration_class_weights.py src/adapters/torch_recurrent_weighted.py src/adapters/torch_recurrent_learner.py src/adapters/torch_demonstration_reward_learner.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check src/core/demonstration_class_weights.py src/adapters/torch_recurrent_weighted.py src/adapters/torch_recurrent_learner.py src/adapters/torch_demonstration_reward_learner.py
```

Type checks use `-m mypy --follow-imports=silent --no-incremental --no-sqlite-cache
--cache-dir=NUL --platform win32` with the same paths; also check `--platform linux`.
The native test file performs bounded engineering work under its
[prospective scope](../artifacts/runs/r42-weighted-demonstration-20261016/SCOPE.md).
No new dependency, environment variable or configuration file is required.

Why this: released demonstration labels are imbalanced, and the reserved decision
measurement showed majority-action endpoints. Weighting is a train-only hypothesis.
Mathematical/ownership correctness cannot establish improved skill, retention,
recovery or circadian benefit. Keep earlier negative baselines and mean-CE behavior;
freeze and review a separate small comparative protocol before scientific use.
