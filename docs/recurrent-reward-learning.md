# Recurrent reward updates and policy-generated training episodes

The existing recurrent learner now has an explicit reward-trajectory update:
`train_reward_episode(episode, behavior_state, discount=1.0)`. Its original
`train_batch`/`train_episode` behavior-cloning math and snapshot format stay intact.
The new app collector samples actual actions in the existing key/door environment.
This component is the prerequisite for R4.2's separately evaluated recurrent RL
recipe. That [separate runtime/CLI route](recurrent-rl-key-door.md) is now accepted,
including its zero-reward negative case; useful navigation and full R4.2 remain open.

## Objective

For a complete episode starting with zero hidden state, define
`G_t = sum(k=t..T-1, gamma**(k-t) * reward_k)` and optimize
`L = -sum(t=0..T-1, gamma**t * G_t * log p(action_t | observation_0:t))`.
The explicit outer discount makes each reward k carry gamma**k in every earlier
eligible score term. Gamma is a finite native number in[0,1], default1. There is
no division by length, baseline, advantage normalization, entropy term, bootstrap,
clipping or repair. Native SGD/BPTT uses the existing bound rate/momentum. Zero
reward still commits one update/T rows and existing momentum may move parameters.
The diagnostic ID is `torch_recurrent_reward_logprob_sum_v1`, distinct from CE.

Core `PolicyGradientEpisode` is a separate exact immutable record containing
physical transitions and per-step behavior probabilities. It shares the existing
physical `DemonstrationTransition` shape; it is not a teacher demonstration.
One to16 ordered continuous steps, observation width1..16, class count2..16 and
bounded finite observations/rewards are validated. Endings are terminal,
truncated, or unapplied illegal_action; false terminal flags are never fabricated.
A sampled action needs positive probability; row mass may differ from1 by<=1e-6
for native float32 roundoff. Copying revalidates frozen-record tampering.

Native admission accepts only train-role reward records matching policy geometry.
It owns/revalidates the full behavior checkpoint and compares current policy,
parameters, gradients, optimizer configuration/momentum and update/example clocks
exactly. Any different/stale state refuses before SGD; reward episodes are not
replayed across policy updates. Recomputed behavior probabilities must match within
absolute1e-6. The finite complete staged SGD state validates before publication,
so late optimizer/capture failure leaves live native state unchanged. This API
uses no RNG; existing owned snapshots/forks retain complete optimizer continuation.

## Actual collection and randomness

`collect_key_door_reward_episode` takes an owned frozen-session source, a declared
training layout/cue, a unique ID and an exact caller-owned NumPy Generator(PCG64).
Each episode starts a fresh zero-activation session. One validated prediction
consumes one uniform; inverse CDF uses the raw cumulative mass, scaled by its
actual total for float32 roundoff. There is no legal-action mask, resampling,
teacher lookup, argmax substitution or hidden-cue reconstruction. Illegal actions
end unapplied with reward0; exhausting max_steps records truncation.

The layout ID is captured before callbacks. The report keeps detached behavior
state and RNG before/after snapshots, attempted/returned predictions, own sampler
draws, attempted/returned physics steps, actual transitions and total return.
PCG64 progression checks reject changes outside recorded draws, including callback
side effects; actual spent state is retained, never rewound. Unknown external
sampler work is not included in the collector's own sampling_draws count.
Operational faults raise `KeyDoorTrainingFailed` with a partial owned report;
only successful complete records can be passed to the native reward API.

The opaque source port relies on the existing owned frozen-session contract.
A returned collection is recorded behavior plus a captured checkpoint, not an
attestation that arbitrary caller implementations preserved source state. If a
callback changes the actual native source, subsequent reward admission refuses
the stale checkpoint before any reward update. Roles are declared provenance,
not independent data attestation. Never relabel benchmark/final data into train.

## Use through existing boundaries

```python
from time import monotonic
import numpy as np
from src.adapters.torch_recurrent_learner import TorchRecurrentLearner, TorchRecurrentPolicy
from src.app.key_door_training import collect_key_door_reward_episode
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.key_door_navigation import KeyDoorLayout

learner = TorchRecurrentLearner(TorchRecurrentPolicy(8, 4, 5, 17))
sampler = np.random.Generator(np.random.PCG64(29))
budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1, max_wall_seconds=90), monotonic)
record = collect_key_door_reward_episode(
    learner, KeyDoorLayout("train-only-room"), 1,
    episode_id="policy-train-0", sampler=sampler,
    before_prediction=budget.before_final,
)
budget.before_update()
diagnostic = learner.train_reward_episode(record["episode"], record["behavior_state"])
budget.record_update()
```

Collection does not create/renew budgets, train, promote or evaluate final data.
The caller counts committed updates using the original budget and accounts for
sampling/prediction/physics attempts from reports. Save an owned native checkpoint
and `sampler.bit_generator.state` for continuation; restore the private generator
explicitly. Failure does not refund sampled draws or native work. A serving
`StableActor` continues owning its separate unchanged fork.

## Structure and focused verification

```text
src/core/policy_gradient_episode.py           owned records and scalar return weights
src/adapters/torch_recurrent_reward.py        explicit native objective and atomic commit
src/adapters/torch_recurrent_learner.py       existing learner plus reward entry point
src/app/key_door_training.py                  actual bounded sampled collection
tests/test_policy_gradient_episode.py       pure record/return arithmetic
tests/test_torch_recurrent_reward.py        independent NumPy BPTT/SGD and state faults
tests/test_key_door_training.py             fake sampling/physics/partial-cost boundaries
tests/test_recurrent_reward_integration.py  actual policy sampling/native ownership
```

From repository root, use the installed environment:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_policy_gradient_episode.py tests/test_torch_recurrent_reward.py tests/test_key_door_training.py tests/test_recurrent_reward_integration.py tests/test_torch_recurrent_learner.py tests/test_recurrent_stream_integration.py
```

Scoped Ruff check/format --check and mypy Windows+Linux commands are recorded in
[the execution results](../artifacts/runs/r42-recurrent-reward-20261016/RESULTS.md).
There is no new dependency, environment variable, disk codec, CLI scenario or
orchestration framework. Why this: real sampled reward learning is a distinct
native objective; it should not silently reinterpret supervised class labels.
[ADR-0330](adr/ADR-0330-separate-recurrent-reward-trajectories-from-demonstrations.md).

The [existing-runner RL route](recurrent-rl-key-door.md) now composes this update
and collector. [Explicit train-only phase routing](demonstration-reward-phases.md)
now verifies full optimizer continuation and separate costs in engineering.
The [matched BC-only control](warm-start-control.md) now measures one frozen pair,
with explicit momentum accounting and no live success. Next add per-decision/class
reserved-probe diagnostics. Preserve old BP/PC/neutral/active CPC results and all
negatives. Do not tune from previously scored finals.

## Executed acceptance and limits

175 distinct cases/525passing phases FIRSTPASS, final eight-path Ruff/format and
Windows+Linux types, independent SOURCE/GO/ACTUAL acceptance. Initial author/root
typing-check failures remain preserved; the central runtime needed no retry.
The sole engineering suite uses51SGD/442combined native+reference timesteps,
96native predictions and127physics entries in12.2297191s worker/12.9106007s parent,
with326942720B sampledRSS. All17 frozen source/controller texts stay exact.

Actual native first/next/restored-next episodes take3/1/1decisions,2/0/0legalsteps;
all end illegal_action with zero reward and no success. Three explicit reward
updates charge candidate2/branch1 and preserve exact full optimizer/private RNG
continuation, source/serving actor/global RNG. A source changed during collection
refuses stale reward admission before additional SGD. Independent analytic rewards
exercise nonzero/negative gradients separately; they are not observed game returns.
No scientific comparison or useful navigation claim follows. All previous negative
results remain intact. The existing-runner RL recipe is now accepted separately;
full R4.2 and release remain open. The explicit train-only demonstration→reward
phase preserves the pure-RL route and its negative result; the matched control
is now measured separately, with useful learning still unproven.

[Raw integration evidence](../artifacts/runs/r42-recurrent-reward-20261016/focused-temp/test_should_collect_real_polic0/reward-engineering.json),
[all commands, failures and budgets](../artifacts/runs/r42-recurrent-reward-20261016/RESULTS.md),
[independent review](../artifacts/runs/r42-recurrent-reward-20261016/REVIEW.md).
