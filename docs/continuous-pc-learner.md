# Owned continuous PC learner

`src/adapters/numpy_continuous_pc.py` supplies a CPU NumPy learner through the
existing `NativeLearner` interface. It performs a bounded continuous PC update,
owns its parameters/momentum/counters, and supports detached snapshots, restore
and fork. The existing delayed-label `ActorShadowRuntime` can use it unchanged.
There is no second runner, replay, checkpoint IO, promotion, or data selection.

```text
src/adapters/numpy_continuous_pc.py   owned policy/state/update port
src/core/coupled_inference.py        shared guarded steps and private progress
tests/test_numpy_continuous_pc.py    state, math, role and runtime checks
scripts/continuous_pc_learner_example.py   one bounded runnable update
```

## Configure and use

```python
base = ContinuousBCPolicy(spec, hidden_dim=2, seed=17)
learner = NumpyContinuousPCLearner(ContinuousPCPolicy(base))
report = learner.train_batch_with_work(released_features, released_targets)
actions = learner.predict(unlabeled_features)
child = learner.fork()
```

`ContinuousPCPolicy` wraps the existing `ContinuousBCPolicy`: identical action
geometry, one-hidden-layer architecture, standardization, initialization, seed,
learning rate and momentum. It adds precision (default1), inference cap (4),
inference rate (.05) and the existing `CoupledInferencePolicy`. All fields are
validated and copied. The default BC half mean active-axis normalized loss is
supported; the optional BC binary-gripper likelihood is refused before RNG.
Deeper energy/inference primitives exist, but this adapter deliberately matches
the current one-hidden robot BP architecture. No new dependencies/environment
variables are introduced.

Raw array training requires release authority from the caller. `train_episode`
additionally requires an exact matching `train` or `probe_train` policy/episode
role. Final, guard and probe-test labels cannot enter that conversion. A private
probe learner must remain detached from deployed/benchmark learners. Final data
must remain unavailable to training, promotion, tuning and probe selection.
Observation standardization is supplied frozen through the base policy; this
learner never fits or revises it.

## Update and publication

1. Own/revalidate the prior state, raw features and bounded targets; preflight
   update and row counters.
2. Apply frozen standardization, then initialize free latents without targets.
3. Run the accepted simultaneous guarded [continuous inference](continuous-pc-inference.md).
4. Convert the accepted parameter partials to the BP vector-bias layout. Compute
   `new_velocity = momentum * old_velocity + partial_gradient`, then
   `new_parameters = old_parameters - learning_rate * new_velocity`.
5. Validate the whole candidate and run one target-free candidate prediction.
   Unrepresentable arithmetic is refused; task improvement is not a gate.
6. Prepare the diagnostic/report, then publish exactly one owned state.

A first-proposal inference fallback can still produce an ordinary parameter
update from its valid initial evaluation. Hidden partials may be zero while
existing momentum still moves hidden weights. There is no claim that these
partials equal BP total gradients or that the update lowers task loss.

`train_batch` returns the exact generic `TrainingDiagnostic`: the continuous
energy ID and preupdate inferred energy. `train_batch_with_work` returns
`ContinuousPCUpdateResult(diagnostic, task_loss, inference, work)`; task loss is
reported separately. It does not evaluate a post-update task metric. Work dicts
are caller-owned and mutable; no success record or trajectory is retained.

## Ownership and failed work

`ContinuousPCSnapshot` contains policy, the same four vector-bias parameter and
velocity arrays as BC, and update/row counters. Snapshot/restore validate exact
types, policy agreement, finite float64 shapes and consistent bounded counts.
Restore prepares a complete detached candidate before assignment. Fork copies
policy/state without RNG; restoring a child does not alter its source. Predict
uses features only and the accepted finite-before-tanh bounded predictor.

After admission, normalization, initialization, energy attempts/returns, latent
attempts/acceptances, norm/comparison work, parameter proposals, candidate
predictions and publication are counted at their actual boundaries. Initial or
late failures return `TrainingUpdateFailed` with copied native work counters,
while parameters/momentum/model counters remain unchanged. Candidate numerical
inference fallback retains its consumed work. Failed transforms are counted even
if no initialization occurs. These native counters do not refund outer runtime
budgets or restore original app charges.

Input/state validation, geometry calculations, array copies, Python overhead,
wall time and process memory are outside the counters. Every energy computes
both latent and parameter partials. Candidate prediction costs one extra forward.
No whole-process memory or deep-scaling measurement is claimed. The private
shared `_InferenceWork` progress record exists because failures can precede an
inference return; public binary/continuous APIs and guard math remain unchanged.
Historical timing results remain tied to their original source/runtime, not
requalified for the small new accounting overhead.

## Example and focused checks

```powershell
python -m scripts.continuous_pc_learner_example
python -m pytest tests/test_numpy_continuous_pc.py tests/test_continuous_pc_inference.py tests/test_coupled_inference.py tests/test_coupled_inference_integration.py
python -m ruff check src/adapters/numpy_continuous_pc.py src/core/coupled_inference.py tests/test_numpy_continuous_pc.py scripts/continuous_pc_learner_example.py
python -m ruff format --check src/adapters/numpy_continuous_pc.py src/core/coupled_inference.py tests/test_numpy_continuous_pc.py scripts/continuous_pc_learner_example.py
python -m mypy --follow-imports=silent src/adapters/numpy_continuous_pc.py src/core/coupled_inference.py tests/test_numpy_continuous_pc.py scripts/continuous_pc_learner_example.py
```

The fixed example performs one synthetic two-row/four-action update and a detached
fork restore. It reports actual inference/parameter work and target-free actions,
without requiring improvement or comparing learners.

Next, add the matched neutral CPC bridge and verify identical full PC state and
updates before any active-CPC or comparative protocol. Reuse the existing stream
runner and frozen metrics/seeds; preserve all negative results.

[Protocol](../artifacts/runs/continuous-pc-learner-20261020/PROTOCOL.md) ·
[Executed evidence and limits](../artifacts/runs/continuous-pc-learner-20261020/RESULTS.md) ·
[Independent acceptance](../artifacts/runs/continuous-pc-learner-20261020/REVIEW.md).

## Follow-up: optional matched CPC wake state

The same native learner now has [optional neutral/active CPC modes](continuous-cpc.md).
The last optional policy/snapshot fields preserve default PC behavior while adding
owned chemistry and exact-control checks. Stream learner selection and comparative
benefit remain later work. CPC transition counters exclude validation's repeated
gate calculations and vector copies; preserve those cost distinctions.
