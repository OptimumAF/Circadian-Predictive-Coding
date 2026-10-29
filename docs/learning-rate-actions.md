# Per-update native learning-rate actions

A finite wake action may request one bounded learning-rate multiplier through
the existing shared gate and causal inbox. The effect lasts for that update;
original policy, optimizer base rate, snapshot schema, budgets and permissions
remain authoritative. There is no controller-supplied callback or persistent
learning-rate setting.

## Use existing owners

```python
from src.app.resource_sharing import ResourceSharedRuntime
from src.core.adaptation_action import AdaptationAction, AdaptationEnvelope

shared = ResourceSharedRuntime(
    runtime, gate,
    action_envelope=AdaptationEnvelope(
        (AdaptationAction("wait"),
         AdaptationAction("half", wake_updates=1, learning_rate_multiplier=0.5),
         AdaptationAction("wake", wake_updates=1)),
        max_decisions=8, max_wake_updates=2,
        max_learning_rate_multiplier=1.0,
    ),
)
poll = shared.train_action("half")
print(poll.requested_learning_rate_multiplier, poll.training.updates,
      poll.training.deferred_reason, poll.progress)
```

`runtime` must be an ordinary TRAIN ActorShadowRuntime with a native candidate
supporting the optional `RateControlledLearner` port. The four TorchFeatureLearner
families support `train_batch_with_rate_multiplier(features, targets, multiplier)`.
Generic learners keep their original `train_batch`; nonunit effects require the
optional port. Unity uses the original public method, preserving subclass counters.

The multiplier must be an exact native int/float in(0,4], finite and within the
owner's immutable envelope. Four is an implementation ceiling, not a tuned dose.
The effective parameter-learning rate must remain finite and positive. Bool,
NumPy scalar, NaN, infinity, overflow/underflow to zero and unsupported owners
are refused. Rate-only zero-wake, replay+rate and sleep remain
unsupported. A joint rate+gate wake requires the optional
`RateGateControlledLearner` port and a compatible bound native gate policy;
both effects execute in one staged update with one charge. It cannot use two
separate updates as a fallback. See [gate actions](gate-actions.md) and the
[combined qualification](reports/combined-rate-gate-20261025.md).
There is no implicit replay scaling. A rate wake may interleave
with separately selected fixed replay under its original predecessor checks.

## Native state and failure semantics

The original scalar is forwarded through ResourceSharedRuntime,
ActorShadowRuntime, ExperienceInbox and `update_learner`. Readiness, training
roles, labels, serving priority, admission quotas, clocks and budgets retain
their existing rules. Future-label/paused/serving/resource deferrals train no
rows. Known catalog choices spend decisions; admitted attempts remain paid.
Returned polls identify the requested multiplier even on a no-work deferral.
AppliedExperience and durable state formats are unchanged; callers retaining
action provenance should retain the finite catalog and poll alongside receipts.

BP stages a complete native head and optimizer, changes staged SGD LR for one
step, and restores the base param-group LR before complete state validation and
publication. Momentum and gradients are the actual computed values. PC, neutral
CPC and active CPC change parameter-learning rate only; inference steps/rate,
gates, chemical dynamics and policy are unchanged. No snapshot validation is
relaxed and no historical fit/receipt/budget authority is renewed.

Failed staged work cannot partly mutate the live native payload. At the app
boundary a lost native return remains conservatively unknown, consumes admission
and stops the inbox; no implicit retry. A completed update followed by a late
resource stop retains its receipt and budget charge. App entry counts and actual
physical optimizer/latent work differ when validation rejects before computation.
Native staging is not hard preemption, hostile-Python isolation or a guarantee of
process-tree peak memory. No final/probe/promotion data port is added.

Why this: full snapshots pin optimizer LR to the original policy. A temporary
effect on staged state enables the finite action without mutating that policy or
discarding its momentum. Original default math is checked against native references.

## Run and test

```console
python -m scripts.bounded_rate_action_example --source artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json --output rate-example.json
python -m pytest -q tests/test_learning_rate_actions.py
python -m ruff check src/core/learner_ports.py src/adapters/torch_feature_learner.py src/app/learner_step.py tests/test_learning_rate_actions.py
python -m ruff format --check src/core/learner_ports.py src/adapters/torch_feature_learner.py src/app/learner_step.py tests/test_learning_rate_actions.py
python -m mypy --no-incremental --explicit-package-bases --follow-imports=silent src/core/learner_ports.py src/adapters/torch_feature_learner.py src/app/learner_step.py tests/test_learning_rate_actions.py
```

Use a fresh output path. Example source is the exact qualified96update/576row BP
artifact, not a new fit. The executed case defers then performs half-rate wake6
and default wake6, ending98updates/588rows with source/actor/RNG/base rate intact.
Windows/Linux each passed203selected combined checks, including four-family
complete-state unity equivalence, nonunit native references, failed staged work,
original default math and causal/resource/fixed-replay regressions. Ten-file
scoped static commands, costs and retained failures are in
`artifacts/runs/bounded-rate-action-20261023/RESULTS.md` and raw JSON receipts.
Five unchanged full checkpoint/consolidation parity cases were reused/deselected.

This is single-agent self-review and scoped engineering evidence. No rate
optimization, retention/recovery, controller efficiency or CPC advantage is
demonstrated. Full R7.1 (remaining sleep/replay combinations), R7.7 and release/science goals
remain open. Add effects through original native/runtime ports with bounded
failure accounting; never select parameters from sealed task outcomes.
