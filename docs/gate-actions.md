# Per-update native plasticity-gate actions

A finite action can select a plasticity setting for bounded wake updates through
the existing TRAIN inbox and serving-priority gate. It changes no base policy,
permissions, budget, persistent gate setting or snapshot schema.

```python
from src.app.resource_sharing import ResourceSharedRuntime
from src.core.adaptation_action import AdaptationAction, AdaptationEnvelope

shared = ResourceSharedRuntime(
    runtime, gate,
    action_envelope=AdaptationEnvelope(
        (AdaptationAction("unit", wake_updates=1, gate_setting="neutral"),
         AdaptationAction("native", wake_updates=1, gate_setting="active"),
         AdaptationAction("wake", wake_updates=1)),
        max_decisions=8, max_wake_updates=3,
        gate_settings=("unchanged", "neutral", "active"),
    ),
)
poll = shared.train_action("unit")
print(poll.requested_gate_setting, poll.training.deferred_reason,
      poll.training.updates, poll.progress)
```

`runtime` is an existing ordinary TRAIN `ActorShadowRuntime` with a compatible
native candidate. Controllers still supply only a declared action ID. The owner
freezes the finite catalog and hard envelope. Generic learners need the optional
`GateControlledLearner` port for nondefault settings; unsupported choices fail.

## Exact semantics

| Bound TorchFeatureLearner | unchanged | neutral | active |
| --- | --- | --- | --- |
| BP / PC | Original public update | Refused | Refused |
| neutral CPC | Original public update | Original unit-gate behavior | Refused |
| gated CPC | Original public update | Unit plasticity for this update | Original bound plasticity rule |

The scalar must be an exact native string. No callable, array, boolean, custom
object or new parameter vector is accepted. Nondefault gate choices require
positive wake, unit learning-rate multiplier, zero replay and zero sleep.
Gate-only no-wake, gate+rate, gate+replay and gate+sleep combinations fail.

For gated CPC, neutral replaces only the per-unit plasticity multiplication in
the existing hidden-output, input-hidden and hidden-bias gradients. The output
bias retains its native ungated update. Chemistry, reward scaling, importance,
traffic, age, energy history, latent relaxation and sleep clocks evolve under the
original policy. **Neutral does not switch gated CPC into neutral CPC or PC.**
In particular, reward modulation remains active when its bound config enables it.

The effect is applied to staged native state. Complete state validation succeeds
before the native payload is published; failed work leaves the live native
weights, gradients, optimizer and chemistry/clocks unchanged. Subsequent default
updates use the original policy with the actual resulting adaptive history.
Active and unchanged are checked against complete default continuations; neutral
CPC retains the existing PC parameter reference. No baseline was tuned or changed.

## Admission and accounting

`validate_gate_setting` checks scalar capability without model construction or
payload reads. The shared wrapper, direct inbox/runtime route and native step
refuse unsupported owners before copying training data or calling resources.
Original role, consent, causal labels, source/version and runtime checks still
apply. Probe owners cannot enroll finite action catalogs; no final/probe data port
or promotion is added.

Future-label, paused, serving-active and resource deferrals train no rows.
Each known selection still spends its decision. Wake bounds, sharing admissions,
original native budgets, clocks, receipts and unknown failures retain their rules.
A lost native return stops the inbox and remains paid/unknown; a completed update
followed by a late stop keeps its receipt. No authority is renewed. Polls expose
`requested_gate_setting`, including no-work deferrals. Durable AppliedExperience
receipts remain unchanged; retain the catalog/poll if action provenance is needed.
The existing counted ticket learner also charges gate ports exactly once.

## Focused verification

```console
python -m pytest -q tests/test_gate_actions.py -k "not native_fixed_three_update_gate_case"
python -m ruff check src/core/learner_ports.py src/core/resnet50_variants.py src/adapters/torch_feature_learner.py src/app/learner_step.py src/app/experience_inbox.py src/app/actor_shadow.py src/app/resource_sharing.py src/core/adaptation_action.py scripts/ticket_workflow_native.py tests/test_gate_actions.py
python -m ruff format --check src/core/learner_ports.py src/core/resnet50_variants.py src/adapters/torch_feature_learner.py src/app/learner_step.py src/app/experience_inbox.py src/app/actor_shadow.py src/app/resource_sharing.py src/core/adaptation_action.py scripts/ticket_workflow_native.py tests/test_gate_actions.py
python -m mypy --no-incremental --explicit-package-bases --follow-imports=silent src/core/learner_ports.py src/core/resnet50_variants.py src/adapters/torch_feature_learner.py src/app/learner_step.py src/app/experience_inbox.py src/app/actor_shadow.py src/app/resource_sharing.py src/core/adaptation_action.py scripts/ticket_workflow_native.py tests/test_gate_actions.py
```

175 distinct selected software checks qualified per platform, including affected
rate/inbox/resource/default regressions; 26 affected pure checks repeated after a
direct pre-payload admission repair. Scoped statics pass. A separate fixed native
case, launched only after prerequisites passed, paused/deferred then performed
neutral2rows, active2rows and unchanged2rows. Exactly3updates/6rows/6latent
iterations; source+actor0, base config and Python/NumPy/Torch RNG preserved.
This is a tiny engineering validation, not evidence of useful task adaptation,
efficiency, safety or a CPC advantage. Full R7.1 sleep/combinations, controller
generalization, deployment, science and release requirements remain open.

Exact commands, budgets, failures and local evidence:
`artifacts/runs/bounded-gate-action-20261023/RESULTS.md`, issue64, ADR-0580.
No new dependency or environment variable is needed by the product API.
