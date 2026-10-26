# Bounded native component-sleep actions

A finite action can request one component-sleep event on a compatible NumPy
CircadianLearner. The original shared gate admits the attempt; the original
ActorShadowRuntime.consolidate owns copying, quota, commit and receipt. The
controller supplies only an action ID, never a function, native config or budget.

```python
from src.app.resource_sharing import ResourceSharedRuntime
from src.core.adaptation_action import AdaptationAction, AdaptationEnvelope

shared = ResourceSharedRuntime(
    runtime, gate,
    action_envelope=AdaptationEnvelope(
        (AdaptationAction("wait"), AdaptationAction("wake", wake_updates=1),
         AdaptationAction("components", sleep_updates=1)),
        max_decisions=8, max_wake_updates=2, max_sleep_updates=1,
    ),
)
poll = shared.train_action("components")
print(poll.requested_sleep_updates, poll.training.deferred_reason,
      poll.sleep_receipt, poll.progress)
```

Use an ordinary TRAIN runtime whose existing NumPy candidate has components mode,
split/prune/replay disabled, fixed min/current/max width and at least homeostasis
or chemical reset enabled. These are owner-supplied native settings, fixed before
execution. The port does not choose a homeostasis factor or change the config.
TorchFeatureLearner remains a fixed no-sleep adapter; it refuses this action.

## Units, costs and failures

`sleep_updates=1` means one attempted whole component event, **not one optimizer
update or epoch**. The action has zero wake/replay, unit rate and unchanged gate.
Amounts above one and combined effects refuse. Selecting another declared
one-event action can spend another unit only within the original cumulative
max_sleep_updates, sharing admission and consolidation quotas.

Each known selection spends a decision. Pause, serving and resource deferrals
perform no model copies or native events and spend no sleep admission. An admitted
attempt spends a sleep admission before original quota/budget/copy callbacks;
even a later refusal does not refund that admission. Sleep native attempts,
completed preparations and committed receipts are separate counters. Training
budget updates count only TRAIN steps; the original sharing admitted_updates
counter counts admitted wake and sleep attempts. No budget or clock is renewed.

The native port prepares a full detached snapshot with explicit zero replay and
fixed width. It calls existing homeostasis/reset/sleep logic, validates the result
and returns ConsolidatedState. Only the original runtime restores/publishes the
candidate. Source and serving actor are unchanged. A native skip or invalid result
is an error, not an invented successful sleep.

Lost native returns, uncertain restoration or lost publication evidence latch
unknown work and refuse further native actions. A late stop after a verified
commit retains the original receipt and charges. Consolidation versions, quota,
attempt IDs and receipt prefixes are checked before subsequent work using one
bounded metadata pin. Native model success records are not retained in a chain.
Ordinary arrival/wake revision changes remain under their existing runtime rules;
they are not consolidation events. Corrupted or replaced consolidation evidence
is refused. Trusted local Python is not a hostile-code sandbox.

An attached categorical replay executor cannot also execute sleep until a
combined ownership contract is supplied. Managed retention's existing arbitrary
consolidation refusal also remains. Probe owners cannot enroll finite actions;
no final data, replay selection, teacher, controller fitting or promotion is added.

## Verification

```console
python -m pytest -q tests/test_component_sleep_actions.py -k "not native_fixed"
python -m ruff check src/core/actor_ports.py src/core/adaptation_action.py src/adapters/numpy_learners.py src/app/resource_sharing.py tests/test_component_sleep_actions.py
python -m ruff format --check src/core/actor_ports.py src/core/adaptation_action.py src/adapters/numpy_learners.py src/app/resource_sharing.py tests/test_component_sleep_actions.py
python -m mypy --no-incremental --explicit-package-bases --follow-imports=silent src/core/actor_ports.py src/core/adaptation_action.py src/adapters/numpy_learners.py src/app/resource_sharing.py tests/test_component_sleep_actions.py
```

177 distinct software checks qualified per platform, including affected original
action/resource/consolidation/gate/rate checks; 25 affected checks repeated after
the metadata-observer failure repair. Scoped statics pass. A separate first-launch
native case performed two TRAIN updates/four rows and one component event under
the original two-update/one-consolidation/three-admission budget. Recorded input
weights show native 0.8 homeostasis, source/actor/config/RNG are preserved, and
the second wake continues after sleep. This proves scoped execution/accounting,
not useful task learning, comparative efficiency or a CPC advantage.

Evidence: issue65, ADR-0581 and
`artifacts/runs/bounded-component-sleep-20261023/RESULTS.md`. Full R7.1 mixed/
renewal, long-duration deployment, scientific and release requirements stay open.
