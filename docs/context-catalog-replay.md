# Contextual TRAIN replay through the existing executor

`scripts/context_catalog_replay.py` supplies the fixed D9 old-v1 TRAIN bank to
the existing `CategoricalReplayAction`. It supports the already implemented
conventional paired-head learner, with its original unit-mistake diagnostic.
No model, optimizer, replay policy, controller, stream runner or session is added.

Why this:the existing full catalog requires wait, wake and replay-then-wake.
Its old ticket composition uses D7 BP source/teacher/diagnostic assumptions;
wait/wake-only contextual selection already exists and is not a replacement
for that three-action contract. This module supplies the missing context replay
capability. It does **not** yet attach a catalog to the context live command or
train/apply a contextual catalog chooser.

## Inputs and ownership

`compile_context_bank(spec, old_train)` requires the exact existing
`DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG)` and ordinary `train` old-v1
six-row prototype returned by `context_head_serving.prepare_data(spec)`.
The schema is9float64 inputs/six classes, ordered v1invoice then v1password,
with original `update-1-0`..`update-1-5` identities, no preexisting episode.
Each observation/action is checked against the existing authentic sandbox
`_ticket_trace`; mismatched labels, context, order or features are refused.
Nothing is normalized, repaired, clipped, inferred from evaluation or selected.

The result is two detached immutable complete three-step TRAIN episodes in that
order. All transitions retain public policy context, actions, terminal flags and
reward evidence. The terminal next-observation is the existing codec's explicit
zero placeholder after commit, **not** an observed final environment frame.
Only observation/action pairs train. The compiler checks declared provenance and
semantic consistency; callers remain responsible for real TRAIN data provenance.
Probe/final/guard records are not remapped to TRAIN.

`compose_context_replay(runtime, learner, replay_budget, spec, old_train)`
requires the existing paired-head learner with the fixed D9 routing policy. It
returns the existing executor using exact-state equality, native unit-mistake
diagnostics and `ReplayRetentionBudget(2,4096)`. Two retained encoded episode rows
plus zero dummy targets occupy2608numeric bytes. Retention counts episodes;
replay exposure counts all6training rows per event. Python objects, validation,
temporary decoded arrays, source/model copies and process RSS are additional.

The caller supplies the existing ordinary TRAIN `ActorShadowRuntime`, original
independent finite replay budget and wake budget. No session is created or
renewed. Generic executor guards still own source/candidate/actor state, original
budget clocks, previous receipts and bank identity. A failed native return keeps
paid exposure and attempted work, refuses later native calls and creates no
successful consolidation receipt. Replacing even equal-valued bank/success owners
is refused. No unbounded model/success history is retained.

## Composition

Use after constructing the existing runtime/source and before any work. Source
qualification and runtime/budget/actor ownership remain the caller's obligations.
The helper does not load, qualify or restore a historical source.

```python
from scripts.context_catalog_replay import compose_context_replay
from scripts.context_head_serving import prepare_data
from scripts.run_ticket_workflow import CONTEXT_FEATURES
from src.app.resource_sharing import ResourceSharedRuntime
from src.core.adaptation_action import AdaptationAction, AdaptationEnvelope
from src.core.catalog_contextual_controller import ContextualArmCatalog
from src.core.decision_features import DecisionFeatureSpec
from src.core.ticket_workflow import CATALOG

spec = DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG)
training, _ = prepare_data(spec)
replay = compose_context_replay(runtime, source, replay_budget, spec, training[1])
catalog = ContextualArmCatalog(AdaptationEnvelope(
    (AdaptationAction("wait"), AdaptationAction("wake1", 1),
     AdaptationAction("replay6-wake1", 1, 6)),
    max_decisions=8, max_wake_updates=2, max_replay_examples=12,
))
shared = ResourceSharedRuntime(
    runtime, gate, action_envelope=catalog.envelope, replay_action=replay,
)
# Original causally arrived TRAIN labels must be recorded before a wake.
# The original runtime permits2consolidations; sharing quota covers2wake+2replay.
result = shared.train_action("replay6-wake1")
```

This demonstrates composition, not a command to run another frozen study.
The caller must declare its own original caps before any model/teacher work.
The example's replay6-then-wake pays one6-row replay pass plus its ordinary wake
packet, under separate original charges. No rate/gate/sleep capabilities are
granted to the conventional learner. All existing entry points/defaults remain.

## Verification and extension

```powershell
python -m pytest -q tests/test_context_catalog_replay.py
python -m ruff format --check scripts/context_catalog_replay.py tests/test_context_catalog_replay.py
python -m ruff check scripts/context_catalog_replay.py tests/test_context_catalog_replay.py
python -m mypy --explicit-package-bases --follow-imports=silent scripts/context_catalog_replay.py tests/test_context_catalog_replay.py
```

The focused suite exercises the real conventional learner and actual replay/wake
executor, including permanent failed-attempt accounting. It is not a demonstration
of task success, controller utility, comparative efficiency, generalization or
CPC benefit. Reuse [the existing learned context control](context-head-control.md)
and [its private further-learning probe](context-head-probe.md) within their
original assumptions. Old D9 BP fit and catalog-native failures stay unchanged.

[Executed results, exact budgeted commands and limitations](reports/context-catalog-replay-20261027.md).
Next extend the current context live selection seam with the **full** existing
catalog/isolated trials and this bank, preserving source binding, history clocks,
three arms, replay costs and final/probe isolation. Do not add a duplicate
wait/wake-only controller, new runner or retrospective study retry.
