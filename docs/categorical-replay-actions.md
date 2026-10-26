# Fixed categorical TRAIN replay actions

`CategoricalReplayAction` adds opt-in replay to the existing finite
`ResourceSharedRuntime` API. The controller selects an ID, with no targets,
callbacks, permissions, budgets or replacement bank supplied at selection time.

## Compose existing owners

Given an ordinary TRAIN runtime, original native source/gate, complete three-step
TRAIN episode and separate fresh finite replay `ToyBudgetSession`:

```python
from src.app.categorical_replay_action import CategoricalReplayAction
from src.app.resource_sharing import ResourceSharedRuntime
from src.core.adaptation_action import AdaptationAction, AdaptationEnvelope
from src.core.circadian_predictive_coding import ReplayRetentionBudget

replay = CategoricalReplayAction(
    runtime, source, replay_budget, (episode,), ReplayRetentionBudget(1, 2048),
    states_equal=same_complete_native_state, definition=native_diagnostic_id,
)
shared = ResourceSharedRuntime(
    runtime, gate,
    action_envelope=AdaptationEnvelope(
        (AdaptationAction("wait"), AdaptationAction("wake", 1),
         AdaptationAction("replay", replay_examples=3)),
        max_decisions=8, max_wake_updates=2, max_replay_examples=3,
    ),
    replay_action=replay,
)
poll = shared.train_action("replay")
print(poll.replay_receipt, poll.training.deferred_reason, poll.progress)
```

The amount must equal the whole bank's transition count. Bank limits are1..4
distinct complete TRAIN episodes,64transitions, and original encoded retention
caps no greater than128examples/1MiB. Existing codec validates episode geometry;
native policy dimensions/classes must match. Numeric encoded bytes exclude
Python objects and native temporaries; sampled RSS is reported separately.

Why this: continuous rehearsal has a different action representation. Reuse the
categorical codec, exposure budgets, complete native forks and consolidation
rather than adding a bank sampler or another stream runner.

## Ownership, costs and failure

The executor binds once to the original runtime/gate. Source, actor, candidate,
versions, bank, budgets, clocks, sampler and receipt prefixes remain pinned.
One expected full candidate advances only on an observed successful native wake
receipt or this executor's validated replay commit. Wake/replay/wake retains
separate wake/replay charges and aggregate native work. Native counters include
both effects; wake receipts count wakes only.

Pause, serving and resource deferrals consume no replay fork/training/exposure.
Sharing admissions stay paid for subsequent known refusals. An additive actor
`snapshot_state_if_idle` port refuses contention without waiting while a runtime
lease is held; existing blocking methods are unchanged. Pre-event ownership-busy
refusals allow a later paid admission; consumed event IDs are never retried.

Replay forks an independent learner, restores the complete candidate and charges
exposure before native entry. It validates diagnostic and clock/row deltas before
original consolidation. Unknown work, invalid returns, late resource stops and
uncertain publication retain charges/known receipts and stop further native work.
A late stop can leave a known committed receipt. Mutated preceding model, bank,
budget, owner or receipt evidence is refused; budgets are never renewed.

Storage holds one expected candidate plus original source/actor snapshots, the
fixed bank and at most1024wake/32consolidation metadata records. It does not retain
a success-model chain. Trusted local composition is not a hostile Python sandbox,
durable continuation format, hard native preemption or process-tree RSS guarantee.
No final/probe/promotion/controller-fit port is supplied.

Default wait/wake remains. Replay without an executor, combined wake+replay,
replay+rate, gate and sleep effects remain refused. Separate per-update rate+wake
is supported by the [optional native port](learning-rate-actions.md).
Full R7.1 and broader goals remain open.

## Run and verify

```console
python -m scripts.bounded_replay_action_example --source artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json --output replay-example.json
python -m pytest -q tests/test_categorical_replay_action.py
python -m ruff check src/app/categorical_replay_action.py scripts/bounded_replay_action_example.py tests/test_categorical_replay_action.py
python -m ruff format --check src/app/categorical_replay_action.py scripts/bounded_replay_action_example.py tests/test_categorical_replay_action.py
python -m mypy --no-incremental --explicit-package-bases --follow-imports=silent src/app/categorical_replay_action.py scripts/bounded_replay_action_example.py tests/test_categorical_replay_action.py
```

Use a fresh output path. Source is the exact qualified96update/576row BP record;
no repeated fit. All seven-file combined static commands and native execution
receipts are in `artifacts/runs/bounded-replay-action-20261023/`.
Windows/Linux each passed174combined checks plus five targeted scalar cases;
five unchanged native math/parity cases were deliberately reused, not rerun.
The sole standalone case passed deferrals then wake6/replay3/wake6, ending99/591
with source/actor/RNG unchanged. This verifies mechanics, not replay utility,
retention, recovery or CPC advantage.

Extend effects through existing native/runtime ports with explicit budgets,
failure accounting and refusal tests. Do not select policies/doses from sealed
task outcomes.
