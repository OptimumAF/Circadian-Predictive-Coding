# Original replay-write observations

`src/core/replay_write_origin.py` owns an explicit invocation scope with bounded
input rows, retained reference count and notification attempts. It reports actual
model/input identities, batch or row ranges, new snapshot references and final
retained references. It performs no array copy, model operation, IO or persistence.
`src/app/replay_write_origin.py` attaches the original managed update record from
the b3a producer port. Neither module certifies provenance, consent or recovery.

Compose `ManagedReplayWriteAccess(model_reference, callback, limits)` as the
`native_observer` of the original owner. The inward model-reference port must
return that learner's original native model. For an already validated NumPy CPC
learner this is its owned `_model`; no model is cloned or restored. The original
model and detached native input references must match at the actual write.
Foreign/nested/reopened windows, callback reentry, invalid ranges, exhausted limits
and foreign thread access refuse. Default calls add no native fields or array copies.

| Stage | Observed reference |
| --- | --- |
| `begin` | Original inputs and prior retained snapshots, before replay prediction/copy. |
| `before_copy` | Native batch or row range, before its array allocation. |
| `copied` | Actual new ReplaySnapshot before insertion/eviction. |
| `retained` | Actual final snapshot references after native storage publication. |

Budgeted oversized rows emit `before_copy` but no snapshot event. Their temporary
array allocation remains native behavior. Dedup and eviction use the existing
policy; origin consumers use actual snapshot identity, never hashes or equality.
Disabled replay emits only begin/retained. Notification attempts are monotone;
limit/fault refusal does not reset them. Native parameters can already have changed
before a replay callback fails; original managed uncertain/committed-failure paths
preserve work and refuse retries. A late write observer fault preserves the buffer.

Reads work synchronously only during their original callback and reject other
threads. The scope owns a ContextVar token and resets it before releasing references;
foreign-context close cannot destroy the original window. The managed bridge closes
on original completed/failure outcomes, including observer failures. A context copy
is not new provenance authority. Callbacks are trusted observational code and must
return, avoid mutation and use owned retention/accounting for references they keep.
Closing releases scope-owned references; it cannot revoke caller-owned copies.

Why this: preserving the legacy model dictionaries keeps snapshot/codec schemas
stable while recording the exact producer-to-copy boundary. This is a prerequisite
for a bounded persistent row ledger. `completed` precedes final resource checks;
every consumer result is provisional until the enclosing call and original consent
rechecks succeed. No row ledger, consent certificate, retained fork linkage, restore
permission or scientific outcome is established by these reports.

## Pure reference example

No native model, array, sampler, worker, budget or update is created.

```python
from src.core.replay_write_origin import ReplayWriteLimits, observe_replay_writes, begin_replay_write

model, features, targets, snapshot = object(), object(), object(), object()
stages = []
def observer(stage, read):
    value = read()
    assert value.model is model and value.features is features and value.targets is targets
    stages.append(stage)

with observe_replay_writes(model, features, targets, observer, ReplayWriteLimits(1, 1, 4)) as window:
    assert begin_replay_write(model, features, targets, [], 1, "batch") is window
    window.before_copy(0, 1)
    window.copied(snapshot)
    window.finish([snapshot])
assert stages == ["begin", "before_copy", "copied", "retained"]
print("closed original replay reference window")
```

## Verification and next extension

The exact scoped test/type/Ruff/guide commands and outcomes are recorded in
`docs/development-log.md` and stage command receipts. Initial fake/current gates
must pass before the separately declared native storage-only tests. Tests preserve
native state byte parity for batch/hash/FIFO/reservoir and directly qualify
duplicates, evictions, oversized rows, disabled buffers and early/late faults.
They train no weights, replay no data and restore no model.

Next: consume this original producer/write pair in a bounded persistent ledger,
using weak payload references or owned retention and original consent/tombstone
authority. Qualify terminal outcomes, retained copies/forks, actors, checkpoints,
promotion and erasure before admitting full compound capture or recovery. Full
R3.5b2e5b3/e5b/e5 remain unchecked.


## Subsequent row-ledger increment

The observations above now feed the separate opt-in
[original candidate row ledger](managed-replay-origins.md). Its original owner
operation and final checks qualify current-candidate metadata only. These callbacks
themselves remain provisional. Retained holders/forks/actors/checkpoints/promotion/
restore/erasure and full compound recovery qualification remain unfinished.
