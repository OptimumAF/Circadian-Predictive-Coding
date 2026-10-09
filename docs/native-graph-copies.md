# Actual copy sequence observation

`core/native_graph_copy.py` observes the real native snapshot/restore, checkpoint
capture/build/restore-input, and inbox capture/materialization copies. Each event
borrows the actual producer and original source before allocation, then the actual
target and copier memo after copying. The observer must validate producer and
source identities; purpose labels and observation records grant no authority.

Why this: one checkpoint operation copies several different roots. A single
window keeps its original attempt, notification and read limits across all of
them. A fault poisons the remaining sequence; selecting another root never renews
its allowance. Borrowed readers expire at callback exit, even between copies.
Scopes reject inherited contexts/threads, nesting, reopening and reentry, and
release all producer/source/target/memo roots. Saved raw references require the
caller's own accounting.

Native state observation and sequence observation use the same actual memo when
both are active. Model constructor observation has its own independent channel.
Unobserved copy sites retain their original `deepcopy(source)` behavior. Native
validation, original checkpoint attempt/retained-model history, inbox consent
guards, enrollment and single publication remain in their original owners.

```python
from src.core.native_graph_copy import copy_graph, observe_graph_copies
from src.core.native_model_copy import ModelCopyLimits

producer = object()
source = {"row": []}
stages = []

def observe(owner, kind, stage, read, lookup):
    assert owner is producer
    if stage == "copied":
        original, target = read().source, read().target
        assert lookup(original) is target
        assert lookup(original["row"]) is target["row"]
    stages.append((kind, stage))

with observe_graph_copies(observe, ModelCopyLimits(2, 4, 16)):
    snapshot = copy_graph(producer, "checkpoint_capture", source)
    restored = copy_graph(producer, "checkpoint_restore_state", snapshot)
assert restored["row"] is not source["row"]
assert len(stages) == 4
```

Run scalar controls with `python -B -m pytest tests/test_native_graph_copy.py`.
After correctness gates, run the separately budgeted actual checkpoint chain
controls in `tests/test_native_checkpoint_graph_chain.py`.

The application still needs original ledger admission before every authorized
copy, weak or owned-accounted retained witnesses, original consent/receipt/holder
checks and ledger anchor transition at actual handoff. Existing constructor-only
witnesses remain invalid after native restore replaces rows or handoff changes
runtime/inbox identity. Nonempty other-holder capture remains refused. Do not
authorize it by matching contents, keys, digests, purpose labels or portable data.
