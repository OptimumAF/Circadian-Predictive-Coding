# Owned payload erasure primitives

R3.6b1 supplies the mechanisms needed by a coordinated deletion API. It does not
complete R3.6b or the original R3.6 privacy lifecycle task.

## Files and boundaries

```text
src/core/data_erasure.py          payload-free tombstones, erasure counts and port
src/core/inbox_cursor.py          canonical format2 validation; format1 compatibility
src/app/experience_inbox.py       private quiescent payload-reference removal
src/core/circadian_predictive_coding.py  whole replay-buffer erasure
src/adapters/numpy_learners.py    owned native adapter erasure methods
tests/test_data_erasure.py        identity/history/handoff/native-state controls
```

## Replay buffer

`CircadianLearner.erase_replay_payloads()` removes every retained replay snapshot
from its owned model. Historical whole batches have no subject index, so erasure
also removes other subjects' retained rows. This is a deliberate conservative
choice: selective removal would need a proven mapping from records to batches.
The return value counts removed snapshots, examples and array payload bytes.
Default and all supported bounded retention policies use the same clear operation.
Weights, topology, RNG, configured policy, cumulative training/replay telemetry and
hash-based exposure metadata remain unchanged. Backprop has no raw replay buffer
and returns zero counts. Repeated erasure returns zero counts.

## Inbox tombstones

`ExperienceInbox._erase_payloads(keys, reason=...)` is an internal primitive. The
outer candidate must hold its exclusive lease and advance its revision. The
primitive refuses an active drain, validates the complete resulting history before
dropping source/label references, and supports quiescent stopped inboxes.

Tombstones retain episode/sample IDs, actor version, observation/label arrival and
event ID when present, erasure tick and reason (`deleted`, `expired`, `opt_out`).
They contain no feature/target payload. Applied receipts and their native diagnostics
remain for cumulative work accounting; tombstones count against identity capacity
and retain duplicate event protection. Source-only and label-first records can be
removed before both halves arrive. Repeated removal keeps the original tombstone.

A canonical format2 cursor contains nonempty tombstone history; unerased inboxes
still emit format1. Both validate before payload copying. Supported owned checkpoint
handoff transfers tombstones and the original consent hooks, clocks and budgets.

```python
from src.core.data_erasure import ErasedExperience, ReplayPayloadErasure
from src.core.inbox_cursor import InboxCursor, validate_inbox_cursor

record = ErasedExperience(("episode", "sample"), "actor-0", 1, "label", 3, 3, "deleted")
cursor = InboxCursor(2, "candidate-0", 1, (), (), (), 3, False, 0, (record,))
validate_inbox_cursor(cursor)
assert cursor.erased[0].key == ("episode", "sample")
assert ReplayPayloadErasure(0, 0, 0).payload_bytes == 0
```

## Explicit limits and next integration

This removes owned Python references; it does not overwrite RAM, erase external
snapshots/caller copies, undo exposure metadata, or unlearn parameter influence.
`ManagedExperienceOwner.opt_out` still blocks future training without deleting data.
Existing checkpoint, retired owner, prepared promotion and rollback copies are not
cleaned by a primitive on another object. Thus no complete public deletion or
checkpoint non-resurrection API is exposed by this increment.

Next R3.6b2 must coordinate original manager consent/quotas/retention lifetimes,
all owned payload copies and failure behavior before reporting deletion complete.
Retired models and pending checkpoint/promotion/rollback state need invalidation
and cleanup, retaining original budgets/clocks/gates and consumed IDs. Add behavior
tests for refused resurrection and partial cleanup failure first. Do not enable
transient/audit-only categories before their actual purge semantics pass.

## Validation

```powershell
python -m pytest tests/test_data_erasure.py -q -k 'not erase_native_replay_payloads'
python -m ruff check src tests scripts
python -m ruff format --check src/core/data_erasure.py src/core/inbox_cursor.py src/app/experience_inbox.py src/core/circadian_predictive_coding.py src/adapters/numpy_learners.py tests/test_data_erasure.py
python -m mypy --platform win32 --no-incremental
python -m mypy --platform linux --no-incremental
```

The six new fixed native wakes are separately reserved once after source freeze.
They compare full native state except the replay deque and preserve the existing
one-update budget. No new external prediction, sleep, sweep or scientific result.
