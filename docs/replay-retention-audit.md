# Replay retention audit (P4.1)

This audit describes the current NumPy circadian replay storage. It is a
deterministic storage fixture, not evidence that one retention policy improves
accuracy or forgetting. `tests/test_replay_retention_audit.py` supplies the
fixed arrays and assertions.

## What each path stores

| Path | Stored unit | Capacity rule | Stored values | Priority behavior |
| --- | --- | --- | --- | --- |
| Historical/default | One copied wake batch per `ReplaySnapshot` | `replay_memory_size` deque entries | Input and target arrays, mean absolute prediction error, positive-target fraction | Computed when the batch is stored; later wake updates do not age or recompute it. Repeated content uses another slot. |
| Opt-in bounded continual | One copied labeled row per `ReplaySnapshot` | `ReplayRetentionBudget` examples and NumPy array bytes | One input/target row, absolute prediction error, target value, stable content ID derived from both arrays | A repeated content ID replaces its stored row and refreshes priority. Survivors are the lexicographically smallest content hashes among observed rows. |

The byte limit counts the retained input and target arrays' `nbytes`. It does
not measure Python object, deque, or allocator overhead. The historical path
has no fixed example or byte limit: a slot can contain any accepted batch
size. Both paths receive training examples only through the wake storage
call; replay consolidation disables recursive storage. The bounded continual
runner additionally checks arrived training IDs before replay and on
checkpoint restore.

## Fixed A→B observations

| Fixture | Input | Final storage | Distinct examples | Phase A survivors |
| --- | --- | --- | ---: | ---: |
| Historical two slots | A: one row; B: four rows then two rows, each row two float64 features plus one float64 target | Two B batch snapshots, 144 array bytes | 6 | 0 of 1 |
| Historical three slots with repeated A | A: one row; B: one row; A: same row again | Three snapshots, including two copies of A | 2 of 3 stored rows | 1 distinct A row in two slots |
| Bounded four examples / 96 bytes | A: four rows; B: four rows, each row two float64 features plus one float64 target | Four one-row snapshots, 96 array bytes | 4 | 2 of 4 |

The fixed hash sample retains two A IDs in this fixture. Content hashes do
not encode task identity, so this result provides no general old-task quota
or class balance guarantee. The legacy A snapshot is evicted by arrival
order. A saved historical priority stays numerically unchanged even after a
later wake update changes model weights; prioritized replay can therefore
rank a stale error estimate. The existing optional class-balanced selection
uses the stored positive fraction, not a declared task label.

## Decision boundary for P4.2

Keep the current paths and protocol identities intact while introducing
explicit comparison controls. Predeclare equal retained-example and byte
budgets, replay exposure, and observed task/label information for recent
FIFO versus reservoir policies. Compare old-task retention and downstream
metrics only under the isolated evaluation protocol; do not infer a winner
from this storage canary.

Reproduce the bounded audit with:

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_replay_retention_audit.py tests/test_continual_bounded_replay.py
```
