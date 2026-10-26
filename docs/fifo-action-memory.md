# Bounded FIFO action memory

This opt-in native learner continues acquiring exact public-observation counts by
retiring its oldest stored entry when a new key arrives at capacity. The original
`observation_memory` control still refuses over-capacity batches. Its accepted
MiniGrid capacity negative is preserved.

## Behavior and cost

- The method is `observation_memory_fifo`, with a distinct policy and full snapshot.
- A hit adds a class count without refreshing queue position. An evicted key
  starts fresh at the tail if it returns. Rows are processed in input order.
- A batch can contain more distinct keys than fit. Entries inserted earlier in the
  same batch may be retired; the native learner publishes the entire valid batch
  atomically. Invalid arrays, labels, state or overflow leave its state unchanged.
- Every retired entry contributes all its stored exposures to `retired_examples`.
  Retained counts plus retired examples equal lifetime training examples.
- `evictions` and `retired_examples` are two bounded scalar totals. No eviction log
  or per-key timestamps accumulate. Both persist through owned fork/restore.

Only the existing16-feature/five-class and296-feature/seven-class shapes are
accepted. Defaults remain64 keys and10752 retained-table bytes. A MiniGrid caller
must explicitly request155136 bytes to permit64 entries of2424 bytes; merely
selecting296/7 allows four entries under the default byte cap. Effective capacity
is the lesser of the entry cap and whole entries fitting the byte cap. Zero
capacity refuses an update.

`FIFO_RETIREMENT_METADATA_BYTES` reports **16 additional logical bytes** for the
two retirement totals, outside the retained-table allowance. This does not make
total model memory identical to the refusal control. Python objects, temporary
batch copies, shared policy/native clock fields and process overhead are outside
that table/extra-counter calculation; actual process RSS must be measured in any
future experiment. Storage updates are not SGD and the diagnostic is not a loss.

## Run a small example

Use the existing NumPy environment; there are no new dependencies or environment
variables. From a Python script run with `.venv/Scripts/python.exe -X utf8 -B`:

```python
import numpy as np

from src.adapters.fifo_action_memory_learner import FifoActionMemoryLearner
from src.core.fifo_action_memory import FifoActionMemoryPolicy

learner = FifoActionMemoryLearner(
    FifoActionMemoryPolicy(max_entries=2, max_payload_bytes=336)
)
features = np.zeros((4, 16), dtype=np.float64)
features[:, 0] = [0.0, 0.25, 0.5, 0.0]
learner.train_batch(features, np.array([[4.0], [1.0], [2.0], [4.0]]))
state = learner.snapshot_state()
print([entry.features[0] for entry in state.entries])  # [0.5, 0.0]
print(state.evictions, state.retired_examples)        # 2 2
```

This example is a deterministic array fixture. It establishes no game skill,
generalization, plasticity or circadian advantage.

## Boundaries and explicit conversion

```text
src/core/fifo_action_memory.py               pure ordered count/retirement rules
src/adapters/fifo_action_memory_learner.py    owned native ports and atomic commit
tests/test_fifo_action_memory.py            core/state/array boundary cases
tests/test_fifo_action_memory_probe.py      existing stream/probe integration
```

The core reuses existing exact geometry, entry and array validation. Dependency
direction stays inward. The adapter exposes the existing `train_batch`, `predict`,
`snapshot_state`, `restore_state` and `fork` ports. Callers still own data roles,
label arrival, budgets, scoring, IO and promotion.

`convert_action_memory_to_fifo(old_snapshot)` explicitly validates/detaches an old
exact-memory snapshot, preserving its ordered table and lifetime counters while
starting retirement totals at zero under the new identity. Construct a matching
FIFO learner from the converted policy and restore that snapshot. Neither native
learner accepts the other snapshot type through ordinary restore. Conversion
resumes no original serving authority, consent or app budget.

The existing delayed TRAIN stream test shows labels arriving before replacement;
source state remains unchanged and four shadow commits have four receipts, with
zero final scoring. Separate probe tests use fresh budgets and disposable copies:
A is initially known, B/C cause FIFO loss of A, and returning A can be learned
again while B is lost. All retained/retired exposures remain counted. These are
fixture-level integration results, not actual MiniGrid measurements.

## Checks and safe extension

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest tests/test_fifo_action_memory.py tests/test_fifo_action_memory_probe.py tests/test_action_memory.py tests/test_minigrid_action_memory.py tests/test_fresh_task_probe.py -q
.venv/Scripts/python.exe -X utf8 -B -m ruff check src/core/fifo_action_memory.py src/adapters/fifo_action_memory_learner.py tests/test_fifo_action_memory.py tests/test_fifo_action_memory_probe.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check src/core/fifo_action_memory.py src/adapters/fifo_action_memory_learner.py tests/test_fifo_action_memory.py tests/test_fifo_action_memory_probe.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/core/fifo_action_memory.py src/adapters/fifo_action_memory_learner.py tests/test_fifo_action_memory.py tests/test_fifo_action_memory_probe.py
```

Repeat the scoped type command with `--platform linux`; this is not a Linux
runtime claim. Final254 focused cases pass, including75 new core/native and four
stream/probe cases; scoped statics and independent source/evidence review pass.
[Exact commands and limitations](../artifacts/runs/fifo-action-memory-20261017/RESULTS.md),
[decision](adr/ADR-0386-bounded-fifo-action-memory.md).

The [matched MiniGrid E8/A8 assay](minigrid-action-memory.md#opt-in-matched-new-task-and-return-comparison)
now executes on disposable saved-state copies. Mature FIFO learns E and relearns A
after partial forgetting, at26evictions/704retired exposures; eightoriginalkeys
remain absent. Known TRAIN teacher-forced scores are not live gameplay. The mature
refusal failure and missing endpoints remain recorded. The separate frozen
[saved-return game readout](minigrid-action-memory.md#frozen-saved-return-game-readout)
now records zero-training actual A/E games: mature FIFO A2/E0→A2/E2 successes.
Independent saved arithmetic verifies379decisions, but the native qualification
remains failed/SPENT because of a launcher runpy warning. No retry ran. Broader
retention/held-out layouts/neural usefulness remain open; raw encoding/defaults,
role isolation, costs and all negative results are preserved. Next is a distinct
public nuisance-invariant encoding control, not another same-case audit/retry.
