# Bounded supervised action memory

`key-door-action-memory` learns exact public observation→action counts through the
existing `NativeLearner`, `AttributionStream` and live key/door episode interfaces.
It is a simple control for acquisition, retention and returning contexts.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario key-door-action-memory --describe
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario key-door-action-memory
```

The runtime requires NumPy and the existing local dependencies; this route does
not import Torch. Preview constructs no learner, episode or resource sampler.
The fixed case has a90-second internal deadline and768MiB process RSS cap.
Use a separately reviewed supervisor/protocol for comparative execution; each
ordinary CLI invocation is new work, not an automatically renewed allowance.

## Structure and boundaries

```text
src/core/action_memory.py                 pure bounded count transition/prediction
src/adapters/action_memory_learner.py      owned snapshot/fork NativeLearner adapter
src/adapters/key_door_memory_pilot.py      fixed recipe using existing stream/live runner
tests/test_action_memory.py               atomic updates/caps/conflicts/ownership
tests/test_minigrid_action_memory.py      explicit296/7 geometry and stream boundary
tests/test_key_door_memory_pilot.py        real learner integration/isolation/faults
tests/test_action_memory_probes.py         isolated E16 copies/charges/faults
```

Exact finite float64 B×16 public initial/current coordinates in[-1,1] are keys.
Hard labels are float64 B×1 classes0..4; batch size1..128. Each key stores five
counts. Known predictions are empirical frequencies, unseen predictions uniform
five; ties choose the first argmax action. Conflicting labels remain distributions.
No context identifiers, legal-action masking, nearest-neighbor matching, eviction,
retries, final data, probes, replay or promotion are used.

Whole batches validate before commit. Malformed later rows, capacity or counter
excess refuse atomically. The fixed learner holds at most64 keys and10752 logical
payload bytes (16float64 coordinates+5int64 counts=168/key). Aggregate exposure
counters cap at2**53−1. Full snapshots validate count sums, update/exposure
consistency and policy; restoration/forks detach every record. Predictions leave
full native state unchanged. Capacity exhaustion is an explicit error.

Eight supervised storage updates expose64 original TRAIN rows in A/B/C/A order,
with delayed labels and checkpoints0/2/4/6/8. All30 TRAIN score calls/240rows
finish before30 known TRAIN live episodes (both cues in A/B/C at each point).
Every episode owns fresh public initial-frame history and frozen native copies,
raw argmax actions and at most12 calls. Original source/actor/candidate, budgets,
receipts, clocks and previously published outputs remain checked. Partial failed
work and unavailable readbacks stay explicit.

Why this: accepted neural controls fail this small known TRAIN task. A production
count learner tests whether a simple bounded learner can use the information
already supplied, before adding more circadian features. It shares observations,
labels and exposure, but has different capacity and computation. Storage updates
are **not SGD**. Neural-parameter and PC-neutral parity fields are null/N/A.
Logical payload excludes Python object overhead, history, snapshots, buffers and
process resources; measured wall time and RSS include those costs.

## Explicit MiniGrid geometry

The same native learner can consume the existing MiniGrid public first/current
encoder's296-coordinate packets with seven native actions:

```python
from src.adapters.action_memory_learner import ActionMemoryLearner
from src.core.action_memory import ActionMemoryPolicy
from src.core.demonstration_episode import episode_arrays
from src.core.minigrid_public_history import encode_public_history_episode

policy = ActionMemoryPolicy(feature_dim=296, num_classes=7, max_payload_bytes=155136)
learner = ActionMemoryLearner(policy)
# train_episode must already be released TRAIN evidence, with148-coordinate frames.
features, targets = episode_arrays(encode_public_history_episode(train_episode))
diagnostic = learner.train_batch(features, targets)
probabilities = learner.predict(features)
```

Only the exact geometry pairs16/5 and296/7 are admitted. Each MiniGrid entry
costs2424 logical bytes (296float64 keys+7int64 counts); the absolute64-key
ceiling costs155136B. Request that allowance explicitly. Changing only dimensions
retains the default10752B, enough for four distinct MiniGrid keys; a fifth key
refuses the whole batch. `policy.entry_payload_bytes` reports the geometry's law.
The legacy `ACTION_MEMORY_ENTRY_BYTES` constant remains168 for old16/5 callers.
Existing CLI recipes, caps, snapshots and saved scientific protocols stay16/5.

Snapshot admission binds every entry's exact feature/count shape to its policy.
Prediction misses are uniform1/7, conflicts remain empirical distributions, and
all atomicity/counter/ownership rules still apply. This adds no final-data access,
new role release, eviction, recurrent state or nearest-neighbor behavior. The
caller and existing stream continue to own data-role isolation. No MiniGrid live
skill, retention, generalization or comparative benefit follows from shape support.
A separately frozen runner case is required for those observations.

## Validation and extension

Run focused tests:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest tests/test_action_memory.py tests/test_minigrid_action_memory.py tests/test_key_door_memory_pilot.py tests/test_action_memory_probes.py tests/test_public_history_inspector.py -q
.venv/Scripts/python.exe -m ruff check src/core/action_memory.py src/adapters/action_memory_learner.py src/adapters/key_door_memory_pilot.py
.venv/Scripts/python.exe -m ruff format --check src/core/action_memory.py src/adapters/action_memory_learner.py src/adapters/key_door_memory_pilot.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/core/action_memory.py src/adapters/action_memory_learner.py src/adapters/key_door_memory_pilot.py
```

The recorded final gate covers all nine affected paths and both type platforms.
Engineering tests execute additional real count updates/tiny physics; these costs
are separate from the single qualification case, which records eight updates.
Use repository Ruff format/check and scoped mypy with both win32 and linux
platform targets. Linux-target typing is not Linux runtime execution.

Add another learner through the existing native port and an explicit method
identity. Preserve immutable benchmark ownership, input roles, original budgets
and private episode publication. Freeze a distinct small protocol before changing
capacity, miss/conflict behavior, exposure, seeds or evaluation distribution.
Current evidence and remaining claims are recorded in the
[stage results](../artifacts/runs/bounded-action-memory-20261017/RESULTS.md).

## Isolated E16 capacity assay

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario key-door-action-memory-probe --describe
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario key-door-action-memory-probe
```

This opt-in alternative runs the original eight benchmark storage updates and
30 TRAIN score calls, then uses the existing checkpoint/fresh-task probe helpers
on disposable full native copies at benchmark0/end8. Each copy gets its own
16-update budget and original reserved E training/test packets, with relative
readouts0/4/8/16 and decision diagnostics. Known-TRAIN live evidence is reused;
this mode runs no live episodes. The ordinary memory CLI remains unchanged.

Only probe_train labels train the copies. Probe_test labels are for metrics;
final data is never constructed, and no probe learning is promoted or used to
select benchmark behavior. Original source/actor/candidate, receipts, clocks,
budgets and preceding reports remain checked. Private cohort publication retains
partial work and refuses changed/replaced earlier results before more work.

The same64-key/10752-byte cap and fixed miss/conflict laws apply. A refusal is a
valid result. Reported benchmark snapshots are observed state; reconstructed
probe table growth is derived from saved inputs and count laws, not a native
allocation trace. Whole-process time/RSS include diagnostics, data collection,
forks and guards. Encoded arrays use8704 TRAIN bytes plus19584 reserved bytes;
these exclude Python records, owned copies and table overhead.

A completed case uses40 storage updates/320rows (8benchmark+32probe),46score
calls/368rows,26teacher episodes/208physics steps. Engineering tests execute
additional real count/physics work separately. No SGD, replay or serving update
is introduced. These reserved deterministic E packets measure capacity to fit
this task, not held-out environment generalization or a circadian advantage.
Each comparative execution needs its own frozen small protocol and review;
previous SPENT cases do not authorize retries. See the
[E16 results](../artifacts/runs/action-memory-probes-20261017/RESULTS.md).

The one reviewed local E16 case completed with both copies improving from12.5%
to100% reserved accuracy (CE1.6094379124→0). The mature copy's derived table fills
64keys/10752logical bytes. All benchmark non-live fields and original reserved
packets match accepted evidence exactly. This is deterministic reserved fit;
there is no unseen-environment or circadian benefit claim. The case used2.8821s
internal time and49,098,752B external peak RSS; see the linked evidence for
engineering costs, separate charges, review and remaining requirements.
