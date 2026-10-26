# Owner-local attribution stream continuation

`AttributionStream` in the existing attribution application exposes its ordinary
delayed-label loop. The old `run_attribution_stream(...)` API runs this same loop
to completion and preserves its report schema. Native objectives and replay
selection rules are unchanged.

```text
src/app/coupled_attribution.py              existing loop, session and facade
src/app/stream_rehearsal.py                 owned replay preparation/publication
src/adapters/torch_feature_rehearsal_pilot.py fixed resumed native scenario
scripts/run_coupled_attribution.py          explicit continuation CLI route
tests/test_attribution_stream_continuation.py native pairs/refusal controls
tests/test_rehearsal_continuation.py         independent helper behavior
```

```python
from src.app.coupled_attribution import AttributionStream

stream = AttributionStream(
    learner, training, heldout,
    checkpoint_updates=(0, 2, 4, 6), label_delay=1,
    budget=original_wake_budget, rehearsal=original_rehearsal,
)
stream.run_until(2)                   # absolute committed wake count
pause = stream.pause()
observation = pause.snapshot_state()  # detached data; no restore authority
stream.resume(pause)                 # same original owner; token consumed
stream.run_until(6)
report = stream.finish()             # all final scoring occurs here, once
```

## What resume restores

A pause owns complete native candidate state, inbox events/labels/applied
receipts, delayed tick, checkpoint states, replay rows/partition order/rotation
cursor/events and original consolidation metadata. At one committed boundary,
resume stages an independent candidate, restores the complete native snapshot,
materializes an independent inbox using its existing inward cursor machinery,
and prepares an independent replay buffer. Complete staged state must match.

The original runtime lease covers publication of these prepared payloads.
Original runtime, actor, source learner, budgets, event/wall clocks and receipts
remain binding. The old inbox is retired. No constructor renews a session or
allowance. Replay's native updates, optimizer state and CPC ages/chemistry remain
part of the candidate snapshot; application wake/replay charges remain separate.

The local token is single-owner and single-use. A detached observation cannot
authorize continuation. Corrupted/replaced snapshots, unobserved live changes,
foreign tokens, changed policies/limits, reset counters, original-clock regression,
exhausted deadlines and uncertain native failures refuse further admission.
Opaque staged restore returning incorrect state is also refused before swap.
Scoring callbacks cannot publish completion after changing original authority.
Only current pause state and at most32 planned checkpoint states are retained;
consuming a token clears its payload/owner reference, with no success-record chain.

Why this: restoring a model while resetting replay charges or forgetting pending
delayed labels changes the continuation even when one prediction still matches.

## Supported scope

This is trusted local continuation against the same live owner and authority.
It is not a restart from exported data, a disk codec or restored consent. Pauses
accept supported owned native state trees: exact scalars, dataclasses, dictionaries,
lists/tuples, NumPy arrays and native Torch tensors. Unsupported state types refuse.
Ordinary one-call execution retains its original opaque learner contract.

Renewal still works through the ordinary runner. Pausing a renewal stream or a
managed lifecycle owner is explicitly unsupported. Trajectory/transition replay
and process/durable recovery remain separate unfinished requirements. Heldout
data stay private to scoring and are not included in token observations or passed
to candidate restore, replay or checkpoint observers.

## Fixed CPU verification

```powershell
.\.venv\Scripts\python.exe -m scripts.run_coupled_attribution --describe --scenario torch-feature-continuation
.\.venv\Scripts\python.exe -m scripts.run_coupled_attribution --scenario torch-feature-continuation
```

This executes the previously frozen native feature case with actual pause/resume
at wakes2/4: eight arms,16 reconstructions,60 native calls/108 rows/64 scores,
one CPU/interop thread,60 seconds/768 MiB process RSS/2 MiB JSON. No probes,
teacher, backbone or baseline experiment rerun. A separate saved-data comparison
checks every common arm field against the accepted uninterrupted artifact.
The inherited raw protocol's conservative phrase about no whole-stream resume
does not denote unrestricted process recovery; `restore_scope` states the new
owner-local reconstruction precisely. Raw protocol/evidence is preserved.

Focused native pairs compare full state (including optimizer/gradients/local RNG),
inbox and receipts, checkpoint observations, replay plans and global RNG with
uninterrupted execution. Fake controls exercise refusal and failure boundaries.
Two unchanged tiny NumPy/renewal regressions cover the refactored default loop.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_attribution_stream_continuation.py tests/test_rehearsal_continuation.py tests/test_torch_feature_rehearsal_pilot.py
.\.venv\Scripts\python.exe -m ruff check src/app/coupled_attribution.py src/app/stream_rehearsal.py src/adapters/torch_feature_rehearsal_pilot.py scripts/run_coupled_attribution.py tests/test_attribution_stream_continuation.py tests/test_rehearsal_continuation.py tests/test_torch_feature_rehearsal_pilot.py
.\.venv\Scripts\python.exe -m ruff format --check src/app/coupled_attribution.py src/app/stream_rehearsal.py src/adapters/torch_feature_rehearsal_pilot.py scripts/run_coupled_attribution.py tests/test_attribution_stream_continuation.py tests/test_rehearsal_continuation.py tests/test_torch_feature_rehearsal_pilot.py
.\.venv\Scripts\python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/app/coupled_attribution.py src/app/stream_rehearsal.py src/adapters/torch_feature_rehearsal_pilot.py scripts/run_coupled_attribution.py tests/test_attribution_stream_continuation.py tests/test_rehearsal_continuation.py tests/test_torch_feature_rehearsal_pilot.py
```

Linux typing uses `--platform linux`. Ordinary pytest alone does not impose the
artifact worker's CPU/work limits. The recorded gate qualifies99 cases via98+1,
preserving a broad invocation's guard failure. Full commands/failures/skips and
scientific limits: [results](../artifacts/runs/r24-stream-continuation-20261016/RESULTS.md).
For another feature, extend the existing loop/helper and freeze its supported
state/authority and work envelope before native execution. See ADR-0314.
