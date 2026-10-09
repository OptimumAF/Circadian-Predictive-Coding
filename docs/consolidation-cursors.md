# Complete candidate consolidation observations

`ActorShadowRuntime.capture_consolidation_cursor()` captures the complete
consolidation ledger under the existing nonblocking candidate lease. It can
observe stopped or retired owners without reopening them. It reads no native
state, model payload, inbox, clock or budget callback.

## Structure and responsibilities

```text
src/core/consolidation_cursor.py             complete immutable ledger validation
src/core/consolidation_codec_policy.py       independently supplied original bounds
src/app/actor_shadow.py                     original owner lease and detached capture
src/adapters/consolidation_checkpoint_codec.py explicit canonical byte schema
tests/test_consolidation_checkpoint_codec.py complete preservation/refusal controls
docs/adr/ADR-0228-capture-complete-consolidation-ledgers.md
```

The cursor has ten fields: format version, original actor and candidate versions,
original consolidation limit, every consumed attempted event ID, ordered complete
`AppliedConsolidation` receipts, stopped, retired, revision and payload ready.
Each receipt retains its event ID, actor/candidate versions, attempt number and
complete native diagnostic definition/value. Failed transforms consume attempts
even when no receipt is committed; those IDs must survive. The native owner uses
a set for attempted IDs, so their canonical sorted order does not invent a
chronology for failed attempts. Receipt numbering and ordering retain the actual
successful attempt history, including gaps.

Why a separate component: this ledger belongs to the candidate lease. Managed
lifecycle catalog, consent, revocation, declaration clocks, retention driver,
cumulative copy charges, owner enrollments, sharing and actor authority have
different owners. A consolidation observation cannot substitute for them.

## Wire contract and bounds

`ConsolidationCheckpointCodec` implements the existing inner
`CheckpointCodec[ConsolidationCursor]` port. Wire version 1, kind
`consolidation_full_v1`, uses explicit complete cursor/receipt/diagnostic fields.
Unknown or missing fields, unknown versions, duplicate JSON keys, noncanonical
bytes, corrupt records and unsupported values refuse with `ValueError`.

Caller supplies the independently known original `ConsolidationCodecPolicy`:
exact original lifetime attempt limit (zero is supported) and maximum UTF8 bytes
per identifier/diagnostic definition. Wire policy must match it exactly. Original
source/policy hashes, content digest and authority reference hash must also match.
These values identify expected references; matching hashes alone neither verify
the live owner nor grant restore authority. No lock, model, callback, authority
token or worker is serialized.

Counts are checked before history copies, strings before bounded UTF8 encoding,
and canonical encoded fragments against the aggregate wire limit before joining
the complete byte output. Counter/quota representation is an exact nonnegative
integer below `2**63`. Diagnostics preserve the native finite float or Python int
without normalization; wider finite integers such as `2**100` are supported.
The native `isfinite` rule determines integer finiteness, with at most 1024 bits.
Float signed zero and integer versus float type remain exact. Shared diagnostic
identities use canonical references to their first receipt; distinct equal
diagnostics remain distinct. Their full definition/value is retained and checked
at each occurrence, including exact numeric type and signed-zero bits.

All decoded records are detached, with original shared diagnostic aliases retained.
Bounds cover supported record structures and wire bytes, not total process RSS,
global Python overhead or the original cumulative lifetime copy allowance.
Caller-owned cursor mutation requires its original lease; frozen data classes do
not provide a source, graph or malicious-mutation seal.

## Example (zero native work)

```python
from hashlib import sha256

from src.adapters.consolidation_checkpoint_codec import ConsolidationCheckpointCodec
from src.core.actor_ports import AppliedConsolidation
from src.core.checkpoint_codec import CheckpointCodec, CodecBinding, CodecLimits
from src.core.consolidation_codec_policy import ConsolidationCodecPolicy
from src.core.consolidation_cursor import ConsolidationCursor
from src.core.learner_ports import TrainingDiagnostic

diagnostic = TrainingDiagnostic("fixture_native_definition", -0.0)
saved = ConsolidationCursor(
    1, "actor", "learner", 3, ("a", "b", "failed"),
    (AppliedConsolidation("a", "actor", "learner", 1, diagnostic),
     AppliedConsolidation("b", "actor", "learner", 3, diagnostic)),
    True, False, 9, True,
)
binding = CodecBinding(sha256(b"fixture source").hexdigest(), sha256(b"fixture policy").hexdigest())
authority = sha256(b"fixture original authority reference").hexdigest()
limits = CodecLimits(16384, 1, 1, 1)
codec: CheckpointCodec[ConsolidationCursor] = ConsolidationCheckpointCodec(
    ConsolidationCodecPolicy(3, 128), authority_sha256=authority,
)
raw = codec.encode(saved, binding=binding, limits=limits)
detached = codec.decode(raw, binding=binding, expected_sha256=sha256(raw).hexdigest(), limits=limits)
assert codec.encode(detached, binding=binding, limits=limits) == raw
assert detached.attempted_ids == ("a", "b", "failed")
assert detached.stopped
assert detached.consolidations[0].diagnostic is detached.consolidations[1].diagnostic
assert detached.consolidations[0].diagnostic is not diagnostic
```

This example uses synthetic historical metadata. It executes no transform,
training, prediction, native snapshot/restore, sleep, structural operation or
worker. The actual runtime test constructs an untrained native learner with an
original zero-update budget and spies that refuse every model operation during
capture. Synthetic historical records do not claim executed native work.

## Checks

Use a freshly declared engineering envelope and a new unused basetemp for each
future invocation. Existing spent test, native and worker scopes are not renewed.

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m pytest tests/test_consolidation_checkpoint_codec.py -q -o addopts= -p no:cacheprovider --basetemp=artifacts/runs/consolidation-codec-unused
.\.venv\Scripts\python.exe -B -X utf8 -m mypy --platform win32 --no-incremental --cache-dir nul
.\.venv\Scripts\python.exe -B -X utf8 -m mypy --platform linux --no-incremental --cache-dir nul
.\.venv\Scripts\python.exe -B -X utf8 -m ruff check src tests scripts
.\.venv\Scripts\python.exe -B -X utf8 -m ruff format --check src/core/consolidation_cursor.py src/core/consolidation_codec_policy.py src/adapters/consolidation_checkpoint_codec.py tests/test_consolidation_checkpoint_codec.py src/app/actor_shadow.py
```

## Next extension

Complete `R3.5b2e4b` under the original managed owner and holder leases: full
catalog/consent/optout/revocation/declaration tick and elapsed time/quota,
retention faults/driver/epoch/history, monotonic copy charges and enrollments.
Inventory live ports and aliases explicitly. Current driver observations omit
the original `created_at`, held token, stop/wake/purge events and thread authority;
those gaps require a complete capture design before composition or recovery.
Full lifecycle, composite, actor, sharing, Torch, single owner, disk/live/native,
model/coordinator loss, scientific and human acceptance remain unfinished.
