# Durable recovery metadata admission

## Scope and module boundary

`src/core/recovery_admission.py` is a pure validation boundary. It accepts exact,
bounded metadata and independently trusted coordinator facts, and returns typed
accounting observations. It has no filesystem, callback, native state or copy
access. Its output grants no restore or training authority.

Why this: the existing candidate checkpoint retains live original budget, clock,
RSS sampler and gate objects. Serializing its observation cannot preserve that
authority across process loss. Validate the proposed relationships before adding
an adapter that can read payload bytes.

## Proposed supported adapter policy

The initial durable policy is restricted to a single local coordinator and one
OS boot epoch. All original limits are mandatory. A future adapter must:

1. Persist the original session, source and policy identity, immutable ceilings,
   original monotonic start, last observation, event cursor, spent admissions,
   completed work, copy reservations, declarations, checkpoint attempts and
   observed absolute RSS peak in independent authoritative storage.
2. Bind complete versioned native, inbox, consolidation, lifecycle, actor and
   sharing components plus the complete payload. Lifecycle includes original
   consent/provenance, consumed IDs, grants, revocations, tombstones and retention
   anchors. Digest presence alone proves neither completeness nor authenticity.
3. Atomically acquire the next owner epoch against that authoritative record;
   establish the previous owner has ended. A timeout alone cannot prove this.
   Retain a live fencing lease through payload validation and publication.
4. Supply cross-process monotonic nanoseconds from the same authenticated OS boot
   epoch. Count downtime from the original start. Reboot, rollback or unsupported
   epoch adapters refuse; wall time or a new process-relative origin is invalid.
5. Sample current absolute process RSS before payload access, carry the original
   observed peak and ceiling, and continue monitoring after admission. RSS is an
   observed absolute metric, not cumulative allocated bytes or hard preemption.
6. Verify exact bytes against the authoritative digests before decoding, then
   validate complete state/cursors, restore without quota renewal and recheck the
   live owner fence immediately before publishing. Recovery attempts and failed
   copy reservations must themselves be durably charged before execution.

No coordinator, authenticated OS epoch adapter, durable state codec, lease
watchdog or process-restart integration is implemented here. The current runtime
has no disk restore path through this module. A future integration must refuse
when those adapters are absent; constructing `RecoveryFence` is not evidence of
authentic time, storage, RSS or ownership.

## Admission checks

Every fixed field is revalidated at the gate, including nested frozen records
and independently supplied expected metadata. Counters are exact nonnegative
integers below 2**63, identifiers are at most 128 characters, digests are exact
lowercase SHA-256 strings, and schema version is exactly integer 1. Unsupported
record subclasses and opaque values are refused before accessing their hooks.

Saved metadata must equal the independent authoritative expected record. This
rejects changed source/policy/component bindings, starts, caps, spent counters,
IDs represented by component bindings and generation/cursor rollback. It does
not detect an authority service that itself supplies forged or rolled-back facts.

The new owner must differ, its epoch must be exactly one greater, its lease must
be live and its predecessor ended. Clock epoch must match and current time must
not precede the last saved observation. Elapsed time includes restart downtime;
elapsed at or beyond the ceiling refuses. RSS above the original ceiling refuses;
an exact RSS ceiling is allowed. The larger old/current absolute RSS is carried.

Stopped, uncertain or unequal admitted/completed update counts refuse candidate
resume. Partial work requires separate reconciliation; this gate never retries
or refunds it. All usage stays within original caps. Exhausted update/copy quotas
can produce zero remaining observations, and confer no permission for more work.
Actor-only recovery of a stopped candidate is outside this first boundary.

## Synthetic example

This is an executable relationships fixture, **not an authentic coordinator**.
The strings below do not bind real model bytes; there is no payload to restore.

```python
from src.core.recovery_admission import (
    RecoveryFence, RecoveryLimits, RecoveryManifest, RecoveryMetadata,
    RecoveryUsage, validate_recovery_admission,
)

manifest = RecoveryManifest(*[f"{n:064x}" for n in range(1, 10)])
saved = RecoveryMetadata(
    1, "fixture-session", "fixture-boot", "fixture-owner-a", 7, 4,
    100, 400, 12, manifest,
    RecoveryLimits(10, 1000, 100, 10, 4, 1024),
    RecoveryUsage(3, 3, 40, 3, 2, 512), False, False,
)
fixture_fence = RecoveryFence(
    saved, "fixture-owner-b", 5, "fixture-boot", 600, 640, True, True,
)
facts = validate_recovery_admission(saved, fixture_fence)
assert facts.elapsed_ns == 500  # Includes time since saved observation.
assert facts.remaining_updates == 7 and facts.remaining_copy_bytes == 60
assert facts.peak_rss_bytes == 640
```

## Verification and extension

From the repository root with the existing environment:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_recovery_admission.py -q
.venv\Scripts\python.exe -m mypy --platform win32 --no-incremental
.venv\Scripts\python.exe -m mypy --platform linux --no-incremental
.venv\Scripts\python.exe -m ruff check src tests scripts
.venv\Scripts\python.exe -m ruff format --check src/core/recovery_admission.py tests/test_recovery_admission.py
```

The platform type targets run on the local Windows interpreter. They do not
establish a Linux runtime or clean-clone result. The new controls and affected
same-process checkpoint/inbox/sharing/ownership tests use bounded fake fixtures;
previous native captures are explicitly deselected and not renewed.

Next implement the independent transactional authority and supported clock/RSS
adapter with tests for missing authority, stale fences and process loss, before
any durable payload reader. Preserve original acceptance for R3.5b2 and R3.7:
actual OS-crash/corruption/recovery, complete lifecycle state, single ownership,
stable actor and no quota/consumed-ID renewal remain unproved. This metadata
boundary does not finish those tasks.
