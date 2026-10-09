# Coordinator metadata journal

R3.5b2c adds a local metadata transaction boundary under a surviving trusted
coordinator. It is a prerequisite for R3.5b2 durable recovery. Actual worker
dispatch, live lease enforcement, native payload recovery and coordinator loss
remain unfinished.

## Modules and trust boundary

```text
src/core/recovery_authority.py            exact records, transitions, authority port
src/infra/recovery_authority_codec.py     bounded canonical JSON metadata
src/infra/sqlite_recovery_journal.py       private SQLite read/compare-and-swap
tests/test_recovery_authority.py          pure transition controls
tests/test_sqlite_recovery_journal.py      local storage/fault/contention controls
```

The adapter implements `RecoveryAuthorityPort.read()` and `advance(change)`,
plus `RecoveryReportingPort.report()` for observation/terminal transitions.
See [terminal authority](recovery-terminal-authority.md).
The inner port has no filesystem or model dependency. Input is independently
known coordinator state and trusted host observations. Output is a typed record
or a successful compare-and-swap boolean; stale proposals return false. Validation,
locking and storage errors raise before any dispatcher may act.

The coordinator must retain an independent high-water record outside worker
checkpoint data. The private journal requires trusted local storage. A caller
can construct the typed records; their types do not authenticate their source.
The journal does not obtain OS facts or manufacture a live `RecoveryFence`.
The coordinator's retained registration handles and observation adapter still
need an application integration that checks ownership through publication.

## Policy

Original session, source, policy, all component hashes, event cursor, anchor,
clock epoch, start and caps stay fixed. Reservations increase sequence by exactly
one, advance observed time and preserve absolute RSS high water. Admissions,
copy bytes, grants and checkpoint attempts cannot decrease or exceed original
caps. One admitted update becomes uncertain before work; no retry, handoff or
completion transition can clear that flag in this first protocol.

Metadata handoff requires a matching registered prior worker reported ended,
a different owner ID and worker identity, and exactly the next owner generation.
It charges one checkpoint attempt and preserves every other spent counter.
This excludes PID-only or timeout-only death facts when supplied through the
trusted observation port. It does not independently authenticate injected facts.

Why this: charge reservations transactionally before a future dispatcher performs
work. A failed or unknown completion must not renew resources. Until a complete
native completion protocol exists, uncertain work deliberately stops progress.
Immutable component hashes and event cursor also prevent this metadata-only
adapter from pretending it can publish a new native checkpoint.

## Storage and failure behavior

Canonical JSON uses an exact schema, nested type validation and duplicate-key
refusal, with a 16 KiB limit. There is no pickle or native payload decoding.
The database has one bounded record, a 256 KiB file bound, 4096-byte pages and
a 64-page limit. Creation is exclusive; ordinary opening uses existing-file
`mode=rw`. Missing or malformed authority is never silently recreated.
Connection, transaction or decoding failure disables that adapter instance.
Explicit terminal-only reconciliation requires the surviving independent
coordinator witness;it preserves actual committed charges and cannot resume work.

SQLite uses DELETE journal mode, FULL synchronization, explicit `BEGIN IMMEDIATE`
and zero busy timeout. The database transaction compares the entire expected
record and commits the proposal before returning true. No automatic retries
occur. These choices follow the [Python sqlite3 interface](https://docs.python.org/3/library/sqlite3.html),
[SQLite transactions](https://www.sqlite.org/lang_transaction.html) and
[SQLite synchronization policy](https://www.sqlite.org/pragma.html#pragma_synchronous).
FULL synchronization is a configured policy; power-loss durability on this host
has not been experimentally proven.

An error before COMMIT rolls back the transaction. An error after COMMIT can
leave the charge persisted even when the caller received no success acknowledgment;
the failed instance is disabled and the charge stays spent. A surviving coordinator
floor detects older counters or sequence and same-sequence changes. Lost
coordinator state, hostile higher-sequence rewrites, reboot and authenticated
external anti-rollback require a separate policy and implementation.

Use one adapter instance per coordinator execution lane. Test contenders use
separate adapters/connections; this API does not synchronize shared Python
instance fields or act as a distributed owner lease.

## Executable metadata-only example

Run from the repository root with the local interpreter. The named directory
must be new; the example refuses overwrite and leaves its small artifact for
inspection. Synthetic process facts and nanoseconds below are test values.

```python
from pathlib import Path
from src.core.recovery_admission import (
    RecoveryLimits, RecoveryManifest, RecoveryMetadata, RecoveryUsage,
)
from src.core.recovery_authority import AuthorityRecord, plan_reservation
from src.core.recovery_observation import (
    RecoveryHostObservation, RecoveryProcessIdentity, anchored_clock_epoch,
)
from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal

folder = Path("artifacts/runs/r35b2c-journal-20261007/guide-fixture2")
folder.mkdir()
anchor, worker = RecoveryProcessIdentity(10, 100), RecoveryProcessIdentity(20, 200)
metadata = RecoveryMetadata(
    1, "example", anchored_clock_epoch(anchor), "owner-a", 7, 4, 100, 400, 12,
    RecoveryManifest(*[f"{n:064x}" for n in range(1, 10)]),
    RecoveryLimits(10, 1000, 100, 10, 4, 1024),
    RecoveryUsage(3, 3, 40, 3, 2, 512), False, False,
)
record = AuthorityRecord(metadata, anchor, worker)
observation = RecoveryHostObservation(metadata.clock_epoch, 600, 640, 640, anchor, worker, False)
journal = SqliteRecoveryJournal.create(folder / "authority.sqlite", record)
change = plan_reservation(record, observation, updates=1, copy_bytes=1)
assert journal.advance(change)
assert journal.read() == change.proposed
assert journal.read().metadata.uncertain_work
assert not journal.advance(change)
print("PASS: metadata charge persisted; stale proposal refused; native work remains uncertain")
```

## Validation and extension

```powershell
python -m pytest tests/test_recovery_authority.py tests/test_sqlite_recovery_journal.py -q
python -m ruff check src tests scripts
python -m ruff format --check src/core/recovery_authority.py src/infra/recovery_authority_codec.py src/infra/sqlite_recovery_journal.py tests/test_recovery_authority.py tests/test_sqlite_recovery_journal.py
python -m mypy --platform win32 --no-incremental
python -m mypy --platform linux --no-incremental
```

Current evidence is in `artifacts/runs/r35b2c-journal-20261007/`, including tests-first
failures, preserved failed sources and commit-fault receipts. This scope uses
fake host observations, local SQLite and two threads in one process. It runs no
new worker, native model, training or sweep. Linux is a type target on the Windows
interpreter, not a Linux runtime validation.

The coordinator service now obtains retained-handle observations,owns the
independent floor,commits reservations/reports and persists terminal stops.
Next add a publication lease and supported live composition before claiming
ownership through publication. Reserve a separate bounded cross-process journal/crash capture after
its correctness gates. Native completion, versioned component codecs, lifecycle
consent/revocation/tombstone/copy ledgers and actual model recovery remain open.
