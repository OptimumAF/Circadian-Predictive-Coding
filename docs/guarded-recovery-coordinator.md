# Guarded recovery coordinator publication

R3.5b2d3b connects the surviving coordinator to a private-journal publication
lease. Full R3.5b2d3 still requires live Windows composition and separately
budgeted actual journal/worker crash evidence. Model/native completion and
recovery after coordinator loss remain unfinished.

## Modules and dependencies

```text
src/core/recovery_publication.py          guard/lease/report-only inner ports
src/app/recovery_coordinator.py           retained probe checks, guarded publication
src/infra/sqlite_recovery_lock.py         permanent one-byte native writer lock
src/infra/sqlite_recovery_publication.py  scoped durable observation-report lease
src/infra/sqlite_recovery_journal.py      bounded private authority transactions
tests/test_recovery_publication_coordinator.py  fake facts, real local contention/faults
```

App depends on `RecoveryLeasedPublicationPort` and `RecoveryPublicationLease`
in core. Infrastructure implements these ports. The legacy transaction-only
`RecoveryPublicationPort.publication_guard` stays supported. The lease adapter's
journal type import is for type checking; journal delegation loads it only when
requested, avoiding a runtime circular import.

Inputs remain the independently known original authority and retained original
anchor/worker registrations with a trusted observation adapter. Checkpoint values
cannot create authority. Typed values cannot authenticate arbitrary callers,
callbacks, private data or host facts.

## Why a separate writer lock

The transaction-only guard cannot commit fresh observations before publication
while retaining its database writer transaction. The original d2 criterion
requires those facts durably visible before the callback. We preserve that
criterion by holding a separate permanent native writer lock while SQLite reports
commit. Supported ordinary reservation/report, legacy guard and explicit terminal
reconciliation all acquire that same lock on the canonical private database path.

The file is `<database>.writer.lock`, at most one byte. Acquisition is nonblocking
(`msvcrt` byte lock on Windows, `fcntl.flock` on POSIX), with no retries. Windows
local behavior is exercised; Linux is a type target in this increment. Never
delete, replace or unlink the lock file while the journal is in use. Copy/hard-link
aliases, remote filesystems and hostile file replacement are outside this private
path contract. A lock file by itself does not prove ownership; the held OS handle
does. Owned handles close on ordinary and exceptional exits, including unlock
errors. Process termination/crash behavior requires the later actual capture.

## Publication order

1. Commit original costs before preparation, retaining native uncertainty.
2. Check/persist fresh observations around preparation and before lease acquisition.
3. Acquire publication ownership and compare the whole independent original record.
4. Under the lease, check retained anchor/worker identity and liveness around fresh
   sampling; validate clock epoch/start, monotone time/RSS, original caps and flags.
5. Durably commit that observation-only report and read back its exact result.
6. Invoke the trusted publication callback while competing supported writers are
   excluded. Check and durably report fresh observations afterward.
7. Release ownership and perform the ordinary current-state check/report.

The lease permits only nonterminal observation reports. It cannot reserve native
work, change source/manifest/policy/owner/start/caps/spent counters, stop/reopen
authority, or issue completion. An inactive lease or use from another thread is
refused. Stopped or uncertain state cannot acquire publication authority. Ordinary
journal writes from the callback contend; use the restricted lease only through
the trusted coordinator's observation orchestration.

## Failure and limits

Lease failure disables the journal adapter and coordinator lane. Terminal reporting
occurs after lease release. If the failed adapter cannot confirm it, the original
exception chains `TerminalReportingFailure` and retains the exact independent
`FailureWitness`. Explicit fresh terminal reconciliation accepts only its known
disk states, preserving committed counters/uncertainty and bound measured facts.
It stops work and cannot resume a coordinator or refund charges.

Tests inject report failure before and after COMMIT, callback BaseException, stale
acquisition, bad entry/exit facts and native unlock failure. They preserve the
actual committed record and release ownership so explicit reconciliation can run.
The original before-publication durable-clock/RSS assertion remains unchanged.

Callbacks remain synchronous and trusted. Post-publication process loss or cap
exhaustion can be detected, but publication cannot be undone or atomically committed
with a model by this metadata journal. There is no arbitrary callback preemption,
generic authentication, live native completion, stable actor ownership or
coordinator-loss recovery proof.

## Executable local example

This is a small test-fake/local SQLite composition, with no new OS worker or model.
Use a new directory for every execution; the example refuses overwrite.

```python
from pathlib import Path
import sys
from test_recovery_coordinator import fixture
from src.core.recovery_coordination import RecoveryCosts
from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal

directory = Path(sys.argv[1])
directory.mkdir()
coordinator, journal, observer, anchor, worker, log = fixture(
    lambda record: SqliteRecoveryJournal.create(directory / "authority.db", record)
)
published = []
def publish(value):
    current = journal.read()
    assert current.metadata.observed_ns == observer.now
    assert current.metadata.usage.copied_bytes == 41
    published.append(value)
with coordinator:
    assert coordinator.execute(RecoveryCosts(copy_bytes=1), lambda: "prepared", publish) == "prepared"
    assert published == ["prepared"]
assert anchor.closed and worker.closed
assert coordinator.terminal_confirmed and journal.read().metadata.stopped
assert (directory / "authority.db.writer.lock").stat().st_size == 1
print("PASS:guarded publication,durable fresh facts,spent charge,terminal cleanup")
```

## Validation and extension

```powershell
.venv/Scripts/python.exe -m pytest tests/test_recovery_publication_coordinator.py -q --basetemp=<new-owned-directory>
.venv/Scripts/python.exe -m ruff check src tests scripts
.venv/Scripts/python.exe -m ruff format --check src/core/recovery_publication.py src/app/recovery_coordinator.py src/infra/sqlite_recovery_lock.py src/infra/sqlite_recovery_publication.py src/infra/sqlite_recovery_journal.py tests/test_recovery_coordinator.py tests/test_recovery_publication_coordinator.py
.venv/Scripts/python.exe -m mypy --platform win32 --no-incremental
.venv/Scripts/python.exe -m mypy --platform linux --no-incremental
```

Exact commands/outcomes: `artifacts/runs/r35b2d3b-integration-20261008/command-*.json`.
The next extension is a supported live Windows composition root through these
inner ports and retained registrations. Validate it before declaring a fresh,
separate actual journal/worker crash capture. Preserve strict physical PID,
creation identity, original clock/RSS/caps, owned process/pipe/thread/handle cleanup
and all spent prior scopes. Full native/lifecycle codecs and model recovery are
still required before native restore/completion or coordinator-loss admission.
