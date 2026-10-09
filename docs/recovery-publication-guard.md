# Private recovery journal publication guard

R3.5b2d3a adds the transaction-only guard. R3.5b2d3b connects
`RecoveryCoordinator.execute` through the observation-report lease described in
[guarded coordinator](guarded-recovery-coordinator.md). R3.5b2d3 remains unchecked until live Windows composition and its separately budgeted
actual journal/worker crash evidence satisfy the original acceptance criteria.

## Modules and boundaries

```text
src/core/recovery_publication.py        inner port, pure host observation validation
src/infra/sqlite_recovery_journal.py    private SQLite guard and existing journal
tests/test_recovery_publication.py     fake host facts, actual SQLite contention
```

`RecoveryPublicationPort` extends the inner reporting port. Infrastructure depends
on core. Inputs are an independently known original `AuthorityRecord` and a trusted
callback returning fresh `RecoveryHostObservation` values. That callback must
sample the retained anchor/worker registrations and check their identity and
liveness. Typed values alone do not authenticate process facts or callers.

## Behavior

Why this: a full-record read/CAS cannot exclude another writer during publication.
The adapter acquires `BEGIN IMMEDIATE`, validates the exact original record under
that reservation, obtains fresh host facts, yields to the trusted callback,
checks fresh facts again, and releases the transaction. Reservations, reports and
terminal reconciliation all use the same SQLite writer reservation on the same
private database. Contention fails immediately at the existing zero timeout.

No authority rows are changed inside the guard. Original manifest/source/policy,
identity/owner generation, clock epoch/start, limits, terminal/uncertain flags and
every spent counter remain exact. Validation refuses stale records, expired
elapsed time, excessive RSS, wrong identities/epochs, ended or unknown workers,
regressing observations, stopped state and uncertain native work. Successful exit
does not claim a native completion or durable observation report.

Any exception after opening the guard disables that adapter and closes its
connection, including callback `BaseException` or transaction-release failure.
There is no automatic retry. Explicit surviving-coordinator terminal reconciliation
remains a separate operation using the existing exact independent witness.

During publication, callbacks must not reserve/report/reconcile or enter another
guard on this journal. A separate supported writer can read, but cannot change
authority until release. Writer serialization assumes the supported private SQLite
file and adapters; it does not protect against hostile file replacement or create
a generic live process lease.

The callback can finish after a process ends or an elapsed/RSS cap is crossed.
Exit validation detects those failures but cannot preempt arbitrary callbacks,
undo side effects or atomically commit model publication with database changes.
The coordinator integration must retain failed host facts and persist terminal
authority after guard release without refunding any already committed charges.

## Executable local example

This example uses trusted fake host facts and one small new local SQLite file.
The fixture import is for illustration; application composition must independently
register original processes and supply authentic fresh observations.

```python
from pathlib import Path
import sys
from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal
from src.core.recovery_publication import RecoveryPublicationPort
from src.core.recovery_reporting import plan_report
from test_recovery_authority import fixture

directory = Path(sys.argv[1])
directory.mkdir()
record, observation = fixture()
journal = SqliteRecoveryJournal.create(directory / "authority.db", record)
port: RecoveryPublicationPort = journal
published = []
with port.publication_guard(record, lambda: observation):
    published.append("prepared metadata")
assert published == ["prepared metadata"]
assert journal.read() == record
assert journal.report(plan_report(record, observation, terminal=True))
print("PASS:guard released;original authority preserved;explicit terminal report")
```

## Validation commands

Use a new `--basetemp` on each test invocation. Captured commands and outputs are
in `artifacts/runs/r35b2d3a-guard-20261008/command-*.json`, `.stdout` and `.stderr`.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_recovery_publication.py -q --basetemp=<new-owned-directory>
.venv/Scripts/python.exe -m ruff check src tests scripts
.venv/Scripts/python.exe -m ruff format --check src/core/recovery_publication.py src/infra/sqlite_recovery_journal.py tests/test_recovery_publication.py
.venv/Scripts/python.exe -m mypy --platform win32 --no-incremental
.venv/Scripts/python.exe -m mypy --platform linux --no-incremental
```

The coordinator now uses `RecoveryLeasedPublicationPort` to durably report fresh
facts under a separate shared writer lock. The legacy guard also acquires that
lock and retains its transaction-only semantics. Next validate live composition
and a separate bounded crash capture;see guarded-recovery-coordinator.md.

Native restore/completion, lifecycle codecs, stable actor ownership and recovery
after coordinator loss remain unfinished. No scientific claim follows from these
engineering controls.
