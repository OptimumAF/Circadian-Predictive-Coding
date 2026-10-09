# Surviving coordinator sequencing

R3.5b2d1 implements the local application sequencing prerequisite of R3.5b2d.
R3.5b2d2 adds persisted observations and terminal authority;see the
[terminal reporting guide](recovery-terminal-authority.md). The full task still
uses guarded publication;full acceptance still requires live composition and bounded real
cross-process crash evidence. Native completion/component codecs,
model recovery and coordinator loss remain unfinished.

## Modules

```text
src/core/recovery_coordination.py        exact costs, closable registration port, host validation
src/app/recovery_coordinator.py          local serialized coordinator and retained registrations
tests/test_recovery_coordinator.py      fake process facts and local SQLite boundary controls
```

Dependencies point inward: app depends on core ports. The coordinator accepts
an independently known `AuthorityRecord`, a report-capable authority journal
(`RecoveryLeasedPublicationPort`), an observation
port and independently registered anchor/worker probes. Infrastructure adapters
implement these ports; application code does not import native APIs or SQLite.

Inputs and callbacks come from trusted coordinator composition. Worker checkpoint
metadata does not create authority. Typed ports and bounded records do not
authenticate an arbitrary caller, callback, private graph or process fact.
No `RecoveryFence` is synthesized. Costs are declared by the trusted dispatcher;
this prerequisite does not measure or attest arbitrary callback work.

## Operation order

`execute(costs, prepare, publish=None)`:

1. Acquire the local lane without waiting; reject nested or simultaneous calls.
2. Validate costs, independent witness, exact persisted state and fresh host facts.
3. Commit the reservation through full-record CAS and verify its persisted result.
4. Recheck registered identities, liveness, time, RSS and journal;persist validated
   observation high water before preparation.
5. Run trusted preparation and recheck/persist those facts afterward.
6. If publication is requested, require no uncertain native work and recheck before
   and after the trusted publication callback under its lease, durably reporting fresh facts.

The observer receives the retained worker probe, never a caller-supplied ended
flag. Anchor/worker bindings are checked on both sides of observation. Clock and
RSS high water cannot regress within this coordinator lane. Elapsed time includes
all time since the original start; original caps and cumulative charges stay bound.
The coordinator owns its retained probes from successful record validation onward.

Why this: persist costs before preparation and check again after callbacks because
work can change process facts or exceed a bound. A stale journal, unknown commit
result, changed registration, failed observation or callback/BaseException disables
the lane. Committed charges are never refunded. A busy rejection occurs before
entry and leaves an already running lane intact. Invalid costs fail before work.

One admitted update becomes uncertain in the journal before preparation. With no
publication callback its result can return, while the admission stays uncertain
and blocks further operations. A publication callback for that update is refused:
there is no completion receipt or native codec that could justify clearing the
flag. Metadata-only actions may publish through trusted callbacks.

## Handoff and cleanup

`handoff(replacement_registration, owner_id)` requires the original registered
worker ended and a different registered live replacement identity. It commits
the exact next owner generation and checkpoint-attempt charge, then rechecks
the new binding before releasing the old handle. If the replacement ends during
commit, the generation/charge stays persisted and the lane fails without dispatch.

`close()` persists a stop during normal close and terminally releases owned
registrations. After operation failure it cleans up without retrying work or
repeating terminal reporting. Initialization failure attempts both retained closures. Cleanup attempts
every owned registration and reports errors as a `BaseExceptionGroup`, preserving
initialization failures when cleanup also fails. It never resumes work. The caller
retains ownership when the supplied original record itself fails validation before
the coordinator takes ownership. A busy lane refuses concurrent close.

## Limits of the current boundary

The local gate serializes this coordinator. Publication now uses a held writer
lease shared by supported adapters on the canonical private journal path, with
durable observation reports before and after the callback. See guarded-recovery-coordinator.md.
Post-publication failure detects loss and disables the lane; it does not roll
back a callback that has already changed external state. Callbacks are synchronous
and trusted; there is no preemption of long or arbitrary native work.

Validated time/RSS high water and terminal stops now persist through report CAS.
If storage is unavailable or its commit outcome is unknown,terminal authority is
unconfirmed:the original error chains a `TerminalReportingFailure` with an
independent witness. Explicit fresh-connection reconciliation accepts only exact
witnessed states and stops work;it never grants restart admission. A new coordinator
must not bootstrap from worker checkpoint data. Live composition,
real crash/native recovery and coordinator loss keep full parent acceptance open.

Current tests use synthetic process identities and observations plus real local
SQLite; no new OS worker or native model runs in this scope. A later bounded
Windows composition/cross-process capture must preserve original caps and
identity checks, follow current correctness gates and use a new allowance.

## Executable test composition

This example intentionally uses the test fakes. Run from the repository root
with test dependencies present. The directory must be new; it leaves a small
SQLite artifact for inspection and refuses overwrite.

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path("tests").resolve()))
from test_recovery_coordinator import fixture
from src.core.recovery_coordination import RecoveryCosts
from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal

folder = Path("artifacts/runs/r35b2d3b-integration-20261008/coordinator-guide-fixture")
folder.mkdir()
coordinator, journal, observer, anchor, worker, log = fixture(
    lambda record: SqliteRecoveryJournal.create(folder / "authority.sqlite", record)
)
published = []
with coordinator:
    result = coordinator.execute(RecoveryCosts(copy_bytes=1), lambda: "prepared", published.append)
    assert result == "prepared" and published == ["prepared"]
    assert journal.read().metadata.usage.copied_bytes == 41
assert anchor.closed and worker.closed
assert coordinator.terminal_confirmed and journal.read().metadata.stopped
print("PASS: fake coordinator/local SQLite charge precedes preparation; callbacks checked; handles closed")
```

## Validation commands and next step

```powershell
python -m pytest tests/test_recovery_coordinator.py tests/test_recovery_authority.py tests/test_sqlite_recovery_journal.py -q
python -m ruff check src tests scripts
python -m ruff format --check src/core/recovery_coordination.py src/app/recovery_coordinator.py tests/test_recovery_coordinator.py
python -m mypy --platform win32 --no-incremental
python -m mypy --platform linux --no-incremental
```

Evidence: `artifacts/runs/r35b2d-coordinator-20261007/`. Linux is a type target on
the Windows interpreter, not Linux runtime proof. See
[ADR-0218](adr/ADR-0218-sequence-coordinator-actions-under-original-authority.md)
and the [metadata journal](recovery-authority-journal.md).

The publication lease is connected through inward ports with durable facts.
Next connect the service to a supported live composition
root with retained Windows registrations and independently known original journal
state, followed by a separately budgeted real worker/journal/crash capture.
Complete native and lifecycle codecs before any native payload restore.
