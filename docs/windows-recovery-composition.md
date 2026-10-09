# Retained Windows recovery composition

R3.5b2d3c supplies the supported infrastructure composition boundary, validated
with concrete Windows adapters using deterministic private API hooks and local
SQLite. Actual physical worker/pre-COMMIT journal interruption evidence is now recorded
in [the process capture](windows-recovery-process-capture.md). Factory controls
alone do not prove process crash,native model restore or coordinator loss.

## Modules

```text
src/infra/windows_recovery_composition.py  original retained registration composition
tests/test_windows_recovery_composition.py  API faults, cleanup, local journal controls
```

Infrastructure composes `RecoveryCoordinator` from app, with the existing bounded
private `SqliteRecoveryJournal`, `WindowsRecoveryObserver` and original concrete
`WindowsProcessHandle` registrations. App continues to depend only on core leased
publication/report ports. Existing domain algorithms, protocols and native APIs
are unchanged.

## Input and ownership contract

`compose_windows_recovery(known_authority, journal_path, anchor, worker)` accepts:

- An independently known original `AuthorityRecord`, including exact source/policy/
  manifest, owner/worker generation, clock epoch/start/caps, spent work and flags.
- An existing private journal on its canonical path, containing that exact record.
- Original retained Windows anchor/worker registrations, created while alive with
  expected physical PID and process creation identity through the same API instance.

Why this: repinning or reconstructing authority from worker checkpoint values
could turn stale/unknown state into a new session. This boundary registers no
process, launches/terminates no worker, creates no journal and grants no native
completion or retry. The trusted dispatcher must establish the original record
and retain its registrations independently before calling it.

Record/nested-type and concrete registration-type validation happen before taking
ownership. Failure there leaves both registrations with the caller. Once those
types are accepted, the factory owns the supplied registration handles. Any later
failure closes each distinct handle, reports every close error with the primary
error as a `BaseExceptionGroup`, and returns no coordinator. Successful composition
transfers them to the returned coordinator; call its `close()` or use its context.
Underlying process ownership/termination stays with the original dispatcher.

Unsupported platforms, stopped/uncertain original state, wrong PID/creation,
different API instances, ended/unknown processes and changed observing identity
refuse composition. The anchor must be this physical surviving coordinator, not
an arbitrary alive process. The observer's independently pinned current identity
must match it too. Missing/stale journal state refuses without bootstrap or work.

Initial coordinator observation failures use existing terminal reporting. Bound
elapsed/RSS overshoot becomes stopped evidence with unchanged original caps/spent
counts; unavailable/unbound observations use only last known facts. No failed
initialization returns a usable coordinator. A failed close is reported and never
treated as confirmed cleanup or a reusable handle.

The returned coordinator uses the shared private writer lease and durably reports
fresh retained identity/clock/RSS facts before/after trusted publication. It cannot
undo side effects or preempt an arbitrary callback. Typed values/private test
hooks do not authenticate arbitrary callers or disk/source/graph contents. The
private canonical path and permanent writer lock contract remain required.

## Executable fake composition example

Run on supported Windows with repository test dependencies. This example uses
the concrete adapters and deterministic private API hooks; it creates no OS worker
or model and refuses an existing destination. A trusted live dispatcher must use
real native registrations and independently budgeted original authority instead.

```python
from pathlib import Path
import sys
from test_windows_recovery_composition import fixture
from src.core.recovery_coordination import RecoveryCosts
from src.infra.windows_recovery_composition import compose_windows_recovery

directory = Path(sys.argv[1])
directory.mkdir()
record, kernel, api, anchor, worker, journal = fixture(directory)
coordinator = compose_windows_recovery(record, journal.path, anchor, worker)
published = []
with coordinator:
    assert coordinator.execute(RecoveryCosts(copy_bytes=1), lambda: "prepared", published.append) == "prepared"
    assert published == ["prepared"]
    current = journal.read()
    assert current.metadata.started_ns == record.metadata.started_ns
    assert current.metadata.manifest == record.metadata.manifest
    assert current.metadata.usage.copied_bytes == 41
assert not kernel.handles and coordinator.terminal_confirmed
assert journal.read().metadata.stopped
print("PASS:concrete adapters/fake API,original authority,guarded publication,owned cleanup")
```

## Verification and next action

```powershell
.venv/Scripts/python.exe -m pytest tests/test_windows_recovery_composition.py -q --basetemp=<new-owned-directory>
.venv/Scripts/python.exe -m ruff check src tests scripts
.venv/Scripts/python.exe -m ruff format --check src/infra/windows_recovery_composition.py tests/test_windows_recovery_composition.py
.venv/Scripts/python.exe -m mypy --platform win32 --no-incremental
.venv/Scripts/python.exe -m mypy --platform linux --no-incremental
```

Windows positive/refusal/cleanup cases execute here. Twenty-two supported-runtime
cases are explicitly skipped on non-Windows; invalid initial types and unsupported
platform refusal and invalid observer-peak seeds remain portable. Linux type checking is not Linux runtime proof.
Exact commands/results are in `artifacts/runs/r35b2d3c-composition-20261008/`.

The separate R3.5b2d3d actual process capture has passed after current gates;
its original worker allowance is spent. Future capture requires a new allowance. Use one owned direct base-executable Windows
worker, strict physical PID/creation/interpreter identity, original genuine clock/
RSS/caps, actual writer ownership across worker termination, durable spent/stopped/
uncertain facts and explicit stop-only reconciliation. Record cleanup for every
owned pipe/thread/process/registration/lock handle. Do not rerun old spent fixtures.

Complete native/inbox/consolidation/lifecycle/actor/sharing codecs, stable actor
and single-owner evidence, model restoration and coordinator-loss admission remain
unfinished requirements of the original parent tasks. No scientific win follows
from this engineering composition.
