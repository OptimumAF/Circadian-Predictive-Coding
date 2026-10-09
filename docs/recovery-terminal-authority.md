# Durable terminal and observation authority

R3.5b2d2 extends the metadata journal and coordinator to persist observed time/RSS
high water and terminal stops. It preserves original caps, ownership and spent
work. Full live publication lease/composition, real process crash, native recovery
and coordinator loss acceptance remain unfinished.

## Module boundaries

```text
src/core/recovery_reporting.py           exact reports, failure witness, reporting port
src/core/recovery_authority.py           existing records; terminal RSS facts retain overshoot
src/app/recovery_coordinator.py          progress/terminal reporting and independent witness
src/infra/sqlite_recovery_journal.py      same bounded CAS; explicit terminal reconciliation
tests/test_recovery_reporting.py         transition, persistence, commit fault and reopen controls
tests/test_recovery_coordinator.py       existing sequencing controls with report-capable fake
```

`RecoveryReportingPort` extends the existing read/advance port with
`report(AuthorityReport)`. App depends on inner ports; infrastructure owns SQLite.
Custom coordinator adapters must implement reporting. The SQLite adapter now does;
there is no fallback to a memory-only journal.

## Reporting policy

An exact report changes only sequence, observed clock, absolute RSS peak and
the stopped flag. Original session/source/policy/all component hashes, start,
clock epoch, anchor/worker, owner generation, caps, event cursor and cumulative
admitted/completed/copy/grant/checkpoint-attempt counters remain unchanged.
Stopped and uncertain history never clears. There is no completion receipt.

A nonterminal report requires bound monotone host observations, a live registered
worker and observations within original elapsed/RSS caps. An uncertain admission
can report new observations, while its uncertainty still blocks publication or
another admission. Progress reporting grants no extra work.

A terminal report may retain an authentic measured elapsed/RSS overshoot or an
ended registered worker. Why this: preserving a negative resource result is more
accurate than clamping the measurement or discarding it. Original caps stay fixed;
every dispatch still refuses the stopped record. Other spent counters remain
subject to their original caps. Terminal status does not grant a larger allowance.

If host facts are unavailable or unbound, stopping can use only the last known
stamp. No clock/RSS value or death fact is invented. Regression and foreign epoch,
anchor/worker or observer identities are refused even for terminal reports.

## Coordinator behavior

The coordinator persists fresh validated observations around callbacks before
publication. On callback/host/commit failure or normal close it attempts a terminal
report. Its original failure type is preserved. Reentry after failure refuses
before attempting another terminal report or work action.

`terminal_confirmed` is true only after a successful terminal write and exact
readback. If reporting fails or its outcome is unknown, the original exception
chains a `TerminalReportingFailure` carrying the independent `FailureWitness`.
The coordinator also exposes `failure_witness` for explicit reconciliation.
Cleanup still attempts all owned registrations; no callback or native work retry
occurs. Constructor failure attempts terminal reporting before releasing probes.

Reports add sequence increments, so sequence is an authority revision rather
than a count of native updates. Update/copy/grant/attempt counters retain their
separate meanings. A stopped record prevents a newly constructed coordinator
from executing work, even when its adapter and retained handles are new.

## Explicit terminal-only reconciliation

`SqliteRecoveryJournal.reconcile_terminal(path, witness)` opens the existing
private database through a fresh connection. It accepts only the exact states
known to the surviving coordinator: the independent pre-attempt record, validated
attempt proposal, attempted stop, and their bounded terminal forms (at most six).
An arbitrary higher sequence, changed owner/source/policy or another writer's
unwitnessed record is refused.

Within one immediate transaction, reconciliation preserves the actual committed
charges and uncertain flag and sets stopped. A before-COMMIT failure leaves no
new reservation; an after-COMMIT failure retains that committed reservation.
Neither case resumes work. Reconciliation of an already confirmed matching stop
is idempotent. Missing authority is not recreated. There is no automatic retry
of a failed action or implicit reconstruction from a worker checkpoint.

The witness must be the independent surviving coordinator's evidence, not
caller checkpoint data. Typed values and trusted private storage do not provide
cryptographic authentication or arbitrary callback/source attestation. Losing
that independent evidence is an unsupported recovery condition. A terminal-only
write does not establish a live lease or admit a restarted native learner.

The existing bounded canonical JSON/schema and SQLite DELETE/FULL/immediate/no
busy retry policy remain. No dependency, schema migration or environment setting
is added. Power-loss durability and actual OS crash recovery are still unproven;
current tests cover local SQLite transactions and injected commit faults.

## Executable test composition

Run from the repository root with test dependencies. The named directory must
be new and leaves a small local artifact. These process facts are synthetic.

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path("tests").resolve()))
from test_recovery_coordinator import fixture
from src.core.recovery_coordination import RecoveryCosts
from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal

folder = Path("artifacts/runs/r35b2d2-terminal-20261008/guide-fixture")
folder.mkdir()
coordinator, journal, observer, anchor, worker, log = fixture(
    lambda record: SqliteRecoveryJournal.create(folder / "authority.sqlite", record)
)
def fail():
    raise OSError("deliberate preparation failure")
try:
    coordinator.execute(RecoveryCosts(copy_bytes=1), fail)
except OSError:
    assert coordinator.terminal_confirmed
else:
    raise AssertionError("failure was not propagated")
stopped = journal.read()
assert stopped.metadata.stopped and stopped.metadata.usage.copied_bytes == 41
assert stopped.metadata.usage.updates_completed == 3
assert coordinator.failure_witness is not None
assert SqliteRecoveryJournal.reconcile_terminal(folder / "authority.sqlite", coordinator.failure_witness) == stopped
coordinator.close()
assert anchor.closed and worker.closed
print("PASS: failed preparation retains charge; terminal stop persists; reconciliation cannot resume work")
```

## Validation and next extension

```powershell
python -m pytest tests/test_recovery_reporting.py tests/test_recovery_coordinator.py tests/test_recovery_authority.py tests/test_sqlite_recovery_journal.py -q
python -m ruff check src tests scripts
python -m ruff format --check src/core/recovery_reporting.py src/core/recovery_authority.py src/infra/sqlite_recovery_journal.py src/app/recovery_coordinator.py tests/test_recovery_coordinator.py tests/test_recovery_reporting.py
python -m mypy --platform win32 --no-incremental
python -m mypy --platform linux --no-incremental
```

Evidence: `artifacts/runs/r35b2d2-terminal-20261008/`. Current captures use fake
process facts and real local SQLite, with no new worker or native action. Linux
is a type target on Windows, not a Linux runtime gate.

Next implement a publication lease/guard against legitimate competing writers,
then supported live Windows composition and a separately budgeted actual journal/
worker-crash capture after correctness gates. Complete native/inbox/consolidation/
lifecycle consent/consumed-ID/revocation/tombstone/retention/copy-ledger and actor/
sharing codecs before native payload restore. Preserve full parent acceptance.
