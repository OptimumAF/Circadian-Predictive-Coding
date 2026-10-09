# Actual Windows journal and worker capture

Evidence: `artifacts/runs/r35b2d3d-crash-20261008/`. This is one prospectively
budgeted metadata and process capture. It is not a native model restore,
power-loss test or recovery after coordinator loss. The worker allowance is spent;
do not rerun the saved helper or old physical-worker test against this scope.

## Implementation correction before launch

Fresh composition previously initialized the observer's RSS peak at zero. An
independently saved peak of640 with current RSS512 was then incorrectly refused
as a regression. Five new controls reproduce that refusal and validate the repair.
`WindowsRecoveryObserver(..., peak_rss_bytes=...)` accepts an exact bounded
nonnegative initial peak;composition supplies the independently known original
usage peak. Fresh observations use the maximum of that seed and measured RSS.
Original caps and regression checks remain unchanged. Default0 preserves ordinary
new-observer behavior. This argument does not create authority from checkpoint data.

```text
src/infra/windows_recovery_observer.py      preserve independently known RSS peak
src/infra/windows_recovery_composition.py   seed it from original journal witness
tests/test_windows_recovery_composition.py  five new regression/type controls
artifacts/runs/r35b2d3d-crash-20261008/       frozen capture/worker helpers and receipts
```

Before launch:522current tests passed,7spent fixtures deselected;both694file type
targets,whole lint,scoped formatting and helper compilation passed. All previous
budgets/failures and unrelated sources/packages remain preserved.

## Prospective limits and original identity

One direct existing Python3.14.7 base interpreter worker;original30-second elapsed
cap from an authentic anchor observation before launch;512MiB coordinator and
worker observed RSS caps;5-second protocol/wait/cleanup bounds;at most32bounded
messages of4096bytes. Aggregate metadata reservations:one uncertain admission,
8copy bytes,one grant,one checkpoint attempt. No native update was executed or
completed. Two fixed private cells use the same original worker/start/caps to test
publication and uncertain-stop history;neither resumes work or transfers budget.

Windows assigns physical PID/creation identity at launch. Source/policy/component
bytes,caps and original anchor/start were frozen first. The dispatcher then built
the exact records from independent retained native registrations before any journal
work. Child hello only cross-checks identity;it does not supply or reset authority.

The manifest's native/component files are explicit `ABSENT` markers with real byte
digests. They are not complete native,inbox,consolidation,lifecycle,actor or sharing
codecs. Their presence grants no native restore or scientific admission.

## Observed outcome

Worker physical PID13240,creation FILETIME134359233749095316,matched the independent
parent registration. The direct executable path and version3.14.7 matched. Original
anchor PID24228/creation134359233744118557 remained live throughout the capture.
Authentic original interrupt-clock start was36235855506800ns;the measured scenario
elapsed440038800ns,within30seconds. Parent/worker observations stayed within the
declared RSS caps and shared the original anchored clock epoch.

1. The supported composition factory created both coordinator cells through inner
   leased/report ports with retained registrations and exact independently known
   authority. Publication committed fresh observations and one copy-byte charge.
2. A valid report from the real worker was refused while the coordinator held
   publication ownership. The worker's later ownership also refused a surviving
   writer. These were actual cross-process native lock checks.
3. A second cell durably reserved one uncertain admission,one copy byte,one grant
   and one checkpoint attempt. Completed admissions remained0;no model ran.
4. The worker entered the supported observation-report lease and an actual SQLite
   write transaction. A COMMIT hook interrupted it before COMMIT,with a4,616-byte
   rollback file present. It exited via `os._exit(17)` without Python cleanup.
5. The surviving reader obtained the exact prior committed live record. Both
   coordinators refused the ended original worker and persisted stops without
   refunding either cell's committed charges or clearing uncertain work. Exact
   independent stop-only reconciliation assertions passed;no retry was granted.

The capture field `hot_journal_record_recovered_exact` records equality of the
surviving record. The helper did not attest the rollback header or prove main-page
mutation/replay. That field name is broader than the evidence. The accepted result
is an actual process exit during a pre-COMMIT observation transaction,with the
prior committed record intact. Hot-journal replay,power loss and lost acknowledgement
after a real committed transaction remain unproven by this capture.

## Cleanup and evidence binding

Exit code17 and retained ended-process facts were confirmed. The reader joined;
all three owned pipes and the Popen native process handle closed. Twelve successful
parent native CloseHandle calls were recorded,with zero remaining tracked process
handles. The dead worker's native lock/SQLite ownership released:survivor reads,
terminal writes and reconciliation succeeded. Worker-owned OS handles terminate
with that process;this does not claim explicit child Python finalizers ran.

The exact executed helpers are pinned in source-freeze.json. Preserve them and the
launch reservation,capture-result.json,original policy/component marker bytes,
SQLite cells and command receipts unchanged. Source/capture acceptance audits
bind the final source/test/type/static/guide and actual identity/accounting/cleanup
evidence. The capture is not authorization to replay its spent process allowance.

## Current commands and next extension

Use a new test temporary directory on every invocation;native fixtures remain
excluded unless a separate fresh process envelope is explicitly declared.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_windows_recovery_composition.py -q --basetemp=<new-owned-directory>
.venv/Scripts/python.exe -m ruff check src tests scripts
.venv/Scripts/python.exe -m ruff format --check src/infra/windows_recovery_observer.py src/infra/windows_recovery_composition.py tests/test_windows_recovery_composition.py
.venv/Scripts/python.exe -m mypy --platform win32 --no-incremental
.venv/Scripts/python.exe -m mypy --platform linux --no-incremental
```

Next inspect complete in-memory candidate/native/inbox/consolidation/lifecycle/
actor/sharing checkpoint state and implement complete versioned bounded durable
codecs. Do not treat absent markers,metadata journals,same-process pickle or scalar
summaries as complete restore authority. Preserve original native payloads and
source/policy bindings,all consent/consumed-ID/revocation/tombstone/retention/copy
ledgers,stable actor/single-owner admission and cumulative resource criteria.
Full R3.5b2/R3.7/R3.8/G3/native model/coordinator-loss/human acceptance stays open.
