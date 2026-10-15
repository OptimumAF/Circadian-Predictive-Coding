# Physical Windows coordinator transfer

The `WindowsTransferSupervisor` adapter retains independently known completed
authority and Windows registrations while the original coordinator and worker
are alive. It survives their exit and binds the replacement coordinator's own
observation through a directly owned, trusted `BoundedJsonChild` channel.

## Run the fixed example

```powershell
.\.venv\Scripts\python.exe -m scripts.run_windows_transfer --describe
New-Item -ItemType Directory -Path .\artifacts\my-transfer
.\.venv\Scripts\python.exe -m scripts.run_windows_transfer --run --directory .\artifacts\my-transfer
```

Use a new private directory; the example creates one authority database without
overwriting an existing file. Windows and the existing physical base Python
interpreter are required. Four directly owned children, CPU threads1, zero native
model work, original60-second/512MiB observation limits, copy10/attempt2 caps.

The original child binds a real `RecoveryCoordinator` and executes a zero-cost
callback/report. The supervisor receives its complete authority over the private
channel while retaining live registrations. That coordinator then deliberately
exits17 without orderly close, and its worker exits0. A new coordinator/worker
pair is pinned while alive. The new coordinator sends its own native observation;
the supervisor brackets it and commits the explicit SQLite transfer before
sending the independently retained authority for adoption. The new coordinator
executes one copy1/attempt1 callback, then the original attempt limit refuses
another callback and stops. All owned children are reaped and registrations closed.

This example uses a **zero-native metadata fixture**, including explicitly labeled
synthetic manifest digests. It proves the physical coordinator/clock/channel and
metadata continuation boundary. It does not restore a checkpoint, perform native
learning, prove learning benefit or establish complete crash/power-loss recovery.

## Boundaries and failure behavior

`core/coordinator_transfer` stays pure. Its optional `supervisor_after` observation
retains the later supervisor time and RSS after the remote reply; existing
three-observation callers keep their original behavior. No remote fact is
relabeled, elapsed start rebased, peak discarded or cumulative allowance renewed.

`infra/windows_transfer_supervisor` owns registrations, not processes. The caller
owns the `BoundedJsonChild` lifetimes. Construct while the original handles are
live; no constructor can rebuild this continuity from a saved database. Its
`plan_transfer` and `commit_transfer` are single-use, with a nonblocking operation
gate and permanent failure state. The current physical supervisor, full pinned
identities, live replacement handles, owned child PID, clock bracket and complete
reply schema must agree. Trusted child code and a private local journal remain
explicit prerequisites; JSON and dataclass construction do not authenticate them.

Only acknowledged CAS permits adoption. On ambiguity, `failure_witness()` captures
fresh supervisor facts for the existing stopped-only reconciliation. It never
retries the transfer or resumes work. If the child already advanced the journal,
the old before/after transfer witness cannot overwrite that newer state; the
caller must preserve the uncertainty and use independently retained current
authority in a later explicit protocol.

## Test and extend

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_windows_transfer_supervisor.py tests/test_coordinator_transfer.py tests/test_windows_recovery_observer.py tests/test_windows_recovery_composition.py
```

The focused tests use deterministic native API fakes and real SQLite, with no
physical children or native model work. Source, tests, actual physical example and
commands are recorded in `artifacts/runs/windows-transfer-20261015/`.

R3.7p now composes independently retained genuine native checkpoint bytes with
this physical path; see [native continuation](native-coordinator-transfer.md).
The metadata-only example here remains unchanged. Full managed recovery remains
open. See ADR0285.
