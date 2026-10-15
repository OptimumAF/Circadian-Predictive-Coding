# Explicit coordinator transfer

`src/core/coordinator_transfer.py` defines an explicit transfer from a completed,
native-only checkpoint authority to a new coordinator and worker. The SQLite
journal persists that transition through `transfer_coordinator(change)`.

## Required trust

A separate supervisor must independently retain the completed authority and
continuously survive both coordinators on the same running Windows interrupt
clock. It must prove that the old coordinator and worker ended. A trusted new
coordinator observation must bind its live worker. These records validate those
relationships; constructing them or reading a database does not prove the facts.
The later physical supervisor adapter and fixed zero-native example are described
in [Windows coordinator transfer](windows-coordinator-transfer.md). Genuine native
checkpoint restoration remains an additional integration gate.

The transfer preserves the original absolute start, session, full manifest,
limits, event tick and cumulative work. It changes the anchor, worker, clock
identity tag and owner fence, advances sequence once, charges one checkpoint
attempt, and retains the maximum observed RSS. The new clock tag does not reset
elapsed time. Live supervisor/coordinator/worker PIDs must differ; ended PIDs may
be reused only with different full process identities.

## Journal use

1. Retain a `SupervisorRecoveryWitness` independently of disk.
2. Obtain trusted `CoordinatorTransferObservation` facts and call
   `plan_coordinator_transfer(witness, observation, new_owner_id)`.
3. Use a `SqliteRecoveryJournal` whose independent floor is exactly the witnessed
   authority. Call `transfer_coordinator(change)` once.
4. Continue existing coordinator ports only after acknowledged success and
   independently binding the new host facts. Existing original quotas apply.

Generic `read`, `advance`, `report`, `complete` and authority-floor checks retain
their original rules. An old journal cannot learn the new anchor from disk.

If a transfer raises before or after COMMIT, the adapter remains failed. Obtain a
fresh trusted supervisor observation binding the replacement anchor, construct
`CoordinatorTransferFailure`, and call
`SqliteRecoveryJournal.reconcile_coordinator_transfer(path, failure)`.
It accepts only the exact witnessed before/after state or its matching terminal
state, charges the attempted transfer once on either branch, and returns stopped
metadata. Repeating the same terminal reconciliation is idempotent. Unknown or
newer disk states are refused. This operation grants no resumed work and does not
reset the failed adapter. Terminal elapsed/RSS overshoot is retained truthfully.

## Focused checks

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m pytest -q tests/test_coordinator_transfer.py tests/test_sqlite_recovery_journal.py tests/test_recovery_authority.py tests/test_recovery_completion.py
```

Tests use real local SQLite and synthetic trusted process observations. They
exercise existing coordinator ports with no native model work. They establish
the typed transition and journal behavior, not physical process replacement,
checkpoint restoration, scientific benefit or complete crash recovery. See
`artifacts/runs/coordinator-transfer-20261015/` for the budget and actual results.

The optional fourth supervisor observation retains the physical adapter's later
time and RSS after the reply; three-observation callers retain their behavior.
Next integrate genuine completed checkpoint restoration into the physical path
before claiming native coordinator-loss recovery.
