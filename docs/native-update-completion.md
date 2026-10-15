# Explicit native update completion

The coordinator has an opt-in completion path for one admitted native-only
candidate update. It retains the original reservation, checks the same live
registered worker, and records completion only after a trusted parent callback
verifies a complete candidate checkpoint. Actor A remains unchanged.

```powershell
python -m scripts.run_coupled_update_completion --describe
python -m scripts.run_coupled_update_completion --directory artifacts/my-new-completion-example
```

Description performs no operational IO/models/processes. Actual execution needs
Windows and a new dedicated directory, CPU1. It runs PC, neutral CPC and active
CPC success cases plus active CPC validation refusal after valid B bytes were
written. Sixteen native updates/64 energy evaluations include twelve preparatory
reference updates before authority sessions and four admitted child updates.
Fixed seed23/D2/H(3,2)/precision(1,1), train data and external rates are in the
description. No final evaluation data or comparative learning claim is involved.

## Corrupted candidate checkpoint control

```powershell
python -m scripts.run_coupled_update_completion --corrupt-checkpoint --describe
python -m scripts.run_coupled_update_completion --corrupt-checkpoint --directory artifacts/my-new-corruption-example
```

This explicit engineering mode runs PC, neutral CPC and active CPC, each with a
healthy child update. Twelve training calls/48 energy evaluations include nine
preparatory reference calls and three admitted child updates; twenty-one
predictions cover reference B, child B and five unchanged actor A checkpoints.
The fixed training fixture and external rates are the same as the default mode.

The parent verifies the complete response and saves valid B, then replaces only
the final `}` byte with `!`, flushes and fsyncs before bounded reload. Reload uses
the original independently expected digest; its actual content-mismatch error
must refuse completion. A successful reload fails the engineering control.
The row retains complete expected B and the valid worker response separately from
the damaged forensic file and its byte offset/digests. `actual_b` remains null:
no corrupt state or valid-before-corruption file is treated as completed B.

The child stays live through terminal checks and exits0 on exact stop. Authority
remains stopped/uncertain, admitted1/completed0/copy262144/attempt1/grants0;
the validator runs once, publication probe and refused retry callbacks never run.
The original entire manifest, A file, complete actor A and all predictions remain
unchanged. Corruption grants no retry, refund, actor promotion or bundle recovery.
Fault injection uses only a quiescent owned new-directory checkpoint; it does not
claim security against a hostile or concurrently changing filesystem.

Why this: matching the worker's response is insufficient when the parent-owned
checkpoint is damaged before reload. Exercise the actual bounded store refusal
and coordinator failure path together without weakening expected digest checks.
No core authority, journal, model equation, metric or scientific protocol changes.

For focused controls use `python -m pytest -q tests/test_coupled_checkpoint_corruption.py`.
Include the existing completion tests for healthy behavior; see ADR0277 and
`artifacts/runs/coupled-checkpoint-corruption-20261015/` for the prospective scope,
commands, results and independent review. Full recovery criteria remain open.

## API and trust boundary

```python
result = coordinator.execute_native_update(
    RecoveryCosts(updates=1, copy_bytes=copy_ceiling, checkpoint_attempts=1),
    prepare,
    validate_checkpoint,
)
```

`prepare()` is a trusted callback invoked after the admission/copy/attempt charges
are persisted. `validate_checkpoint(result)` is also trusted and returns an exact
`RecoveryManifest`. It must fully verify the parent-owned checkpoint, not trust
digests or counters from an untrusted JSON request. The example verifies full B
against an independently prepared one-update reference, writes B at a distinct
path from A, bounded-reloads it and checks its complete state before returning.

This path accepts only an original native-only payload whose payload/native
hashes are equal. Completion may change that pair, again equal, and preserves
source/policy/inbox/consolidation/lifecycle/actor/sharing bindings. It does not
flatten a full managed bundle into model bytes. A legitimate zero-effective
update may retain the same hash. Matching hashes alone never prove completion.

The inner `AuthorityCompletion` carries the original exact `AuthorityChange`
reservation. Current pending state may differ only by monotone report sequence,
time and observed RSS. A fresh same-worker live observation follows validation.
The explicit CAS increments completed by one and clears uncertainty while all
admitted/copy/grant/attempt charges remain spent. Old `execute(updates=1)` retains
its previous uncertain-work behavior and does not infer completion.

## Floor and failure behavior

The journal's ordinary recovery floor is unchanged. Only its own successful
validated `complete()` CAS advances that instance's trusted floor to B. A stale
original or pending witness reading disk B still refuses. The example checks
this and performs a zero-cost publication **probe** after success; it does not
replace actor A. A second update exceeds the original cap and refuses before
either native callback.

Preparation, validation or host failure before completion leaves the admission
uncertain and terminally stops the lane. The failure example retains a fully
valid B file but does not complete it, run the probe, retry or refund charges.
If completion COMMIT succeeds and its acknowledgment fails, the journal may hold
completed B. `FailureWitness` retains only known pending/completed outcomes for
explicit terminal-only reconciliation. That ambiguity never grants automatic
resumption, and a committed completion is not rolled back or charged twice.

## Structure and extension

```text
src/core/recovery_completion.py              pure explicit transition + port
src/app/recovery_coordinator.py              admitted callback/completion sequence
src/core/recovery_reporting.py               completion-aware terminal witness
src/infra/sqlite_recovery_journal.py          explicit validated completion CAS
src/adapters/coupled_update_worker.py         one fixed admitted native update
src/adapters/coupled_update_completion.py     native/Windows/SQLite composition
scripts/run_coupled_update_completion.py      CLI
tests/test_recovery_completion.py            independent core/failure/SQLite controls
tests/test_coupled_update_completion.py      independent full native integration
tests/test_coupled_update_completion_cli.py  worker/CLI boundaries and full CLI
tests/test_coupled_checkpoint_corruption.py  owned fault/CLI/full native refusal
```

The new worker reuses bounded transport and compatible restore helpers; the
existing read/stage worker remains unchanged and never trains. It enforces exact
D2/H(3,2)/precision(1,1) before constructing a model, independently caps encoded/
array allowances at65536B, performs one fixed update and waits for stop while the
parent validates. All existing model equations, authority wire fields and generic
floor checks remain unchanged. No dependency or environment configuration added.

```powershell
python -m pytest -q tests/test_recovery_completion.py tests/test_coupled_update_completion.py tests/test_coupled_update_completion_cli.py
python -m pytest -q tests/test_coupled_checkpoint_corruption.py
python -m pytest -q tests/test_recovery_authority.py tests/test_recovery_coordinator.py tests/test_recovery_reporting.py tests/test_sqlite_recovery_journal.py tests/test_recovery_publication_coordinator.py tests/test_recovery_publication.py tests/test_windows_recovery_composition.py
$completionFiles = @('src/core/recovery_completion.py', 'src/app/recovery_coordinator.py', 'src/core/recovery_reporting.py', 'src/infra/sqlite_recovery_journal.py', 'src/adapters/coupled_update_worker.py', 'src/adapters/coupled_update_completion.py', 'scripts/run_coupled_update_completion.py', 'tests/test_recovery_completion.py', 'tests/test_coupled_update_completion.py', 'tests/test_coupled_update_completion_cli.py')
python -m ruff check @completionFiles
python -m ruff format --check @completionFiles
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 @completionFiles
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform linux @completionFiles
```

Native/process checks require a prospective local allowance. Current scope and
receipts live in artifacts/runs/coupled-update-completion-20261015. Next extensions
must preserve original admission/ownership/failure uncertainty. Full managed
component completion, actor promotion and coordinator-loss recovery remain open;
valid model bytes do not supply lifecycle/consent/cursor/budget authority. ADR0275
records the design choice and remaining boundaries.
