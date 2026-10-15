# Supervised native checkpoint writer interruption

The Windows example keeps a parent PC/CPC actor available when its checkpoint
writer is terminated after staging a replacement. A replacement child restores
the previous complete model for prediction under the same parent coordinator.

```powershell
python -m scripts.run_coupled_checkpoint_interruption --describe
python -m scripts.run_coupled_checkpoint_interruption --directory artifacts/my-new-writer-example
```

Description performs no operational IO, model or process work. Actual execution
requires Windows and a new dedicated directory; an existing directory refuses
before native models or children. It leaves one checkpoint, a bounded SQLite
authority database and its permanent writer lock per family, plus JSON on stdout.
The worker's known orphan is inspected and removed after its ended proof.

## Sequence and accounting

1. Prepare actor A (two updates) and candidate B (one fork update) for each of
   PC, neutral CPC and active CPC. This nine-update fixture preparation happens
   **before** the recovery authority sessions; it is not resumed worker training.
2. Save A. Launch the physical base Python worker with the current environment's
   existing package path. Validate its hello PID/version and retain its Windows
   process registration while alive.
3. The existing coordinator charges a checkpoint attempt and a fixed copy ceiling
   before sending B. The child decodes/reencodes B through the accepted codec and
   file store, then pauses immediately before replacement, after fsync and close.
4. Complete the coordinator call while the child is still alive. Parent actor A
   serves unchanged predictions. Then terminate/reap only that owned child, prove
   it ended with its retained handle, and verify actual A and staged B bytes.
5. Remove only the verified owned staging file. Register a new child and perform
   the supported handoff. Existing handoff accounting charges attempt 2; the
   original session, start, limits, manifest and spent copy charges remain.
6. Reserve the read copy ceiling, restore complete A in child 2 and predict without
   updating. Verify every snapshot field and predictions against the parent.
7. While child 2 remains alive, request another checkpoint attempt. Limit 2 is
   exhausted; refusal precedes its callback and terminally stops the coordinator.
   No spent work is refunded. Stop/reap the reader and verify actor A again.

Each full example prepares nine native updates/36 energy evaluations, launches
six children and performs no child training. Per-family copy reservations are
twice the encoded ceiling for staging and twice for restoration; they remain
charged even when actual payloads are smaller. Authority updates/grants stay 0,
attempts advance 0→1→2, owner epoch advances once. Elapsed/observed RSS high water
are retained by the surviving parent. Fixture component labels describe this
model-only engineering session, not authenticated full-runtime provenance.

## Structure and boundaries

```text
src/infra/bounded_json_child.py                 owned bounded pipe/process IO
src/adapters/coupled_checkpoint_worker.py       stage or restore/predict protocol
src/adapters/coupled_checkpoint_interruption.py outer native/authority composition
scripts/run_coupled_checkpoint_interruption.py  CLI
tests/test_bounded_json_child.py                independent transport controls
tests/test_coupled_checkpoint_interruption.py   independent native integration
tests/test_coupled_checkpoint_interruption_cli.py CLI boundary/integration checks
```

The transport uses only the standard library. It has bounded stdout JSON lines,
aggregate bytes, message queues, retained stderr and a dedicated stdin writer.
Operations have finite timeouts; failure cleanup has its own finite timeout
(default 5 seconds each). Cleanup terminates/reaps the owned Popen, then joins IO
threads before closing buffered pipes. Inherited descendant pipes may prevent
cleanup; that reports an error within the cleanup deadline. No process-tree
termination or descendant ownership is claimed.

The worker rejects duplicate/nonfinite/nonobject JSON, invalid controls and
oversized lines. It independently caps declared encoded/array allowances at
65,536 bytes and supports restored input dimension 2 for the fixed prediction
fixture. It never trains. Stage mode cannot publish B even if its pause is stopped
or receives invalid input. Existing codecs, file store, native equations and
coordinator/SQLite/Windows authority code are reused unchanged.

## Verification commands

```powershell
python -m pytest -q tests/test_bounded_json_child.py tests/test_coupled_checkpoint_interruption.py tests/test_coupled_checkpoint_interruption_cli.py
$writerFiles = @('src/infra/bounded_json_child.py', 'src/adapters/coupled_checkpoint_worker.py', 'src/adapters/coupled_checkpoint_interruption.py', 'scripts/run_coupled_checkpoint_interruption.py', 'tests/test_bounded_json_child.py', 'tests/test_coupled_checkpoint_interruption.py', 'tests/test_coupled_checkpoint_interruption_cli.py')
python -m ruff check @writerFiles
python -m ruff format --check @writerFiles
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 @writerFiles
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform linux @writerFiles
```

Execute native/process checks under a declared local allowance. Evidence and the
current finite scope are in artifacts/runs/coupled-checkpoint-interruption-20261015.

## Remaining recovery requirements

This verifies a checkpoint writer crash while the original parent survives.
Existing native update reservations remain uncertain and cannot resume; the
example preserves that rule. No uncertain update completion, B publication,
coordinator-loss restart, full cursor/lifecycle/consent/budget reconstruction,
hostile filesystem/multiwriter/CAS or power-loss recovery follows. Extend the
existing inner completion/admission contracts before enabling resumed training;
valid native bytes alone never grant authority or renew allowances. See ADR0274.
