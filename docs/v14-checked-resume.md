# Checked local v14 trial resume (P5.3b)

The fixed NumPy v14 study has six seed/arm trials. A resumable run
stores each completed, unscored trial prefix under the ignored
`artifacts/runs/.<run-id>.resume/` directory. The public run directory
appears only after all six trials pass the original global preflight,
final roles are released and scored, and complete artifact files are
atomically published.

From the repository root, with an unchanged Git workspace and runtime:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --run-id p53-local --resumable --capture-wake-diagnostics
.\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --run-id p53-local --resume --capture-wake-diagnostics
.\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --verify-run artifacts/runs/p53-local
.\.venv\Scripts\python.exe -m scripts.project_v14_observations --run-measured artifacts/runs/p53-local
.\.venv\Scripts\python.exe -m scripts.project_v14_observations --verify-measured-run artifacts/runs/p53-local
```

The first command creates the cursor. Use the second after an
interruption; on an already completed run it verifies and returns the
existing bundle. Repeat the same diagnostic capture flag. The CLI
refuses an occupied public ID or a fresh run over an existing resume
state. Put a custom output root outside the source tree or under an
ignored path: the workspace identity is checked before and after the
run and again on resume.

`run-state.json` records `incomplete` while active, `failed` after a
caught error, `canceled` after KeyboardInterrupt/SystemExit, and
`completed` only after bundle and optional sidecar publication. It
stores the exact immutable checkpoint filename/SHA-256, next cell,
source and environment facts, config/protocol hashes, and capture mode.
A process crash can leave `incomplete`; the OS lock is released on
process exit. A crash during a trial repeats that trial. A crash after
checkpoint file creation but before cursor replacement leaves an
unreferenced checkpoint file, which is retained for inspection.
Publication stages and their separate statuses remain governed by
[P5.3a](atomic-artifact-publication.md).

Only trusted local checkpoint files may be loaded: format-10 payloads
use pickle behind a checksum and exact run-state file hash. Saved
trials contain no deferred final source references or final-role
release. On resume, the runner validates the whole stored prefix,
including role/replay/work/capacity facts and matched arms, before
running another trial. After all six cells, it reconstructs the fixed
source references, preflights all six again, then opens final roles.
It will not overwrite a changed published bundle or sidecar.

The scope is same-environment repeatability for the fixed v14 NumPy
study. There is no mid-trial cursor, sweep scheduler, cross-version
pickle compatibility, or signed attestation. The baseline methods,
seeds, metrics, trigger settings, and negative/mixed result are
unchanged (ADR-0124).
