# Module: `src/adapters`

## Responsibilities

- Parse command-line arguments
- Invoke app-level use cases
- Print user-facing experiment summaries
- Expose a dedicated CLI for ResNet-50 speed benchmarking
- In `scripts/run_continual_arrived_confirmation.py`, prepare an immutable
  local v7 study request and invoke ordinary/checkpointed app use cases
  under fixed correctness and work checks
- In `scripts/profile_cifar_representative_feasibility.py`, prepare a
  fixed CUDA development-only cost request and run one bounded child
  feature extraction; `scripts/prepare_cifar_representative_study.py`
  saves the later matched-budget request without training;
  `scripts/run_cifar_representative_selection.py` checks its frozen digests,
  quiet-device gate, and six development-only candidates and writes a
  result, synced attempt journal, manifest, or failure artifact
- `scripts/restore_cifar_representative_selection.py` verifies exact
  result/journal/manifest bytes and source hashes without dataset access;
  it returns a typed frozen manifest plus the audited local evidence
- `scripts/run_cifar_representative_confirmation.py` enforces the restored
  freeze, quiet CUDA gate, separate scope/total time limits, and local
  attempt/seed/result/failure persistence; `scripts/audit_cifar_representative_confirmation.py`
  checks complete typed scope reports and paired identity/work/memory evidence
- `scripts/run_structural_rank_comparison.py` runs the fixed v12 train-only
  gate or globally sealed comparison and writes deterministic JSON to an
  exclusive new local path; it does not implement factor training or
  release final roles
- `scripts/run_sleep_trigger_comparison.py` writes fixed v13 train-only
  or globally sealed outcome JSON to an exclusive new local path; the
  app owns trigger decisions and final-role release
- `scripts/run_continual_trigger_replay_schedule.py` writes fixed v14
  train-only replay opportunities to an exclusive new local JSON path;
  it serializes potential rows/work without model training or scoring

## Inputs / Outputs

- Inputs: process arguments and optional environment-backed defaults
- Outputs: console output; the confirmation script also writes an ignored
  request, trusted checkpoint files, and a complete JSON result
- The CIFAR probe and study-preparation scripts write ignored request,
  measured result or failure, and frozen future-study JSON artifacts
- The bounded CIFAR confirmation writes ignored gate, fixed-data journal,
  per-seed wall-time/memory, final result, or failure artifacts

## Non-Responsibilities

- Dataset generation
- Model training internals
- General-purpose persistence or network IO

The confirmation script is a local experiment adapter. Its explicit file
IO is confined to request/result artifacts and the existing trusted
checkpoint store; it does not implement model training or selection rules.
