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
- `scripts/run_continual_trigger_replay_training.py` writes fixed v14
  guarded train-only work and typed decision facts for all six trials to
  an exclusive new local JSON path, omitting durations and final scores
- `scripts/run_continual_trigger_replay_outcomes.py` writes all globally
  sealed v14 scored rows and paired contrasts to an exclusive local
  JSON path without implementing training, role release, or selection
- `scripts/run_versioned_v14_bundle.py` runs the fixed v14 study once,
  serializes train-only bytes before the global final gate, and writes
  or verifies an opt-in local P5.1 bundle. Its opt-in resumable mode
  validates an unscored format-10 prefix before further training or
  final release. It does not choose a model or alter v14 metrics
  (ADRs 0120, 0124). It declares the sole `fixed-v14` preset and rejects
  unknown settings before training (ADR-0126).
- `scripts/project_v14_observations.py` projects or verifies a completed
  P5.1 bundle in an exclusive local directory. It does not train,
  score, or choose a result (ADR-0121).
- `scripts/build_v14_artifact_report.py` publishes or verifies one
  source-checked descriptive JSON/CSV table beside a completed P5.1
  bundle. It does not train, choose a winner, or edit the dashboard
  (ADR-0137).
- `scripts/build_v14_dashboard.py` publishes or verifies one static
  dashboard derived from a checked P5.6a table. It does not train or
  modify the historical dashboard (ADR-0138).
- Opt-in `--capture-wake-diagnostics` on `run_versioned_v14_bundle.py`
  and `--run-measured` / `--verify-measured-run` on
  `project_v14_observations.py` publish genuine train-update metrics
  only after the completed global scored bundle exists (ADR-0122).
- `scripts/run_continual_shift_benchmark.py` parses typed repeatable
  `--override` values for its existing named profiles and optionally
  writes an exclusive fully resolved config record. The app validates
  fields before the adapter invokes training (ADR-0125).
- `scripts/run_multiseed_resnet_benchmark.py` retains its descriptive
  unmatched flags and output names, parses repeatable JSON settings
  after them, checks the app's typed config before training, and embeds
  complete per-seed settings in the JSON result (ADR-0127).
- `predictive_coding_experiment.py` delegates to `src/adapters/cli.py`.
  The adapter keeps the toy baseline/indepth flags, parses typed JSON
  overrides, rejects invalid requests before training, and saves a full
  exclusive resolved-config record or an enriched baseline JSON result
  after completion (ADR-0128).
- `src/adapters/toy_budget_cli.py` handles opt-in toy baseline update,
  time, replay, circadian hidden-width, and process-RSS limits. It claims a local
  lifecycle path, binds resume to the exact resolved config and trusted
  checkpoint hash/cursor, records observed and
  durable wake, replay, and width work plus per-invocation absolute process-RSS
  start/peak/sample facts for completed/incomplete/error attempts,
  and publishes a completed JSON only after final scoring. It does not
  select settings, change training rules, or open final roles (ADRs 0133–0136).
- `resnet50_benchmark.py` delegates to `src/adapters/resnet_benchmark_cli.py`.
  All old defaults come from a typed preset, existing flags remain, and
  repeatable JSON overrides are checked before the Torch runner. The
  adapter writes an exclusive full config or completed result JSON and
  retains the original stdout report (ADR-0129).

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
