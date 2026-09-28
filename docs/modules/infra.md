# Module: `src/infra`

## Responsibilities

- Generate deterministic synthetic datasets for experiments
- Generate deterministic synthetic vision datasets for ResNet benchmarks
- Persist trusted local fixed-feature, toy, continual, and seeded vision
  checkpoint files

## Inputs / Outputs

- Inputs: sample count, noise, seed
- Outputs: train/test split dataclass
- `circadian_checkpoint_files.py` takes a runner checkpoint and a
  local path, writes a checksummed temporary file then replaces the target,
  and loads a verified payload. Pickle loading is restricted to trusted
  local files; protocol/config/data validation belongs to the app layer
  (ADR-0053, ADR-0055–0056, ADR-0058).

## Non-Responsibilities

- Training orchestration
- Model internals
- CLI concerns
