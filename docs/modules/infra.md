# Module: `src/infra`

## Responsibilities

- Generate deterministic synthetic datasets for experiments
- Generate deterministic synthetic vision datasets for ResNet benchmarks
- Persist trusted local fixed-feature, toy, continual, and seeded vision
  checkpoint files

## Inputs / Outputs

- Inputs: sample count, noise, seed
- Outputs: train/test split dataclass
- `datasets.py` can carry a deferred final-test role when the caller sets
  `hash_test=False` and `defer_test_access=True`. It reads source test
  fields only when the caller later validates, hashes, or scores that role;
  it does not control when the synthetic generator allocates test arrays.
- `circadian_checkpoint_files.py` takes a runner checkpoint and a
  local path, writes a checksummed temporary file then replaces the target,
  and loads a verified payload. Pickle loading is restricted to trusted
  local files; protocol/config/data validation belongs to the app layer
  (ADR-0053, ADR-0055–0056, ADR-0058). Continual format 5 stores
  unscored trained seed records; the store remains unaware of test roles
  and defers their identity checks to the app layer (ADR-0072).

## Non-Responsibilities

- Training orchestration
- Model internals
- CLI concerns
