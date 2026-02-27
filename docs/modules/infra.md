# Module: `src/infra`

## Responsibilities

- Generate deterministic synthetic datasets for experiments
- Generate deterministic synthetic vision datasets for ResNet benchmarks

## Inputs / Outputs

- Inputs: sample count, noise, seed
- Outputs: train/test split dataclass

## Non-Responsibilities

- Training orchestration
- Model internals
- CLI concerns
