# Architecture

## Boundaries

- `src/core`
  - Responsibilities: model math, training rules, traffic summaries, adaptation interfaces
  - Inputs/outputs: numpy arrays and typed dataclasses
  - Non-responsibilities: CLI parsing, environment loading, file/network IO
- `src/app`
  - Responsibilities: compose dataset + models into a reproducible experiment
  - Inputs/outputs: config objects in, single-run and aggregate experiment reports out
  - Non-responsibilities: low-level math implementation
- `src/infra`
  - Responsibilities: synthetic dataset generation
  - Inputs/outputs: deterministic train/test split
  - Non-responsibilities: model optimization logic
- `src/adapters`
  - Responsibilities: command-line argument handling and output formatting
  - Inputs/outputs: process arguments in, text output out
  - Non-responsibilities: core training logic
- `src/config`
  - Responsibilities: map environment variables to typed defaults
  - Inputs/outputs: process environment in, `Settings` out
  - Non-responsibilities: training orchestration

## Dependency Graph

```text
adapters -> app -> core
config   -> adapters
infra    -> app
app      -> core + infra
core     -> (no inward dependency on app/infra/adapters)
```

## Key Choices

1. Predictive coding implementation uses iterative hidden-state inference.
Reason: this is the minimum mechanism that makes it conceptually different from standard backprop.

2. Same dataset and near-similar architecture for both models.
Reason: keeps comparison focused on learning rule differences rather than dataset/model mismatch.

3. Neuron adaptation policy defined as an interface with no-op default.
Reason: enables future growth/pruning experiments without destabilizing the baseline comparison now.

4. In-depth comparison runs repeated seed/noise scenarios.
Reason: single-run metrics are noisy; aggregate stats make differences between learning rules clearer.
