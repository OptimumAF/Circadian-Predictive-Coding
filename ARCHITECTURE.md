# Architecture

## Objective

The repository is designed to evolve Circadian Predictive Coding as the main algorithm while preserving reproducible comparisons with:

- traditional backpropagation
- traditional predictive coding

## Layer Boundaries

- `src/core`
  - Pure model logic, learning dynamics, and typed model configs
  - No CLI parsing, environment loading, or dataset IO
- `src/app`
  - Use-case orchestration for experiment runs and benchmark workflows
- `src/infra`
  - Dataset and dataloader construction only
- `src/adapters`
  - User-facing CLI parsing and text formatting
- `src/config`
  - Environment variable mapping into typed settings
- `src/shared`
  - Small runtime helpers shared across modules (for example optional torch loading)

## Dependency Direction

```text
adapters -> app -> core
config   -> adapters
infra    -> app
shared   -> core + app + infra

core must not depend on app/infra/adapters.
```

## Core Domain Components

- `BackpropMLP`
  - Baseline one-hidden-layer backprop model for toy tasks
- `PredictiveCodingNetwork`
  - Baseline predictive coding model with iterative hidden-state inference
- `CircadianPredictiveCodingNetwork`
  - Primary algorithm with:
    - chemical-gated plasticity
    - wake/sleep phases
    - split/prune structural adaptation
    - replay/homeostasis/threshold control knobs
- `resnet50_variants.py`
  - Head-to-head benchmark implementations for all three model families on a shared ResNet-50 backbone

## Data Flow

### Toy baseline

1. `infra.datasets` creates deterministic two-cluster data
2. `app.experiment_runner` trains all three toy models
3. `adapters.cli` exposes baseline and in-depth modes

### ResNet benchmark

1. `infra.vision_datasets` creates synthetic or torchvision dataloaders
2. `app.resnet50_benchmark` runs all three models with aligned evaluation metrics
3. `adapters.resnet_benchmark_cli` exposes benchmark configuration
4. `scripts/run_multiseed_resnet_benchmark.py` aggregates cross-seed results

## Design Decisions

1. Circadian-first with mandatory baseline comparisons
   - Why: improvements are only meaningful when measured against stable references.
2. Separate wake training and sleep consolidation
   - Why: mirrors the circadian concept and keeps adaptation logic explicit and testable.
3. Configuration-heavy experiment control
   - Why: enables reproducible sweeps and ablations without branching code paths.
4. Deterministic seed handling
   - Why: avoids flaky claims in model comparisons.

## Extension Rules

- New adaptation strategies should be added via policy/config extension points, not by hardcoding branches across modules.
- New datasets must be added in `infra` and wired via `app`, never directly from `core`.
- Major algorithmic changes require an ADR in `docs/adr/`.
