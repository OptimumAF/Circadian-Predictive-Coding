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
  - Dataset/dataloader construction and trusted local checkpoint files
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
   - `app.toy_checkpoint` validates run identity and model-order progress;
     `infra.circadian_checkpoint_files` stores the trusted payload
3. `adapters.cli` exposes baseline and in-depth modes

### Continual shift

1. `infra.datasets` regenerates phase-A and phase-B roles per seed
2. `app.continual_shift_benchmark` trains both phases, freezes A-only
   state, and scores final tests after each seed finishes training
3. `app.continual_checkpoint` validates seed/phase progress and committed
   reports; `infra.circadian_checkpoint_files` persists its trusted payload

### ResNet benchmark

1. `infra.vision_datasets` creates synthetic or torchvision dataloaders
2. `app.seeded_vision_loader` reconstructs v3 train order and augmentation
   streams from a logical epoch/batch cursor (ADR-0057)
3. `app.shared_vision_loader` replays the v1/v2 shared sampler without
   reseeding or changing their ordinary process streams (ADR-0059)
4. `app.vision_checkpoint` binds runner/model state and exact development
   data; `infra.circadian_checkpoint_files` persists a trusted local file
   (ADR-0058–0059)
5. `app.resnet50_benchmark` runs all three models with aligned evaluation
   metrics and resumes CPU unmatched protocols before final-test scoring
6. `adapters.resnet_benchmark_cli` exposes benchmark configuration
7. `scripts/run_multiseed_resnet_benchmark.py` aggregates cross-seed results

The fixed-feature matched runner has separate CPU checkpoint protocols. The
capacity-only route verifies equal head width and guarded sleep without a
memory claim (ADR-0060). Explicit checkpoint-memory routes store per-process
RSS segments in the fixed-feature payload and expose maximum observed absolute
RSS across committed segments; the original memory-enabled protocols retain
their scope (ADR-0061). CUDA checkpoint evidence remains open.

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
