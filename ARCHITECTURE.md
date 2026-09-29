# Architecture

## Objective

The repository is designed to evolve Circadian Predictive Coding as the main algorithm while preserving reproducible comparisons with:

- traditional backpropagation
- traditional predictive coding

## Layer Boundaries

- `src/core`
  - Pure model logic, learning dynamics, typed model configs, and validated sleep-event facts
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
- `replay_retention.py`
  - Pure eviction choice for explicit bounded FIFO or seeded bottom-k
    retention; the model owns copied rows, budgets, and snapshots
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
4. Opt-in v6 uses `infra.continual_roles` for four phase-local identities,
   `app.continual_arrived_benchmark` for role release and final reporting,
   `app.continual_arrived_transactions` for active model/sleep continuation,
   and `app.continual_arrived_checkpoint` for its separate format-6
   identity and cursor checks. `app.continual_arrived_sleep_history`
   validates the separate typed sleep history against phase roles and the
   existing guard ledger before restoration; the same trusted file layer
   persists it. A failed sleep attempt keeps the `before_sleep` model cursor;
   ordinary v6 can retry locally by explicit bound, while checkpointed v6
   raises and resumes later without repeating wake work
5. Opt-in v7 uses `app.continual_arrived_selection` above the v6 training
   boundary to predeclare equal-work candidates, score only arrived outer
   roles, freeze independent per-method choices, and then release final
   tests. `app.continual_arrived_selection_checkpoint` defines a distinct
   candidate-manifest cursor; `app.continual_arrived_selection_resume`
   embeds v6 active transactions and validates completed trials and their
   independent sleep-history provenance before continuation. The trusted
   file layer persists format 8 separately
6. `scripts.run_continual_arrived_confirmation` fixes and persists one
   small study request, invokes the v7 app through ordinary and interrupted
   checkpoint paths, and writes a full result artifact. Its source seals
   and state comparisons are experiment checks, not alternate training
   logic (ADR-0079)

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
   metrics and resumes CPU/CUDA unmatched protocols before final-test scoring;
   the vision payload binds the selected CUDA process stream and seeded-fork
   outer stream beside the classifier's local split generator (ADR-0086)
6. `adapters.resnet_benchmark_cli` exposes benchmark configuration
7. `scripts/run_multiseed_resnet_benchmark.py` aggregates cross-seed results

The opt-in development-only torchvision loader omits final CIFAR source
construction and final role identity while preserving the ordinary loader
default. `app.matched_head_tuning._build_seed_bank` can request that
boundary for feature-cost probes. The local probe script measures only
development feature setup under a quiet-device and worker-time gate;
`scripts.prepare_cifar_representative_study` freezes the subsequent
matched selection/confirmation request without training (ADR-0080).
`app.matched_head_tuning` accepts a development-only source and an optional
attempt observer; the latter lets the local
`scripts.run_cifar_representative_selection` adapter persist each candidate
before its bounded CUDA worker exits. That adapter verifies source/request
digests, quiet GPU, equal trials, shared identities, and the confirmation
manifest while final CIFAR source construction remains sealed (ADR-0081).
`scripts.restore_cifar_representative_selection` performs a read-only
artifact and source-digest preflight before any later confirmation adapter
may construct the final dataset (ADR-0082).
`scripts.run_cifar_representative_confirmation` repeats that preflight,
records the quiet CUDA gate, and launches separately bounded fixed-data,
wall-time, and fresh-child memory workers. Each worker repeats the freeze
check; completed attempts or seeds are persisted before later work.
`scripts.audit_cifar_representative_confirmation` restores typed saved
scope reports and checks complete seed/head coverage, paired identities,
work, deadlines, and scoped memory before the parent accepts a result
(ADR-0083). These scripts own local experiment orchestration and artifact
IO; the app modules still own training and scoring.

The fixed-feature matched runner can restore CPU or CUDA learning state.
On CUDA, the combined checkpoint binds the process CUDA RNG to the live
head's device while its head snapshot retains the local split generator
(ADR-0084). The capacity-only route verifies equal head width and guarded
sleep without a memory claim (ADR-0060). Checkpoint-memory routes store
per-process RSS segments; CUDA routes also store device-bound allocated and
reserved starts and peaks. Each resumed report retains its segments and
exposes maximum observed absolute peaks across committed processes, without
a common circadian start. The original memory protocols retain their scope
(ADR-0061/0085).

## Design Decisions

The opt-in v8 ordinary replay-policy comparison sits in `app` above the
existing v6 arrived-role boundary. Its fixed manifest binds FIFO and seeded
bottom-k trials to one source/role configuration and seed list; `core`
retains only policy and exposure mechanics, while `infra` writes the local
JSON artifact. A separate `app` format-9 completed-trial prefix and active
A/B cursor, with an `infra` trusted local store, preserve unscored
policy/seed state across restarts. The app validates role, model, sleep,
and exposure provenance before continuation (ADRs 0103–0105).

The P4.4a `core` shared replay buffer applies the existing FIFO/bottom-k
eviction under one copied-array cap and a prediction-independent recent
sampler. An `app` arrived-role session opens only Phase A train rows first,
then Phase B after its declared A epochs; it emits one ordered selection and
separate planned work for all three methods. It does not train or score
models. A bounded `scripts` adapter writes a schedule-only local artifact
(ADR-0106).

The P4.4b `app` training runner uses that session before each guarded
NumPy sleep and preflights the circadian buffer's retained and selected
IDs. Once the inner guard accepts circadian replay, it feeds separate
copies of those rows to PC and backprop and records actual per-method
calls and inference-loop work. It holds unscored A/B models and role/sleep
audit without opening final tests. A bounded `scripts` adapter writes a
deterministic local training trace (ADR-0107).

The P4.4c `app` outcome runner validates every unscored trial against a
fresh train-only schedule before final-role release. It then binds all
final A/B roles, compares their hashes across policies, and reuses the
existing continual-shift scoring calculation. The `scripts` adapter
persists every fixed policy/seed result without timing fields in its
deterministic JSON. No model is chosen from the result (ADR-0108).

The P4.3b NumPy `core` model offers a pretraining opt-in replay side-effect
policy. Historical models retain their snapshot fields and training
behavior. The wake-only policy updates weights/exposure using pre-row
adaptive state without advancing chemistry, structural usage/progress, or
wake clocks. The `app` v10 ablation reuses the v9 schedule and guarded
training boundary, audits equal applied rows and baseline states across
policies, and opens final roles only after all eight trials freeze. The
`scripts` adapter writes the bounded local result (ADR-0109).

For structural changes, NumPy's existing `NeuronAdaptationPolicy` receives
detached layer traffic and returns typed proposals. The core parses these,
checks active budgets and eligible sources, ranks candidates from its
separate usage scores, then mutates tensors only after validation. Torch
keeps separate methods because it may prune a newly created child after
a noisy split. Shared protocol or config fields would obscure that
different ordering (ADR-0110).

The P4.6b v11 `infra` source supplies deterministic A/B development rows
and deferred final properties. The existing four-role splitter owns row
identity and final release. A pure `core` diagnostic computes clipped
train error and BCE without changing updates. The `app` runner copies one
B train row per corruption condition, hashes effective train content,
trains matched reward-switch arms, and checks all 24 role/work traces
before final release. The `scripts` adapter writes every outcome to new
local JSON. Historical model/config/checkpoint protocols remain intact
(ADR-0112).

The P4.7b v12 `app` trial uses that sealed A/B source and the existing
NumPy/CPU-Torch core heads. Its post-update controls separately remove
reward scaling from the applied parameter delta or the newly added
importance EMA contribution; the third factor controls importance's
existing structural score mix. One A-boundary sleep retains each core's
native split/prune ordering. The comparison app preflights all 32
unscored cells, including role hashes, actual work, initial/pre-sleep
state matching, stable neuron lineage, and capacity, before releasing
shared final roles. A `scripts` adapter writes train-only or scored JSON
to a new local path. This experiment adds no core/config/checkpoint
identity or setting-selection path (ADR-0114).

For P4.8a, `infra` generates paired stationary/axis-shift phase rows
with final fields deferred behind the existing role release boundary.
The `app` fixes per-epoch train jitter, compares existing runner/core
periodic, adaptive, and no-sleep decisions, and checks equal wake work,
capacity, train streams, and actual chemical-reset sleep events in all
twelve cells before final scoring. Its fixed-width, replay-free control
isolates trigger timing; P4.8b retains full sleep-component attribution.
The `scripts` adapter only writes train-only or scored local JSON to a
new path. Historical trigger/config/checkpoint identities are unchanged
(ADR-0115).

For P4.8b1, a separate `app` schedule advances the existing arrived
four-role A→B source one train-only wake epoch at a time. It uses the
`core` shared FIFO replay buffer to emit prediction-independent potential
rows and per-method planned work at *every* epoch. A fixed v14 manifest
binds seeds, source, trigger arms, replay caps, guard tolerance, and
structural limits; the periodic subset is checked against v9 without
changing v9's identity. The `scripts` adapter writes only train-role
facts to an exclusive local JSON file. Actual guarded training and final
release remain a separate P4.8b2 app boundary (ADR-0116).

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
