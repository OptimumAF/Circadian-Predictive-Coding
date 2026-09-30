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
  - Pure prelaunch work estimates for synthetic vision sweep candidates
  - Opt-in toy execution ceilings and typed incomplete stops at checked wake/sleep cursors
- `src/infra`
  - Dataset/dataloader construction, trusted local checkpoints, and exclusive toy run-state files
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
3. `adapters.cli` exposes baseline and in-depth modes. Its opt-in budget
   handler binds a checked checkpoint to an exclusive local lifecycle state
   and publishes a result only after final scoring

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
release are separate P4.8b2 app boundaries (ADR-0116).

The v14 guarded runner is a separate `app` use case: it derives only
arm-specific sleep scheduling, trains all three NumPy methods on the
same arrived wake rows, checks core replay retention against each
all-epoch offer, and commits detached baseline replay after accepted
guarded circadian sleep. A second `app` study independently rederives
roles/opportunities and validates all six unscored work and structure
traces before its `scripts` adapter writes deterministic local JSON.
This train-only boundary does not release final roles or score models;
the completed P4.8b2b use case owns the global final-release boundary
(ADR-0117).

The v14 outcome `app` use case reruns the six-trial preflight, including
stable structural-ID reconstruction against saved A/B model lineage,
then releases all twelve final roles and compares within-seed final
identities before scoring any model. It reports all method metrics,
applied work/capacity, and signed paired contrasts. A `scripts` adapter
writes deterministic scored JSON to an exclusive local path; no
selection or new trigger rule is made (ADR-0118).
The v14 artifact's NumPy-only scope is recorded in the current
[backend capability matrix](docs/backend-capability-matrix.md) and a
hash-bound [result metadata sidecar](docs/result-backend-metadata.json)
without changing the scored artifact bytes (ADR-0119).

P5.1 adds a pure `core/run_manifest.py` field and identity validator.
The `app/versioned_v14_run.py` use case binds an already preflighted v14
study, its scored comparison, and their original-format bytes to that
contract. `infra/run_environment.py` captures the executing Git/workspace,
dependency, and CPU facts; `infra/versioned_run_files.py` writes complete
local JSON files and verifies their hashes and v14 cell/role identities.
The `scripts/run_versioned_v14_bundle.py` adapter orchestrates one study
and publishes `manifest.json` last. Dependency direction remains script
→ app/infra → core, and no historical v14/checkpoint format changes
(ADR-0120).

P5.2a adds `app/v14_observation_projection.py` as a pure transform of
the verified raw training/outcome records. It preserves epoch order and
labels missing wake metrics explicitly. `infra/observation_projection_files.py`
verifies the completed P5.1 source, writes derived JSONL/CSV beside it,
and verifies each byte against a fresh derivation. The
`scripts/project_v14_observations.py` adapter exposes local create/verify
commands. No app logic depends on infra; the fixed v14 scoring and role
release paths are unchanged (ADR-0121).

P5.6a adds `app/v14_artifact_report.py` as a pure aggregate over all
declared v14 seeds, arms, and methods. `infra/v14_artifact_report_files.py`
first verifies the completed P5.1 bundle, then atomically publishes an
exclusive JSON/CSV summary and re-derives both files when verifying it.
`scripts/build_v14_artifact_report.py` exposes create/verify commands.
The report identifies the synthetic NumPy track and source commit, shows
seed count and observed range, and limits its zero-failure statement to
cells inside the completed bundle. It does not rank models or change the
fixed v14 payload or historical dashboard (ADR-0137).

P5.2b passes existing core `train_epoch` return values through
`app/continual_shift_benchmark.py` and opts in to collection only in
the v14 runner. `app/wake_diagnostic.py` identifies finite metric,
definition, and timing facts; `app/v14_measured_observations.py`
preflights and serializes the full measured grid and derives an
additive projection. `infra/measured_observation_files.py` binds
that sidecar to a completed P5.1 bundle, then verifies a distinct
ten-file measured projection. The two existing scripts expose the
opt-in commands. The dependency path remains script → app/infra →
core; final-role release still occurs only in the scorer (ADR-0122).

P5.3a adds `infra/atomic_artifact_directory.py` beneath the P5.1
and P5.2 file adapters. It stages already validated byte sets,
records incomplete/failed/canceled publication state outside the
public path, and renames a complete directory into view. No app or
core module depends on this infra helper (ADR-0123).

P5.3b adds a pure `app/v14_trial_checkpoint.py` format-10 prefix and
validation boundary. The `scripts/run_versioned_v14_bundle.py` adapter
orchestrates the fixed trial runner, app preflight, scoring, and infra
stores. `infra/circadian_checkpoint_files.py` writes immutable trusted
checkpoint files; `infra/v14_resume_files.py` atomically replaces a
hidden run cursor under an OS lock. Checkpoints omit deferred final
sources. The app reconstructs them from the fixed seed only after the
whole unscored prefix passes validation, then the existing global
preflight guards final release. Dependency direction remains script →
app/infra → core; the original v14 public artifact schema is unchanged
(ADR-0124).

P5.4 keeps the fixed v14 manifest closed to overrides. For the existing
configurable continual study, `app/continual_experiment_config.py`
validates a small allowlist of typed preset overrides and constructs
the full resolved record. `scripts/run_continual_shift_benchmark.py`
parses repeatable JSON values, applies them after the preset and legacy
flags, and writes an exclusive config artifact after a successful run.
The JSON result embeds the same typed config; no app module imports a
CLI or filesystem adapter (ADR-0125).
`app/v14_experiment_config.py` resolves the sole typed `fixed-v14` preset
before the versioned bundle runs; its `resolved_config` and original
training/outcome bytes remain unchanged. The active CLI audit records the
separate unmatched ResNet reference configuration boundary (ADR-0126).
`app/resnet_experiment_config.py` now owns that route's typed historical
preset and complete per-seed resolved record. Its script adapter retains
legacy flags, validates JSON overrides before any Torch runner call,
and writes the record into the existing descriptive JSON result;
baseline rates and winner selection are unchanged (ADR-0127).
`app/toy_experiment_config.py` owns the root toy CLI's typed historical
preset, strict existing-field overrides, and complete baseline/indepth
execution record. The adapter parses flags and delegates validation before
the runner opens data; `indepth_comparison.py` shares its per-cell config
constructor with the record builder. Infra writes the completed report or
config to a new local JSON path. Neither app module reads scores to choose
settings (ADR-0128).
`app/toy_execution_budget.py` owns typed runtime stops and observed wake
and applied replay-example progress. The NumPy core preflights the exact
selected replay batch lengths before sleep mutation when a limit is passed;
the toy app restores exposure from checked sleep telemetry. This budget is
separate from replay retention's stored-row bound (ADR-0134).
The optional run-level circadian hidden-width cap checks selected splits
before sleep and final proposal mutation. The app restores current and
historical transient peak width from the checked snapshot and sleep history;
the CLI separates observed and durable width (ADR-0135).
The opt-in toy process-RSS cap uses `shared/process_memory.py` from before
dataset/model construction through the scored invocation. It samples at
checked wake/sleep/final boundaries and in a 5 ms background thread. The
absolute process segment is per invocation; the CLI reports it separately
from durable checkpoint work and never attributes it to a model. An observed
over-cap peak stops at the next check or before result publication (ADR-0136).
`adapters/toy_budget_cli.py` ties each budget attempt to the
complete resolved toy config and trusted checkpoint identity.
`infra/toy_run_state_files.py` claims an exclusive state path and atomically
advances it under a local lock. The lifecycle record can be `running`,
`incomplete`, `error`, or `completed`; the old unbudgeted CLI path and fixed
v14 files use none of these new paths (ADR-0132–0133).
The P5.6 report path is script → `infra/v14_artifact_report_files.py` →
`app/v14_artifact_report.py` after the completed P5.1 verifier passes.
The independent presentation path is script →
`infra/v14_dashboard_files.py` → `app/v14_dashboard_projection.py` →
`app/v14_report_plot.py`. Infra verifies the exact report, writes an
exclusive static HTML/PNG directory, and re-derives every byte on verify.
The app owns fixed metric labels and descriptive presentation; it has no
file access or result-selection rule. Pillow is already a project
dependency, so the four plots need no new plotting service or package.
The old `docs/index.html` remains a separate historical snapshot and
now visibly links its provenance warning
(ADRs 0137–0138).
The P6.3 development-only gating pilot follows script → app → existing
arrived-source app and NumPy core. `app/continual_gating_pilot.py` composes
three seeded shallow PC heads on the same arrived development roles,
checks exact ordinary/neutral parity after every update, and reports
outer-selection metrics plus work/capacity facts. It has no filesystem
write or final-role release path. `scripts/run_p63_gating_pilot.py` owns
exclusive request/result/audit files, source and manifest hashes, the
subprocess wall limit, and artifact readback. Reusing the existing arrived
source builder preserves its split/arrival semantics while a distinct
pilot protocol keeps fixed v14 identities and result bytes untouched.
The P6.3 replay factor follows script → `app/continual_replay_factor_pilot.py`
→ the existing arrived-source app and `core/shared_replay_schedule.py` plus
three NumPy cores. The app fixes source/seed/arm/work identities, gives all
on arms the same detached recent-row IDs, and checks neutral PC/circadian
parameter parity at every wake and replay boundary. It scores only arrived
outer-selection roles, with no final-role release or file IO. The pure
`core/continual_metrics.py` validates two-task arithmetic without knowing
the experiment. `scripts/run_p63_replay_factor_pilot.py` owns source-hash
preflight, an exclusive request, a bounded subprocess, result readback,
and audit/failure files. The planned width-12 controls have different
capacity by design; replay on/off pairs have the same width but different
optimizer work. Other sleep mechanisms and guard/rollback are outside this
pilot and stay explicit in later protocols.
`app/continual_sleep_factor_preflight.py` extends the staged matrix at the
app layer without changing pinned cores or older result identities. It
constructs nine no-replay controls on arrived train roles, calls the existing
guarded sleep helper at one A boundary, and returns deterministic work,
role, proposed/applied structure, and capacity facts. It never reads outer
or final values. `scripts/run_p63_sleep_factor_preflight.py` preflights
source/config/output identities and owns the bounded child, observed
whole-process RSS, exclusive request/result/audit or failure files, and
readback. The source config retains v5 replay fields solely to satisfy
arrived-role validation; the actual arm configs disable replay and memory.
`app/continual_sleep_factor_development.py` reuses the frozen preflight's
training/guard/fact helpers and retains after-A model copies. It completes
all 27 train-only cells, compares the entire fact object with the saved c3
reference, then reads outer roles and computes the pure two-task metrics and
three paired contrasts. It has no file or final-role access. The separate
`scripts/run_p63_sleep_factor_development.py` validates the reference
request/result/audit and pinned source bytes, enforces a bounded child,
and owns exclusive scored artifacts and readback. This dependency on c3's
private helpers is deliberate while their bytes and train facts remain
frozen (ADRs 0143–0144).
`app/single_resnet_experiment_config.py` owns the root ResNet CLI's
unchanged 110-field typed preset, strict existing-field override resolver,
and complete descriptive request record. The CLI translates old flags to
that preset, validates before the Torch runner, and asks infra to write
an exclusive config or completed result JSON. The app resolver has no
score or file access (ADR-0129).

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
