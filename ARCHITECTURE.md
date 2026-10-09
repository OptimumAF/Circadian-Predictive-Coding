# Architecture

## Objective

The repository is designed to evolve Circadian Predictive Coding as the main algorithm while preserving reproducible comparisons with:

- traditional backpropagation
- traditional predictive coding

## Current scope and chronology

This guide includes historical milestone observations and current implementation
boundaries. Historical test counts, timings, payload hashes and numerical results
below retain their original scope; they are not fresh validation of this checkout.
The current audit is P9.4f in [the development plan](DEVELOPMENT_PLAN.md), with
evidence and the session handoff in [the log](docs/development-log.md).
The full R0 baseline, G0 and independent source/native/runtime scientific admission
remain open. P6.7d2b2j6b2 correctness is reopened for premature instrumentation
release; its owning-with repair P6.7d2b2j6c is deferred at the user's request.

Current portable fixture boundary (ADR-0196): factor CLI tests bind exact
test-local request/audit/file and internal result pins in the parent and real
worker. Production canonical pins and historical files remain unchanged.
These tests establish same-environment engineering invariance, not historical
reproduction or scientific admission. The module register below preserves its
older request/audit-only fixture description as historical chronology; the public
CLI continues to require its declared canonical files.

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
Arrow means imports / depends on:

adapters -> app -> core
adapters -> config
infra    -> app + core
core + app + infra + adapters -> shared

core must not depend on app/infra/adapters.
```

The diagram describes the intended inward dependency policy. The current source
also contains 44 direct app-to-infra import statements, chiefly dataset and
continual-role types/functions. This is existing boundary debt, recorded as P9.4g;
the static audit finds no direct core import of app/infra/adapters/config. Counts
include guarded and type-only statements. Literal import inspection does not prove
dynamic or native runtime isolation.

Why this: moving declarations to inner ports and injecting outer dataset/role
implementations requires a separate behavior-preserving change. Preserve protocol,
role, checkpoint and result identities while doing that work; do not claim the
existing app imports satisfy the policy in AGENTS.md.

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
2. Descriptive v1-v4 paths in `app.continual_shift_benchmark` train both
   phases, freeze A-only state, and score final tests after each seed finishes
   training. The separate v5 global-seal path trains every declared seed before
   scoring any final test; older paths do not inherit that stronger seal
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
   boundary to predeclare candidates with the same fixed training/role/guard
   settings except method learning rates and the same candidate/seed trial count.
   It scores only arrived outer roles, freezes independent per-method choices,
   and then releases final
   tests. `app.continual_arrived_selection_checkpoint` defines a distinct
   candidate-manifest cursor; `app.continual_arrived_selection_resume`
   embeds v6 active transactions and validates completed trials and their
   independent sleep-history provenance before continuation. The trusted
   file layer persists format 8 separately. The bounded selector permits two to
   four candidates and at most eight candidate/seed trials per method; the first
   predeclared candidate wins an exact outer-score tie. Trial records retain
   training, guard, outer-score and sleep/replay facts; disclose their actual costs
   rather than infer equal work across methods
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
commands. This pure app projection does not depend on infra; the fixed v14
scoring and role release paths are unchanged (ADR-0121). The statement does not
describe the legacy dataset/role imports elsewhere in app.

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
`app/continual_schedule_factor_preflight.py` builds the fixed-width matched
schedule controls and shared all-epoch replay supply. The existing guard
helper owns neutral-model rollback; baseline replay happens only after
commit. Rejected neutral replay executions remain in the cost ledger.
`app/continual_schedule_factor_validation.py` independently rederives
readiness, spacing, guard decisions, shared IDs and method work from the
complete deterministic fact object. Neither module reads outer/final
arrays or writes files. `scripts/run_p63_schedule_factor_preflight.py`
pins the selected sources/manifest, bounds the worker, and owns exclusive
request/result/audit/failure files and finite readback (ADR-0145).
`app/continual_schedule_factor_development.py` reuses c5's training/fact
helpers and c4's score types/arithmetic, holding after-A model copies until
all 33 c5 cells and every checkpoint hash match globally. Only then does it
read outer roles and report all nine within-method policy pairs per seed.
It has no file IO or final-role release. The separate
`scripts/run_p63_schedule_factor_development.py` verifies the canonical c5
bundle and resource facts, pins the selected 23-source map, bounds its child,
and owns exclusive scored files and complete readback (ADR-0146). The frozen
private-helper dependencies are deliberate: equality against the unscored
reference detects any orchestration drift without editing c5 source bytes.
`app/continual_combined_factor_manifest.py` freezes the existing combined
switches, seven removals and matched replay/capacity references before data.
`app/continual_combined_factor_preflight.py` composes arrived training,
the existing guarded sleep helper and shared replay supply; only full commits
feed baseline replay. It returns complete unscored state/lineage/work facts
and charges rejected replay, retaining exact neutral PC parity. Complete
snapshot fingerprints include RNG, memory, chemistry and clocks.
`app/continual_combined_factor_validation.py` independently rederives all
JSON fact/role/clock/capacity/ID and cost invariants without data or file IO.
Dependency direction is script → combined app modules → existing app source/
guard helpers and NumPy cores. `scripts/run_p63_combined_factor_preflight.py`
owns selected-source/manifest pins, an exclusive request, bounded worker,
whole-worker RSS through validation/serialization, and result/audit/failure
files. The app has no outer/final value access or artifact writes. Keeping
the current component APIs intact isolates combined correctness from the
missing scheduled/random parent selector, retained as separate c9 work
(ADR-0147). Changing those helpers requires a new protocol identity.
`app/continual_combined_factor_development.py` reuses the frozen c7 helpers
and c4 score types/arithmetic, retaining after-A models. It completes all
51 train cells and independently validates exact equality with the saved
c7 facts, then globally checks every checkpoint's parameter/width and
complete circadian snapshot hash before any outer value. It checks copies
again after scoring and reports all 22 declared pairs/seed with neutral
PC parity. It has no file IO, final-role release or selection logic.
`scripts/run_p63_combined_factor_development.py` validates the entire c7
request/result/audit and source/work/RSS identities before launch, adds a
selected 29-source map, bounds the child and owns exclusive scored/failure
artifacts. RSS covers training, held copies, scoring, serialization and
output validation; stdout/parent writes are outside the interval. These
deliberate frozen dependencies allow drift detection without editing c7
or core behavior (ADR-0148). Explicit parent-selection controls remain c9.
`core/controlled_parent_selection.py` adds the explicit-proposal parent
ranking extension (ADR-0149). It inherits the original eligibility, budgets,
split/prune, lineage and complete sleep transaction. Usage delegates the
old ranker; cyclic stable-ID and separate seeded PCG64 modes preserve its
chemical-preference tiers. Immutable decision views describe selected IDs
and cursor/RNG fingerprints. Model-owned selector state participates in
snapshot compatibility/rollback, including a new transaction around direct
proposals because selection precedes a transient-width check. Split-capable
sleep requires an explicit policy. This core module imports only core and
standard/NumPy code; it has no scheduling, guarding, datasets, scores or IO.
Experiment orchestration and separately gated scoring belong to app.
`app/continual_parent_factor_manifest.py` fixes c9b's eight cells, arrived roles,
fresh seeds, common explicit counts and planned-width/work limits without
constructing data. `app/continual_parent_factor_preflight.py` composes existing
schedule decisions, inner-guard semantics/telemetry and core snapshot/restore
around explicit policy sleep, capturing proposed selector state before rejection.
It checks complete wake/before/proposed/applied snapshots and all retained row
contents/order. No replay, outer/final scoring or artifact IO occurs in this app.
`app/continual_parent_factor_validation.py` independently rederives parent ranks,
PCG64/cursor state, counts, phase/guard decisions, lineage, clocks and costs.
`scripts/run_p63_parent_factor_preflight.py` binds the selected 31-source/manifest/
adapter identities, bounded child process and exclusive result/audit/failure
artifacts. These modules extend the existing pieces without patching earlier
frozen helpers (ADR-0150).
`app/continual_parent_factor_development.py` composes those pinned helpers
and existing score arithmetic, retaining every after-A copy before B arrival.
It globally compares complete train facts and parameter/width/full circadian
state hashes before outer values, including parent selector and both RNGs,
then verifies copies after scoring (ADR-0151). It owns no IO or final access.
`scripts/run_p63_parent_factor_development.py` binds exact canonical c9b
request/result/audit bytes, the selected 34-source map, all twenty pairs per
seed and bounded child/RSS/exclusive artifact gates. Temporary test bundles
substitute only fixture request/audit pins within isolated tests; the public
CLI requires the declared canonical files. Independent confirmation remains
outside these development modules.
`app/continual_confirmation_manifest.py` binds the six existing resolved
factor configurations, every reserved seed/cell/pair, known source/result
digests and prospective joint budgets without data/model/score/IO access
(ADR-0152). `scripts/inspect_p67_confirmation_scope.py` verifies canonical
and repeated development evidence, inventories actual reserved-seed usage
and publishes an exclusive finite scope record. It creates no confirmation
source or model. `app/continual_confirmation_state.py` binds sealed role
metadata and complete live checkpoint fingerprints, validating baseline
parameter aliases and exact shallow shapes. `continual_confirmation_simple.py`
and `continual_confirmation_periodic.py` compose six existing family phase
helpers and preserve raw unscored costs/decisions, adding full sleep/schedule
rejection witnesses. `continual_confirmation_training.py` validates the frozen
scope/settings, holds independent A copies, completes all A work before any B
source, and rechecks every held checkpoint (ADR-0153). These components own
no artifact IO or scoring. Independent all-seed JSON/resource/artifact gates
and final scoring remain later work; old scientific sources/validators stay
unchanged. Fixtures use existing development sources, never reserved seeds.
`continual_confirmation_json.py` verifies finite/types/canonical schemas,
`continual_confirmation_checkpoints.py` verifies declared model/state/view and
seeded initial fingerprints, and `continual_confirmation_validation.py`
enforces complete reserved envelope/role/witness metadata (ADR-0154). The
checkpoints carry exact hashes for all three original parameter contracts;
`continual_confirmation_parameter_links.py` binds raw endpoints, supplemental
sleep/schedule witnesses and combined/parent wake/guard/epoch chains to held
boundaries. Hashes are captured from live bytes and independently derived for
seeded initial state. Intermediate hashes cannot reconstruct tensor contents.
`continual_confirmation_fact_schema.py` closes raw dataclass/TypedDict keys and
types. `continual_confirmation_simple_work.py` independently derives simple
family rules; `continual_confirmation_work_validation.py` reuses unchanged
periodic seams, binds raw full held state/clock/lineage/selector/supply and
baseline traffic to applied work, and exposes only whole-scope
`verify_confirmation_payload` with derived work records/caps. These pure app
components construct no model/source and own no IO/score. No app imports CLI
validators. `app/continual_confirmation_execution.py` binds strict request
metadata and compares observed optimizer/model-kind counts and RSS to pure
facts. `infra/continual_confirmation_io.py` owns unambiguous finite JSON,
source-file checks, exclusive claims/files and intended byte identities.
`infra/continual_confirmation_runtime.py` observes original optimizer seams
outside rollback snapshots and enforces fixed work/wall/observed RSS caps.
`scripts/run_p67_confirmation_training.py` binds the saved scope and complete
79-file local code closure, validates the child request before data and live
held states after JSON validation/serialization, and publishes/readbacks
complete request/result/audit/failure bundles (ADR-0155). App owns no IO or
method wrapping. Both full 560-cell reserved processes and independent
readbacks now verify exact deterministic equality; source-bound live capture
binds actual values/arrays to fingerprints before separate scoring. P6.7c
still requires the predeclared P6.11 uncertainty/contrast contract and a new
independent final-release boundary.
`core/seed_statistics.py` owns pure ordered observations, explicit nulls,
conditional observed summaries and predeclared interval arithmetic. It
imports no app/infra/adapters or distribution library.
`app/continual_confirmation_analysis_contract.py` binds all original cells,
pairs, seed IDs, final-role counts, metric/sign/missing/constant rules and
fixed interval constants to the frozen manifest/train digest.
`app/continual_confirmation_analysis.py` validates the whole scheduled scope
and shared within-seed final identities before summaries, recomputes metrics
from raw accuracies and retains every arm/contrast/seed/failure. Analysis has
no data/model/IO/source-proof authority; the future scored boundary must
establish actual provenance and join costs by the exact P6.7b reference.
ADR-0156 records P6.11a's tested declaration; independent scoring and actual
seed reports remain separate gates.
`app/continual_confirmation_scoring_manifest.py` binds both complete saved
train bundle identities, the full analysis declaration and every original
endpoint/cell/pair/cap without IO. `continual_confirmation_scoring_state.py`
incrementally encodes the reproduced training dataclasses in the original
artifact byte contract, requires the complete production inventory and
exact held fact attachments, and checks all live models/original sealed
roles. It returns a state proof without final-release authority or source/
file/resource provenance. Reuse this check before and after later evaluation;
separate final views require their own content checks. ADR-0157 records why
the proof avoids a second full decoded result without changing the memory
cap. Final composition is implemented below; the complete scored process/
artifact boundary remains unfinished.
`core/confirmation_final_roles.py` validates released binary float64 arrays,
ASCII IDs, original role-byte hashes and typed exact count/numerical-failure
results. It imports no app/infra/adapters or IO. `app/continual_confirmation_scoring.py`
declares release/evaluation/checkpoint ports, globally checks training before
release, binds every final/shared-source view before any evaluation, dispatches
all three fixed endpoints in manifest order and retains failed/null cells.
It rechecks training, original/released roles and endpoint/cell links before
returning app-local accounting; external provenance/execution authority is
explicitly false. `infra/continual_confirmation_final.py` composes unchanged
final release, validates release-only metadata and uses the original binary
prediction threshold to derive exact correct counts. Only declared numerical
prediction failures become nulls; other errors propagate (ADR-0158). The
available 87-source composition closure is frozen before fabricated fixtures,
with a linked correction record for an initial refactoring error. C1c must
extend this to the full worker/request/resource/artifact closure and independent
actual observations; fixtures do not authorize reserved final execution.
`app/continual_confirmation_scoring_validation.py` independently reads the
whole ordered scored JSON matrix, exact global training proofs and final
role links; derives every cell/failure/total from decoded endpoint counts;
and preserves app-only authority. No source/model/live verification or IO.
`infra/continual_confirmation_training_references.py` streams bound file
identities, composes an injected complete training reader sequentially,
binds decoded canonical bytes and original declarations, and checks markers/
bytes after both reads. Only small reference/cost/historical resource metadata
survives each large graph. The inspection script supplies the unchanged public
training reader and publishes exclusive metadata (ADR-0159). Available source
closure extends to 90 unchanged/pinned files before fabricated scored fixtures;
the full scored request/worker/observation/artifact boundary remains c1c2.
`infra/continual_confirmation_final_runtime.py` keeps independent ordered
source/release/prediction facts outside model snapshots. Scoped original-role
source observers and outer guards preserve sealed metadata; direct prediction
hooks bind the actual model/input and independently derive exact outcomes.
It retains views/source-returned arrays through serialization and restores
owned guards on every exit. Resource policy and full live-state verification
remain external. `app/continual_confirmation_final_observation.py` independently
links the whole fixed scored JSON to every declared observation/model kind,
with no live/IO/resource authority. ADR-0160 records the callback-order repair
and limits of these 92-source component freezes. They are extended by the
97-source full scored composition below; no partial scientific scope or core edit.
`app/continual_confirmation_scoring_execution.py` owns closed full request,
observed worker and saved audit links without IO or live authority.
`infra/continual_confirmation_scoring_bindings.py` checks exact current sources,
scope/reference files, request bytes, command and environment.
`infra/continual_confirmation_scoring_worker.py` composes unchanged training,
original optimizer/resource observation, complete state gates and direct final
observation; retained views survive scientific serialization and later checks.
`infra/continual_confirmation_scoring_artifacts.py` owns exclusive parent
request/result/audit/failure publication and complete independent readback.
The new CLI supplies both unchanged complete training readers sequentially;
the child checks bytes instead of decoding another large graph beside models.
ADR-0161 and the worker report preserve the V1 failure-marker regression and
prospective V2 repair. Development/fabricated correctness closes c1c2/c1c/c1;
both actual full scored processes/resources/repetition/readbacks now pass c2,
with exact scientific result-byte equality. The subsequent P6.11b consumer
below completes exhaustive seed/interval/cost publication and readback.
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

## Confirmation cost consumer

`app/continual_confirmation_report_costs.py` binds the entire original training
result and preserves raw method cost fields, three checkpoint capacities and
shared seed context. Streaming fingerprints surround projection and complete
work validation. `infra/continual_confirmation_report_cost_references.py`
composes the two unchanged complete training readers sequentially, derives
the cost join before each held-checkpoint graph is discarded, and requires
full repeated cost/audit/reference equality. The cost CLI extends the unchanged
97-source closure to 100, publishes exclusive inspection metadata and fully
rederives it on readback. It has no scientific execution or statistical-report
authority. Full guard proof context makes this evidence large; recorded local
validation budgets remain separate from the original scientific caps.
ADR-0162 and the cost-join report document the boundaries and original costs.

## Exhaustive confirmation report consumer

`app/continual_confirmation_report_cost_binding.py` binds the complete compact
original cost identity and exact references to every shared proof context.
`app/continual_confirmation_report.py` verifies both entire scored payloads,
reuses the frozen analysis twice, requires full equality and joins all 560
outcome/cost cells. `app/continual_confirmation_report_rendering.py` renders
every raw endpoint, cost and all 626 seed vectors/116 primary statements with
raw precision, null/eligibility reasons and no selection or ranking. These
three pure modules own no IO, model/data access or experiment execution.

`infra/continual_confirmation_report_bindings.py` owns current 106-source,
environment, request and input identities, complete sequential reader ports,
and original run/resource provenance. `infra/continual_confirmation_report_artifacts.py`
owns exclusive request/result/Markdown/audit/failure/claim publication and
independent full reconstruction. The fixed report CLI supplies the unchanged
complete scored readers; each gets a fresh full cost readback, itself using
both original complete training readers. Dependency direction remains CLI →
infra → app/core. Infrastructure never imports its CLI adapter.

ADR-0163 and the report record the prospective freeze, 101 new/446 related
passing tests, and four bounded actual operations with exact whole JSON and
Markdown repetition. All original scientific pins and caps remain unchanged.
The 180-second consumer budget measures derivative validation only; original
whole-process RSS/wall is retained and grants no per-arm resource claim.
Add future matrix/resource presenters over the verified report boundary with
their own input/output scope and evidence; retain every unmeasured field and
unfinished parent criterion.

## Stored confirmation matrix consumer

`app/continual_confirmation_matrix_inputs.py` bridges the stored report to the
existing public scored/report declaration validators and rederives every
original endpoint/cell/summary/cost link. Its reconstructed expected training
declaration grants no source or execution authority. `app/continual_confirmation_matrix.py`
then preserves every individual 2×2 stage/task matrix, raw endpoint/count/role
record and original metric/failure policy, with B-after-A explicitly unmeasured.
`app/continual_confirmation_matrix_rendering.py` presents every row and exact
original JSON pointer with raw float precision and descriptive transfer signs.

`infra/continual_confirmation_matrix_bindings.py` extends the unchanged
106-source report closure to a prospectively frozen 112. It binds both
original report bundles and all their original inputs, invokes one unchanged
complete official report reader and rechecks current bindings. That reader
itself validates both complete scored bundles and four complete original
training references. `infra/continual_confirmation_matrix_artifacts.py`
owns exclusive request/result/Markdown/audit/failure/claim publication and
independent full reconstruction; the fixed CLI supplies the complete port.
Direction remains CLI → infra → app/core, without an infra-to-adapter import.

ADR-0164 and the matrix report record scope, prospective source/request budget,
110 new/556 related tests and actual complete publication/readback evidence.
No original model/dataset/scoring/metric/interval rule or cap changes. The
original reporting audit independently closes P6.11, and the subsequent
scope audit closes P6.9 under its prospective three-endpoint contract.
P6.9a passes both actual publications/readbacks with exact whole result bytes;
the future-task slot remains unmeasured. Resource or hypothesis
presenters should extend the verified report boundary via separate modules.

## Original resource field and scope inventory

`app/continual_confirmation_resource_contexts.py` independently derives
context work/guard/storage and named capacity points from complete stored
facts. `app/continual_confirmation_resource_fields.py` assigns every arm field
its unit, scope, measurement status and original pointer; no shared wall/RSS
is attributed to an arm. `app/continual_confirmation_resources.py` binds the
entire original cost/report/audit declarations, preserves all cells/contexts,
reconciles work/capacity/storage and exposes explicit gaps.
`app/continual_confirmation_resource_rendering.py` renders every field/status/
scope without outcome ranking. Large original proof objects stay referenced.

`infra/continual_confirmation_resource_bindings.py` preserves all original
106 report/scientific pins and adds the existing pure report-input validator
plus seven new inventory files. All 114 sources are pinned, covering all 90
runtime imports plus retained evidence producers. It checks current original
input/report/scope/recorded-reader identities before/after two complete saved
cost/report/audit reads and independent inventory reconstruction.
`infra/continual_confirmation_resource_artifacts.py` owns exclusive publication,
failure/audit ownership, late byte checks and full reconstruction. The fixed
CLI supplies paths; dependency direction remains CLI → infra → app/core.

ADR-0165 distinguishes current saved-evidence inventory authority from another
complete scientific reader run. P6.10b preserves its separate official-reader
requirement. The consumer's 120-second derivative budget does not change
scientific caps. The resource document/log record prospective source/input
binding, 99 new/576 related tests, actual bounded acceptance and unknowns.
No model/source construction, profiling, training or new final view. Add
outcome-versus-cost or hypothesis presentation through separate modules over
verified inputs; preserve measurement scopes and all unfinished criteria.

## Complete original checkpoint retention ledger

`app/continual_confirmation_retention_checkpoints.py` extracts every original
nullable view, replay owner field, ordered array fingerprint and exact
canonical mapping pointer, with shared FIFO facts kept separate.
`app/continual_confirmation_retention_costs.py` requires the whole original
training and inventory identities, validates every checkpoint/state/role/work
and reconciles every stage/group/arm against original inventory facts.
`app/continual_confirmation_retention_rendering.py` deterministically presents
every arm/checkpoint and shared context; companion JSON keeps all proof fields.

`infra/continual_confirmation_retention_references.py` composes two unchanged
complete original training reader ports, verifies entire decoded-part bytes
through the existing reference boundary and compares full projections. Each
large training graph is discarded before the next reader.
`infra/continual_confirmation_retention_bindings.py` freezes all 121 sources
(111 runtime imports plus preserved original evidence) and binds complete
current input/request/environment/recorded inventory authority.
`infra/continual_confirmation_retention_artifacts.py` owns exclusive parts,
failure/audit ownership, late marker/byte/budget checks and full independent
reconstruction. Only `run_p610_retention_costs` composes the actual original
training adapter; infra never imports it. Direction remains CLI → infra → app/core.

ADR-0166 requires verified closed-state evidence before deriving zero owned
bytes from a null view. Configured empty and disabled storage remain distinct.
Shared FIFO, array geometry, checkpoint copies and process RSS have separate
scopes; no samples are reopened or content IDs rehashed. The prospective
180-second derivative budget preserves all original scientific caps. All
83 new/506 related tests and four actual full-reader operations pass, with
complete byte repetition and eight original reader calls. P6.10b1 is complete;
add outcome-versus-cost presentation over the complete official report and
current bound ledgers via separate modules, retaining all unfinished criteria.

## Complete original outcome versus cost presentation

Three new pure app modules join the accepted whole report/resource/retention
bodies. `continual_confirmation_outcome_cost_inputs` validates complete scope,
original report declarations, counter/capacity/state/parameter/owned/shared
links and historical process scopes. `continual_confirmation_outcome_costs`
requires every original canonical body identity and retains full metrics,
endpoints, costs, proofs, histories and original interval/replication records.
`continual_confirmation_outcome_cost_rendering` presents all 560 cells against
compute and memory, with every shared context and four separate process segments.
No IO, model/data/training/scoring/profiling or resource allocation belongs here.

The public interface is `build_outcome_cost_presentation(report, inventory,
retention) -> body`, then `render_outcome_cost_presentation(body) -> Markdown`.
The private development seam grants no original artifact/source/reader authority.
Exact original `_work_fields`/`_array_proofs` validators are reused under frozen
sources rather than copying their work/array semantics. Complete JSON retains
all 25,400 history points and 626 original metric vectors; tables do not rank
or select models. No new dependency or environment variable.

P6.10b2's prospective 124-source map preserves all earlier 121 sources and
covers 85 pure imports. All 45 new/417 related tests/static gates and both
complete pure saved-input derivations pass, exact whole repetition and zero
scientific calls. Recorded prior full-reader authority is preserved; no fresh
official reader is executed by the pure API itself.

`infra/continual_confirmation_outcome_cost_bindings` binds all six complete
original/companion bundles, both pure outputs and 127 current sources, then
invokes the unchanged whole official report reader and public pure builder.
The just-validated current request passes only within the operation: complete
saved request bytes are checked before the reader, every current binding is
reconstructed after the whole derivation and at final publication/readback.
There is no cross-operation cache. This avoids immediate duplicate traversals
identified after a preserved first 240-second derivative timeout. No full check,
reader scope, scientific setting or cap is dropped.

`infra/continual_confirmation_outcome_cost_artifacts` owns four exclusive
parts, failure/claim markers and full independent reconstruction. Distinct
claim ownership preserves replaced foreign claims and other artifacts; failure
cleanup revokes only provably owned parts. Only `run_p610_outcome_costs` composes
the actual whole report adapter. Infra depends inward on unchanged app/core.
All 112 new/359 related tests and static gates pass after a repaired prospective
freeze; all 120 local runtime imports are pinned. Four actual unoccupied v2
operations pass under the unchanged 240-second derivative envelope (215.65–
216.26 seconds), with the original inner reader 180-second/scientific caps.
Four whole report/sixteen training/eight scored readers and zero scientific
guards per operation, exact whole JSON/Markdown repetition and complete current
bindings pass. The original criterion audit closes b3/b/P6.10 at the declared
sampled whole-process RSS, named capacity history and recorded/derived work/
owned-shared storage scopes. Historical false-parent/gap flags remain immutable;
the first timeout and occupied partial claim survive. P6.12 and broader work
remain. See ADR-0167/presentation guide/original acceptance audit.

## Complete primary confirmation findings

`app/continual_confirmation_findings` requires the whole original P6.11 report,
reuses its complete declaration validator and preserves every raw field and
all 116 primary statements. Each statement keeps its exact interval/seed/
direction/eligibility and three secondary endpoint summaries. Fixed hypothesis
context maps distinguish combined-system comparisons from isolated factors;
no hypothesis vote, favorable subset or composite score is introduced.

`app/continual_confirmation_findings_rendering` rebuilds and compares the whole
body before deterministic exhaustive Markdown. Exact round-trippable numbers,
all statement rows and the entire original report appendix remain. Public API:
`build_confirmation_findings(whole_report) -> body`, then
`render_confirmation_findings(body) -> Markdown`. The private fixture seam
grants no artifact/source/current IO authority. Both modules are pure app
logic with no infra/adapter import, dependency or environment setting.

ADR-0168 freezes 129 current sources covering all 75 pure runtime imports and
44 test dependencies before new fixtures. All 36 new/331 related tests/static
gates and two actual saved-input derivations pass, with exact whole output
repetition and zero scientific/fresh reader calls. The original a acceptance
audit passes. P6.12b must separately bind all development/tuning, activity,
cost and operational failure evidence and prove complete current publication/
readbacks; P6.12 remains unfinished. Extend through separate modules, preserving
these frozen sources, every original statistical rule and all unfinished work.

## Complete fixed development evidence for findings

The pure `continual_findings_development` app module retains every original
input and projects 168 development cells/174 prospective within-seed pairs.
`continual_findings_development_bindings` owns whole filesystem/source checks;
its development/preflight validators are explicit ports. The outer
`inspect_p612_development_findings` CLI composes the unchanged original
validators. A fixed JSON input catalog lives in `src/config`. App has no
infra/adapter import; no dependency, environment setting or scientific
override was introduced.

ADR-0169 explains this separately tested input boundary. All original outer
roles and metric names remain, including the gating pilot's five fields.
Twelve gating/replay pairs are marked as projections; 162 stored contrasts
remain unchanged. Twenty complete bundles and both original scope records
are retained. Tests and two bounded actual derivations establish b1, without
fresh confirmation-reader authority. Add b2 synthesis and b3 complete IO
through separate modules; keep all frozen sources and original criteria.

## Complete original confirmation activity

`app/continual_confirmation_activity` accepts two whole canonical original
bodies and preserves the complete transaction/replay-offer/cell records. Its
pure helpers check original guard direction, rollback commits, skipped work,
attempt counters and the neutral controller's three-applier scope. No
filesystem, infra or adapter imports occur in this module.

ADR-0170 explains why original raw decisions must precede interpretation;
final counters cannot supply missing individual commit records. The b2a
source/input freeze, meaningful tests and two complete actual derivations
pass. Extend through separate b2 synthesis and b3 IO modules; this projection
does not grant fresh current confirmation-reader authority.

## Complete fixed findings synthesis

Four pure app modules separate whole identity/declaration checks, operational
failure interpretation, exhaustive synthesis and rendering. The fixed
`p612_findings_inputs.json` catalog binds all original bodies/repeats and 93
complete raw records. Existing complete primary/development/activity/matrix
declaration validators remain unchanged. Historical matrix reader provenance
is independently verified against its original complete source/request
template and grants no fresh execution authority.

The public API is `build_complete_findings(inputs)` followed by
`render_complete_findings(body)`. Every original raw input is retained in the
companion JSON; exhaustive Markdown uses exact numbers and complete indexes.
H1–H4 remain unresolved at the fixed measured scopes. The timeout/claim and
failed synthesis attempt remain; unobserved counters are not replaced by zero.
ADR-0171 records the rationale. All tests/statics and two bounded actual pure
derivations pass. b3 owns current filesystem/source checks and reader ports,
exclusive publication and independent readbacks; no app-to-infra import or
new dependency/environment/scientific override is introduced.

## Complete current findings IO

`app/continual_findings_readers.py` defines three complete reader ports.
`infra/continual_findings_current_bindings.py` binds the full current physical
scope; `current_inputs` composes fresh ports and unchanged public synthesis;
`artifacts` owns exclusive claims, whole outputs and independent readback.
The fixed `run_p612_complete_findings` CLI wires unchanged outcome/cost,
matrix and development adapters. Each report reader dispatches separately.
There is no app-to-infra import, cross-operation cache or scientific override.
ADR-0172 and [the IO guide](docs/p612-current-findings-publication.md) describe
budgets, corruption/ownership gates and complete actual acceptance evidence.
Two publications and two fresh independent readbacks retain every pure result
byte; the original b3/b/P6.12 audit passes. Failed observer parts remain separate
and preserved; historical pure scope/pending flags are not rewritten.

## Complete retrospective pilot variability

`core/pilot_precision.py` reuses ordered seed summaries and projects conditional
SE without IO, critical-value selection or a sufficient-count decision.
`app/continual_pilot_variability.py` first validates the entire original
development input ledger, retains all116 primary paired vectors and binds
whole input/ledger/manifest/analysis identities. It uses unchanged df-nine
critical values only as conditional half-width scaling at the original count.
Missing/constant forecast reasons and weak three-seed/role-dispersion limits
remain explicit. No source/model/train/score/final/IO layer is added.

The dependency direction is app → original app validators and core summaries;
core → core seed statistics. A private fabricated seam grants no original
or current reader authority. Actual evidence dispatches the unchanged complete
development reader once, with every original port and120-second gate. ADR-0173
and the pilot guide preserve the missing original prospective sample-size
criterion; a future planning/feasibility module must supply a separate
prospective count/untouched-role contract before new confirmation.

## Complete prospective precision feasibility

`core/seed_precision_budget.py` owns validated numeric sensitivity and
bounded-mean sufficient-count arithmetic using complete ordered pilot rows.
`app/continual_precision_contract.py` fixes the target/ranges/assumptions;
`app/continual_precision_feasibility.py` first validates that contract and
the entire original development/pilot boundary, then retains all116 vectors,
original provenance and exact additive six-family work. Output is detached
from its input. Core depends only on core summaries; app depends inward on
core and existing complete app validators. No new IO/source/scoring layer.

Why this: a separate bounded check covers constant/discrete pilots without
pretending a three-seed SD certifies final-role precision. No sample-count
selection, new interval or confirmation launch belongs to these modules.
ADR-0174 and [the feasibility guide](docs/p67-precision-feasibility.md) preserve
the negative planning result and the original unchecked sample-size criterion.
Extend next through the separate untouched-role/execution contract and fixtures,
preserving all informative factors/matched controls and fixed caps.

## Complete retained seed declaration inventory

`core/seed_usage.py` traverses decoded regular/canonical metadata, embedded
configurations and CLI arguments. Typed declarations retain locations and
representations; unresolved values remain visible. It owns no IO or RNG.
`infra/seed_usage_inventory.py` discovers and freezes all retained JSON files,
checks complete membership/whole bytes before and after, and retains parse
failures. Runtime/cache exclusions and the newly owned output are explicit;
external/symbolic or unowned output paths fail before exclusion.
Dependency direction is infra → core. No app/scientific import is introduced.

The whole 593-file actual inventory/repeat and 30 new/100 related fixtures pass.
Declarations alone cannot prove prior execution or independence. D2b2 owns
schema/text/source/default/derived-stream/release classification; d2b3 owns
full execution fixtures. ADR-0175 and [the guide](docs/p67-seed-usage-inventory.md)
explain the split. Original P6.7 acceptance and scientific caps remain open.

## Complete retained prior seed context and history

`core/seed_declaration_context.py` gives every original declaration/issue a
typed context and preserves structured seed-map base/stream evidence.
`core/seed_source_evidence.py` retains text/CSV witnesses and all Python AST
calls, assignments/defaults and seed mapping values without executing source.
`infra/prior_seed_evidence.py` freezes every retained physical file and all
local Git objects/commit-path aliases, verifies full membership/whole bytes
before and after, and parses identical bodies once with every alias retained.
Opaque/unparsed/symbolic/history uncertainty remains explicit. Infra depends
on core; these modules own no scientific dispatch or fresh-role admission.

Why this: seed-named JSON fields alone miss structured maps and historical
source/default/stream evidence. Context prevents treating quantity fields or
valid nonrandom policy nulls as executed seed identities while preserving the
original raw records. ADR-0176 and [the guide](docs/p67-prior-seed-evidence.md)
record the boundary and extension path. 163 selected tests and static checks
pass. The first whole audit passes, but its independent repeat exceeds 180s;
the resource/reproducibility acceptance remains unfinished. Extend through
bounded complete saved-input timing diagnosis, then original b2 chronology/
ambiguity/role contract and b3 fixtures with the original scientific caps.

## Current checkout bindings and historical seed-evidence timing

The local saved-evidence diagnosis owns durable phase events and current whole
physical/Git/source/proof pins. It uses the existing process RSS sampler and
standard-library JSON/profiler; it adds no public module or dependency. It
re-encodes the entire saved report with exact canonical bytes while retaining
all historical content/aliases and declared uncertainty.

Why this: a later user commit changes live HEAD without changing the preserved
historical source bytes. The new current binding is the complete old freeze
cloned with only its head parameter changed to prospectively frozen28e71ee;
the identical full validator and every original source/test/input/proof/guard
check run unchanged. Current complete Git and historical objects/aliases are
both bound. Historical freezes/functions/failed receipts remain immutable.
The diagnostic supplies measurements; original repeat/resource/fresh-role
acceptance remains unfinished. Extend via measured, parity-checked lexical
work before another explicitly declared protocol choice or scientific source.

## Extension Rules

- New adaptation strategies should be added via policy/config extension points, not by hardcoding branches across modules.
- New datasets must be added in `infra` and wired via `app`, never directly from `core`.
- Major algorithmic changes require an ADR in `docs/adr/`.


### P6.7d2b2a2: preserve the measured lexical boundary

The pure seed_source_evidence text/AST/CSV contract is unchanged. A complete
fixed benchmark rejects whole-word scan plus substring search: exact token/
column/line-hash parity, about5.69% slower across both complete-input passes.
Why this: optimize only against a declared complete workload and behavior
regressions; a valid negative result retains the existing simple implementation.
Eleven additional pure fixtures cover Unicode case/words/combining marks,
whole-word boundaries, line separators, normalized UTF8 hashes and character
columns. Local metadata producers and snapshots stay under the owned evidence
transaction. Source release/admission remains a separate unfinished contract.


### Complete saved chronology boundary (P6.7d2b2b1)

Add core/seed_stream_screening.py for the eight fixed source, role split, exposure, initialization, parent selection and circadian local-noise streams; app/prior_seed_corpus.py for complete membership; app/prior_seed_chronology.py for every saved witness and its unknown chronology; and infra/saved_prior_seed_evidence.py for whole-byte IO and before/after membership. Dependency direction is infra -> app -> core. No expression execution, unpickling, RNG sampling, real candidate selection or positive admission path. No new dependency or environment variable.

Why this: exact input pointers and canonical full-record digests retain millions of original witnesses without copying them into a second report. Numeric declarations, inventory lineage, static literals and Git copies do not prove execution, independent replications or actual final role release. The screen always denies admission; verified chronology and the complete prospective protocol are separate unfinished contracts. ADR-0177 records alternatives and consequences.


### Fixed original release readback boundary (P6.7d2b2b2)

Add core/seed_release_chronology.py for supported observer order, app/scored_release_witnesses.py for complete decoded audit/result witnesses, and infra/original_release_witnesses.py for the fixed actual original reader boundary. Dependency direction is infra -> app -> core. No dependency or environment variable is added.

Why this: retain whole source-bound original evidence before deriving partial order. Pure dictionaries and private fixture spies leave reader verification false; only the fixed public adapter calls the original scoring reader and both original full training readers. Each node binds an exact original pointer and full record digest. Full current/historical preservation is a separate closing gate. Documentary clocks, copies and shared numeric seeds establish no stronger chronology or independence. ADR-0178 records context, alternatives and consequences. Future work must complete the independent prospective protocol and original resource acceptance before fresh authority.


### Complete pending confirmation contract

Add core/prospective_replications.py, app/prospective_confirmation_design.py, app/prospective_confirmation_evidence.py and infra/prospective_confirmation_inputs.py plus four test files. Dependencies point infra -> app -> core; no new dependency, environment variable or execution adapter. See docs/p67-prospective-confirmation-contract.md and ADR-0179 for inputs/outputs/non-responsibilities, commands, rationale and extension.

Why this: validate every future setting and role requirement before binding any actual source. A separate complete saved-file consumer retains unresolved prior effects and negative precision without turning copies or declarations into fresh authority. Current source/history/proof preservation remains a supervised boundary concern; actual original readers were executed in the preceding component.


## Prospective external stream boundary

`src/app/prospective_stream_declarations.py` consumes the existing app design and core declaration types; no infra/adapters imports. One pure public function validates whole canonical design identity and all external streams.

Public `validate_prospective_stream_declarations` verifies the expected whole
design identity, all60 ordered bindings/50 planned groups and every400 external
typed stream claim through unchanged public full-design/replica validators.
Missing/extra/reordered/duplicate/mutable/detached/symbolic/unknown-record and
late type/value drift are rejected; exact types defeat value-equal booleans,
floats and string subclasses. All560 cells/480 roles/116 contrasts, original
matched settings/eight offsets/analysis/count/caps/stopping remain. Return
independent replications unknown/fresh authority false; no source/RNG/model/IO/
scoring/execution or historical-domain proof. Original25 helper cases are unbound.

Current full metadata suite120 passes, including42 new and78 existing cases,
0errors/0failures/0skips in5.4621946s. Ruff/two-file format/full no-incremental
mypy560/diff pass. Preserve initial missing-module red1error and full mypy1error,
exact old fixture versions and the local-only annotation repair; no exclusions,
ignores, mock adapter or cap reset. Complete preservation remains pending.

Dependency flow: caller -> app stream validator -> app fixed design/public replica checks -> core replica/stream metadata. The future outer request adapter must prove full provenance and compose this boundary without granting scientific authority from numeric claims. See ADR-0180.


## Complete prospective role/source/request boundary

Core holds frozen declarations and inspection result; app consumes fixed design and public stream checks, never infra/adapters.

`inspect_prospective_role_request` composes the existing full-design and400-stream
checks with all100 ordered phase-source descriptors and480 ordered role records.
Bind complete design/source-map/request metadata identities, expected code
declaration and every original data argument/geometry/derived data seed. Exact
types/order/counts/IDs/availability/use policies and planned shared views are
checked; resealing cannot repair foreign, partial, mutable, overlapping, reordered,
early-release, unknown-schema or array/execution claims. Preserve A120 development
positions and B60 retained original positions in0..119, final IDs0..39 and all560
cells/116 contrasts/original settings/analysis/count/caps/stopping. No existing
source/test changed. Return all actual provenance/chronology/authority flags false,
independent replications unknown and all12 actual-proof obligations still required.

Full fixed metadata suite174 passes (54 new/120 existing),0errors/0failures/0skips
in14.7607819s. Ruff/all3-file format/full
no-incremental mypy563/diff pass. Retain initial missing API, reserved pytest fixture
name and mypy local-variable failures; fix only fixture/local names and formatting,
with exact failed versions preserved. No exclusions, weakened tests or cap reset.
Full preservation is pending; original25 helper cases remain unbound. Metadata
hashes and expected code declarations do not verify physical source or release.

Dependencies: caller -> app role inspector -> app full design/stream checks -> core role/replica/stream records. Actual whole-file/code/UTC/owner/prior/resource proofs belong to outer adapters through inner proof ports; no physical proof is inferred here. Why this: keep metadata inspection safe before any source while preserving original B row IDs. See ADR-0181.


## Prospective bundle proof port

Core contains immutable spec/snapshot/result and read/recheck protocol. App validates fixed design/full typed request and declarations; infra implements strict whole-file proof. Dependency flow: caller -> app preflight -> core protocol/types and app e gate; infra -> app decoder/core protocol. No app/core import infra/adapters.

`preflight_prospective_request_bundle` composes the full e inspector through an
inner read/recheck port. The real outer reader binds canonical whole request,
source metadata map, closed expected code manifest and every declared code file
before and after app inspection. Full100 sources/480 roles/400 streams/60 views
are preserved. Exact schemas/types/membership/identities and distinct regular
paths, failed/pending markers, canonical UTC/owner and original full resource/
same-request repeat declarations are checked. Retain all560 cells/116 contrasts/
settings/analysis/count/caps/stopping, all12 actual-proof obligations and every
prior unknown. Physically matching declared files do not prove runtime closure,
exclusive ownership, before-source chronology, untouched sources, independence,
actual resource fit/repetition or b3. All actual/fresh/execution/precision flags
remain false; independent replications unknown;25 original helper cases unbound.

Full304 behavioral tests pass (55 new/249 related),0errors/0failures/0skips in
56.0459501s. Ruff/all5-file format/full
no-incremental mypy568/diff pass. Preserve missing API, two tuple-pop fixture
failures, duplicate module-name failure and two deliberately invalid payload
typing failures. Correct fixtures/import and validate foreign rows before
attributes; add six regressions. Final local Any annotation only has identical
runtime AST after erasure; no optional56s pytest rerun, scope exclusion/ignore,
test removal or cap reset. Whole preservation is pending. No scientific work.

Why this: existing saved-evidence reader checks historical inputs, while current code/request/map files require separate complete physical checks. Future lease/runtime/provenance admission must compose the port without treating physical metadata as fresh source or execution proof. See ADR-0182.


## Prospective generation recipes

Core has immutable recipe/request/inspection records; app composes existing complete fixed-design/stream/source/concrete-role gates. Dependency flow: caller -> app generation -> app gates/core records. App/core import no infra/adapters. A future V2 physical reader and live sequential arrival observer must use inner ports.

The V2 generation request freezes480 ordered role recipes/100 source declarations/
400 streams/60 bindings against the unchanged complete fixed design (560 cells/
116 contrasts). Assignment/exposure rules and seeds, original/retained geometry,
counts, allowed uses, availability, all120 canonical final-ID tuples and all12
actual-proof obligations are frozen; all360 development role ID realizations are
unset. The separate full concrete-role declaration bridge composes e's partition/
shared-view checks after strict type rejection and binds its result to the V2
request. It cannot prove actual arrival, seeded assignment, runtime code, owner,
chronology, source independence/freshness or execution. All actual/fresh/execution/
precision flags remain false; independence unknown. Existing e/f remain intact.

Initial full382 cases pass (78 new/304 related) in
99.9559949s. Six added sentinel cases then
expose deepcopy before rejection; preserve every failed version/receipt and reject
full concrete schema before encoding. All84 current new cases pass in
35.8429476s. Current unique coverage388:
all304 related cases match earlier complete receipts; all776 older physical Python
files unchanged, no existing source/test/script references new modules or symbols,
and all78 earlier generation fixtures have identical runtime AST and rerun. No
optional full repeat: related-suite-reuse.json binds full reuse evidence without
scope/test removal. Ruff/all4-file format/full no-incremental mypy572/diff pass in
63.6039481s. Whole preservation pending.

Why this: original label-dependent split/exposure rules cannot be realized before source arrival; freeze rules then bind realizations without early scientific work. See ADR-0183.


## V2 generation files and shared physical proofs

Core defines immutable V2 snapshot/result and reader protocol. App owns full exact decoder/g recipe/envelope preflight through that port. Outer V1/V2 typed adapters delegate identical whole-file IO to one infra module. Dependency flow: caller -> app/core port -> injected infra adapter -> shared infra IO; app/core import no IO.

The V2 physical boundary binds the full request/source map/code manifest and every
closed expected code file to all480 recipes/100 source declarations/400 streams/
60 bindings, unchanged560 cells/116 contrasts/settings/analysis/stopping/count/caps
and all12 actual-proof obligations. Immutable snapshots and inner read/recheck port
keep app/core free of IO. Both versions share whole bytes, canonical strict JSON,
root-contained regular paths, physical alias/publication-marker checks and late
rechecks. V1 public constructor/read/recheck behavior is preserved. All360
development realizations stay unset;120 final-ID tuples declare geometry only.
UTC/owner/resource/same-generation-request repeat are checked as declarations.
Matching physical metadata proves neither runtime code closure, exclusive owner,
actual sources/roles/arrival/assignment/chronology, unrecycled independence, prior
effect resolution, resource fit, independent repeat nor b3 admission. These stay
required; independence unknown and fresh/execution/precision flags false.

Initial full454 cases pass (66 new/388 related),0skips in
138.3763548s. Two added V1/V2 direct source
numeric-alias recheck cases expose comparing the rebuilt snapshot with the source
map instead of the supplied snapshot. Restore the original supplied-snapshot
comparison; all68 current new cases pass in
30.2081090s,0skips. Current unique
coverage456 is bound by both full receipts and exact V1 specialization: all six
original private IO/constructor bodies, whole read/recheck bodies after controlled
callback substitution and three public signatures match the saved V1 AST; all42
other lead files unchanged, all66 prior fixture ASTs unchanged and rerun. This is
coverage across receipts, not a single456-case pytest invocation. Ruff/all7-file
format/full no-incremental mypy578/diff pass in
45.6565722s. Whole preservation pending.

Why this: share actual repeated physical policy while keeping typed schema/scientific rule validation separate; one permitted older V1 implementation edit preserves exact prior semantics and original150 closure. See ADR-0184.


## Live V2 ownership proof port

Core contains immutable whole scope/observations and live claim/observe protocols. App composes unchanged full H/g file preflight through injected reader/owner ports. Infra owns native locking/path/handle/whole-file IO, standard platform modules and non-scientific nonce/time. Dependency: caller -> app -> core ports -> injected outer owner/reader. No app/core imports infra/adapters.

Complete V2 file preflight now composes a live owner port with a real native local
lease. Contention keys use the entire inner generation request, so different
outer owner declarations/file copies of that request contend in one configured
canonical registry. All480 recipes/100 sources/400 streams/60 bindings, unchanged
560 cells/116 contrasts/settings/analysis/stopping/count/caps and all12 admission
obligations remain. Reread after acquiring and recheck whole physical files/owner
before yield and on success/failure exit. Immutable observations bind scope,
native single-link file identity, nonce and monotone sequence/UTC at the recorded
time held. Native handles release on exception and process death; permanent lock
paths stay in place and are never reclaimed/unlinked by the adapter. Observations
remain historical after exit and cannot authorize execution. Runtime code closure,
actual arrival/assignment/chronology, cross-host ownership/source independence,
prior effects/resource/repeat/b3 remain unverified; all fresh/execution/precision
flags false and independence unknown.360 development IDs remain unset and120
final-ID tuples declare geometry only. Existing49 lead files/APIs unchanged.

Full502 cases pass (46 owner/456 related),0errors/failures/skips in
131.0861267s. Real native Windows controls
cover same-process and subprocess contention, different outer owners for the same
whole inner request, normal release and os._exit73 process-death release, failures,
clock rollback/type rejection, physical/registry drift and early/late invalid
observations. All new cases guard scientific source/final/model/scoring/RNG/array
work; IO/native non-scientific UUID/time are intentional. Ruff/all4-file format/
full no-incremental mypy582/diff pass in
25.8889968s. Preserve initial missing
API and static fixture mapping-type failure, all exact versions/receipts. No
optional repeated tests; shared tests132.2325310/180spent, static50.7586573/180spent.
Whole preservation pending.

Why this: ownership lifetime requires a live native holder, while actual runtime code/import closure remains a separate next gate. Permanent request keys prevent owner/file-copy bypass; observations are historical after exit. See ADR-0185.


### Prospective runtime process proof port (ADR-0186; acceptance open)

Core immutable RuntimeCodeObservation/LiveRuntimeCodeLease/GenerationRuntimeObserver
-> app full V2/native owner/runtime composition <- injected outer process observer.
Focused Python/native modules inspect actual namespaces/code/files/executable memory.
App/core perform no process IO or source/model work. Actual code observations and
source-version/full prior/arrival/resource/repeat/b3 admission remain distinct.
All53 prior lead files/original150 closure unchanged; no new dependencies/config.


### Runtime capture cost diagnosis (ADR-0187)

Owned evidence tools profile the unchanged real V2/native/process boundaries and
transform a complete saved JSON body through a pure experimental codec. No source
layer/dependency/config change or actual source/model work. Raw actual membership
remains complete; metadata compression cannot attest source/runtime/provenance.
Measure and prove full parity before any exact-type namespace optimization.


### Exact-type native namespace optimization (ADR-0188)

Only outer runtime_python_objects._namespace changes: exact native container type
identity returns no instance dictionary; all other native descriptor branches and
full GC/function/file/native/process boundaries retain behavior. No persistent
cache/core/app API/config/dependency change. The complete frozen GC adapter is
owned parity evidence only, preserving unfiltered actual inputs for both complete
implementations. Every current544 regression case/current590 static file passes;
full source-version/transient/nested-schema/continuity admission remains open.


### Passive complete runtime structure and static V2 callback (ADRs0189/0190)

App -> pure core record/value/Python/native/payload validation; infra retains whole
files/native process/owners/audit through existing ports. Per-body identity registry
and passive repr codec/fixed module-load patterns add no persistent observation
cache or source authority. V2 uses one typed metadata projection instead of two
transient lambdas, preserving identical sources/public ports/file checks. Tests
report complete temp paths; supervisor copies whole runtime groups after finish.
No dependency/config/environment change. Broader source-version/reference-count/
transient/arrival/prior/science admission remains required and unproved.


### Marshal diagnostics before observer retention

The new tests compare complete native public code/constant contents and raw marshal
bytes without editing the infra observer, pure schema or inward dependency direction.
Native metadata sharing requires its own retention control; private interpreter and
source/version correspondence remains unproved (ADR-0191).


### Native code-value lifetime before runtime baseline

The infra observer composes a focused retention collector with its existing native
namespace reader. The strong-reference tuple lasts for the actual lease; core/app
validation and raw hashes/audit remain unchanged. Pure pytest-free fixture helpers
serve unit and real-process controls. Private build/source/transient admission remains
open; no inward dependency bypass (ADR-0192).


### Admission counterexamples preserve proof boundaries

Real V2/native owner/observer fixtures save actual records and separately labeled
edited structural controls. No production or core/app dependency changes. Continuous
code/binding enforcement and complete source/native/build attestation require separate
trusted composition; green boundary/schema checks cannot supply it (ADR-0193).


### Installed monitoring measurements inform enforcement

The pytest-free capability fixture composes real V2/native ownership and observers,
with primitive monitoring event identities and full unfiltered runtime artifacts.
It changes no core/app/infra production dependency. Separate continuous enforcement
must reject unsupported callback/native/tracing paths before admission and remain
effective after a caught denial. VM events and boundary equality grant no native/
source authority (ADR-0194).


### Global mutation mechanism before complete admission

Optional infra enforcement captures the prepared GC-visible function/recursive-code
catalog and monitors delivered global before-call/instruction events over explicit
lifetimes. Its bounded controls reject unsupported tools/tracing/native calls and
retain poison after caught denials (ADR-0195). Their measured scope remains.
The confirmed foreign nested owning-with exit can release instrumentation early,
allow parameter mutation and body execution, then fail cleanup. General j6b2
correctness is reopened; j6c repair is user-deferred. Exact owning-boundary release
is therefore an unfinished requirement. Existing observer behavior is unchanged.
Trusted support-operation/source/native/private/full-lifetime integration remains
required; these controls grant no fresh scientific or runtime admission.

### First local pilot resource planning

The new standard-library-only `core/local_pilot_budget.py` owns typed resource
values and fixed local/simulation/CPU ceilings. Its CLI adapter maps flags to
requests and emits JSON/status; the public script delegates to that adapter.
Dependency direction is script → adapter → core, with no execution app/infra
path yet. Why this: resource policy can be tested independently of native model
objectives. Measurement, enforcement and promotion remain separate R3 work.
See `docs/local-pilot-budget.md` for limits, commands and extension boundaries.


### Native learner ports (R3.1)

`core/learner_ports.py` defines generic input/target/prediction/state types and
native diagnostic IDs. `app/learner_step.py` composes one complete wake call with
unchanged ToyBudgetSession. `adapters/numpy_learners.py` owns detached existing
CPC/backprop models; it reuses CPC snapshots and validates full ordinary state
before restore. Dependencies are adapter -> core and app -> core/app budget. No
core/app import bypasses that boundary. Existing44 app-to-infra statements remain
P9.4g debt; this addition does not claim to migrate them. Losses, tensor layouts,
old models/checkpoints/protocols and scientific admission remain unchanged.
Actor/shadow state, experience permissions and serving resource contention need
separate contracts; see docs/learner-ports.md and ADR-0197.

Current dependency status: R3.1 remains unchecked pending the reopened R0.3/G0
clean-checkout/native CI diagnosis. Its local port implementation and85-case
validation are preserved; no further dependent implementation or experiment is
launched from this work. See the active RESEARCH_ROADMAP.md.


## Historical presentation boundary (P9.5a)

`src/app/historical_outcome_view.py` is a pure ordered presentation projection
with original uncertainty/cost preservation and padded, unclipped plot limits.
`scripts/export_historical_outcome_figures.py` binds the complete accepted local
artifact before optional ReportLab rendering and write-once filesystem export.
Dependency direction: script -> app -> stdlib; optional graphics remain at the
outer boundary. No app import of infra/adapters, scientific reader dispatch,
source admission, changed model/objective or new estimator. Full original state
proofs stay in the immutable source; the JSON view retains every plotted vector,
all resource fields/history points and original analysis records. See
[the workflow and limits](docs/historical-outcome-figures.md) and ADR-0198.


## Local experience arrival boundary (R3.2)

`core/experience.py` holds immutable role/permission/sample/episode/candidate/action/
reward/version metadata and a logical event clock. `app/experience_inbox.py` owns
bounded train-role delivery, detached payload copies, identity history, stable
eligibility ordering and failure bookkeeping. It calls the existing native
`app/learner_step.py`; optional start/completion hooks preserve old callers and
record completed work before the existing post-update budget check.

Dependency direction: inbox -> core contracts and app learner/budget; outer
native adapters -> core port. No infra import or IO is added. Trusted tags do not
attest physical provenance or release evaluation/final roles. Byte/RSS bounds,
actor concurrency, promotion, persistence/replay/privacy and scientific/native
guards remain separate. See ADR-0199 and docs/experience-contracts.md.


## Stable actor and shadow ownership (R3.3)

core/actor_ports.py adds a separate owned-fork port and versioned results; the
old native port remains unchanged. Native adapters preserve full model state
and policy through existing owned constructors. app/actor_shadow.py keeps a
private stable serving fork and independently gated candidate fork. Actor
reads and candidate writes have separate locks; wake work reuses the accepted
experience inbox and budget. Trusted consolidation receives detached state,
never a mutable learner handle. App -> core/app contracts; outer adapters ->
core. No infra dependency, algorithm or service is added. Construction requires
a quiescent source; promotion/rollback and live contention remain R3.4/R3.5.
See ADR-0200 and docs/actor-shadow-runtime.md for complete boundaries.


## Matched promotion guard boundary (R3.4a)

core/promotion_guard.py owns pure policy/evidence/decision and role-tagged guard
metadata. app/promotion_guard_evaluation.py restores independent native predictor
copies and measures the same new/old inner inputs under identical utility/action
callbacks. App -> core ports; no infra, IO, actor setter or training is added.
Role/time/ID/base-version refusals precede payload access. Frozen reports bind
policy, snapshot digests, versions/revision and declared data IDs; they remain
trusted local evidence, not global release or a promotion ticket. R3.4b retains
atomic complete serving transitions/rollback. See ADR-0201 and promotion-guards.md.


## R3.4b complete serving transactions

`core/serving_ports.py` owns configuration, detached serving/cache records and local ticket/receipt types. `app/serving_promotion.py` composes the native builder and existing guard evaluator with optional actor composition and candidate leases in `app/actor_shadow.py`. Dependency flow: serving app -> actor/evaluator app -> core native/experience/promotion/serving contracts; core imports no infra/adapter/IO. One serving slot owns current model/cache/context/version, monotone generation and one complete prior bundle. All reads/cache fills/swaps use the same gate. Candidate/controller gates refuse reentrance; preparation does not hold the serving gate. See ADR-0202 and docs/serving-promotion.md. Metadata is passive audit context; algorithm-affecting routing is unsupported pending matched guard extension.


## R3.5a cooperative work admission

`core/resource_sharing.py` owns positive sharing limits and detached admission/poll/counter records. `app/resource_sharing.py` composes actor serving and candidate updates with a short-lock priority gate; native ownership remains in the existing runtime. `ExperienceInbox.drain` optionally bounds polls and leases each update through an exact-bool context before payload/native access. Flow: sharing app -> actor/inbox/serving app -> core ports. Core and inner app import no infra/adapter. Resource probes run outside the gate lock then contention is rechecked. Active native calls finish; subsequent updates defer. Complete supported checkpoint/restore and actual live p50/p95 remain R3.5b/c; see ADR-0203.


## R3.5b1 inbox history format

`core/inbox_cursor.py` owns exact-version complete metadata validation and opaque train payload records. `app/experience_inbox.py` checks quiescent owner indexes, validates metadata before payload copying and returns a detached cursor. Flow: inbox app -> cursor/experience/native diagnostic core; no infra/adapter imports. Model/time/budget/sharing owner transfer remains a separate outer R3.5b transaction. See ADR-0204; a cursor grants no resume/final authority.


### Complete candidate checkpoint ownership

Supported same-process full candidate handoff is implemented through app/candidate_checkpoint.py; see docs/candidate-checkpoints.md and ADR-0205. Retain original cumulative budget/clocks/RSS sampler/resource gate and stable actor, restore independent native state plus full inbox/consolidation histories, retire old owner and invalidate its promotion authority. Identity tokens and preparation work are bounded. R3.5b acceptance remains pending current full validation; durable process recovery and actual live latency remain unfinished.

R3.5b current supported owned checkpoint acceptance:371 tests,647-file Windows/Linux types,full scoped/static/source/resource/guide gates pass. Evidence:artifacts/runs/r35b-owned-handoff-20261007/. Actual live R3.5c latency and durable R3.5b2 recovery remain unfinished.


### Actual live serving measurement

The generic app native observer/shared-request harness and pure core timing/
nearest-rank/overlap records are documented in docs/live-serving-measurement.md
and ADR-0206. The reserved script boundary executes the finite declared matched
two-native protocol after full correctness/source binding. Retain all raw requests,
native windows and incomplete worker status; no automatic repeat/outcome filtering.
Current measurement acceptance is pending; no scientific advantage is claimed.

R3.5c/original R3.5 current acceptance:396 cases,full651-file platform types/static/source/resource gates and one reserved actual native serving run pass;96/96 shared requests per method fully native-contained. Negative circadian p95 slowdown retained. Evidence:artifacts/runs/r35c-live-serving-20261007/measurement-summary.md. Next R3.6 privacy/replay lifecycle;durable R3.5b2 and broader guards remain open.


## 2026-10-07 — R3.6a permanent data admission (validation pending)

Optional managed local admission: [guide](docs/managed-experience.md). Pure src/core/data_lifecycle.py metadata and src/app/managed_experience.py original authority install permanent hooks on a fresh ExperienceInbox; legacy uninstalled behavior stays intact. Requires declared training/replay consent and permissions, bounded lifetime grant counts and explicit synthetic/unverified policy. Supported checkpoint handoff retains authority. Opt-out stops future training; native/inbox/checkpoint erasure and unlearning are not claimed. No new environment variables/dependencies. Full R3.6b deletion controls remain the next extension.


R3.6a acceptance:433 passing cases,current654-file Windows/Linux types/static/AST/executed guide/source/resource gates. See artifacts/runs/r36a-data-admission-20261007/admission-summary.md and validation.json. Full R3.6/R3.6b erasure remains unchecked. Exact next action: Prospectively scope R3.6b actual native replay/inbox erasure: inspect replay snapshot identity and full InboxCursor/applied receipt references; design payload-free tombstones preserving consumed IDs, enforce retention lifetimes/quotas, invalidate pending checkpoint payload copies, and test refused resurrection across owned handoff with original clocks/budgets/gates. Start with fake deletion controls, then fixed native controls; do not call erasure parameter unlearning or erase caller-held copies by implication. Preserve original full R3.6 acceptance and durable R3.5b2/R3.7/G3/human-deferred/scientific work.


## 2026-10-07 — R3.6b1 erasure prerequisites (validation pending)

[Payload erasure primitives](docs/data-erasure.md): core data_erasure.py provides tombstones/counts/optional ReplayPayloadOwner port;InboxCursor format2 and private ExperienceInbox erasure preserve consumed IDs/applied work;NumPy adapters expose native whole-buffer erasure. Only raw replay references are removed;native weights/RNG/policy/counters and exposure hashes remain. Format1 unerased histories remain supported. No new dependencies/environment variables. Full original-authority deletion,quotas/lifetimes/checkpoint/promotion cleanup/non-resurrection is unfinished R3.6b2.


R3.6b1 acceptance:488 passing cases,current656-file win/linux types/static/AST/guide/source/resource gates. Evidence:artifacts/runs/r36b1-erasure-primitives-20261007/validation.json and erasure-summary.md. Full original R3.6/R3.6b/R3.6b2 remains unchecked. Exact next action: R3.6b2: prospectively scope original-authority coordinated deletion. Inspect all owners of payload copies: live/retired candidate and inbox, pending/inspected checkpoints and prepared/failed models, serving promotion/rollback bundles and pending tickets. Define owned-versus-caller copies and bounded supported native/payload measurement ports; implement deletion/expiry/opt-out coordination plus declared record/byte/time policies under original manager/candidate/serving/checkpoint leases. Tests first for partial cleanup failure and refused resurrection; retain budgets/clocks/gates/consumed IDs and weights. Keep transient/audit-only admission disabled until purge semantics pass, and original R3.6/R3.6b unchecked until the complete criteria are proven.


## 2026-10-07 — R3.6b2a retained-copy ownership (validation pending)

[Retained payload ownership](docs/payload-ownership.md):core metadata/reference ports and app weak registry enumerate supported actor/candidate/checkpoint/promotion owners under nonblocking all-holder quiescence. One opt-in live/lifetime registration allowance survives handoff and GC without renewal. Existing native/budget/serving semantics remain;no new dependencies/environment variables. This is a cleanup prerequisite,not complete deletion or byte/time policy;R3.6b2b retains original-authority all-copy cleanup/non-resurrection acceptance.


R3.6b2a acceptance:512 current tests,full659-file win/linux types/static/AST/guide/source/resource gates;zero NEW native work. Evidence:artifacts/runs/r36b2a-payload-ownership-20261007/validation.json and ownership-summary.md. Full original R3.6/R3.6b/R3.6b2 remains unchecked. Exact next action: R3.6b2b: prospectively scope original-authority coordinated cleanup and retention. Bind current ownership registry and manager plus native replay/inbox tombstone ports; use all-holder references under the existing nonblocking lease, deduplicate by identity, acquire the original sharing/consent authority, and avoid public methods that reacquire leased locks. Define supported native/payload byte measurement and cumulative holder/record/byte/time policies; tests first for pending checkpoint/prepared/failed/retired/promotion/rollback copies, partial cleanup failure, stopped/retired inbox ledger accounting and refused resurrection. Integrate deletion/expiry/opt-out without resetting budgets/clocks/gates/IDs or parameters; keep transient/audit-only admission disabled until actual purge semantics pass. Preserve full original R3.6/R3.6b/R3.6b2 criteria and caller-copy/RAM/unlearning limits.


## Managed cleanup integration (R3.6b2b acceptance open)

Core data_retention defines immutable policies/reports. App managed_data_lifecycle coordinates the original manager,actor ownership registry and paused sharing leases;it accepts trusted native footprint/erasure and ingress measurement ports. Outer adapters/numpy_learners.make_managed_data_lifecycle composes exact NumPy ports;dependency arrows point inward. Why:unkeyed replay requires explicit conservative whole-buffer deletion,token invalidation and fail-closed partial cleanup. Guide/ADR-0210 document caller-copy/unlearning exclusions and unfinished aggregate retained-byte/automatic-time enforcement.


### Owned payload copy bytes

Core payload_bytes defines immutable limits/observations;app payload_copy_budget owns a nonblocking monotonic reservation ledger. ManagedDataLifecycle composes original-authority admission at existing copy boundaries;outer NumPy ports inspect supported graphs and builder sources. Why:conservative no-refund accounting avoids reclaiming capacity while retired/failed owners retain copies. Measured raw replay/inbox/checkpoint and auxiliary arrays are bounded;parameter/temporary/caller/Python/RSS memory and arbitrary graph attestation are excluded. Guide/ADR-0211 document limits;automatic elapsed purge/full R3.6b2b acceptance remain open.


### Bounded elapsed retention

Core retention_driver defines immutable policy/results;app retention_expiry owns one non-daemon bounded worker per original lifecycle. Lifecycle anchors original elapsed/logical clocks;sharing adds an independent hold and quiescent retry. Why:preserve original consent/budget/manual pause authority while clearing overdue owned copies. Callback publication guards revalidate deadlines. No inner-to-adapter dependency or new package. Physical purge requires responsive quiescent owners/process;caller copies/RAM/unlearning/durable recovery are excluded. Scalar-only auxiliary metadata has a confirmed missing anchor and prevents full R3.6 acceptance. Guide/ADR-0212 and acceptance audit record the exact repair.


### Auxiliary content presence versus array bytes

ManagedDataLifecycle now validates every exact initial auxiliary dictionary and anchors nonempty content,including scalar metadata/nested empty values/zero arrays,independently of measured numeric bytes. Promotion uses the same focused helper. Why:array size cannot identify owned data and short-circuit validation skipped later graphs. Existing inward measurement ports,original clocks/quotas and byte metric remain unchanged. No new interface/dependency. Full R3.6 acceptance audit passes;runtime matrices,durable restart,R3.7/R3.8/G3,caller/RAM/unlearning remain separate. Guide/ADR-0213 document usage/limits.


### Finite repeated runtime faults

No production architectural change:R3.7a composes existing trusted fake/native ports in deterministic finite tests. Original ownership,budgets,IDs and actor availability are checked across rejection,corruption,stopped handoff and growth refusal. Why:isolated failures cannot establish repeated behavior;processRSS,traced allocation and owned-array accounting are distinct. Eight-cycle Native checks are not authentic OS-crash recovery or sustained deployment. Existing label-first version-neutral behavior is preserved. Guide/ADR-0214 and acceptance audit retain full R3.7/R3.5b2 unfinished authority/recovery work.


### Durable metadata admission boundary

`src/core/recovery_admission.py` validates bounded original-accounting/component/epoch/next-owner relationships and returns typed accounting facts. No outer imports,IO,callback/native state or copy/restore capability. Why:restart observations cannot recreate original authority. Independent transactional coordinator,OS epoch/resource/ownership adapters and complete state codec remain future infra work through inward ports;full crash acceptance stays open. See recovery-admission guide and ADR-0215.


### Windows observation adapter boundary

`core.recovery_observation` defines bounded identity/observation values and inward port;`infra.windows_process_handles` owns documented API bindings/live registered handles;`infra.windows_recovery_observer` returns time/RSS/liveness under retained coordinator anchor. No fence/codec/native restore integration. Why:PID reuse/timeout cannot establish predecessor death. Coordinator loss remains unsupported/unfinished. Negative venv launcher capture retained;corrected harness unrun,full durable authority/crash acceptance stays open. See guide/ADR-0216.


### Windows observation validation accepted

No source/interface changes. A separately declared one-worker successor validates the original observation port/adapter policy with exact same-version direct interpreter identity,time bracketing,original live anchor,retained exit and bounded RSS/cleanup. Prior failed launcher capture retained. R3.5b2b scoped acceptance complete;full transactional authority/live fencing/codec/durable model recovery/coordinator loss unfinished. Next add journal adapter through inward authority ports.


### Coordinator metadata transaction boundary

`core/recovery_authority` defines exact records,monotone transitions and read/advance port;`infra/recovery_authority_codec` encodes bounded metadata and `infra/sqlite_recovery_journal` implements transactional CAS. Dependency direction:infra -> core. Independent surviving coordinator witness/trusted host observations/private disk are required. Commit charges before future app dispatch;uncertain outcomes cannot retry. Why SQLite:existing stdlib transaction boundary without a dependency. Live handle ownership,app dispatch/publication,native completion/component codecs and coordinator-loss recovery remain future boundaries. See [ADR-0217](docs/adr/ADR-0217-persist-monotone-authority-before-worker-actions.md).


### Local recovery coordinator boundary

`app/recovery_coordinator` -> `core/recovery_coordination`,authority/observation ports;infrastructure implements the ports. App retains independent original witness and registrations,commits costs before trusted callbacks and rechecks around publication. Why:a local sequencing prerequisite before native codecs/live lease integration. No native controller changes. Current lock is one lane;terminal/high-water memory and callback side effects do not establish restart safety or an external-writer publication lease. See [ADR-0218](docs/adr/ADR-0218-sequence-coordinator-actions-under-original-authority.md).


### Durable reporting and terminal reconciliation

Core recovery_reporting defines report/witness/extended reporting port. App persists observations/stops through the inner port;infra SQLite shares bounded CAS and offers explicit fresh-connection terminal-only reconciliation of exact independently known states. Dependencies point inward;storage schema unchanged. Why:retain unknown-commit and measured overshoot evidence without refunds,retries,cap increases or invented completion. Actual publication lease/live composition/native/coordinator-loss recovery remains separate. See [ADR-0219](docs/adr/ADR-0219-persist-terminal-facts-without-renewing-authority.md).


## Private journal publication serialization

`RecoveryPublicationPort` in core extends reporting;the SQLite adapter holds a
writer reservation across trusted publication and fresh entry/exit validation.
App coordinator now depends on RecoveryLeasedPublicationPort without importing
infrastructure;its lease commits fresh observations under shared writer ownership. See docs/recovery-publication-guard.md and
ADR-0220. Guard writes no authority and cannot undo callback side effects.


## Durable observations under publication ownership

Core defines RecoveryLeasedPublicationPort/RecoveryPublicationLease;app checks its
retained probes and reports under that lease. Infrastructure uses a permanent
one-byte native writer lock shared by every supported private journal writer,
plus existing bounded SQLite observation transactions. Legacy publication_guard
stays supported. See docs/guarded-recovery-coordinator.md and ADR-0221.


## Original Windows recovery composition boundary

Infrastructure windows_recovery_composition.py combines the existing private
journal,original retained Windows registrations and observation adapter with the
app coordinator through inner leased/report ports. The current physical anchor
identity and exact original journal are required. No registration/process launch/
journal bootstrap or native restore is performed. See docs/windows-recovery-composition.md
and ADR-0222 for ownership/error cleanup and separate actual crash validation.


## Metadata boundary actual process validation

Original surviving-coordinator/leased publication boundary now has one strict
actual Windows worker pre-COMMIT interruption/contended lock/released ownership
capture,with exact prior committed records and no refunds. Fresh observers retain
independent original RSS high water. Native/component digest markers in that
metadata fixture are explicitly absent states;they cannot grant model restore or
complete codec authority. See docs/windows-recovery-process-capture.md. Native
complete codecs/stable actor/coordinator-loss/model acceptance remains open.


### Explicit durable component codecs (R3.5b2e1)

`core/checkpoint_codec` owns typed binding/limit/byte ports. The separate
`adapters/backprop_checkpoint_codec` implements an exact complete native schema
using the existing adapter snapshot type; dependency direction stays inward.
No disk/network/live owner or learner-policy reconstruction enters core. Unknown
fields/aliases fail before copies. Array/wire size limits do not reserve original
lifetime copy capacity or certify cumulative native restoration. Full component
and actor/lifecycle ownership work remains in R3.5b2e. Why/alternatives:
ADR-0223 and docs/durable-checkpoint-codecs.md.


### Complete circadian native byte component

Three separate adapter responsibilities are NumPy frame validation/encoding,
frozen circadian native schema/relations,and byte orchestration. All depend
inward on existing native/core contracts;no core-to-adapter dependency or disk
operation added. Complete native payloads remain independent of runtime/consent/
resource authority. ADR-0224 and docs/durable-checkpoint-codecs.md record why
explicit current variants precede composite recovery admission.


### Circadian checkpoint C/F layout qualification

CPC codec wire v2 (`circadian_full_v2`) records exact C/F array order and rejects
old v1,unknown/missing order and unsupported noncontiguous storage. Native
snapshotv2/API/equations remain unchanged. Separate34layout controls compare
fixed native structural continuation across8variants. Frame validation,
native schema and byte orchestration retain their existing inward boundaries;
no dependency/configuration change. See docs/durable-checkpoint-codecs.md and
ADR-0225 for tree,commands,limits,ownership and extension details. Backprop layout
qualification and full composite/native/live recovery remain unfinished.


### Explicit Backprop layout codec qualification

`LayoutBackpropCheckpointCodec` adds separately selected `backprop_layout_v2`
wire frames preserving supported C/F array storage,canonical bytes and native
required aliases. The existing v1 codec remains available for its original
noncontiguous logical-value scope. Exact schema/shared C/F frames depend inward
on existing native contracts and typed codec ports. Every bounded payload is
validated before detached NumPy materialization. Fixed native continuation uses
existing Backprop BCE loss/state/predictions;no model metric/API/equation or
configuration/dependency change. Newmodule tree,examples,commands,limits and
extension path:docs/durable-checkpoint-codecs.md;decision:ADR-0226. Full durable
composite/owner/copy/lifecycle/model/scientific acceptance remains unfinished.


### Complete supported NumPy inbox byte component

NumpyInboxCheckpointCodec wirev2 preserves complete native cursorv1/v2 events,
permissions,unmatched/consumed/applied/tombstone history and actual supported real
numeric C/F payload dtype/byteorder/bytes/ownership. Inner InboxCodecPolicy binds
original independent shape/dtype/record/string/candidate bounds. Role/permission/
metadata checks precede payload inspection;all raw validation precedes detached
NumPy materialization. Existing native/inbox/source/test/API bytes retained;
no dependency/configuration/learning change. Tree/commands/zero-native example/
extension and authority limits:docs/durable-checkpoint-codecs.md andADR0227.
Lifecycle/consolidation/consent/revocation/retention/copy/actor/sharing/Torch/full
composite/singleowner/native/model/coordinatorloss/scientific/human work stays open.


### Consolidation observation boundary

ActorShadowRuntime -> core ConsolidationCursor validates complete ledger metadata;
ConsolidationCheckpointCodec -> core cursor/policy/checkpoint port encodes explicit
canonical bounded bytes. Core imports no outer layer. Capture uses the existing
nonblocking candidate lease;typed records and codec hold no clock,budget,model,
callback or live authority. Lifecycle/retention/copy/owner capture remains a
separate required component before composition. See docs/consolidation-cursors.md
and ADR0228 for tree,independent bindings,aliases and rationale.


### Lifecycle record and authority boundary

managed_lifecycle_schema (app) -> managed_lifecycle_state/validation (core) ->
existing inner policy/consent/retention/copy/ownership records. Exact source-field
guards cover all66 current private fields. Metadata copies preserve original
policy aliases;47 reference slots retain original live objects without copying
or invoking them. No live capture lease,IO,serialization or restore authority
is provided. Driver mutator synchronization and actual capture remain required.
Why and alternatives:ADR0229;tree and workflow:docs/managed-lifecycle-state.md.

Lifecycle capture:src/app/managed_lifecycle_capture.py orchestrates original app owner leases and inward core records/validation. Driver state transitions share a short original RLock;cleanup callbacks and joins run outside it. No infra or adapters import into core. See ADR-0230 and docs/managed-lifecycle-state.md.

Lifecycle byte component:core/lifecycle_codec_policy holds independent original policies;adapters/lifecycle_checkpoint_schema owns explicit wire preflight/immutable aliases;adapters/lifecycle_checkpoint_codec implements the inner CheckpointCodec[ManagedLifecycleCapture]. No outer import enters core. Original app capture and every prior source/API byte remain unchanged. See ADR0231.

Paired record capture:core/managed_record_state defines full immutable records and pure bounded relationship checks;app/managed_record_capture uses the original lifecycle lease interval and internal leased cursor reader. One metadata graph is detached;59live reference slots retain identity. Existing public capture signatures remain stable. Promotable slot versions are read under the already-held actor gate. See ADR0232.

Paired byte component:core/managed_record_codec_policy validates independent complete component policies;adapters/managed_record_checkpoint_schema enumerates24native records and raw relationships;adapters/managed_record_checkpoint_codec implements CheckpointCodec[ManagedRecordCapture]. Existing lifecycle walkers accept trusted schema/prefix parameters with unchanged defaults. Entire original metadata/alias graph and59original refs are independently bound;no inward layer imports adapters or app. See ADR0233.


### Original-owner composite capture

`core/managed_composite_state.py` defines bounded records and inward projection,
preflight and copy ports. `app/managed_composite_sources.py` enumerates exact
source-field roles; `app/managed_composite_capture.py` owns original lease and
admission orchestration. `adapters/numpy_composite_capture.py` validates complete
native states and copies one assembled graph. App/core do not import NumPy
adapters. Why: one admitted graph copy preserves cross-component aliases without
renewing original live authority. Full pending/sampler/native-variant qualification
and canonical bytes/restore remain unfinished;see ADR-0235 and module guide.


Managed capture resource admission (ADR-0236): shared ProcessRssSampler supplies
an expiring internal observation port under its original nonblocking gate. The
app consumes that port with the original budget/progress and rechecks retention
after payload copying. The inward copy port receives one shared memo; the outer
NumPy adapter reuses detached payloads while final observation records refresh.
No replacement sampler/owner or dependency reversal is introduced.


Managed capture retained-source bindings (ADR-0237):explicit original enrolled
runtime/controller/pending/token relationships and installed consent/copy guards
now precede holder payload ports and projection. Complete graph bounds precede
checkpoint measurement,integrity and pure promotion checks. Retained checkpoint
payloads require live consent even when the current inbox is empty. Final pending
references and consent are rechecked without invoking controller restore guards.
See docs/managed-composite-capture.md and src/app/managed_composite_bindings.py.
Portable tickets preserve fields/aliases;original ticket and rollback receipt
remain live authority references. Actual promotion issuance/native provenance/
all variants/replay consent/bytes/recovery remain unfinished.


Original managed native update observation (ADR-0238):optional native_observer
ports in inbox/runtime/sharing/managed owner expose original source/label/learner,
actual detached inputs and committed receipt/spent count through synchronous
expiring access. See docs/native-update-origin.md. Original consent/admission,
owner instance fields,default calls and update order remain unchanged. Callback
faults preserve original failure/receipt/resources;returned references remain
caller-owned. No new configuration/dependency/environment variable. Core defines
the reference contract;app manages lifetime;neither imports adapters/infra.
Persistent replay origin and every storage/retention/dedup/eviction/fork/checkpoint/
promotion/restore/erase path remain open under R3.5b2e5b3. Complete compoundcapture,
canonical bytes and recovery gates remain unchecked. Do not infer row lineage
from content hashes or treat these observations as consent/restore permission.
Tests:fixed fake-only origin controls +five selected fake inbox controls +990
current composite/resource/codec controls;both742types/wholeRuff/check format.
For safe extension:add a bounded original replay-write/row port with weak or
owned-accounted payload references;preserve terminal failures and original gates.


Original replay-write observation (ADR0239):core/replay_write_origin defines
a bounded original model/input window;app/replay_write_origin composes it with
the original managed producer. Native copy ranges,new snapshot references and
final retained identities are observed without extra array copies/native fields.
Explicit original ContextVar token/thread/callback lifetime survives refused
close;all default storage/policy/RNG rules preserved. See docs/replay-write-origin.md.
Pure/current gates precede a separately declared tiny native storage-only parity
fixture. No dependency/env/config changes. This is not a persistent row ledger or
consent/restore certificate. Full b3/e5b/e5 retained variants/lineage/bytes/recovery
remain open. Next consume actual copy identity under bounded weak/owned-accounted
retention and original consent/terminal-outcome authority before broadening capture.


### Bounded current-candidate replay origins

`src/core/replay_origin.py` defines metadata/admission and injected ports;
`src/app/managed_replay_origins.py` composes original owner/runtime gates and
weak row records; `src/adapters/numpy_replay_origins.py` implements native reference/
integrity ports. Dependencies point inward. Existing source/schema fields stay
unchanged. Why this: exact observed copy identities establish producer binding;
hashes only verify already-bound contents. See ADR-0240 and the row-origin guide.


### Replay capture boundary

App `replay_capture_origins` validates the original ledger under already held
owner/runtime leases. Adapter `numpy_replay_capture` supplies bounded original
model/snapshot rows before projection/copy. Capture repeats checks around callbacks;
ledger capture uses its own nonblocking gate and existing lifetime charges.
Native schemas/fields unchanged. Rationale: ADR-0241 and replay-capture guide.


Replay capture lifecycle repair: original runtime open/consent checks use
leased elapsed access during capture; public default reads keep ordinary access.
See docs/replay-capture-origins.md and current development log for qualification.


### Original model-copy observation boundary

`src/core/native_model_copy.py` defines local source-bound observation ports;
`src/adapters/numpy_learners.py` supplies the actual native copier memo. No model
fields or adapter policy change. Original-context Token validation is required
because inherited ContextVar values alone do not establish original context.
Borrowed references expire per callback;caller retention requires separate owned
accounting. Original managed consent,copy budgets and registered holder lineage
remain application responsibilities and are unfinished for copied replay holders.


### Managed replay-copy witnesses

`src/app/managed_replay_copies.py` composes the original ledger and registered
checkpoint controller with `src/core/native_model_copy.py` observations. The
NumPy builder-source port lives in `src/adapters/numpy_replay_copies.py`. Metadata
uses the original row ledger admission;raw replay copies use the original lifecycle
payload budget. Copies hold original owner/runtime/registry/time/resource gates
through admission and copier completion;persistent targets/rows are weak.
Why this:failed checkpoint preparations can retain genuine copied models before
policy checking. Actual memo and holder position/history bind their row origins.
The observer closure keeps raw references and leases local to one invocation;
helpers handle admission,binding and history validation. Copied metadata slots
stay reserved after faults or collection. Full restored snapshot/inbox/receipt
lineage and ledger transition remain separate unfinished work.


### Native state copy boundaries

`core/native_state_copy.py` reuses the bounded copy-window implementation in
`core/native_model_copy.py` through an independent context channel. CPC snapshot
and restore depend inward on that core port. Original validation and native state
publication remain in CPC; admission and retained lineage remain application
responsibilities. No native field or checkpoint schema is added. Why this: fork
and state copies have different roots and can occur within the same preparation.


### Actual copy sequences

`core/native_graph_copy.py` reuses a single bounded copy window across changing
actual roots;app checkpoint/inbox copy sites and core native-state copies depend
on this inward port. State/sequence observers share the original copier memo.
Why this: separate allowances for each root would renew limits within one real
checkpoint operation. Original holder fields,validation/enrollment and publication
remain in their owners;original authority/admission/ledger transition remain app
responsibilities. Purpose labels cannot serve as provenance.

`core/replay_graph_origin.py` defines borrowed inventory ports;
`adapters/numpy_replay_graphs.py` implements native state dictionary/snapshot
inspection through existing exact NumPy payload validation. Dependencies point
inward. It retains no payload, performs no native model operation and grants no
authority. Replay-only byte counts are separate from complete model/heap/RSS
accounting. See docs/replay-graph-origins.md for caller responsibilities.

### Managed checkpoint replay publication

`app/managed_replay_checkpoints.py` composes original admission and native/fork/
inbox witnesses; `app/checkpoint_replay_handoff.py` defines the trusted lease at
the controller's original publication point. `core/checkpoint_content.py` reads
bounded exact core records and numeric buffers after opaque ports; it provides
local integrity, never provenance. Adapters supply original borrowed graph
ports. Dependencies point inward; no adapter is imported by the application.
The original controller owns preparation and the four publication statements.
Why this: final probes run after graph-copy observations, so authority must be
rechecked and held through publication. Exclusive cleanup releases the actual
lexically acquired lock even if a callback replaces its field. ADR-0242 records
this boundary; full copied-holder/compound/canonical/recovery gates remain open.
