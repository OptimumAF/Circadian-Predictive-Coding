# Module: `src/app`

## Responsibilities

- Orchestrate end-to-end experiment flow
- Build comparable reports across learning approaches
- Run in-depth aggregate comparisons across seeds and dataset noise levels
- Run ResNet-50 speed/latency benchmarks with target-accuracy and circadian sleep metrics
- Run bounded, equal-trial tuning of frozen shared-feature heads with validation-only selection
- Freeze validation-selected matched-head confirmations across predeclared seed and budget scopes
- Label NumPy toy, continual, and dynamics outputs with algorithm identity and comparison scope
- Estimate planned synthetic-vision candidate training batches and examples before external resources open

## Inputs / Outputs

- Inputs: `ExperimentConfig`, optional adaptation policy
- Outputs: `ExperimentResult`, `InDepthComparisonResult`, `ResNet50BenchmarkResult`,
  `MatchedHeadTuningResult`, `ProcessIsolatedMemoryResult`, and
  `RepeatedConfirmationResult`

`sweep_work_estimate.py` accepts resolved synthetic `ResNet50BenchmarkConfig`
candidates and a seed count. It returns the maximum planned optimizer calls
and row exposures from epochs and full/short batches; a separate predicate
checks an explicit launch ceiling. It does not train, open datasets, estimate
validation/inference work, measure elapsed time or memory, or stop a running
sweep. Both legacy circadian-policy and multi-seed Pareto scripts use it before
Torch initialization, passing the same ordered candidate lists to training
after a launch ceiling succeeds.

`toy_execution_budget.py` accepts non-negative total wake-update and
replay-example ceilings, a positive adaptive circadian hidden-width cap,
an absolute per-invocation process-RSS cap, per-invocation wall time, and an
injectable monotonic clock. The toy runner checks them before each model
update, before sleep, and before final scoring, raising typed incomplete-stop
facts with a checked existing cursor when available. The budget is outside
`ExperimentConfig`, so it does not alter scientific identity or old
checkpoint bytes. An optional mutable `ToyExecutionProgress` exposes
committed wake/replay work and the last saved
cursor to an outer adapter even if a later operation raises. The toy runner
preflights exact selected replay batch lengths before sleep and restores
applied examples from checked sleep telemetry on resume. This module
also records current and historical transient peak width from the checked
circadian snapshot and sleep history. It stops before selected sleep or
final-proposal growth exceeds the run-level width cap and does not change
the core's split/prune choice (ADR-0135).
The RSS sampler starts before toy dataset/model construction, observes
5 ms background samples and explicit checked boundaries, and records its
start/peak/sample count. A sampled over-cap peak stops at a boundary; short
unobserved peaks cannot be ruled out (ADR-0136). This module does not train,
score, persist a lifecycle artifact, or interrupt an
individual update/sleep operation; the adapter/infra boundary owns CLI
publication (ADRs 0132–0136).

`matched_head_tuning.py` takes a guarded, frozen-backbone base configuration,
predeclared per-head candidate configurations, common seeds, and an equal
candidate count. It caches train/guard/validation features once per seed,
records every candidate trial and its guard/validation exposure, selects by
mean outer-validation accuracy, then scores only selected heads on final test.
With `confirm_test=False`, it returns the selected candidates without reading
final test. Failed attempts remain available on `MatchedHeadTuningError`;
trial rows have no final-test metric. The module does not define candidate
search spaces, rank head families, run large sweeps, or attribute peak memory.

`repeated_head_confirmation.py` takes a validation-only tuning result and
freezes selected configs, disjoint confirmation seeds, metrics, and the
fixed-data, wall-time, and isolated-memory scopes in a digested manifest.
Its runner verifies per-seed hash/capacity pairing and returns raw reports
plus separate descriptive summaries. It does not choose seeds or winners,
retune after test, or treat process RSS as incremental head memory. The
small repeatable entry point is `scripts/run_repeated_confirmation_smoke.py`.

For P1.8o's feature-only cost measurement,
`matched_head_tuning._build_seed_bank(..., include_final_test=False)` asks
the outer loader boundary for development roles only. This opt-in path
does not construct the final CIFAR source or train a head; ordinary tuning
and confirmation retain their existing defaults and outputs. The probe
script records feature cost and the separate study-preparation script
freezes a future matched request without scoring it.

`experiment_runner.py` accepts an optional validated model order in
`ExperimentConfig` for the NumPy toy comparison and records the order in its
result. Per-model seeds, role splits, and the default execution order stay
fixed. The option is for reproducibility checks, not candidate selection;
it does not make the toy and image benchmarks comparable.

`toy_experiment_config.py` takes environment-backed defaults, the original
toy CLI's existing fields, and explicit typed overrides. It returns a
validated `ExperimentConfig` and a complete ordered baseline/indepth
request record. It does not parse command syntax, train, write files, or
inspect scores. `indepth_comparison.py` supplies the single per-cell config
constructor used by both execution and recording (ADR-0128).

`single_resnet_experiment_config.py` takes the existing 110-field
`ResNet50BenchmarkConfig`, named historical preset, and explicit typed
overrides. It returns a validated config and complete unmatched request
record. It does not parse flags, load images, run Torch, inspect scores,
or write files. The existing runner validator still owns protocol and
model constraints; this module enforces them before the runner starts
(ADR-0129).

`numpy_sleep_decisions.py` combines a NumPy core result with the runner's
schedule, attempt duration, and optional measured guard facts. It retains
the complete core proposal when the arrived inner guard restores a rejected
attempt. For an unscheduled epoch it reads chemistry without calling sleep.
It does not choose evaluation roles, score a guard, or change legacy counts.
`toy_checkpoint.py` stores this sequence with its model/order cursor and
rejects incomplete or old version-1 histories and invalid replay usage
before model restoration. It derives current and transient peak adaptive
width for budget resume from the saved model and validated sleep history.

`comparison_scope.py` takes a NumPy hidden-width tuple, or `None` when a
standalone figure builder lacks architecture provenance. It returns an
immutable scope with hidden depth, three stable algorithm IDs, and a
descriptive attribution status. Toy, in-depth, continual, and dynamics
reports carry it alongside their unchanged evaluation protocol IDs. It
does not change training, infer causal equivalence from matched widths, or
authorize a deeper circadian-mechanism claim. See ADR-0024.

`continual_shift_benchmark.py` accepts the same kind of validated order
control for both NumPy training phases. It still builds Phase B roles only
after Phase A and deep-copies the A-only states before B training. Per-seed
results record the order; final-test scoring remains after Phase B. This
control checks reproducibility and does not establish a strict-online study.
Its v0–v5 routes now attach one sleep decision per phase A/B epoch to each
seed report. The runner history stays outside the trained-state object so
variable timing does not change the model-state serialization used by
final-label and order invariance checks. These routes preserve their original
absence of a guard; arrived v6 uses the separate typed guarded path and v7
selection carries its candidate histories into trials and the frozen choice.

`continual_checkpoint.py` defines the typed NumPy continual checkpoint,
phase-specific data digests, and pre-restore validation; it owns no file IO.
An independent sleep-history extension version on the v0–v5 payload rejects
older eventless files before restore. Active event sequences live beside
the model state; completed v5 unscored seeds carry their own sequence.
The opt-in `continual_phase_arrival_v2` orchestration in
`continual_shift_benchmark.py` constructs Phase A roles first and upgrades
to the combined Phase A/B digest only after Phase A training finishes.
Its checkpoint training context excludes final-test objects and defers
test-role hashing until both training phases complete.
The trusted file implementation remains in `infra`. Existing v1 recovery
and the default offline protocol retain their format. This boundary does
not define strict-online replay budgets or hide the known full A+B schedule.
The separately named `continual_phase_local_schedule_v3` route keeps the
same phase-arrival data boundary and uses checkpoint format 3, while Phase A
passes only its own epoch horizon to sleep in both execution paths. Its
Phase B schedule remains the arrived A+B horizon. It does not yet implement
the replay, guard, label-arrival, or reporting requirements for a full
strict-online protocol (ADR-0067).
The opt-in `continual_bounded_replay_v4` adds a distinct config/result
type and checkpoint format 4. The app supplies explicit replay caps to
the NumPy circadian core, reports retained content IDs/count/array bytes
at both phase boundaries, and rejects saved active/frozen examples from
unarrived or nontraining roles before resume. Its phase-local schedule
and final-test seal are inherited from v3. Guard and outer-selection
arrival remain outside this increment (ADR-0068).
The ordinary v4 path now sets `hash_test=False` for both phase splits
and binds the final-test hashes after B training, matching checkpointed
v4. The held-out roles are not passed into either training helper;
the splitter now carries a deferred reference so neither v4 path asks the
source for test input/label fields until B training ends. The generator
still allocates those arrays early, and the runner scores each completed
seed before later seeds train. P1.3c3b still needs global setting freeze,
disjoint guard/outer roles, and label-arrival reporting (ADR-0069–0070).
The opt-in `continual_global_test_seal_v5` ordinary route holds each
trained seed and its deferred test roles until all configured seeds
finish, then scores in seed order. It preserves v4 per-seed reports on
the same fixture but adds memory proportional to pending seeds. It
persists pending trained states under checkpoint format 5 without final
test hashes or scores, then validates and scores only after all configured
seeds finish. It does not define inner guard, outer selection, or a
setting search (ADR-0071–0072).

`continual_arrived_benchmark.py` takes one fixed v5 training configuration,
inner/outer split fractions, guard drop tolerance, and predeclared seeds.
It returns a separately identified v6 report with role IDs/hashes, observed
runner access and label-release events, guard decisions, method task
information, and descriptive final metrics. The app asks `infra` for each
phase's four-role split and final release, and passes only train and inner
guard roles into existing model helpers. Its ordinary path holds all
trained seed states and typed phase A/B sleep decisions until global
final-test release. `continual_arrived_checkpoint.py` stores the distinct
v6 transaction plus a versioned typed history and digest bound to its
existing role ledger. `continual_arrived_sleep_history.py` validates phase
progress, guard identity/scores, and typed facts before restoration.
`continual_arrived_transactions.py` saves model and sleep boundaries, keeps
failed attempts at a retryable `before_sleep` cursor, and resumes only
remaining updates after preflight. Ordinary v6 optionally retries a typed
error a bounded number of times; its default and checkpointed paths raise
and require an explicit later resume. These modules do not open
final source fields, select settings, implement CLI parsing, or establish a
full strict-online comparison (ADR-0074–0076, ADR-0092–0093).
The small runnable example is `scripts/run_continual_arrived_smoke.py`.

`continual_arrived_selection.py` takes a bounded, ordered candidate set
and seeds, using the v6 app training boundary to hold every unscored
candidate state. It validates equal fixed work and disjoint learning-rate
options, scores only arrived outer roles, records full trial identities,
access/work/guard/replay exposures, and freezes one choice per method
before releasing final fields for selected models. Its output is the
v7 selection ledger and selected final report. It also exposes a separate
typed sleep history for every candidate/seed and carries the selected
circadian history into final seed metrics. Each circadian trial carries
its history, and the frozen choice binds ordered candidate histories through
an independent provenance digest. Histories remain outside the original
selection score and trial digests because measured durations vary between
equivalent runs (ADR-0094–0095). `continual_arrived_selection_checkpoint.py`
defines the distinct format-8 candidate-manifest payload and source-free
header checks. `continual_arrived_selection_resume.py` embeds existing v6
seed/active transactions in that run-level cursor, validates completed
candidate states and recomputed outer trials, and freezes the choice before
the selection module can release final fields. These app modules do not
read checkpoint files, construct source splits, or tune seeds/metrics after
final results. The small runnable example is
`scripts/run_continual_arrived_selection_smoke.py` (ADR-0077–0078).

`matched_head_benchmark.py` also exposes a fixed-width capacity route. It
requires equal head widths, scheduled circadian sleep with guard rollback,
and a fixed epoch budget. It verifies initial/final head parameter counts
before final test, reports explicit capacity metadata, and enables observed
memory telemetry. It does not attribute process RSS to an individual head.
Its circadian report now carries typed sleep decisions; backprop and ordinary
PC reports carry empty histories. `torch_sleep_decisions.py` combines the
head's core event with a runner schedule and measured inner-guard facts,
retaining proposed stable IDs when a guard rejects sleep. It reads immutable
chemistry for schedule misses and does not choose evaluation data, mutate a
head, or write files (ADR-0096).

Matched-head and frozen-backbone vision guards validate finite pre/post
scores and restore the head after a rejected event or scoring error. The
vision guard remains head-only; accepted event results and historical
split/prune report counts retain their meaning. Runner attempt and rollback
counts remain outside restored learning state (ADR-0050).

`sleep_schedule.py` also owns a versioned rollback cooldown state. After
a rejected component-mode sleep, a new due attempt requires one elapsed
epoch and a later successful wake batch. Legacy mode resolves to zero
cooldown unless explicitly overridden. Matched-head and vision reports
expose attempted and suppressed counts plus the resolved setting; the
state is separate from the head snapshot (ADR-0051).

`circadian_checkpoint.py` captures an in-memory continuation boundary for
a NumPy circadian model or CPU/CUDA Torch head. The caller supplies protocol,
config, data digest, and wake/pre/post-sleep progress; the result includes
the model snapshot, optional retry state, and process RNG streams. CUDA
capture binds the actual head device and the process CUDA stream; the head
snapshot owns its local split generator. Restore validates identity, CUDA
state, and a detached candidate before changing live state (ADR-0052/0084).

`fixed_feature_checkpoint.py` defines the fixed-feature runner payload and
storage port. The CPU/CUDA three-head route saves the circadian head after each
wake batch and on both sides of sleep. It checks complete runner config,
initial-head, split, and cached train/guard/validation hashes plus report
counters before restoration. Format 2 also carries an ordered typed sleep
history, checking its cursor, role digest, and legacy counters before
restoring the head. Failed guarded attempts save typed errors at the
`before_sleep` cursor and preserve their order when the same epoch is retried.
The validator checks exact completed guard-batch prefix counts and rejects
impossible partial exposure before model restoration. The sleep transaction
restores the head and process random streams on failure; its error record
retains known guard scores and a completed core proposal, while the legacy
guard counter continues to count completed decisions (ADR-0097). Older
format-1 files are incompatible (ADR-0096). The wall-time route carries a cumulative
active deadline and pauses checkpoint I/O outside it. Full-image classifier
resume remains separate. The checkpoint-memory opt-in stores typed RSS
observations through each saved boundary; on CUDA it also stores allocator
starts and peaks bound to the process and device. The completed report
retains each segment and its maximum absolute observed peak, with no common
circadian start. Segments start after feature setup and do not attribute
shared bytes to a head (ADR-0061/0085).
The fixed-width capacity control can opt into a distinct checkpoint protocol
without a memory report, or a separate segmented RSS/allocator protocol.
Both verify unchanged capacity and guarded sleep before final test
(ADR-0053–0054, ADR-0060–0061/0085; P3.9b2b2b/c).

`seeded_vision_loader.py` owns the v3 image train-loader epoch/batch cursor.
It captures sampler and process augmentation streams, recreates skipped
logical batches on a fresh ordered DataLoader, and verifies sampler state
before yielding resumed work. It does not own dataset identity, classifier
state, checkpoint files, or any held-out role (ADR-0057).

`shared_vision_loader.py` captures the v1/v2 shared sampler's epoch-entry and
current streams. It reconstructs skipped logical batches on a fresh loader
without reseeding ordinary iterations. It does not own models, held-out data,
or files (ADR-0059).

`vision_checkpoint.py` captures detached completed model states, active
circadian classifier/cursor/report progress, and process random streams for
the CPU unmatched runner. Its inputs are the three development-role
loaders, run config, and model outcomes; its output is a versioned payload.
It does not read final test or write a file. `resnet50_benchmark.py` validates
and restores that payload before training and scores final test only when all
three variants finish. CUDA checkpoints bind the selected device's process
stream; seeded v3 also restores its outer RNG-fork stream, while the full
classifier snapshot retains the head-local split generator. Supported CPU
files omit the optional CUDA fields (ADR-0058–0059/0086).
Format-2 vision checkpoints also carry Torch sleep decisions at
active and completed cursors. `vision_sleep_history.py` validates one resolved
decision per epoch, guard role/split hash, selected-batch exposure, and
runner-owned counters before active model restoration or completed final
scoring. It does not evaluate loaders or mutate models. The shared
`torch_sleep_decisions.py` adapter describes v1 validation-guard and v2/v3
inner-guard outcomes while keeping rejected core proposals distinct from
applied work. Failed pre/core/post attempts retain completed guard-batch
exposure, known scores, and any returned core proposal at a retryable
`before_sleep` cursor. `torch_sleep_transaction.py` captures and restores
process random streams for vision and fixed-feature attempts; it does not
own model snapshots or checkpoints. Actual-device CUDA continuation now
covers fixed-feature and v1/v2/v3 unmatched-vision typed histories,
including error retries, local JSON, process/head RNG, and allocator-memory
modes (ADR-0098–0100).

`isolated_head_memory.py` runs each fixed-width head in a fresh spawned
process. Inputs are one guarded synthetic or local CIFAR-10 config and a
per-child timeout; CIFAR requires zero workers and `download=False`;
outputs are per-head setup/trainer RSS observations, cache bytes, work counts,
and hashes. The parent verifies the same feature bank and initial tensors
across children. Synthetic v1 and CIFAR-10 v2 carry distinct protocol IDs.
No child reads final test or exports a model ranking. The small repeatable
entry points are `scripts/run_isolated_head_memory_smoke.py` and
`scripts/verify_cifar_isolated_memory.py`.

`repeated_head_confirmation.py` restores a saved manifest from plain JSON
values, verifies its schema and digest, and runs the already selected heads
under fixed-data, wall-time, and isolated-memory scopes. The app layer does
not read files; the CIFAR scripts supply the saved request, selection, and
manifest and write success/failure artifacts. The manifest carries the seed,
metric, and scope decisions before final-test access (ADR-0062).

`matched_head_tuning.py` also offers an opt-in CIFAR development-only source
and an attempt observer. The source option omits the final CIFAR dataset,
labels, role IDs, and hashes during selection; the observer reports start,
completion, and failure rows to a local adapter without adding file IO to
the app layer. Neither option changes the ordinary matched-head API path
(ADR-0081).

`sleep_schedule.py` takes completed runner epochs, an epoch interval,
adaptive readiness, periodic-force policy, and sleep mode. It returns a
pure decision with periodic/adaptive readiness, whether to attempt, and
whether the core call is forced. It does not mutate a model, evaluate a
guard, or count completed sleep events. Toy and continual component runs
can attempt adaptive sleep between intervals; legacy NumPy runs preserve
their interval-only schedule. Both vision tracks use the same decision,
and disabled mode avoids sleep guard attempts. See ADR-0034.

Each runner constructs `SleepEpochProgress` from its completed outer
epoch and run cap when it calls core sleep. Continual training uses the
global completed epoch for warmup/progress while its periodic schedule
remains phase-local. The model tracks successful wake batches rather
than outer epochs; ADR-0035 records the distinction.

Existing runner `total_prunes` and `circadian_total_prunes` fields count
selected `pruned_indices` requests. A NumPy gradual mark contributes at
sleep time even if removal occurs during a later wake update; the old
fields retain this meaning for report comparability. Core `PruneOutcome`
separates requested, scheduled, and removed stable IDs (ADR-0048).

## Non-Responsibilities

`continual_replay_policy_comparison.py` accepts one fixed arrived-role
manifest with two retention policies and ordered seeds. It invokes the
existing train-only arrived boundary for each trial, holds all unscored
models until global final-test release, and reports every outcome plus
observed duplicate/replay IDs and baseline state hashes. It does not
choose a policy or read files (ADR-0103). The separate
`continual_replay_policy_checkpoint.py` defines the format-9 completed-trial
prefix and active A/B cursor; `continual_replay_policy_resume.py` validates
both against arrived roles, exact duplicate counts, replay updates, and
policy model state before training remaining work. Neither module opens
final tests (ADRs 0104–0105).

`continual_matched_replay_schedule.py` validates a new v9 planning
manifest, builds only arrived train roles in A→B order, and emits one shared
selection plus planned examples, updates, and distinct PC/circadian
inference counts at each periodic boundary. It rejects changed manifest or
train data before advancing. It does not train models, checkpoint, use
guard/selection/final labels, or claim a matched accuracy comparison
(ADR-0106).

`continual_matched_replay_runner.py` trains the three NumPy methods on
arrived A then B rows and applies the schedule's detached replay rows to
PC/backprop only after a successful guarded circadian sleep. It preflights
buffer identity before replay, checks actual sleep telemetry and wake
clocks, and returns unscored models plus per-method applied-work records.
It does not open final tests, tune a policy, or checkpoint (ADR-0107).

`continual_matched_replay_outcomes.py` takes one fixed all-policy/seed
manifest, trains through the v9 runner, validates every unscored trace
against newly derived train-only schedules, and releases all matched A/B
final roles before scoring any model. It returns every seed's existing
continual-shift metrics, per-method applied replay work, and policy
aggregates. It does not perform file IO, tune settings, or select a
winner (ADR-0108).

`continual_replay_side_effect_ablation.py` binds historical and opt-in
wake-only replay to the same two v9 retention schedules. It verifies each
unscored train-only trace, equal applied boundaries, and unchanged PC and
backprop states before releasing any final role. It scores all eight
policy/seed combinations with the existing metrics and returns every
outcome. It does not choose a policy, checkpoint, or write files; the
bounded `scripts` adapter writes local JSON (ADR-0109).

`difficulty_matched_benchmark.py` runs the fixed v11 two-seed, three-train-
condition modulation comparison for shallow NumPy and CPU Torch heads.
Inputs are a fixed manifest and deferred phase sources; outputs are all
held-out A/B accuracies, signed forgetting, per-update train-only
diagnostics, and work/role hashes. It verifies every unscored trial before
opening final source fields. It does not tune settings, choose a heuristic,
checkpoint, or write files; the `scripts` adapter writes local JSON
(ADR-0112).

`structural_rank_trial.py` runs one fixed v12 shallow-head factor cell on
sealed A/B train roles and returns unscored post-A/post-B model states,
reward scales, stable structural IDs, and actual work/capacity facts.
It removes wake or importance-history reward weighting at the app
boundary after each existing core update so the factors are independent;
it does not change a historical core, score final data, or write files.
`structural_rank_comparison.py` checks all 32 cells against the fixed
manifest and common roles before releasing final A/B roles once per
seed and reporting every outcome. It neither tunes a setting nor writes
an artifact (ADR-0114).

`sleep_trigger_trial.py` runs one fixed v13 NumPy periodic/adaptive/no-
sleep arm on the same noisy train-only batches and returns unscored
post-A/post-B states, decision facts, stream hashes, and work. It does
not score final data, tune thresholds, or write files.
`sleep_trigger_comparison.py` checks the complete twelve-cell manifest,
actual train streams, decisions, clocks, model states, and capacity
before releasing common final roles and reporting accuracy/BCE for all
cells. It does not select a trigger policy or perform file IO (ADR-0115).

`continual_trigger_replay_schedule.py` binds the fixed v14 full-stack
source, three arms, and budgets, then offers a prediction-independent
FIFO selection after every arrived train-only wake epoch. It records
train role identity, retained/selected IDs, detached replay rows, and
potential work for each NumPy method. It does not train, decide/guard
sleep, open decision/final roles, or score; the `scripts` adapter writes
only those prospective facts to new local JSON (ADR-0116).

`continual_trigger_replay_runner.py` trains one fixed v14 arm on arrived
A then B roles, preflights circadian retention/selection before each
guarded sleep decision, and gives PC/backprop the same detached replay
rows only after acceptance. It records typed events, work, structure,
and capacity but does not open final roles. The separate
`continual_trigger_replay_training_study.py` rederives all six train-only
trials and source/work facts before the `scripts` adapter saves local
JSON; it does not score or choose an arm (ADR-0117).

`continual_trigger_replay_outcomes.py` rechecks the six-trial train-only
study, releases and compares all common A/B final roles before any
prediction, then reports every method's accuracy/BCE/forgetting,
capacity, work, and paired arm contrast. It does not tune or choose an
arm, alter historical protocols, or perform file IO; the `scripts`
adapter saves the fixed local result (ADR-0118).

`versioned_v14_run.py` accepts the fixed study, scored comparison,
their original-format JSON bytes, and execution facts. It assembles a
P5.1 manifest with all source-role hashes and exact seed derivations,
rejecting changed study/payload identities. It does not capture Git,
write files, select an arm, or resume training (ADR-0120).

`v14_observation_projection.py` converts verified v14 seed-level JSON
objects into deterministic typed JSONL and final-row CSV bytes. It
checks cell/epoch order and train-only role access, and marks genuinely
unrecorded wake metrics unavailable. It does not read files, estimate
metrics, aggregate seeds, or choose an arm (ADR-0121).

`v14_artifact_report.py` accepts a completed v14 manifest and scored
outcome object after the file boundary has verified them. It preserves
the declared seed/arm/method grid and returns deterministic JSON/CSV
tables with observed final-metric mean/minimum/maximum/range, source and
track labels, and a strictly within-bundle failure count. It does not
read files, train, select a result, estimate unpublished failures, or
make causal comparisons (ADR-0137).

`v14_dashboard_projection.py` accepts a verified P5.6a summary and
renders a standalone HTML page and four PNGs with every source-order
arm/method row, fixed metric labels, observed spread, and scoped
provenance. `v14_report_plot.py` renders means and observed minimum–maximum
bars using the existing Pillow dependency. Neither module reads files,
estimates uncertainty, ranks cells, or edits historical pages (ADR-0138).

`wake_diagnostic.py` identifies the metric returned by each successful
NumPy wake update and validates its order, definition, timing, and
finite value against the six-trial work grid. `v14_measured_observations.py`
serializes that train-only grid and derives an additive measured
projection. Neither opens decision/final roles, performs another
update, or selects a result (ADR-0122).

`v14_trial_checkpoint.py` constructs and validates a typed format-10
Cartesian prefix of complete unscored v14 trials. It binds source,
config, protocol, and capture identity; rechecks trial, matched-arm,
and wake facts; strips deferred final sources before persistence; and
reconstructs fixed sources only for a full six-trial study. It does
not read checkpoint files, train, score, or release final roles
(ADR-0124).

`continual_experiment_config.py` takes an existing typed continual
preset config and explicit JSON-compatible override values. It rejects
unknown and type-invalid fields, keeps protocol/baseline/model-order
identity fixed, applies validated values, and builds a fully resolved
config record. It does not parse CLI syntax, train, or write files
(ADR-0125).

`v14_experiment_config.py` resolves only the typed `fixed-v14` preset to
the historical manifest and rejects any other ID before training. It
does not parse CLI syntax, open a source, or change the old fixed result
identity (ADR-0126).

`resnet_experiment_config.py` owns the typed historical unmatched
multi-seed preset, validates only the already exposed override fields,
and builds a complete ordered per-seed config record. It does not parse
CLI syntax, load images, run Torch training, or write files (ADR-0127).

- Low-level math routines
- CLI argument parsing
- Environment variable parsing


`experience_inbox.py` accepts trusted train-role source/label events in either
transport order and applies eligible pairs through the native learner step. It
copies inputs, bounds lifetime identities, orders each drain, records committed
work and stops on uncertain learner failure. Inputs are core metadata plus the
existing native learner/budget and an injected event clock; outputs are immutable
applied-event diagnostics. No file/network IO, held-out scoring, scientific seal,
actor/promotion or replay service belongs here. `learner_step.py` now offers
optional synchronous start/completion bookkeeping hooks; omitted hooks preserve
its existing behavior. See docs/experience-contracts.md and ADR-0199.


actor_shadow.py composes private stable actor/candidate forks with separate
read/write gates, arrived training and bounded detached-state consolidation.
Inputs: trusted forkable learner, versions, core clock/events, existing budget
and snapshot transforms; outputs: detached versioned actor results, candidate
snapshots and committed receipts. It does not promote, schedule background work,
score held-out roles or grant scientific/source authority. experience_inbox.py
exposes read-only stopped status and poisons uncertain BaseException/native
cancellation while preserving exception and budget-stop phase semantics.
See docs/actor-shadow-runtime.md and ADR-0200.


promotion_guard_evaluation.py accepts frozen versioned actor/candidate state,
declared inner guard batches/training IDs, a prospective policy and trusted
native builder/probes. It refuses role/time/overlap/stale metadata before copying
payloads, restores independent copies and measures identical inputs/utility/action
rules. Outputs are immutable scalar/identity/state-digest reports; no training,
actor mutation, final-label release or promotion ticket. Atomic serving/rollback
remains R3.4b. See docs/promotion-guards.md and ADR-0201.


## Complete serving promotion

`src/app/serving_promotion.py` owns compatible promotable serving bundles, real bounded TTL cache, configured guard issuance, exact candidate/model/generation checks, atomic publish and complete one-step rollback. `ActorShadowRuntime` optionally composes that actor and leases detached candidate state through trusted callbacks while freezing its original base version. No learner handle leaves these boundaries. Inputs are trusted native builder/probes, inner guards, arrived training IDs and passive metadata; outputs are owned frames/snapshots/local tickets/receipts. No native algorithm, physical final-release authority, arbitrary Python graph certification, IO or live latency-sharing guarantee. See ADR-0202 and docs/serving-promotion.md.


## Cooperative serving priority

`src/app/resource_sharing.py` gates actual actor predict/cached serve and individual candidate native updates. Inputs are an owned runtime, limits and trusted exact-bool resource monitor; outputs are versioned/atomic actor frames, committed polls and detached admission counters. `ExperienceInbox.drain(max_updates=..., before_each_update=...)` preserves ready/duplicate/arrival/work identity while yielding before native work; defaults retain original behavior. No hard cancellation, arbitrary allocation guarantee, hidden threads, final/outer release, durable checkpoint or live p50/p95 belongs to this leaf. See docs/resource-sharing.md and ADR-0203.


## Inbox checkpoint capture prerequisite

`ExperienceInbox.capture_cursor()` refuses in-flight drain, checks duplicate/dictionary identities and constructs a validated domain cursor before deep-copying owned payloads. It captures stopped histories without reopening them. Call from a quiescent owner (candidate gate in the future complete handoff); no independent standalone transport locking, native restore, clock/resource sampling or budget renewal. Full R3.5b restore remains unfinished; see docs/inbox-cursors.md.


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

Optional managed local admission: [guide](../managed-experience.md). Pure src/core/data_lifecycle.py metadata and src/app/managed_experience.py original authority install permanent hooks on a fresh ExperienceInbox; legacy uninstalled behavior stays intact. Requires declared training/replay consent and permissions, bounded lifetime grant counts and explicit synthetic/unverified policy. Supported checkpoint handoff retains authority. Opt-out stops future training; native/inbox/checkpoint erasure and unlearning are not claimed. No new environment variables/dependencies. Full R3.6b deletion controls remain the next extension.


R3.6a acceptance:433 passing cases,current654-file Windows/Linux types/static/AST/executed guide/source/resource gates. See artifacts/runs/r36a-data-admission-20261007/admission-summary.md and validation.json. Full R3.6/R3.6b erasure remains unchecked. Exact next action: Prospectively scope R3.6b actual native replay/inbox erasure: inspect replay snapshot identity and full InboxCursor/applied receipt references; design payload-free tombstones preserving consumed IDs, enforce retention lifetimes/quotas, invalidate pending checkpoint payload copies, and test refused resurrection across owned handoff with original clocks/budgets/gates. Start with fake deletion controls, then fixed native controls; do not call erasure parameter unlearning or erase caller-held copies by implication. Preserve original full R3.6 acceptance and durable R3.5b2/R3.7/G3/human-deferred/scientific work.


## 2026-10-07 — R3.6b1 erasure prerequisites (validation pending)

[Payload erasure primitives](../data-erasure.md): core data_erasure.py provides tombstones/counts/optional ReplayPayloadOwner port;InboxCursor format2 and private ExperienceInbox erasure preserve consumed IDs/applied work;NumPy adapters expose native whole-buffer erasure. Only raw replay references are removed;native weights/RNG/policy/counters and exposure hashes remain. Format1 unerased histories remain supported. No new dependencies/environment variables. Full original-authority deletion,quotas/lifetimes/checkpoint/promotion cleanup/non-resurrection is unfinished R3.6b2.


R3.6b1 acceptance:488 passing cases,current656-file win/linux types/static/AST/guide/source/resource gates. Evidence:artifacts/runs/r36b1-erasure-primitives-20261007/validation.json and erasure-summary.md. Full original R3.6/R3.6b/R3.6b2 remains unchecked. Exact next action: R3.6b2: prospectively scope original-authority coordinated deletion. Inspect all owners of payload copies: live/retired candidate and inbox, pending/inspected checkpoints and prepared/failed models, serving promotion/rollback bundles and pending tickets. Define owned-versus-caller copies and bounded supported native/payload measurement ports; implement deletion/expiry/opt-out coordination plus declared record/byte/time policies under original manager/candidate/serving/checkpoint leases. Tests first for partial cleanup failure and refused resurrection; retain budgets/clocks/gates/consumed IDs and weights. Keep transient/audit-only admission disabled until purge semantics pass, and original R3.6/R3.6b unchecked until the complete criteria are proven.


## 2026-10-07 — R3.6b2a retained-copy ownership (validation pending)

[Retained payload ownership](../payload-ownership.md):core metadata/reference ports and app weak registry enumerate supported actor/candidate/checkpoint/promotion owners under nonblocking all-holder quiescence. One opt-in live/lifetime registration allowance survives handoff and GC without renewal. Existing native/budget/serving semantics remain;no new dependencies/environment variables. This is a cleanup prerequisite,not complete deletion or byte/time policy;R3.6b2b retains original-authority all-copy cleanup/non-resurrection acceptance.


R3.6b2a acceptance:512 current tests,full659-file win/linux types/static/AST/guide/source/resource gates;zero NEW native work. Evidence:artifacts/runs/r36b2a-payload-ownership-20261007/validation.json and ownership-summary.md. Full original R3.6/R3.6b/R3.6b2 remains unchecked. Exact next action: R3.6b2b: prospectively scope original-authority coordinated cleanup and retention. Bind current ownership registry and manager plus native replay/inbox tombstone ports; use all-holder references under the existing nonblocking lease, deduplicate by identity, acquire the original sharing/consent authority, and avoid public methods that reacquire leased locks. Define supported native/payload byte measurement and cumulative holder/record/byte/time policies; tests first for pending checkpoint/prepared/failed/retired/promotion/rollback copies, partial cleanup failure, stopped/retired inbox ledger accounting and refused resurrection. Integrate deletion/expiry/opt-out without resetting budgets/clocks/gates/IDs or parameters; keep transient/audit-only admission disabled until actual purge semantics pass. Preserve full original R3.6/R3.6b/R3.6b2 criteria and caller-copy/RAM/unlearning limits.


### Managed cleanup integration — 2026-10-07

managed_data_lifecycle:original-authority nonblocking coordinated conservative cleanup,revocation/token invalidation and failed-cleanup retry;retired inbox ledgers preserve historical work. Inputs/outputs and non-responsibilities: [managed lifecycle guide](../managed-data-lifecycle.md). No native learning-equation/dependency/environment-variable change. Aggregate retained bytes,automatic elapsed-time purge,caller-copy deletion and unlearning remain unimplemented;full R3.6b2b is unchecked.


### Owned payload copy byte policy — 2026-10-07

payload_copy_budget:nonblocking original monotonic byte reservations;managed lifecycle integrates pre-copy admission and quiescent all-holder observations,with original handoff hooks. Inputs/outputs/non-responsibilities and extension guidance:[retained payload guide](../retained-payload-budget.md). No dependency/environment-variable change. Parameters/temporary/caller/Python/RSS memory,unlearning and automatic elapsed purge are outside this increment;full R3.6b2b unchecked.


### Elapsed retention driver — 2026-10-07

retention_expiry:one bounded original-lifecycle worker,independent sharing hold,quiescent retry and explicit stop/join/unfinished cleanup;managed lifecycle anchors original logical/elapsed clocks and blocks overdue access/publication. Inputs/outputs/non-responsibilities,tree and commands:[retention expiry guide](../retention-expiry.md). No dependency/environment change. Scalar-only auxiliary expiry remains unfinished;full R3.6 parents unchecked. Clock/OS attestation,caller-copy deletion,RAM/unlearning,durable restart are excluded.


### Owned auxiliary age correction — 2026-10-07

ManagedDataLifecycle adds one focused exact-container graph validation/retention helper;initial bind and promotion anchor any nonempty owned auxiliary content independently of measured array bytes. Copies/discard/rejection do not renew age;only actual purge resets. Inputs/outputs/limits/commands:[auxiliary retention guide](../auxiliary-retention.md). No new interface/environment/dependency. Full R3.6 audit passes;durable/R3.7/R3.8/G3 and caller/RAM/unlearning/scientific limits remain separate.


### Runtime failure sequence validation — 2026-10-07

Production app interfaces/dependencies remain unchanged. [Fault sequence guide](../runtime-failure-sequences.md) composes original actor/checkpoint/sharing/consent/copy ports in finite repeated tests and one reserved native stream. Tracing/RSS/owned-array observations remain separate;label-first semantics preserved. R3.7a finite scope accepted,full R3.7/R3.5b2 authentic process-crash/durable authority/sustained coverage unfinished. No dependency/environment/scientific change.


### recovery_coordinator

RecoveryCoordinator owns retained original registrations/witness and sequences trusted callbacks through injected inward journal/observer ports. Nonblocking local lane,commit before prepare,checks around publication,uncertain native refusal,conditional handoff and terminal cleanup. No infra imports/native controller changes. Memory-only terminal/latest-observation and a local lock do not establish durable restart or external publication ownership. See [guide](../recovery-coordinator.md).


### Coordinator durable facts

RecoveryCoordinator now persists validated observations around callbacks and stops on failure/normal close. terminal_confirmed requires exact successful readback;failure_witness and chained TerminalReportingFailure preserve independent evidence when storage is unconfirmed. Failed reentry never retries work/report. Cleanup still closes all retained registrations. No native completion/publication lease/coordinator-loss guarantee. See [guide](../recovery-terminal-authority.md).


## Guarded coordinator callbacks

RecoveryCoordinator uses inward leased publication/report ports. It persists
fresh retained registration checks before/after callback under ownership,checks
after release,and preserves terminal/commit ambiguity witnesses on failure.
No app infrastructure imports or side-effect rollback. See ../guarded-recovery-coordinator.md.


### ActorShadowRuntime.capture_consolidation_cursor

Read-only complete metadata capture under the existing nonblocking exclusive
candidate lease. Validate supported immutable records before detached copying;
retain diagnostic sharing. Stopped/retired observations are allowed without
reopening the owner. No model/inbox/budget/clock call or payload copy. Busy or
invalid-state failure releases the lease and preserves original authority.
Lifecycle capture requires its own original owner/holder leases.


### managed_lifecycle_schema

Guards exact complete native owner/lifecycle/driver/registry/copy field sets
and root identities. detach_lifecycle_record copies only validated immutable
metadata as one graph while retaining the exact original authority tuple. It
does not acquire coherent source capture or replace original authority.
The next integration must cover all driver mutators and original leases.

managed_lifecycle_capture:inputs exact original owner plus independent bounds;output complete detached lifecycle metadata/original live references under nonblocking original leases. No clock/measurement/model callbacks,weak map pruning,cleanup,encoding or restore. retention_expiry state locking excludes callbacks/joins and preserves original operation serialization.

Complete app lifecycle capture can feed LifecycleCheckpointCodec without any additional native work,clock read,measurement call or owner mutation. The byte decoder returns original capture authority references,not a newly installed manager or driver. App capture code remains unchanged.

managed_record_capture:exact original runtime/inbox schema guards;reuse original nonblocking lifecycle lease order;read full cursor without reentrant locking;validate and detach paired metadata once while every original lease remains held. No ports,clock/native/cleanup/worker/encoding/restore operation.

The accepted original common-interval managed record capture can feed ManagedRecordCheckpointCodec without an additional clock/native operation. Existing app capture/runtime/inbox/driver source bytes remain unchanged by paired encoding. Decode returns the original supplied authority tuple;it does not reinstall an owner.
