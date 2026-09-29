# Module: `src/app`

## Responsibilities

- Orchestrate end-to-end experiment flow
- Build comparable reports across learning approaches
- Run in-depth aggregate comparisons across seeds and dataset noise levels
- Run ResNet-50 speed/latency benchmarks with target-accuracy and circadian sleep metrics
- Run bounded, equal-trial tuning of frozen shared-feature heads with validation-only selection
- Freeze validation-selected matched-head confirmations across predeclared seed and budget scopes
- Label NumPy toy, continual, and dynamics outputs with algorithm identity and comparison scope

## Inputs / Outputs

- Inputs: `ExperimentConfig`, optional adaptation policy
- Outputs: `ExperimentResult`, `InDepthComparisonResult`, `ResNet50BenchmarkResult`,
  `MatchedHeadTuningResult`, `ProcessIsolatedMemoryResult`, and
  `RepeatedConfirmationResult`

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

`numpy_sleep_decisions.py` combines a NumPy core result with the runner's
schedule, attempt duration, and optional measured guard facts. It retains
the complete core proposal when the arrived inner guard restores a rejected
attempt. For an unscheduled epoch it reads chemistry without calling sleep.
It does not choose evaluation roles, score a guard, or change legacy counts.
`toy_checkpoint.py` stores this sequence with its model/order cursor and
rejects incomplete or old version-1 histories before model restoration.

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

- Low-level math routines
- CLI argument parsing
- Environment variable parsing
