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

`experiment_runner.py` accepts an optional validated model order in
`ExperimentConfig` for the NumPy toy comparison and records the order in its
result. Per-model seeds, role splits, and the default execution order stay
fixed. The option is for reproducibility checks, not candidate selection;
it does not make the toy and image benchmarks comparable.

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

`continual_checkpoint.py` defines the typed NumPy continual checkpoint,
phase-specific data digests, and pre-restore validation; it owns no file IO.
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

`matched_head_benchmark.py` also exposes a fixed-width capacity route. It
requires equal head widths, scheduled circadian sleep with guard rollback,
and a fixed epoch budget. It verifies initial/final head parameter counts
before final test, reports explicit capacity metadata, and enables observed
memory telemetry. It does not attribute process RSS to an individual head.

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
a NumPy circadian model or CPU Torch head. The caller supplies protocol,
config, data digest, and wake/pre/post-sleep progress; the result includes
the model snapshot, optional retry state, and process RNG streams. Restore
validates identity and a detached candidate before changing live state
(ADR-0052).

`fixed_feature_checkpoint.py` defines the CPU fixed-feature runner payload and
storage port. The CPU three-head route saves the circadian head after each
wake batch and on both sides of sleep. It checks complete runner config,
initial-head, split, and cached train/guard/validation hashes plus report
counters before restoration. The CPU wall-time route carries a cumulative
active deadline and pauses checkpoint I/O outside it. Full-image classifier
and CUDA routes remain outside this checkpoint mode. The CPU memory opt-in
stores typed RSS observations through each saved boundary; the completed
report exposes each process segment and its maximum absolute observed RSS.
The segment starts after feature setup and does not attribute shared resident
bytes to a head (ADR-0061).
The fixed-width capacity control can opt into a distinct CPU checkpoint
protocol without a memory report, or opt into a separate segmented RSS
protocol. Both verify unchanged capacity and guarded sleep before final test
(ADR-0053–0054, ADR-0060–0061; P3.9b2b2b/c).

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
three variants finish. CUDA checkpoint mode remains open
(ADR-0058–0059; P3.9c2b2b).

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

- Low-level math routines
- CLI argument parsing
- Environment variable parsing
