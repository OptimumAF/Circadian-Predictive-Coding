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
active deadline and pauses checkpoint I/O outside it. Full-image classifier,
CUDA, memory, and capacity routes remain outside this checkpoint mode
(ADR-0053–0054; P3.9b2b/c).

`seeded_vision_loader.py` owns the v3 image train-loader epoch/batch cursor.
It captures sampler and process augmentation streams, recreates skipped
logical batches on a fresh ordered DataLoader, and verifies sampler state
before yielding resumed work. It does not own dataset identity, classifier
state, checkpoint files, or any held-out role (ADR-0057).

`vision_checkpoint.py` captures detached completed model states, active
circadian classifier/cursor/report progress, and process random streams for
the seeded v3 CPU unmatched runner. Its inputs are the three development-role
loaders, run config, and model outcomes; its output is a versioned payload.
It does not read final test or write a file. `resnet50_benchmark.py` validates
and restores that payload before training and scores final test only when all
three variants finish. Earlier protocols and CUDA checkpoint mode remain open
(ADR-0058; P3.9c2b2).

`isolated_head_memory.py` runs each fixed-width head in a fresh spawned
process. Inputs are one guarded synthetic config and a per-child timeout;
outputs are per-head setup/trainer RSS observations, cache bytes, work counts,
and hashes. The parent verifies the same feature bank and initial tensors
across children. No child reads final test or exports a model ranking. The
small repeatable entry point is `scripts/run_isolated_head_memory_smoke.py`.

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
