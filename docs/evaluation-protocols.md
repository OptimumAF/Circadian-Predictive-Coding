# Evaluation roles and label timing

## Current guarded protocols

| Protocol | Wake inputs | Repeated decisions | Outer selection | Final reporting |
|---|---|---|---|---|
| `vision_three_head_fixed_width_process_memory_v1` | Each head in a fresh spawned process rebuilding the same frozen train/guard/validation feature roles | Same scheduled fixed-width sleep and disjoint guard rollback policy | Validation runs inside each measured trainer window; no candidate selection | No final-test loader access or test score; reports setup RSS, pretrain RSS, trainer RSS/CUDA peaks, hashes, and feature bytes |
| `vision_three_head_fixed_width_capacity_memory_v1` | One frozen feature bank, equal fixed head width and parameter count throughout, same epoch cap | Scheduled forced circadian sleep with disjoint guard rollback; no structural split/prune capacity | Disjoint validation after each head trains | Capacity invariant checked before final test; test after all heads train, with separate observed RSS/CUDA fields |
| `vision_matched_head_equal_trial_tuning_v1` | One frozen feature bank per declared seed, equal candidate count and seed set for all three heads | Disjoint guard for stopping checks and circadian sleep/rollback; target-accuracy stopping disabled | Mean validation accuracy across all declared seeds, deterministic first-candidate tie break; trial ledger records full config, hashes, guard and validation counts, work, and no test metric | Open test once per seed only after all selections are fixed; score selected heads only and keep confirmations separate from trials |
| `vision_three_head_fixed_feature_wall_time_memory_v2` | Same wall-time matched inputs/deadlines with opt-in RSS/CUDA sampling | Same disjoint guard and sleep decisions | Disjoint validation inside memory sampling, outside time budget | Test after all deadlines; separate telemetry protocol ID |
| `vision_three_head_fixed_feature_wall_time_v1` | One cached train feature bank, common per-head deadline | Disjoint guard stopping checks and circadian sleep/rollback inside each deadline | Disjoint validation after each head trains | Test after all three deadlines; reject epoch caps reached before deadline |
| `vision_three_head_fixed_feature_memory_v1` | Same epoch-limited matched inputs with opt-in RSS/CUDA sampling | Same disjoint guard and sleep decisions | Disjoint validation inside memory sampling | Test after all three heads train; separate telemetry protocol ID |
| `vision_three_head_fixed_feature_v1` | One cached train feature bank, epoch cap | Disjoint guard stopping checks and circadian sleep/rollback | Disjoint validation after each head trains | Test after all three heads train |
| `vision_two_head_fixed_feature_v1` | One cached train feature bank, epoch cap | Disjoint guard stopping checks | Disjoint validation after both heads train | Test after both heads train |
| `vision_guard_separated_seeded_unmatched_v3` | Training loader with reset shuffle/augmentation streams per model and epoch | Separate labeled guard: epoch stopping and sleep rollback | Disjoint validation, measured after each candidate trains | Test after all models train; training order and model-state hashes recorded |
| `vision_guard_separated_unmatched_v2` | Training loader only | Separate labeled guard: epoch stopping and sleep rollback | Disjoint validation, measured after each candidate trains | Official or synthetic test, after all three models train in the main runner |
| `vision_validation_unmatched_v1` | Training loader only | Validation doubles as guard | Same validation | Test after training; explicit reproduction route for the first corrected vision protocol |
| `toy_validation_v1` | Training split only | Training-derived sleep state; no held-out guard | Validation is descriptive, with no tuning loop | Test after all three models train |
| `continual_validation_v1` | Current phase's training split and its permitted replay; validated model order recorded | Training-derived sleep state; no held-out guard | Phase-local validation is descriptive | Both phases' final tests after phase B training |
| `validation_dynamics_v1` | Phase-local training splits | Training-derived state; validation only drives an offline plot | No selection in the plotter | Final test after the last training/sleep event |

The vision guard is held-out labeled data, not retained wake-training data.
Its labels are available at the start of the offline vision run. Synthetic
guard samples are generated independently from training, validation, and
test. CIFAR guard IDs are drawn from the official training source, disjoint
from wake training, outer validation, and the official test source. Guard and
outer validation use deterministic transforms. The corrected vision runner
never uses future-task examples because it has no task sequence.

The v3 route retains the same split roles as v2 but isolates model
initialization and replays shuffle and augmentation streams per epoch.
It is a distinct protocol because those streams can change results. The
v1/v2 routes remain available for reproducing their existing behavior.
The v3 trained-model hash includes backbone tensors, head parameters,
predictive-head traffic state, and circadian snapshot state plus its
structural-noise generator state. It does not serialize optimizer momentum,
so it is an end-model-state comparison rather than a resumable checkpoint.
A two-worker stochastic-image fixture replayed Torch, NumPy, and Python
transform draws on this local Windows CPU. CIFAR transforms and GPU order
behavior remain unverified. A second two-worker fixture replayed torchvision
random flip and crop views from in-memory images without a dataset download.

The default guard costs 64 synthetic images per run. At 96 × 96 × 3 float32
pixels, that is 7,077,888 bytes (6.75 MiB) of materialized image tensors,
plus labels and tensor overhead. The CIFAR guard withholds 1,000 examples from
wake training in addition to the 1,000-example outer validation holdout.
Its index subset and loader add small metadata; it shares the already loaded
deterministic CIFAR source view with outer validation, so it does not add a
third raw-image copy. Training and deterministic evaluation views remain two
separate source dataset objects. During iteration, transforms allocate batch
tensors. These data and memory costs must be included in later fairness
budgets; equal epochs alone do not equalize exposure or compute.

The NumPy corrected protocols reserve 20% of each phase's former training
portion for validation. They have no separately labeled sleep guard. Their
adaptive sleep triggers, thresholds, and replay read training-derived state
and current-phase training examples only. The toy and continual validation
scores are descriptive; they do not select a candidate. The offline dynamics
plot can display phase-B validation during phase A, but those values do not
enter training or sleep decisions. It is not a strict-online result.
The toy runner now records a validated execution-order permutation. A tiny
CPU reversal with prioritized replay and a noisy split preserves selected
replay priorities, split indices, trained state/RNG hashes, split hashes,
and metrics exactly despite unrelated NumPy RNG draws. A direct Torch CPU
head reversal preserves its noisy split indices and state/RNG hash despite
global Torch draws. This local gate does not cover CIFAR loaders, GPU
kernels, or strict-online replay; see ADR-0019.
The corrected continual runner now uses the same validated model order in
both phases, preserving its A-only state copies before B begins. A tiny
reverse-order CPU run matched all phase-role hashes, actual replay choices,
structural decisions, A-only and final state/RNG hashes, and per-model
retention/adaptation metrics. Both orders kept final-test labels sealed until
Phase B ended. This remains an offline local gate with the full A+B sleep
horizon known in advance; see ADR-0020.

## Strict-online continual protocol contract

A future strict-online runner must make phase B data, labels, validation, and
guard examples unavailable until phase A training and its decisions finish.
After phase B arrives, phase-B training and any phase-B guard may be used for
phase-B decisions; replay may contain only examples observed in past or
current training phases under its declared memory budget. An inner guard
must be disjoint from outer selection and final test. Outer selection may
use only examples already available at the selection time. Final-test labels
for all phases remain sealed until the model and configuration are frozen;
they cannot set sleep thresholds, stopping, rollback, or a winner. Record
sample IDs/hashes, guard provenance, label arrival time, retained example
count, and bytes for each run. This contract is a specification, not a claim
that the current offline dynamics plot or continual runner implements the
strict-online study.

## Training-energy diagnostic contract

The NumPy PC/circadian `energy` histories and Torch PC/circadian
`final_energy` are **training diagnostics** computed from relaxed states
using training labels. New reports and JSON/CSV rows identify their exact
formulas with `training_metric_id` or `training_energy_id`; the old numeric
fields remain for compatibility. `final_energy` is the last training batch's
value, whereas vision `final_cross_entropy` uses feedforward logits on the
final test role. NumPy hidden penalties average over all hidden units (PC)
or the final adaptive width (circadian). Torch averages output and hidden
squared residuals separately over class count and hidden width. Their
scales can move as topology changes, and none is a shared head-to-head
optimization loss. See `docs/learning-mathematics.md` and ADR-0021. Keep
selection and comparison on their declared held-out metrics.

## Shallow no-circadian parity control

For a direct core-level control, construct the one-hidden NumPy model with
`CircadianConfig.matched_pc_control()` or the one-hidden Torch head with
`CircadianHeadConfig.matched_pc_control()`, using the same seed, hidden
width, data, and rates as its ordinary PC counterpart. These named presets
fix plasticity and reward scale to one, disable sleep structure and
homeostasis, and disable NumPy replay. A forced sleep event is a no-op.
`tests/test_neutral_circadian_control.py` checks exact four-update local
CPU parity, including trained parameters and predictions, while a
two-hidden NumPy fixture deliberately records the unmatched boundary.
This is a correctness control for future ablations, not a new benchmark
protocol or a claim that binary NumPy and multiclass Torch metrics can be
compared directly. See ADR-0023 and P2.6.

## NumPy algorithm and comparison scope

New toy, in-depth, continual, and hardest-mode dynamics outputs identify
Backprop as `numpy_backprop_tanh_mlp_v1`, ordinary PC as
`numpy_pc_all_latent_fixed_prior_v1`, and circadian PC as
`numpy_circadian_final_latent_feedforward_prior_v1`. Their existing
evaluation protocol IDs still identify data roles and label timing; the
algorithm IDs identify the executed update rules. The output also includes
a `comparison_scope` with hidden depth, an explanatory scope ID, and
`causal_attribution_supported=False`.

One-hidden NumPy runs are `numpy_shallow_descriptive_v1`: the neutral core
control is verified, but toy/continual model seeds and controls differ, so
those reports are descriptive. Runs with two or more hidden layers are
`numpy_deeper_unmatched_descriptive_v1`: ordinary PC relaxes all latents and
uses local earlier-layer updates, while circadian PC relaxes its final
latent and differentiates earlier feedforward priors. A figure builder
called without a known architecture reports
`numpy_architecture_unknown_descriptive_v1`. No NumPy leaderboard or
hardest-mode plot in these routes isolates a circadian mechanism. The
one-hidden neutral-control fixture and separately matched Torch
fixed-feature track are the available attribution foundations, within
their stated budgets and validation gates. See ADR-0024; a common deeper
algorithm and matched protocol remain P2.6a.

## Cross-backend numerical fixture

P2.7 uses a local float64 CPU fixture with one hidden tanh layer. It maps
NumPy binary `sigmoid(z)` to the positive class of Torch
`softmax([-z/2,z/2])`, with identical hidden tensors and mapped output
tensors. The represented binary BCE and two-class CE agree. Actual
production forward probabilities, hidden traffic/one-step updates, and
neutral circadian chemical/plasticity state agree within the declared
float64 tolerance. A separate zero-momentum fixture checks the same
forward/hidden-update boundary for the NumPy and Torch backprop MLP heads.
Their output *parameter* updates do not: with both
Torch logits trainable at the same scalar rate, the output margin moves
twice as far as NumPy's scalar logit. This is an explicit negative parity
result, not a reason to retune rates or compare later benchmark scores.

Separate static, non-tied split-only and immediate-prune-only fixtures
compare ranking and selected indices with matched chemistry, importance,
thresholds, budgets, and zero structural noise. They check width, mapped
parameters, chemistry, and feedforward probabilities after sleep. Combined
split/prune events, stochastic noise, default float32 Torch training,
NumPy-only replay, and production binary-versus-multiclass tasks remain
outside this exact numerical claim. See ADR-0025 and
`tests/test_backend_parity_boundaries.py`.

Historical benchmark figures predate these corrected role protocols. The
reviewed vision path used final-test labels for decisions; its figures are
tagged separately in `docs/historical-benchmark-provenance.md`. The v1 vision
ID above refers to the first corrected validation route, not that historical
test-informed path.

## Fairness budget contract (P1.8 in progress)

Report three comparisons separately. For a **fixed-data/epoch** comparison,
use the same split and feature hashes, batch order, epoch cap, and stopping
rule. Disable target-accuracy stopping when equal exposure is intended;
otherwise report each model's actual epochs, examples, and wake batches.
The matched fixed-feature track now reports those counts, per-batch latent
relaxation iterations, repeated guard-example evaluations, sleep calls, and
replay examples. A guard example used in multiple evaluations counts each
time. The current matched-head route has zero replay examples.

For a **fixed-wall-time** comparison, use a common per-model deadline and
include wake updates, latent relaxation, guard evaluations, sleep, and replay
inside each timed budget. The versioned three-head wall-time route requires
`target_accuracy=None`, a positive finite deadline, and an epoch safety cap
high enough that every head reaches the deadline; otherwise it raises before
final test. Each timer starts after the shared frozen-backbone feature bank
and head initialization. It includes per-head optimizer setup, wake work,
guard checks, and circadian sleep/rollback. It stops launching operations at
batch and guard/sleep boundaries, so an operation already underway can exceed
the deadline. Reports include actual time, overshoot, completed epochs,
partial-epoch samples/batches, stop reason, and work counts. Device work is
synchronized at deadline checks. Outer validation and final test are outside
the timed scope. Shared feature extraction and head initialization must be
reported as separate setup costs; this is a head-training comparison only.
The epoch-limited route retains its prior timing scope and protocol ID.

For a **capacity/memory-aware** comparison, match or stratify initial and
final head parameters, report backbone parameters separately, and include
peak device/host memory, cached feature bytes, and any replay store in the
memory budget. Current `feature_bytes` sums materialized features and labels
for all roles; it is not a peak-memory measurement. With `measure_memory=True`,
the three-head epoch and wall-time routes use distinct protocol IDs and
report per-head starting RSS, highest observed process RSS, and sample count
during trainer work plus outer validation. The original routes leave the
sampler off so their timing behavior stays available for reproduction.
Windows/Linux sampling uses 5 ms polling plus boundary samples; unsupported
hosts report `None`. CUDA runs also report PyTorch allocator starting/peak
allocated and peak reserved bytes; the local CPU run reports `None` for them.
These are observed
process/device statistics, not memory attributable solely to one head or a
guaranteed transient peak. All heads and cached features coexist in one
process, allocator reuse can affect later baselines, and feature setup and
final-test materialization fall outside per-head windows. ADR-0014 records
these limits. The versioned fixed-width capacity route now requires the
predictive and circadian initial width and circadian minimum/maximum to be
equal, `target_accuracy=None`, scheduled forced sleep with an executable
budget, and guard-based rollback. It checks equal initial and final head
parameter counts, unchanged circadian width, no splits/prunes, and at least
one guarded sleep attempt before reading final test. `capacity_control`
records the invariant and the route always samples observed memory. This is
a small fixed-data/epoch control, not a peak-memory attribution result.
The process-isolated route now starts one child per head, checks matching
split, feature, backbone, initialization, and capacity hashes/counts, and
does not access final test. Its setup RSS window starts after Torch import
and device resolution and includes loader, backbone, feature-bank, and head
setup; `pretrain_rss_bytes` marks their resident footprint. A separate
trainer window includes wake, guard/sleep, and outer validation. Cached
train/guard/validation tensor bytes are reported separately; final-test
feature bytes are intentionally absent. Each head is alone in its child,
but process RSS still contains Python/Torch runtime, backbone, feature bank,
temporary allocations, and sampling overhead. Peaks are highest observed
values, not guaranteed instantaneous peaks. CUDA allocator fields cover
trainer work when available; no local CUDA result is claimed. The route is
limited to explicit CPU/CUDA, synthetic data, and zero loader workers for
this reproducibility gate. ADR-0016 records the fixed-width control and
ADR-0017 records this isolation boundary. The repeated confirmation route
now saves a validation-only selection and digested manifest before any final
test access. The manifest fixes one candidate per head, 3–4 new seeds,
accuracy and cross-entropy, a declared fixed-data/epoch scope, a separate common
per-head wall-time deadline, and untimed isolated-memory measurements. It
requires per-seed split/feature/backbone/initialization and fixed-width
parameter parity. The result retains all raw per-seed work and scores and
reports accuracy and observed-RSS dispersion separately; cross-entropy and
work remain in the raw reports. Its digest detects accidental changes, not
malicious fabrication. The tiny random-feature CPU smoke used selection seed
47 and confirmation seeds 53/59/61, with eight examples per role and a 0.05 s
wall-time budget. All three fixed-data means were 0.25; in the wall-time
scope the circadian mean was 0.167 versus 0.5 backprop and 0.458 PC. This is
a retained negative result, not a head ranking. Actual work and process RSS
are in the saved JSON; larger-data and real CUDA evidence remain open under
P1.8. ADR-0018 records the predeclaration rule.

Give each candidate family the same declared tuning trial count and outer
validation access. Preserve all tested seeds and negative results; use final
test only after candidate selection. Keep the unmatched image-level and
end-to-end backprop references labeled separately from the shared-feature
learning-rule comparison. The matched-head tuning route now accepts one to
eight trials per head (candidate count times seed count). All heads use the
same seeds and candidate count. Only each head's learning rates, inference
steps, or backprop momentum may vary; data, backbone, width, guard policy,
epoch cap, and metric stay fixed. A candidate is rejected if its per-head
settings duplicate another trial. The ledger records full attempted configs,
per-seed validation scores, work counts, guard exposures, and validation
example counts. A failed candidate raises with completed and failed attempt
records, completed trial rows, and no final-test access. The selection rule
averages accuracy across the complete
declared seed set; it does not choose a seed or consult final test. Test
features are materialized after selections are frozen, once per seed, and
only selected heads are confirmed. These equal search counts do not equalize
compute within a trial; the work ledger exposes that difference. The bounded
route retains all trained candidates in memory until confirmation and is a
correctness gate, not a scalable search engine. Larger-data confirmation
and real CUDA memory evidence remain open under P1.8; no head-family ranking
is established here.

Sleep-component ablations are opt-in and must record the selected mode
and switches alongside the existing protocol ID. In `legacy`, toy and
continual `sleep_event_count` retains its historical topology-change
meaning. In `components`, it counts executed events, including chemical,
homeostasis, or NumPy replay work with no width change. `disabled` records
no executed events. Toy results and formatted toy, continual, and vision
reports expose the mode. A vision guard still evaluates an attempted
no-topology event when rollback checking is enabled. These distinctions
must be retained when comparing experiments; ADR-0033 documents them.

Periodic sleep intervals are measured in completed runner epochs and
schedule an attempt, not guaranteed consolidation. With periodic forcing
enabled, an interval call bypasses the adaptive criterion but can still
be skipped by warmup; disabled mode schedules no call. Without forcing, a periodic call
requires adaptive readiness. Adaptive-ready component runs may attempt
between intervals; the NumPy legacy route remains interval-only. Vision
`sleep_attempts` therefore counts attempts, while `performed` describes
core execution; guard evaluations can occur for an attempted event that
is later skipped. Disabled mode schedules no guard attempt. ADR-0034
records this rule; width-sensitive diagnostics remain under P3.2c.

Sleep warmup and structural progress use completed runner epochs via
`SleepEpochProgress`. Adaptive minimum spacing uses successful core wake
batches: NumPy's historical `min_epochs_between_sleep` counts
`train_epoch` calls, and Torch's `min_sleep_steps` counts `train_step`
batches. NumPy replay updates do not advance that wake clock or add new
wake examples. `get_sleep_clocks()` exposes wake batches/examples,
replay updates, and performed events for audit; attempts remain a runner
scheduling decision. These clocks are not interchangeable when a runner
has multiple batches per epoch. ADR-0035 defines their units and legacy
input compatibility. Width-sensitive plateau history remains P3.2c.

In component mode, an actual hidden-width change restarts the adaptive
diagnostic window; no split/prune metric is renamed or retrospectively
normalized. While the current-width window fills, adaptive triggers
wait and the adaptive structural budget uses its configured minimum
scale. Legacy keeps its historical mixed-width window and 1.0 fallback
when too few values are present. These routes must not be pooled when
comparing trigger rates. The controlled fixture shows a diagnostic
shift caused solely by the width divisor, not an accuracy change or a
circadian advantage. ADR-0036 records the boundary.
