# Evaluation roles and label timing

## Current guarded protocols

| Protocol | Wake inputs | Repeated decisions | Outer selection | Final reporting |
|---|---|---|---|---|
| `vision_three_head_fixed_width_capacity_cuda_checkpoint_memory_v2` | Same fixed-width frozen CUDA feature bank and trusted device-bound checkpoint | Same forced guarded sleep/rollback and unchanged parameter count | Disjoint validation; checkpoint file carries no final test | Test after capacity verification; per-process RSS and CUDA allocator segments, with max absolute peaks and no common circadian start |
| `vision_three_head_fixed_width_capacity_checkpoint_memory_v1` | Same fixed-width frozen feature bank and trusted CPU checkpoint; the circadian head may span processes | Same forced guarded sleep/rollback and unchanged parameter count | Disjoint validation after each head; checkpoint file carries no final test | Test after all heads and capacity verification; typed per-process RSS segments and maximum absolute observed RSS, with no common circadian start value |
| `vision_three_head_fixed_width_capacity_checkpoint_v1` | Same fixed-width frozen feature bank and trusted CPU or CUDA checkpoint | Same forced guarded sleep/rollback and unchanged parameter count | Disjoint validation after each head | Test after all heads and capacity verification; no RSS or allocator claim |
| `vision_three_head_fixed_width_process_memory_v1` | Each head in a fresh spawned process rebuilding the same frozen train/guard/validation feature roles | Same scheduled fixed-width sleep and disjoint guard rollback policy | Validation runs inside each measured trainer window; no candidate selection | No final-test loader access or test score; reports setup RSS, pretrain RSS, trainer RSS/CUDA peaks, hashes, and feature bytes |
| `vision_three_head_fixed_width_cifar10_process_memory_v2` | Same isolated fixed-width control on a complete local CIFAR-10 cache; zero loader workers and `download=False` | Same scheduled sleep, rollback, and unchanged parameter count | Same validation-only trainer window; identical split/feature/backbone/initial hashes checked across child processes | No final-test loader access or test score; per-child setup/trainer RSS is descriptive process telemetry |
| `vision_three_head_fixed_width_capacity_memory_v1` | One frozen feature bank, equal fixed head width and parameter count throughout, same epoch cap | Scheduled forced circadian sleep with disjoint guard rollback; no structural split/prune capacity | Disjoint validation after each head trains | Capacity invariant checked before final test; test after all heads train, with separate observed RSS/CUDA fields |
| `vision_matched_head_equal_trial_tuning_v1` | One frozen feature bank per declared seed, equal candidate count and seed set for all three heads | Disjoint guard for stopping checks and circadian sleep/rollback; target-accuracy stopping disabled | Mean validation accuracy across all declared seeds, deterministic first-candidate tie break; trial ledger records full config, hashes, guard and validation counts, work, and no test metric | Open test once per seed only after all selections are fixed; score selected heads only and keep confirmations separate from trials |
| `vision_three_head_fixed_feature_wall_time_memory_v2` | Same wall-time matched inputs/deadlines with opt-in RSS/CUDA sampling | Same disjoint guard and sleep decisions | Disjoint validation inside memory sampling, outside time budget | Test after all deadlines; separate telemetry protocol ID |
| `vision_three_head_fixed_feature_wall_time_cuda_checkpoint_memory_v2` | Same CUDA matched inputs with trusted device-bound checkpoint and per-process RSS/allocator observations | Same guarded decisions and cumulative active deadline across resume | Disjoint validation inside measured trainer invocation | Test after all deadlines; max absolute RSS/allocated/reserved peaks across segments, with file I/O excluded from active deadline |
| `vision_three_head_fixed_feature_wall_time_checkpoint_memory_v1` | Same CPU wall-time matched inputs/deadlines with trusted checkpoint and per-process RSS observations | Same guarded decisions and cumulative active deadline across resume | Disjoint validation inside measured trainer invocation | Test after all deadlines; per-process RSS segments and absolute observed peak, with file I/O excluded from active deadline |
| `vision_three_head_fixed_feature_wall_time_v1` | One cached train feature bank, common per-head deadline | Disjoint guard stopping checks and circadian sleep/rollback inside each deadline | Disjoint validation after each head trains | Test after all three deadlines; reject epoch caps reached before deadline |
| `vision_three_head_fixed_feature_memory_v1` | Same epoch-limited matched inputs with opt-in RSS/CUDA sampling | Same disjoint guard and sleep decisions | Disjoint validation inside memory sampling | Test after all three heads train; separate telemetry protocol ID |
| `vision_three_head_fixed_feature_cuda_checkpoint_memory_v2` | Same CUDA fixed-epoch feature bank and trusted device-bound checkpoint | Same guarded decisions with restored process CUDA RNG and head-local split generator | Disjoint validation inside measured trainer invocation | Test after all heads train; per-process RSS/allocator segments and max absolute peaks |
| `vision_three_head_fixed_feature_checkpoint_memory_v1` | Same CPU fixed-epoch feature bank with trusted checkpoint and per-process RSS observations | Same guard and circadian sleep/rollback decisions | Disjoint validation inside measured trainer invocation | Test after all heads train; per-process RSS segments and absolute observed peak |
| `vision_three_head_fixed_feature_v1` with a CUDA checkpoint | Same frozen feature bank and trusted device-bound checkpoint | Same guard and circadian sleep/rollback decisions with restored process CUDA RNG and head-local split generator | Disjoint validation after each head; no checkpointed memory telemetry | Test after all heads train; actual-device wake/accepted/rejected continuation is verified |
| `vision_three_head_fixed_feature_v1` | One cached train feature bank, epoch cap | Disjoint guard stopping checks and circadian sleep/rollback | Disjoint validation after each head trains | Test after all three heads train |
| `vision_two_head_fixed_feature_v1` | One cached train feature bank, epoch cap | Disjoint guard stopping checks | Disjoint validation after both heads train | Test after both heads train |
| `vision_guard_separated_seeded_unmatched_v3` | Training loader with reset shuffle/augmentation streams per model and epoch | Separate labeled guard: epoch stopping and sleep rollback | Disjoint validation, measured after each candidate trains | Test after all models train; training order and model-state hashes recorded |
| `vision_guard_separated_unmatched_v2` | Training loader only | Separate labeled guard: epoch stopping and sleep rollback | Disjoint validation, measured after each candidate trains | Official or synthetic test, after all three models train in the main runner |
| `vision_validation_unmatched_v1` | Training loader only | Validation doubles as guard | Same validation | Test after training; explicit reproduction route for the first corrected vision protocol |
| `toy_validation_v1` | Training split only | Training-derived sleep state; no held-out guard | Validation is descriptive, with no tuning loop | Test after all three models train |
| `continual_phase_arrival_v2` | Current phase's training split; Phase B source is constructed after Phase A training, including in trusted-file resume | Training-derived sleep state with the reviewed full A+B schedule; no held-out guard | Phase-local validation is descriptive | Both final tests after Phase B; Phase A/B checkpoints bind only arrived development roles, with test hashes bound after training |
| `continual_phase_local_schedule_v3` | Current phase's training split; Phase B source arrives after Phase A, including in trusted-file resume | Phase A sleep uses only the Phase A epoch horizon; Phase B uses its arrived full horizon; no held-out guard | Phase-local validation is descriptive | Both final tests after Phase B; checkpoint format 3 binds only arrived development roles before scoring |
| `continual_bounded_replay_v4` | Current phase's training split; Phase B source arrives after Phase A, including in trusted-file resume | V3 phase-local A schedule; component sleep replays only content-identified observed rows under declared example/array-byte caps | Phase-local validation is descriptive | Both final tests after Phase B; format 4 validates active/frozen replay provenance before resume, and each seed reports retained IDs/count/bytes at A and B |
| `continual_global_test_seal_v5` | V4 arrived training/replay roles under one fixed config across predeclared seeds | Training-derived sleep state; one validation role remains descriptive | No candidate selection | Every final test is opened only after all configured seeds train; format 5 persists unscored states |
| `continual_arrived_roles_v6` | Phase-local train only; Phase B source after all Phase A models; original B row IDs retained after exposure cap | Arrived inner guard accepts or rolls back circadian sleep | Outer rows are disjoint and released but unused; no setting selection | Final fields released after all predeclared seeds train; format 6 resumes A/B model and sleep transactions after arrived role/access/replay preflight |
| `continual_arrived_outer_selection_v7` | Same arrived four-role source and training rule as v6 for every predeclared candidate/seed | Same inner-only guard with fixed sleep/replay work budget across candidates | A/B outer roles score every completed candidate; the fixed mean balanced objective chooses one candidate independently per method, with first-declared tie break | Ordinary and format-7 checkpointed routes freeze the full candidate manifest, all trials, and choices before any final field release; only selected models are finally scored |
| `continual_validation_v1` | Current phase's training split and its permitted replay; validated model order recorded | Training-derived sleep state; no held-out guard | Phase-local validation is descriptive | Both phases' final tests after phase B training |
| `validation_dynamics_v1` | Phase-local training splits | Training-derived state; validation only drives an offline plot | No selection in the plotter | Final test after the last training/sleep event |

Trusted unmatched-vision checkpointing retains the v1/v2/v3 protocol IDs,
training order rules, guard roles, metrics, and final-test timing. On CUDA,
the file binds the selected canonical device and its process RNG; the full
circadian classifier snapshot owns the local split generator. Seeded v3 also
stores the CUDA stream at entry to its model-local RNG fork, while v1/v2 keep
their shared CPU loader-generator cursor. Preflight rejects incompatible
device or stream state before live restoration. Bounded RTX 3080 synthetic
fixtures verify wake and accepted/rejected continuation plus a fresh-process
wake restart; ADR-0086 gives the scope and limits.

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

The P1.3a future-phase canary found that ordinary corrected runs finish all
three Phase A models before requesting the Phase B source, but the trusted
v1 checkpoint route requests that source before Phase A to hash both phases.
The new `continual_phase_arrival_v2` route keeps v1 recovery unchanged and
uses a distinct v2 checkpoint format: Phase A checkpoints bind Phase A
development roles only; Phase B development roles and the combined digest
are built at the arrival boundary. Final-test validation and hashing wait
until both training phases finish. Fresh, Phase A resume, Phase B resume, and next-seed
continuation tests enforce that boundary in both model orders, reject
altered arrived roles before an update, and seal final-test scoring until
both phases finish. The generator still materializes Phase A test data and
the sleep schedule still knows the full A+B horizon. There is no declared
retained-memory budget or label-arrival ledger. Thus v2 is a phase-source
isolation checkpoint protocol, not a full strict-online study.

`continual_phase_local_schedule_v3` is an opt-in P1.3c1 increment. It
inherits v2's phase-arrival data boundary and uses checkpoint format 3.
Its Phase A sleep progress is `completed_A_epochs / configured_A_epochs`
in ordinary and checkpointed training, so Phase B's configured duration
cannot change Phase A's split/prune budgets. Phase B may use the arrived
A+B horizon. A tiny forced-sleep fixture showed the older v2 Phase A made
one versus two splits when only Phase B duration changed from 1 to 7
epochs; v3 retained the same sleep events and trained state under both
durations, model orders, and execution paths. V1/v2 schedules are unchanged.
This is still a partial protocol: its runner configuration and checkpoint
identity know future Phase B settings, and it lacks a declared replay
budget, guard/outer-selection timing, and label/retention ledger. Scores
from this route are not full strict-online evidence; see ADR-0067.

`continual_bounded_replay_v4` adds P1.3c2's observed-example replay
boundary without changing the v1/v2/v3 configuration or checkpoint
shape. It requires explicit positive example and byte limits and
`components` sleep with replay enabled. Each retained row has a stable
SHA-256 content ID over its input and training label. The byte cap counts
copied NumPy input and target arrays only, excluding Python/deque and
checkpoint serialization overhead. The route keeps the smallest content
hashes seen so far, up to both caps; duplicate rows update one retained
entry. This fixed hash rule is independent of model errors, phase order,
and final-test outcomes, but it can retain an uneven mix of A and B and
has no learned replay selection policy. At A and B boundaries, the result
reports the actual IDs, count, bytes, and declared caps. A trusted-file
resume checks its active buffer against training examples available at
that phase and checks the frozen A buffer against A only, before another
update. Guard/outer-selection timing, a label-arrival ledger, and full
strict-online confirmation remain P1.3c3–c4; see ADR-0068.

P1.3c3a then closed a timing difference between v4 execution paths.
The ordinary path now builds A and B roles with final-test hashing
disabled and binds both test hashes only after every model finishes
Phase B, matching the checkpointed route. Raising role/hash sentinels
cover both model orders and paths. Flipping only both final-test label
arrays changed final scores and hashes without changing development-role
hashes, trained model states, sleep decisions, or replay selections.
The synthetic source generator still materializes the held-out labels
when it constructs a phase. P1.3c3b1 now lets v4's split helper carry a
deferred source reference instead of requesting its test fields during
training. Raising source input/label sentinels pass in ordinary and
checkpointed paths and both model orders. This is per-seed field release:
the generator still allocates test arrays, and a finished seed is scored
before later seeds train. Disjoint arriving inner guard/outer selection,
global setting freeze, and explicit label-release records remain P1.3c3b.
V1/v2/v3 timing and outputs are unchanged; see ADR-0069 and ADR-0070.

`continual_global_test_seal_v5` is an opt-in run-level boundary for one
declared configuration in ordinary and checkpointed paths. It trains
all A/B models
for all configured seeds before binding or scoring any final-test role.
A two-seed source sentinel confirms this in both model orders; v4
opens seed 17's final test before seed 19 trains. V5's per-seed reports
match v4 exactly on the fixed tiny fixture, and a seed-17 final-label
perturbation leaves both trained states and seed 19's report unchanged.
The pending states increase memory with seed count. The checkpoint route
uses format 5 to store only unscored completed states and development
hashes. Fresh and resumed runs keep final-test roles sealed until every
configured seed finishes, then regenerate deterministic roles for
scoring. Changed prior development data or future-seed replay in a saved
state is rejected before an update. Inner guard/outer-selection roles,
selection across settings, and an arrival ledger are still absent. V5
is not full strict-online evidence (ADR-0071–0072).

The four-role source contract in `src/infra/continual_roles.py`
deterministically separates phase-local train, inner guard, and outer
selection roles and predeclares final IDs without opening final source
fields. The opt-in ordinary `continual_arrived_roles_v6` runner uses that
contract when each phase arrives. It trains on train rows only and accepts
or rolls back circadian sleep using only the arrived inner guard. It records
actual runner release and access events, guard decisions, and per-method
available phase information. After all configured seeds train under one
fixed setting, it releases final fields and produces descriptive scores.
Phase B's exposure fraction is applied before role splitting while stable
IDs retain original source positions. The generator can allocate final
arrays before release; the source-field sentinels prove the runner does not
open them early. Its distinct format-6 route saves unscored completed seeds
and active wake, before-sleep, after-sleep, and A/B arrival transactions.
Resume checks only arrived development roles, replay, model progress, and
the event cursor before another update (ADR-0073–0076).

The separate ordinary `continual_arrived_outer_selection_v7` route takes
two to four ordered candidates and at most eight candidate-seed trials
per method. Only backprop, PC, and circadian learning rates may vary;
data/role splits, model order, inference work, epoch counts, guard policy,
sleep schedule, and replay caps stay fixed. Each method has a distinct
rate for every candidate and the same seed count. After every candidate
and seed finishes training, it measures Phase A pre/post and Phase B post
accuracy on the disjoint arrived outer roles. The fixed choice metric is
the seed mean of `0.5 * (A post + B post)`; the first declared candidate
wins an exact tie. It records all trials, training-role examples seen,
two guard evaluations per attempted sleep, outer example exposures,
arrived role IDs/hashes, access/guard/task events, and circadian replay
retention. These counts describe work; equal candidate counts and fixed
settings do not imply equal runtime or sleep work. The ordered candidate
manifest, complete trial ledger, and three independent choices are
digested into a freeze record before any final source field opens. Only
the selected model for each method receives final scoring. The ordinary
route retains candidate states in memory. Its distinct format-7 trusted
checkpoint stores full ordered candidate configs/seeds, completed unscored
v6 states and outer trial/exposure rows, and the active v6 transaction.
Resume checks the run header before source access, rehydrates and validates
all completed candidates and their recomputed outer trials before another
update, then validates the active v6 cursor. A saved frozen choice must
equal recomputed selection before final release. No final value, hash, or
score enters the file. The fixed P1.3c4 confirmation saves its complete
candidate/order/seed/objective/work request before final access and checks
ordinary against A- and B-interrupted format-7 execution in both model
orders. Its ignored JSON retains all 12 outer trials and two final seed
rows per order, full role/work/replay ledgers, state digests, and signed
circadian-minus-baseline outcomes. The seed-17 final balanced scores were
0.95/0.70/0.00 and seed-19 scores were 0.15/0.50/0.35 for
backprop/PC/circadian. Both orders and resumes matched; the circadian
method lost to PC on both seeds. This tiny synthetic confirmation validates
the protocol boundary and does not establish a representative model
ranking (ADR-0077–0079).

The separate ordinary `continual_replay_policy_comparison_v8` path fixes
one v6 arrived-role configuration, ordered seeds, FIFO and seeded bottom-k
retention, and one manifest digest before source access. Both policies see
the same A/B role hashes, baseline states, count/array-byte caps, wake
epochs, sleep schedule, and replay-step limit. All policy/seed models train
before any A/B final-test source field opens. Public results retain all
scores, retained IDs/bytes, observed duplicate IDs, distinct successfully
applied replay IDs, actual replay updates, and role/guard/sleep ledgers.
The audit's observed/exposed ID memory is outside the retained-array byte
cap. A fixed tiny two-seed run kept all four rows and found different
retention/exposure IDs but identical balanced scores for the two policies;
no winner is chosen. Format-9 trusted local continuation saves a prefix of
completed unscored policy/seed trials and at most one active A/B cursor.
Resume validates the manifest before source access and recomputes arrived
role, duplicate-ID, replay-update, and model/sleep provenance before another
update or final release. Active wake, before-sleep, and after-sleep trials
continue without repeated wake work in both model orders. Replay-capable
PC/backprop controls remain P4.4 (ADRs 0103–0105).

P4.4a starts a separate `continual_matched_replay_schedule_v9` protocol.
Its shared buffer applies FIFO or seeded bottom-k to the same arrived train
rows under the same example and copied-array byte caps. A fixed
`newest_retained_v1` sampler chooses one ordered set at each forced periodic
sleep boundary, independent of model predictions and class balancing. The
plan assigns those IDs to circadian, PC, and backprop and separately records
planned replay examples, optimizer updates, and PC/circadian inference
iterations. It opens B only after the declared A schedule and reads no
guard, outer-selection, or final-test values. The two-seed local artifact is
a schedule correctness check with no model updates, scores, or ranking;
actual matched replay controls remain P4.4 (ADR-0106).

P4.4b uses the same fixed manifest in a separate
`continual_matched_replay_training_v9` train-only runner. Before each
periodic sleep, it compares the circadian buffer's retained IDs/bytes,
ordered IDs, selected IDs, and copied row contents with the shared
schedule. The arrived inner guard determines whether circadian replay is
committed. PC and backprop get detached copies only after acceptance; a
rejected or failed sleep cannot leave an unmatched baseline replay call.
The trace distinguishes applied examples, optimizer calls, and PC versus
circadian inference iterations, and verifies that replay does not advance
wake clocks or refill memory. Its fixed two-seed/two-policy artifact has
no final scores or ranking (ADR-0107).

P4.4c binds both policies and both seeds in one
`continual_matched_replay_outcomes_v9` manifest. Every train-only trial
finishes before final-role access. A fresh schedule audit checks role,
selection, retention, applied-work, exposure, and clock facts before the
global release. All A/B final roles are then released and their IDs and
content hashes compared across policies before scoring starts. The
existing continual-shift metrics are applied to all three methods for
every fixed policy/seed. The local artifact omits volatile sleep durations
only; it retains all scores and work. The two-seed fixture is too small
for a broad ranking and must not drive new seed or metric selection
(ADR-0108).

P4.3a audits historical replay side effects independently of the P4.4 outcome
artifact. Historical NumPy replay updates chemistry,
importance/traffic, reward baseline, cooldowns, and pending-prune TTL,
but not wake age, wake counts, or energy history. Its chemical variance
can change adaptive sleep readiness. These are observed mechanisms,
not a tuned explanation for the negative matched result. P4.3b added a
versioned opt-in wake-only policy and retained the null side-effect
ablation under matched baseline work (ADR-0109).

P4.6a is a train-only signal probe of reward-named difficulty scaling.
The existing rule computes a clipped mean-absolute-error/EMA ratio, not
reinforcement reward. Fixed clean, one-label-flip, and one-feature-outlier
rows show that either corruption reaches the historical 1.5 cap in
NumPy and CPU Torch. A diagnostic 0.5 per-row error clip produces a 1.375
ratio for the corruptions; a constructed same-row post-update BCE
improvement is largest on the outlier and is not available for choosing
the current update. The unmodulated control remains at 1.0. These signal
observations are not held-out accuracy or forgetting comparisons; P4.6b
must freeze matched training and evaluation roles before any outcome
claim or new heuristic (see `docs/difficulty-modulation-audit.md`).

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

## Fairness budget contract (P1.8 scoped confirmation complete)

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

Checkpointed CUDA memory uses the separate v2 protocol IDs above. After
shared feature/head setup, each head-training invocation synchronizes its
device, reads allocated and reserved starts, and resets the PyTorch allocator
peaks once. A checkpoint reads peaks just before saving; an interrupted
process contributes only its last committed observation. The completing
process contributes its final observation. The circadian report retains
PID/device-tagged allocator and 5 ms RSS segments, takes maximum absolute
allocated, reserved, and observed RSS peaks across segments, and has no
single aggregate start. Each baseline has one segment in the completing
process. This is descriptive allocator and process telemetry, not total GPU
use or memory attributable to an individual head. Save work after a pre-save
snapshot requires a later boundary to be observed. ADR-0061 defines the RSS
scope and ADR-0085 defines the CUDA boundary and validation contract.

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
limited to explicit CPU/CUDA and zero loader workers. Synthetic runs keep
their v1 ID; verified local CIFAR-10 with `download=False` uses a distinct v2
ID. The test dataset object may be constructed during setup, but final-test
batches are not iterated in measured children. ADR-0016 records the
fixed-width control, ADR-0017 records the original isolation boundary, and
ADR-0062 records the CIFAR extension. The repeated confirmation route
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
The bounded real-CIFAR continuation used 32/16/16/16 role sizes, selection
seed 73, and confirmation seeds 83/89/97 fixed in a digest-checked manifest
before test access. A first exact-manifest attempt failed at the former
synthetic-only memory gate after fixed-data/wall-time computations, with no
complete result saved. After a versioned local-CIFAR memory gate passed, the
same manifest finished all three scopes in 58.91 s. Fixed-data test-accuracy
means were 0.125 backprop, 0.104 PC, and 0.104 circadian; wall-time means
were 0.125, 0.146, and 0.125. These 16-example-per-seed random-feature
scores are descriptive only. Every per-seed score, work count, cross-entropy,
RSS observation, and the earlier failure remain in the ignored
`artifacts/benchmark_cifar_v2_*_smoke.json` files. No seed, metric, budget,
or candidate was changed after final-test access.

The next CPU matched-head study was independently predeclared after a
no-score ImageNet ResNet-50 V2 feature-setup probe. It uses the verified
local CIFAR-10 archive and pretrained checkpoint, 32-pixel inputs,
1024/256/256/512 train/guard/outer-validation/final-test roles, equal width
16, one fixed-data epoch, selection seed 113, and two learning-rate
candidates per head (base and 0.8× base). Final test was sealed through all
six selection trials and the saved manifest fixed confirmation seeds
127/131/137, accuracy/cross-entropy metrics, and three separate scopes:
fixed-data/work, 0.5 s per-head wall-time, and process-isolated fixed-width
memory. The source selection and manifest digests were checked before any
confirmation test batch. All nine fixed-data rows, nine deadline head
reports, and nine fresh-child memory reports completed from the unchanged
manifest. Fixed-data mean test accuracy was 0.406 backprop, 0.178 PC, 0.152
circadian; equal-wall-time means were 0.579, 0.507, 0.389. The full per-seed
scores, cross-entropy, work, identity, and RSS records are retained in
ignored `artifacts/benchmark_cifar_pretrained_v1_*_smoke.json` files. The
circadian head did not lead either accuracy scope. Observed child RSS includes
setup, backbone, feature cache, and runtime overhead; it is not head-only
memory. One epoch on 1,024 training examples with 32-pixel ImageNet features
and CPU timing cannot establish a general architecture ranking. Matched CUDA
confirmation is reported below; a more representative scale remains open under P1.8;
ADR-0063 records the CPU study design and limits.

An isolated Torch 2.14.0+cu130 environment subsequently ran an actual-CIFAR
seeded v3 CUDA reversal at seed 149. The forward and reverse three-model
orders used identical 8/4/4/4 role hashes and trained-model hashes; all
declared accuracy/cross-entropy deltas were zero within the predeclared
`1e-6` tolerance. Final test stayed sealed until all three models finished
in each order, and both made one forced circadian sleep attempt. The small
run made no structural split; CPU replay/noisy-split gates remain separate
evidence. This v3 route is an unmatched image-level reference, not a
matched-head ranking. Raw reports are ignored
`data/cuda-vision-order-seed149-*.json` artifacts.

The distinct CUDA matched-head selection kept the pretrained CPU study's
roles and equal base/0.8× candidate grid, used seed 151, and sealed final
test through all six validation trials. Its saved manifest fixes seeds
157/163/167, fixed-data, 0.5 s per-head wall-time, and isolated-memory
scopes. The manifest's source-selection digest and saved manifest digest
were verified before confirmation. The first launch was deferred at
30%/27%/37% background utilization without reading final test. The later
unchanged-manifest launch met the three five-second-apart readings at
3%/3%/2% utilization with at least 7,757 MiB free, then completed all three
scopes in 98.172 s. Each seed retained three complete fixed-data/test rows,
three deadline-limited wall-time heads, and three distinct child-process
fixed-width memory reports. Training, guard, and validation hashes,
backbone state, and initial head tensors matched across scopes; final-test
hashes matched between confirmation rows and wall-time reports. All nine
memory children kept 32,954 head parameters and reported CUDA allocator
peaks and observed RSS separately. Fixed-data mean test accuracy was 0.449
backprop, 0.182 PC, and 0.160 circadian; wall-time means were 0.594, 0.410,
and 0.271. The full per-seed work, cross-entropy, overshoot, dispersion,
and memory observations are in ignored
`artifacts/benchmark_cifar_pretrained_cuda_v1_result_smoke.json`. GPU
utilization was 51% after the run, so intervening background load cannot be
ruled out for wall-time interpretation. The training role is still only
1,024 32-pixel examples. This is a retained negative circadian result at a
limited local scale, not a general architecture ranking. ADR-0064 records
the device and quiet-window decisions; the later P1.8 larger-subset study
addresses a separate frozen-feature scope below.

P1.8o's separate development-only feature probe uses an opt-in loader
that never constructs the CIFAR `train=False` source, omits final IDs and
hashes, and raises if its final loader is used. The default loader and old
benchmark outputs retain their behavior. The predeclared 224-pixel CUDA
probe used seed 173 and 4,096/512/512 train/guard/outer-validation roles;
the three quiet readings were 3%/1%/1% utilization with at least 8,139 MiB free.
It processed 128/16/16 batches in 9.522 seconds with zero final source
constructions/iterations. Observed worker RSS peaked at 1,555,025,920
bytes and the worker CUDA allocator at 447,518,208 allocated and
660,602,880 reserved bytes. These are feature-setup scopes, not matched
head training memory. The saved future request scales to 224-pixel
16,384/2,048/2,048 development roles and fixes equal-trial selection,
confirmation seeds, final role size, and separate scope limits before
new final access. Its ~33.84-second per-seed feature projection is a
cost estimate. The unchanged request then ran its six seed-179 candidates
under the saved 180-second cap in 47.551 seconds. A physical CIFAR
`train=False` construction trap and unavailable final loader kept final
labels inaccessible; zero final constructions, iterations, and scores were
recorded. Each attempt has a durable start/completion journal event, and
the complete six-trial result verifies shared role/feature/backbone/initial
hashes and equal fixed candidate configs. Outer accuracies for a/b were
0.8662109375/0.865234375 (backprop), 0.77734375/0.75732421875
(predictive), and 0.716796875/0.67236328125 (circadian); all three chose
a. The negative circadian comparison was retained without retuning. A
digest-checked manifest froze separate confirmation seeds 181/191/193
and the 240/240/600-second fixed-data/wall-time/isolated-memory limits,
with a 1,080-second total. The subsequent bounded confirmation used that
exact freeze (ADR-0080–0083).

P1.8p2's first gate restores the exact saved result, journal, and manifest
bytes before even source hashing. It checks the unchanged study request,
probe, archive, and weights; all six trial and 12 journal rows; zero final
access; the original quiet selection readings; both manifest digests; and
the already frozen three confirmation budgets. A read-only preflight
passed on the local artifacts before final access (ADR-0082).

The representative confirmation's own quiet gate read 2%/3%/8% GPU
utilization with at least 8,199 MiB free. All three frozen scopes finished:
142.232 seconds fixed-data, 178.568 seconds wall-time, and 356.843 seconds
fresh-child memory, totaling 677.735 seconds. A separate typed audit found
nine complete fixed-data training/test rows, nine five-second deadline heads,
and nine different memory-child PIDs. It verified per-seed role/feature/
backbone/initial/capacity identity, fixed-data work, deadline overshoot,
relaxation, guard, sleep and replay counts, plus separately scoped RSS/CUDA
allocator telemetry. Fixed-data mean final accuracy was 0.8793 backprop,
0.7749 predictive, and 0.7087 circadian. Wall-time means were 0.8934,
0.8757, and 0.8040. Per-seed values and population dispersion are saved in
`data/cifar-representative-confirmation-v1-result.json`. The negative
circadian outcome was retained with no post-test setting change. These are
matched shared-feature heads on a 16,384-example CIFAR-10 training subset
with 224-pixel inputs and a frozen pretrained ResNet-50; they do not rank
full-data or trainable-backbone models (ADR-0083).

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
correctness gate, not a scalable search engine. The scoped 224-pixel CUDA
confirmation above supplies larger-subset and real-device memory evidence;
no general head-family ranking is established here.

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

The fixed NumPy replay side-effect ablation uses the v9 arrived-role,
prediction-independent matched schedule, retention caps, seeds, guard,
replay update counts, and existing final metrics. Its v10 manifest adds
historical versus `wake_only_adaptive_v1` as an explicit factor. Every
trial is trained and its applied row IDs/work and PC/backprop states are
checked before any final A/B role is released. The opt-in core policy
leaves chemistry, structural usage/progress, supervised-error baseline,
and wake clocks unchanged during replay; the wrapper still records a
successful sleep event. The fixed two-seed comparison gives unchanged
balanced aggregates across side-effect policies, with circadian below
the matched baselines. This null observation is retained without changing
seeds, metrics, or the historical v9 artifact (ADR-0109).
