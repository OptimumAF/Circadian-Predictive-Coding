# Evaluation roles and label timing

## Scope and chronology

This document was checked against checkout `182077545d12d880e918f73cbf142c2279c211da`
on 2026-10-06. The inventory includes preserved older protocols as well as
later guarded and matched controls. A later implementation does not change an
older protocol's role access, scheduling, output, or acceptance evidence.
Why this: keeping that chronology explicit makes saved results reproducible.

Completed Phase 1, 4, and 5 gates establish their declared local protocol and
artifact scopes. They do not establish independently fresh seeds or sources,
native ownership, or runtime admission for a new prospective scientific run.
The owning-with guard repair P6.7d2b2j6c remains deferred at the user's request;
see the unfinished criteria in [the development plan](../DEVELOPMENT_PLAN.md).
The confirmed premature-release regression also keeps broad
P6.7d2b2j6b2 correctness reopened. Its historical bounded controls do not
grant fresh execution admission.

The current verified environment is Windows Python 3.14.7, NumPy 2.4.6,
Torch 2.14.0+cpu, and torchvision 0.29.0+cpu. CUDA observations below belong to
their saved runs. Preserve their recorded environment and explicit unknowns
in [published run environments](published-run-environments.md) and the
[published experiment register](published-experiment-register.md); current
CPU package versions do not fill missing historical GPU fields.

## Current guarded protocols

| Protocol | Wake inputs | Repeated decisions | Outer selection | Final reporting |
|---|---|---|---|---|
| `vision_three_head_fixed_width_capacity_cuda_checkpoint_memory_v2` | Same fixed-width frozen CUDA feature bank and trusted device-bound checkpoint | Same forced guarded sleep/rollback and unchanged parameter count | Disjoint validation; checkpoint file carries no final test | Test after capacity verification; per-process RSS and CUDA allocator segments, with max absolute peaks and no common circadian start |
| `vision_three_head_fixed_width_capacity_checkpoint_memory_v1` | Same fixed-width frozen feature bank and trusted CPU checkpoint; the circadian head may span processes | Same forced guarded sleep/rollback and unchanged parameter count | Disjoint validation after each head; checkpoint file carries no final test | Test after all heads and capacity verification; typed per-process RSS segments and maximum absolute observed RSS, with no common circadian start value |
| `vision_three_head_fixed_width_capacity_checkpoint_v1` | Same fixed-width frozen feature bank and trusted CPU or CUDA checkpoint | Same forced guarded sleep/rollback and unchanged parameter count | Disjoint validation after each head | Test after all heads and capacity verification; no RSS or allocator claim |
| `vision_three_head_fixed_width_process_memory_v1` | Each head in a fresh spawned process rebuilding the same frozen train/guard/validation feature roles | Same scheduled fixed-width sleep and disjoint guard rollback policy | Validation runs inside each measured trainer window; no candidate selection | No final-test feature iteration or test score; reports setup RSS, pretrain RSS, trainer RSS/CUDA peaks, hashes, and feature bytes |
| `vision_three_head_fixed_width_cifar10_process_memory_v2` | Same isolated fixed-width control on a complete local CIFAR-10 cache; zero loader workers and `download=False` | Same scheduled sleep, rollback, and unchanged parameter count | Same validation-only trainer window; identical split/feature/backbone/initial hashes checked across child processes | No final-test feature iteration or score; the default loader constructs the final dataset; per-child setup/trainer RSS is descriptive process telemetry |
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
| `continual_arrived_outer_selection_v7` | Same arrived four-role source and training rule as v6 for every predeclared candidate/seed | Same inner-only guard with fixed sleep/replay work budget across candidates | A/B outer roles score every completed candidate; the fixed mean balanced objective chooses one candidate independently per method, with first-declared tie break | Ordinary and format-8 checkpointed routes freeze the full candidate manifest, all trials, and choices before any final field release; only selected models are finally scored |
| `continual_validation_v1` | Current phase's training split and its permitted replay; validated model order recorded | Training-derived sleep state; no held-out guard | Phase-local validation is descriptive | Both phases' final tests after phase B training |
| `validation_dynamics_v1` | Phase-local training splits | Training-derived state; validation only drives an offline plot | No selection in the plotter | Final test after the last training/sleep event |

### Later fixed continual controls

| Protocol | Development roles and decisions | Selection and final release | Work and scope |
|---|---|---|---|
| `continual_replay_policy_comparison_v8` | Arrived v6 roles; circadian replay uses the declared FIFO or seeded bottom-k buffer and inner guard | Fixed policies/seeds; all trials train before final fields open | Baselines have wake training only in this preserved route |
| `continual_matched_replay_schedule_v9` | Arrived training rows supply a prediction-independent periodic replay schedule | No candidate choice or final scoring | Schedule audit only; no optimizer updates or model ranking |
| `continual_matched_replay_training_v9` | Same schedule and retained rows; the inner guard commits circadian replay before detached PC/backprop replay | No final fields or scores | Applied row IDs and optimizer counts match; inference counts are reported separately |
| `continual_matched_replay_outcomes_v9` | Rechecks every completed training trial against the fixed schedule, roles, retention, exposure, and work | Global preflight, then all final role IDs/hashes agree before any scoring | All policy/seed/method outcomes retained; unchanged metrics |
| `continual_replay_side_effect_ablation_v10` | Historical versus wake-only replay state policy under the fixed v9 rows/work and guard | All trials preflight before common final release | Fixed null ablation; no baseline, seed, or metric adjustment |
| `difficulty_matched_modulation_v11` | Fixed offline A/B roles; modulation on/off has equal wake exposure and inference, without sleep/replay | No setting selection; all trials preflight before final roles open | NumPy and CPU Torch analyzed separately; retained null accuracy/forgetting result |
| `continual_trigger_opportunities_v14` | A shared replay opportunity follows every arrived wake epoch | No model outcome or final scoring | Offered rows are distinct from applied replay work |
| `continual_trigger_replay_train_only_v14` | Periodic/adaptive/no-sleep arms use arrived roles; PC/backprop replay follows a guard-accepted circadian event | Complete fixed training grid; final fields remain unopened | All applied rows, wake/replay work, and structural lineage preflighted |
| `continual_trigger_replay_outcomes_v14` | Repeats the complete training gate | All final role IDs/hashes agree within each seed before scoring | All method/arm contrasts retained; differing accepted-event counts imply differing work |

These are separate protocol identities. Equal replay rows and optimizer calls
do not make latent-inference work, total compute, or structural capacity equal.
The detailed sections below retain their original small negative or null
observations and the limits on comparisons.

Trusted unmatched-vision checkpointing retains the v1/v2/v3 protocol IDs,
training order rules, guard roles, metrics, and final-test timing. On CUDA,
the file binds the selected canonical device and its process RNG; the full
circadian classifier snapshot owns the local split generator. Seeded v3 also
stores the CUDA stream at entry to its model-local RNG fork, while v1/v2 keep
their shared CPU loader-generator cursor. Preflight rejects incompatible
device or stream state before live restoration. Bounded RTX 3080 synthetic
fixtures verify wake and accepted/rejected continuation plus a fresh-process
wake restart; ADR-0086 gives the scope and limits.

At this checkout, runner checkpoint formats are 2 for toy, unmatched
vision, and fixed-feature heads; continual v1/v2/v3/v4/v5 use 1/2/3/4/5,
v6 uses 6, v7 uses 8, and v8 uses 9. The v14 complete-trial prefix uses
format 10. Completed vision model snapshots use format 1 separately.
Shared numbers do not make different runners' typed payloads compatible.
Each runner's additional configuration, data, and device checks still
apply. These files remain trusted local state, with the historical v7
format-7 confirmation distinguished from current format 8 below.

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
transform draws on this local Windows CPU. A second two-worker fixture
replayed torchvision random flip and crop views from in-memory images
without a dataset download. Those fixtures do not verify actual CIFAR or
GPU execution. The separate bounded actual-CIFAR v3 CUDA reversal recorded
under the fairness contract below supplies its own saved order evidence;
it does not expand these CPU fixtures into general GPU or backend parity.

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

The original NumPy `toy_validation_v1`, `validation_dynamics_v1`, and
descriptive continual v1-v5 routes reserve 20% of each phase's former
training portion for validation and have no separately labeled sleep guard.
Their adaptive sleep triggers, thresholds, and replay read training-derived
state and available training examples. Their validation scores are
descriptive; they do not select a candidate. The offline dynamics plot can
display phase-B validation during phase A, but those values do not enter
training or sleep decisions. It is not a strict-online result.

The later v6 arrived-role route creates disjoint training, inner guard,
outer selection, and final roles. V7 adds outer-only candidate selection
before global final release. V8/v9/v10/v14 replay controls reuse that
arrived role contract; their guard and scoring rules are described below.
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

The strict-online contract requires phase B data, labels, validation, and
guard examples to be unavailable until phase A training and its decisions
finish. After phase B arrives, phase-B training and its guard may be used for
phase-B decisions; replay may contain only observed past or current training
examples under its declared memory budget. An inner guard must be disjoint
from outer selection and final test. Outer selection may use only arrived
examples. Every configured trial and setting choice must finish before
final-test labels can be released; they cannot set sleep thresholds,
stopping, rollback, or a winner. Record sample IDs/hashes, guard provenance,
label arrival, retained example count, bytes, and actual work for each run.

V6/v7 implement and locally verify the arrived-role, decision, global
freeze, order, and trusted-resume boundaries described below, including the
fixed P1.3c4 confirmation. The older v1-v5 limitations remain specific to
those routes. Source-field sentinels prove access timing through the
verified runner paths; some synthetic generators already allocate held-out
arrays. They do not prove independent physical source freshness or satisfy
the still unfinished prospective native/runtime admission gates. The
offline dynamics plot remains an offline visualization.

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
update. V4 itself has no disjoint inner guard/outer selection or complete
label-arrival ledger. The later v6/v7 implementations and fixed P1.3c4
confirmation supply those local boundaries; they do not retrofit the
preserved v4 protocol. See ADR-0068 and the v6/v7 sections below.

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
before later seeds train. V4 retains that per-seed boundary. The completed
v6/v7 routes add disjoint arrived inner guard/outer selection, global
setting freeze, and explicit label-release records as described below.
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

The format-7 description and interrupted-run results above record the
original P1.3c4/ADR-0077–0079 stage. Current v7 uses checkpoint **format 8**:
it binds typed candidate/trial/frozen-selection sleep history independently
of the unchanged score/choice digest (ADR-0095). The public v7 protocol,
objective and seed identities stay fixed. The loader rejects older format-7
files; regenerate local checkpoints under format 8. Historical scores and
state comparisons remain attached to their original execution.

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
continue without repeated wake work in both model orders. V8 retains
circadian-only replay; the separate completed v9 training/outcome routes
below add replay-capable PC/backprop controls (ADRs 0103–0108).

P4.4a starts a separate `continual_matched_replay_schedule_v9` protocol.
Its shared buffer applies FIFO or seeded bottom-k to the same arrived train
rows under the same example and copied-array byte caps. A fixed
`newest_retained_v1` sampler chooses one ordered set at each forced periodic
sleep boundary, independent of model predictions and class balancing. The
plan assigns those IDs to circadian, PC, and backprop and separately records
planned replay examples, optimizer updates, and PC/circadian inference
iterations. It opens B only after the declared A schedule and reads no
guard, outer-selection, or final-test values. The two-seed local artifact is
a schedule correctness check with no model updates, scores, or ranking.
The separate completed v9 training/outcome routes below apply and score
the matched replay controls. The schedule-only protocol remains unscored
(ADRs 0106–0108).

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

The preceding P4.6a paragraph records the earlier signal probe and the
gate required at that stage. P4.6b is now complete in its fixed v11 scope:
all 24 condition/seed/arm/backend trials trained and passed the complete
role/work/diagnostic preflight before final release. All twelve matched
modulation pairs have identical final accuracy and signed forgetting at
the declared 40-row resolution, although actual update scales changed.
NumPy failed to learn B on this budget, and one seed failed A; these
underlearning limits remain part of the null result. No new heuristic
was selected. The [fixed difficulty comparison](difficulty-modulation-comparison.md)
records its unchanged manifest, full outcomes, and original result hash.

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

**Reading the saved progress notes:** P1.8 is complete in its declared
local scope, including the representative-subset CUDA confirmation
recorded later in this section. The original numeric paragraphs retain
earlier statements about remaining larger-data/CUDA work as chronology.
The later study supplies that bounded evidence; it does not establish
a full-dataset, end-to-end architecture ranking or universal GPU behavior.
All saved negative results, seed sets, objectives, and resource limits
remain unchanged.

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

Historical process-isolation and confirmation description (original wording retained):

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

Current isolated-memory scope: "does not access final test" in the preserved
description means no final-test feature iteration or scoring. The default
CIFAR loader constructs the official final dataset during setup, so the
child supplies no physical final-source absence guarantee. Construction-free
development loaders belong to their separately declared selection/probe routes.

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
records this rule; the completed P3.2c width-history boundary is described below.

Sleep warmup and structural progress use completed runner epochs via
`SleepEpochProgress`. Adaptive minimum spacing uses successful core wake
batches: NumPy's historical `min_epochs_between_sleep` counts
`train_epoch` calls, and Torch's `min_sleep_steps` counts `train_step`
batches. NumPy replay updates do not advance that wake clock or add new
wake examples. `get_sleep_clocks()` exposes wake batches/examples,
replay updates, and performed events for audit; attempts remain a runner
scheduling decision. These clocks are not interchangeable when a runner
has multiple batches per epoch. ADR-0035 defines their units and legacy
input compatibility. P3.2c's completed current-width history policy follows below.

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

The fixed v14 full-stack trigger study first offers one train-only shared
replay selection after *every* arrived wake epoch, including epochs when
adaptive or no-sleep arms will not apply it. The FIFO buffer sees only
the arrived training role; Phase B arrives after all A wake epochs, and
the common A/B final roles remain unopened. This separates potential
row supply from actual guard-committed replay exposure. Six periodic
subsets per seed match the older v9 schedule exactly; v9's
periodic-only protocol identity is unchanged. The v14 guarded train-only
runner now checks exact retained and selected rows before each decision
and applies PC/backprop replay only after the circadian guard commits an
event. All six trials are rederived and preflighted without final data.
Global final release and outcome scoring are handled by the separate
P4.8b2b gate (ADRs 0116–0118).

The completed v14 outcome route rechecks the six-trial train-only gate,
including applied structural stable IDs against saved A/B lineage,
then opens all twelve final A/B roles and compares within-seed IDs and
content hashes before scoring. Each of the eighteen method outcomes
and all paired arm contrasts is retained with accuracy, BCE, forgetting,
replay work, and capacity. Different accepted-event counts are reported
as different total work; they are not presented as matched compute.
The adaptive arm attempted no sleep at unchanged thresholds in these
fixed cells, and no winner or new heuristic was selected (ADR-0118).
An opt-in P5.1 [versioned run manifest](versioned-run-manifest.md) now
captures executing source, environment, seed derivations, all four-role
hashes, and exact payload digests for a fresh v14 run. It leaves the
fixed v14 result schema and scored bytes unchanged (ADR-0120).
P5.2a's [observed-record projection](structured-observation-audit.md)
derives typed epoch, decision, topology, replay, role-access, and final
rows only after the completed bundle verifies. Its wake rows explicitly
say per-epoch training metrics were not recorded. The derived CSV uses
all final method rows without seed/arm selection (ADR-0121).
P5.2b's opt-in [measured wake sidecar](measured-wake-observations.md)
now records the returned train-only Backprop loss and two separately
defined PC energies for every successful method/epoch update. It
preflights the complete measured work grid before the original global
final gate and publishes only beside a completed scored bundle. The
additive JSONL/CSV projection retains every P5.2a and final-only
record. No new optimization update, role read, metric selection, or
trigger change is introduced (ADR-0122).
P5.3a [atomic artifact publication](atomic-artifact-publication.md)
stages validated run and observation files outside their public
paths. Verifiers still require the completed manifest; a hidden
pending/failed/canceled stage is never a scored result (ADR-0123).
P5.3b [checked trial-prefix resume](v14-checked-resume.md) persists
only complete unscored seed/arm trials with deferred final sources
removed. It checks source/config/protocol/capture identity and exact
checkpoint bytes before another update, then repeats the six-trial
global preflight before final release. Tests interrupted after trials
1 and 3 repeat the original train, scored, and measured bytes; no
baseline, seed, metric, or trigger setting changed (ADR-0124).
P5.4 [typed continual configuration](configured-continual-experiments.md)
applies explicit, finite overrides only to the existing configurable
descriptive continual route. Its resolved artifact records the exact
config and seeds used, and its completed JSON result embeds the same
config. The v14 matched result remains fixed and is not reinterpreted
from these configurable runs (ADR-0125).


## 2026-10-06 - Saved CUDA order-control figures

See [saved order-control figures](vision-order-control-figures.md): four exact
original seed149 bodies, three pages covering scores/capacity/both timing orders
and original control facts. Nine declared score deltas are zero at original
1e-6 tolerance; all test accuracies zero, unequal capacities, timings vary.
Single-seed unmatched reference only; no fresh isolation/scientific admission.
P9.5b4 complete for saved presentation; parent P9.5/P9.5b remain open.


## 2026-10-06 - Saved CIFAR loader-control figures

See [saved loader-control figures](vision-loader-control-figures.md): two
exact original seed73 bodies, original counts and complete saved identity table.
Same-worker objects repeat; cross-worker batch IDs/role hashes equal but view
hashes differ. Only first file has a separate training_seal. Single-seed saved
metadata only, no fresh isolation/scientific admission. P9.5b5 complete for this
presentation; parents P9.5/P9.5b unfinished.


## 2026-10-06 - Saved CIFAR memory-control figures

See [saved memory-control figures](vision-memory-control-figures.md): one
exact original seed83 body, three pages preserving sampled RSS, separate cached
feature bytes, sampling/capacity/native work and nine null CUDA fields. Equal
saved head/initial/feature identities and three recorded PIDs are descriptive;
no continuous peak, memory winner or fresh isolation/scientific admission.
P9.5b6 complete for saved presentation; parents P9.5/P9.5b unfinished.


## 2026-10-06 - Saved CUDA environment-smoke figures

See [saved environment-smoke figures](vision-environment-smoke-figures.md):
four exact original seed109 failure/request/retry bodies, full failure and
synthetic telemetry. First failure preserved; retry is not independent seed
replication, dataset scoring, supported-baseline or fresh scientific evidence.
P9.5b7 complete for saved presentation; parents P9.5/P9.5b unfinished.


## 2026-10-06 - Saved development-only feature-profile figures

See [saved feature-profile figures](vision-feature-profile-figures.md):
both exact original seed101 request/result bodies, development counts/payload/
setup/weight/hash/counter facts. Recorded zero test iterations is saved metadata,
not fresh whole-lifetime isolation or held-out construction proof. No training,
accuracy, RSS/allocator or supported-baseline conclusion. P9.5b8 complete for
saved presentation; parents P9.5/P9.5b unfinished.


### P9.5b9 saved representative feasibility scope

Both original request/result whole bytes bind to frozen vision-feasibility metadata;
request SHA joins exactly, all200 original leaf facts retained. Seed173 CUDA/CIFAR10/
ImageNet/image224/batch32/development4096/512/512; complete config (including unused
head/test/null fields) stays requested scope. Original measurement9.522131000005174s,
setup1.4156498999800533s and worker12.96794860000955s remain distinct values; saved
setup_seconds already exists, no recalculation or new throughput/mean/CI.
Role batches128/16/16 and payload33587200/4198400/4198400 bytes retained. Absolute
observed RSS start759087104/peak1555025920,612 samples/interval0.01/PID23224 is
sampled process memory, not continuous peak or per-head payload. Six allocated/
reserved CUDA start/peak/end scalars separate, including recorded start zeros;
allocatedpeak447518208/reservedpeak660602880. Device free/used MiB and utilization
three pre-probe readings stay separate from after snapshot, exact UTCs retained.
Saved pre-readings satisfy original5120MiB/10percent/count3/spacing5 declarations
at their historical scope; no whole-worker availability claim. Recorded zero
final-source construction/iteration counters versus no-construct policy distinct;
not fresh lifetime isolation. Archive/weight metadata agrees but actual files/
features/role IDs are not loaded/hashed now. Current writer read only for meanings,
not original source provenance. Original full execution source/image/environment,
raw memory/GPU/access traces and outside failure inventory remain unknown.
Later study/selection/confirmation families separate; no admission, new statistic,
head training/test score/matched-baseline or independent replication claim.
See [guide](vision-feasibility-figures.md). No current evaluation gate changed.


### P9.5b10 saved development selection scope

Four complete originals bind to frozen vision-selection metadata: request/result/
manifest/12-event JSONL. All5218 original leaf facts retained. Requested CUDA/
ImageNet/image224/batch32/development16384/2048/2048, single selection seed179,
two candidates/head; declared confirmation181/191/193 and scopes distinct.
All six trial configs match original grid and all started/complete journal events
match ordered attempts/trials exactly. All a selected under original maximum
outer validation accuracy/first candidate on exact tie rule; no tie in saved data.
Original chosen values BP0.8662109375/PC0.77734375/CPC0.716796875 retained. Original
mean_validation_accuracy is single selected-seed value, trial_count2 describes
candidates, not two independent seeds. No new mean/CI/rate/metric/baseline/seed
choice/tuning. Complete unused/null config/projection/confirmation declarations
stay separate, no new extrapolation or confirmation/test score.
All32954 initial/final parameters and input/initial/backbone hashes agree in saved
metadata; trained hashes differ, actual tensors/weights/role IDs are not loaded/
verified. Native objectives and work differ: guard64/64/192, latent0/1024/1024,
sleep0/0/1 per a/b, all seen16384/wake512/epoch1/validation2048. Eight BP/PC width
nulls retained; cannot infer measured widths from config or capacity equality.
Worker47.55129189998843s versus declared180s distinct; original per-candidate
train seconds retained without full runtime/feature-cost attribution. Three pre-GPU
MiB/utilization/UTCs and after snapshot separate; no RSS/allocator/continuous peak
in this selection family. Pre-readings meet original declared gate at saved scope
only. Zero final-source counters/policy versus absent whole-lifetime trace stays
explicit. No recorded failed candidate/resume event in12-event ledger; outside
attempt/failure inventory unknown. Full execution source/image/environment and
source/archive/weight/feature arrays unverified. Current writer/reader inspected
as text for meanings only, no import/invocation. JSON canonical selection/freeze/
typed-manifest and whole request/journal digest joins checked as saved metadata,
not fresh original semantic/runtime/source/native/scientific admission. Original
confirmation list empty; later family outcomes not substituted.
[Guide](vision-selection-figures.md); existing gates/metrics unchanged.


### P9.5b11 saved synthetic repeated matched-head scope

Three complete original manifest/result/selection bodies bind to frozen
vision-synthetic metadata; all5868 leaf facts retained. CPU/synthetic3classes/
8 examples per role/image32/batch4/frozen untrained backbone/no pretrained
weights/equal32835 initial/final head parameters. Complete unused config stays
config; no CIFAR or end-to-end/full supported baseline interpretation.
Six selection candidates seed47/all complete choose a; BP/PC a validation0.125,
CPC a/b both0 with first-on-exact-tie preserved. Selection distinct from reported
confirmation53/59/61; nine fixed-data test values [0.375,0.125,0.25] tie across
heads, original means0.25/populationSD0.10206207261596575 retained. Wall original
means BP0.5/PC0.4583333333333333/CPC0.16666666666666666 retained without retuning,
new seed/metric/statistic/CI/estimator or winner claim. Nine original accuracy/
RSS summary records preserved, original population SD descriptive, not CI.
All27 head/seed/scope records retained with original native work/guard/latent/sleep/
0.05s deadline/overshoots/config/source protocol. Matched wrapper versus original
unmatched_v2 config label separate. Wall report has no embedded seed field;
series use published manifest/summary order and source indices, not fresh seed
stamps. Fixed trial/final trained hashes and saved wall/memory input/backbone/
role/test/initial hashes join; no actual arrays or fresh role/source validation.
Memory nine distinct recorded PIDs, absolute sampled setup/trainer RSS with three
trainer samples/head, not continuous peak or owned-head storage. Memory payload
196800 bytes and wall-bank262400 remain separate original role scopes, not additive
RAM. CPU CUDA nulls and BP/PC hidden widths unrecorded; two all-null panels have
no axis/extent. Complete26 displayed null entries retained without inference.
Original full environment/source/runtime/final-access/GPU/memory traces/outside
failures remain unknown; current writer read as text for meanings only. No model,
reader/verifier/dataset/weight/archive/device/CI execution or fresh isolation,
independence/source/native/scientific admission.
[Guide](vision-synthetic-smoke-figures.md). No current gate or metric change.


### P9.5b12 saved random-backbone CIFAR scope

Seven whole originals are retained: early/v2 requests and selections, the v2
failure, manifest and retry2 result. CPU CIFAR10, frozen untrained backbone,
weights none, image32/batch4, subsets32 train and16 guard/validation/test. All
saved heads have32954 initial/final parameters. Unused synthetic config stays
config; the original guard_separated_unmatched_v2 label is distinct from the
matched selection/confirmation wrapper and equal saved capacity counts.
Early request confirmation[83,89] remains distinct from v2[83,89,97]; all other
request fields agree. Both six-candidate selections73 choose a (BP0.125; PC/CPC0
with first-on-exact-tie retained). Whole selection bodies agree excluding only
train_seconds; versions are not independent replications. Saved request archive
path/170498071 bytes/MD5/source URL are declarations, not fresh archive proof.
Retained ValueError elapsed15.19s says "The local process-isolated gate requires
synthetic data and zero workers." Its digest binds the v2 manifest and embedded
retry2 manifest. Filename retry2 does not establish a complete retry chronology;
outside attempts/failures are unknown. No missing earlier outcomes invented.
All27 head/seed/scope records and nine original summary records retained. Fixed
accuracy means BP0.125, PC=CPC0.10416666666666667; wall means BP=CPC0.125,
PC0.14583333333333334. Original means/population SD only, no new estimator/CI,
seed/metric/baseline tuning or winner claim. Wall lacks embedded seed stamps:
published manifest/summary order and source index retained. Native-work/config/
guard/latent/sleep/0.05s deadline/overshoot fields stay original. Saved fixed
trial/final hashes and wall/memory backbone/initial/role/test hashes join as
metadata; actual arrays, source/runtime isolation and independence unverified.
Nine distinct recorded memory PIDs, absolute sampled setup/trainer RSS; original
trainer sample counts[3,4,4,3,4,3,3,4,4], not continuous peaks or owned-head bytes.
Memory payload524800 versus wall bank656000 have distinct role scopes and are
not summed as RAM. CPU CUDA and BP/PC width nulls remain unknown; two wholly null
panels have no axes/extent. Full original source/environment/weight/archive/
feature arrays/access/native/memory/GPU traces and outside failures unknown.
[Guide](vision-random-cifar-smoke-figures.md). No fresh gate or metric change.


### P9.5b13 saved pretrained CPU CIFAR scope

Four whole original request/selection/manifest/result bodies bind to frozen
vision-pretrained-cpu metadata. CPU CIFAR10/ImageNet pretrained frozen backbone,
image32/batch32/subsets1024 train/256 guard/256 validation/512 test. Fixed-data
and memory saved heads32954 initial/final parameters. Original unused synthetic
config remains config; original guard_separated_unmatched_v2 label, matched
wrapper and wall capacity_control null remain distinct. No inferred full baseline.
Selection113 six complete candidates choose a: BP0.39453125, PC=CPC0.171875;
PC exact tie retains first candidate. Request candidate configs/attempts/choices
and canonical selection/manifest digests join as saved metadata. Confirmation
127/131/137 separate; all27 head/seed/scope records and nine original summary
records retained. Fixed accuracy means BP0.4055989583333333, PC0.177734375,
CPC0.15234375. Wall means BP0.5787760416666666, PC0.5071614583333334,
CPC0.3893229166666667. CPC below both baselines in both saved accuracy scopes.
Only original means/population SD, no new estimator/CI/seed/metric/baseline tuning.
Original wall0.5s deadline/work/overshoots retained; wall lacks embedded seed
stamps, so manifest/summary order and source index retained without fresh stamps.
Nine distinct recorded memory PIDs, sampled absolute setup/trainer RSS; BP4/PC5/
CPC7 trainer samples per seed. Original RSS mean CPC above BP above PC, but every
seed remains visible without selecting favorable cases. Not continuous peak,
owned-head storage or additive RAM. Memory payload12595200 versus wall16793600
bytes have different original roles. CUDA nulls/BP-PC widths stay unrecorded;
two wholly null panels have no axes/extent. Config guards256 versus observed
memory guard scores64/64/192 remain distinct; no invented complete role trace.
Saved fixed trial/final trained hashes and wall/memory backbone/initial/role/test
hash joins checked without arrays. Request archive170498071 bytes/MD5/source URL
and weight102540417 bytes/SHA256/path/URL declarations retained, no archive/weight
access/validation/download. Request limits selection120/confirmation480 are config,
not measured total execution. Six selection and nine fixed-data attempts complete
in original records; outside attempt/failure inventory and retry chronology unknown.
Full original environment/source/native/runtime/weight/feature arrays/access/GPU/
continuous memory traces/independence/isolation remain unverified. Pretrained CPU
smoke is separate from random-backbone and earlier CUDA feasibility; no old
semantic reader/verifier or model invocation, fresh seal or scientific admission.
[Guide](vision-pretrained-cifar-smoke-figures.md). No fresh gate or metric change.


### P9.5b14 saved pretrained CUDA CIFAR scope

Five whole original deferred/request/selection/manifest/wrapped-result bodies
bind to frozen vision-pretrained-cuda metadata. CUDA CIFAR10/ImageNet frozen
pretrained backbone/image32/batch32/subsets1024train/256guard/256validation/512test.
Fixed-data and memory saved heads32954 initial/final parameters. Original unmatched
config label, matched wrapper and null wall capacity_control remain distinct.
Selection151 six complete candidates choose a: BP0.47265625, PC0.1484375, CPC0.125;
confirmation157/163/167 separate. Request candidate/attempt/choice/fixed-attempt/
canonical selection-manifest/hash joins checked as saved metadata; no arrays.
Fixed accuracy means BP0.44921875, PC0.181640625, CPC0.15950520833333334. Wall
means BP0.5944010416666666, PC0.4095052083333333, CPC0.2708333333333333. Original
CPC below both baselines in both scopes. Nine original summary records retain
means/population SD only; no new estimator/CI/seed/baseline/metric tuning.
RSS original means CPC above PC above BP; allocated peaks BP162577408,
PC128756736, CPC128899584 bytes, reserved BP195035136 versus PC=CPC161480704.
CPC allocator values lower than BP coexist with higher process RSS; no selected
favorable metric, sums or whole-lifetime/owned-head memory claim. All27 original
head/seed/scopes retained. Memory9distinctPIDs, original trainer sample counts
BP14/14/20, PC22/27/28, CPC43/46/54; sampled setup/trainer RSS not continuous peak.
Memory payload12595200 and wall16793600 bytes have different original roles.
Configured guards256 versus observed memory scoring64/64/192 stay distinct.
All measured CUDA peak entries retained; only eight displayed BP/PC width nulls
remain unrecorded. No wholly null plot panels, no inherited CPU CUDA-null claim.
Deferred digest binds v1 manifest and embedded wrapped result. Original deferred
final_test_iterations0 belongs to that record, not later confirmation totals.
Three deferred utilization readings30/27/37 exceed recorded10 limit; their free
MiB exceeds5120. Later quiet readings3/3/2 and post-result51 remain distinct
snapshots with complete UTCs; wrapper elapsed98.172 seconds retained. No missing
outside retry chronology/failures, continuous device trace or fresh gate proof.
Wall0.5s deadline/work/overshoots preserved; no embedded wall seed stamps, so
manifest/summary order and source indices retained. Request archive/weight path/
bytes/digests/URLs/Torch2.14.0+cu130/torchvision0.29.0+cu130/deterministic true/
CUBLAS :4096:8 declarations retained, not fresh archive/weight/environment proof.
Full original source/environment/arrays/access/native/runtime/GPU/memory traces,
independence/isolation unverified. Pretrained CUDA separate from CPU/random-
backbone/earlier feasibility; no old reader/model/device/scientific invocation.
[Guide](vision-pretrained-cuda-smoke-figures.md). No fresh gate or metric change.


## P9.5b15 — registered presentation reconciliation (2026-10-06)

This is saved presentation metadata and derived-artifact consistency validation.
All 62 original family rows and publication bodies are preserved; 13 have declared
accepted saved scopes across 14 stage groups and 15 task IDs. The canonical
P6.10 aggregate view (560 cells/626 vectors) remains separate. The other 49
families retain pending original coverage; no family receives fresh scientific
admission. 463 whole metadata files were bound before joins and 142 accepted
presentation files were checked by exact bytes/hash. This does not revalidate
all original scientific sources, unregistered binaries or private attempts.
Frozen original coverage flags are retained verbatim; current checks are separate receipts.

Concrete next validation executed: nine original legacy-multiseed-charts artifacts
bound before parsing: four PNGs, four HTMLs and docs/index.html. The four complete
PNG images were visually inspected; independent PNG checks validated every chunk
CRC and decompressed scanline extent. Full HTML bodies and exact payloads were
preserved and the three detail vectors matched the overview. Browser/CDN code
was not executed; hardest-case animation/dashboard scope remains unvalidated.
Both benchmark_multiseed_cifar100_summary.csv and benchmark_multiseed_cifar100.json
remain absent. Per-seed source filenames, seed N/IDs, uncertainty, original
execution environment and corrected-protocol provenance remain unknown.
Charts are unchanged, with nonzero throughput/latency axes explicitly unsuitable
for bar-length ratio claims. The dashboard warns about test-label-informed
stopping/sleep rollback and unmatched baselines. No recomputed means, uncertainty,
composite, chosen seeds, changed metrics or new experiment.

Evidence: `artifacts/runs/p95-coverage-reconciliation-20261006/{inputs,coverage-index,derived-chart-validation,readback,static-validation-v2,acceptance,terminal,final-accounting}.json`; guide `docs/research-presentation-index.md`. Six helper static gates pass; failed initial F401 candidate/receipt retained and charged. The v2 preformatter AST proves formatting preservation after the explicit unused-import/path repair, not equivalence with the failed candidate.

Budget: original 600 aggregate engineering seconds, 60-second hard child cap,
64 MiB owned stage, fixed 160-second manual/discovery/visual/closing reserve plus
all captured attempts including the failed lint gate. Final accounting reserves
its own full 60-second child cap; no whole-session walltime or process RSS claim.
Science 350.7925872/360 and runtime 168.7993043/180 remain spent, no reset/rekey.
Full pytest/native/Torch/CI/mypy/clean-clone and old semantic readers skipped in
this ignored helper/additive-document scope; original full gates remain required.
No installs, downloads, new dependencies, datasets, weights, arrays, device jobs,
sweeps, algorithm/config/baseline changes, publication, commit, push or merge.
G0/R0.3/full R3.1 and P9.5/P9.5b remain open. Owning-with j6c repair remains
human-deferred; R0 publication remains separate.

Why this plan change: full membership reconciliation reveals original-source coverage gaps even after accepted vision presentation scopes. Validate the already present legacy charts before selecting the next independently bound text family; retain parent acceptance and unfinished coverage. No rerender of covered scope or missing-source reconstruction.

Exact next P9.5b16: bind the frozen legacy-master-subset original docs/benchmarks/benchmark_master_cifar100_subset_2026-02-28.txt under a fresh small engineering scope before parsing; inspect the complete non-JSON text, historical test-informed protocol, baseline capacities, original units, environment/resource/failure limits and existing figures. Validate and present only uncovered saved values; preserve unknown fields and original acceptance. No model, old semantic reader, CI or scientific execution.


## P9.5b16 — historical master-subset saved presentation (2026-10-06)

P9.5b16 is a saved presentation scope. The original is 2,438 bytes, UTF-16
little-endian with BOM; whole CRLF text roundtrips exactly. SHA256:
d22ec86990860ab4a8535f93a8ac67ab221d1fd9672efd1a56f744c379ec9db2.
All 24 nonempty lines and 44 numeric literals are preserved, including the whole
decoded body/full publication record. Five metadata bodies bound before parsing.
Registered same-family paths contain only this text: no same-family existing
figure. Older multi-seed charts/P6.10 aggregate have different sources, not reused.

Original setup: CUDA, CIFAR100/root data/size96/batch32/augmentationTrue/subsets
20000/5000,12 epochs each. Trainable parameters BP204900 versus PC/CPC825316;
total BP23712932 versus PC/CPC24333348. README labels single-seed but source seed
ID is unrecorded. Historical runner used test-label-informed early stopping/sleep
rollback and unmatched head/backbone states. Historical command declares ImageNet
weights/frozen BP backbone; declarations are not actual version/weight-byte proof.
Dependency inventory, original source commit, exact weight archive, complete
execution environment, energy units, RSS/allocator/sampler, attempts/failure
history unknown. No source failure lines does not prove a complete failure ledger.

Circadian reported throughput delta -107.0 is unchanged; rounded displayed
874.2 minus981.3 is -107.1. This inconsistency is flagged without explaining it
with invented missing sources. PC delta -16.1 retained. Saved accuracy:
PC0.692>CPC0.685>BP0.678; cross-entropy CPC1.1082<PC1.1175<BP1.7144;
throughput BP981.3>PC965.2>CPC874.2; p95 PC20.77<BP23.03<CPC23.27ms.
All values/units and sleep counts/energies are retained. Unrecorded BP energy,
sleep or delta fields are not filled with zero. Six panels show exact saved
literals with explicit zero baselines/historical/unmatched limitations.
No new means/uncertainty/composite/independent replication or scientific admission.

Evidence: docs/legacy-master-subset-figures.md; artifacts/runs/p95-master-subset-20261006/{scope,inputs,view,visual-review,readback,static-validation,acceptance,terminal,final-accounting}.json and saved-values.{md,svg,png}.

Prospective local engineering scope:600 aggregate seconds,60-second hard child,
64MiB owned stage; fixed160-second manual/discovery/visual/closing reserve plus
all captured attempts. The manual documentation SyntaxError is retained separately
and included in that original fixed reserve; no reset/rekey. Final accounting
reserves its own full60-second cap. Not whole-session walltime or process RSS.
Science350.7925872/360 and runtime168.7993043/180 remain spent.
Full pytest/native/Torch/CI/mypy/clean-clone and original semantic readers skipped
in this ignored helper/additive-document scope; original full gates remain open.
Pillow12.3.0 already installed; no new dependency or install/download/model/dataset/
archive/array/device/CI/sweep/algorithm/config/baseline/seed/metric change, guard
repair, publication/commit/push/merge/delegation/other-chat message.
P9.5/P9.5b/G0/R0.3/fullR3.1 remain open; owning-with j6c human-deferred;
R0 publication separate. Source/test/architecture boundaries unchanged.

Plan rationale: uncovered master text has UTF16 encoding, unmatched capacities and an original rounding inconsistency. Preserve rather than normalize or borrow another family. Only this saved scope complete after gates, parent criteria retained. The48-epoch README family has no named original artifact and remains unresolved. Next Pareto JSON and separately registered known-mismatched summary need whole-source comparison.

Exact next P9.5b17: freeze complete legacy-pareto and legacy-pareto-summary publication/coverage metadata and whole benchmark_pareto_hard_results.json plus benchmark_pareto_hard_summary.md before parsing. Independently compare every summary claim against the JSON; preserve known mismatches, all configurations/results/seed/resource/failure/environment limits and historical test-informed protocol. Inspect registered figures and present only uncovered validated saved values. No inferred missing provenance, corrected-protocol/source/execution admission, old semantic reader/model/CI/scientific dispatch.


## P9.5b17 — historical Pareto JSON and mismatched summary (2026-10-06)

Only P9.5b17 saved JSON/summary comparison scope is complete after gates.
Two whole originals pinned before parsing: benchmark_pareto_hard_results.json
1,076,676 bytes/SHA0b5ffedfbee82c7de0a458246dd0cd0a41b0aefb26fa993126b246b639f30f10;
benchmark_pareto_hard_summary.md36,906 bytes/SHA2edd3ccf722b7f0315b7ae1e637bcc2cb9af69605b86360f614d54a511e4d067.
Full original publication bodies remain distinct. Full parsed JSON body and whole
summary text are retained, with independently checked20,210 typed leaves, all
34 primary trials,102 seed-report positions,136 duplicate ranked/front/best trial
copies and four global-report references. Duplicate presentations are not new
experiments. Stored means/std/nulls/energies/capacities/configs and all saved
fronts/rankings/winners are unchanged; no new score/front/selection/statistic.
Registered same-family paths contain those originals only, no existing figure;
older family figures/canonical P6.10 aggregate are separate and not reused.

JSON dataset declaration: hard/noise0.08/train2500/test700/classes10/image96/
CUDA/20epochs/seeds7,13,29. Summary declares14epochs and no seeds; it does not
identify the same run. Numeric seed IDs exist only in JSON dataset declaration;
individual seed reports have positions and no seed ID, so association or fresh
independence is unproved. Actual dataset name, dependency inventory, source/weight
identity, complete attempt/failure history, sampler/environment and corrected
baseline/isolation provenance unrecorded. Do not call this a corrected CIFAR run.
Both families retain historical test-informed provenance/unmatched capacities.
Original BP trainable count20490; PC527114/790666/1054218; CPC varies with source
adaptive state; copied reports preserve float aggregates and every seed-position
integer/null rather than converting types or claiming equal capacities.

All120 summary rows were compared.80 exact-parameter matches (40BP,40PC) disagree
on all320 displayed metric claims at the original summary precision.40CPC rows
have no exact JSON configuration match: summary thresholds/sleep intervals differ
from adaptive percentile/cooldown/dual-chemical/homeostasis JSON configurations.
Front sizesBP5=5,PC4=4,CPCsummary5 versusJSON4; equal counts do not prove equal
front membership. Every global-winner leaf/presence/type difference is recorded,
including summary14epoch/loss versusJSON20epoch/cross_entropy/aggregate fields.
Summary Best balanced score compared explicitly to source global_best_efficiency;
labels differ and identical score semantics are not inferred. JSON reports BP for
accuracy/train/inference globals and CPC for efficiency; summary reports BP for
all four. Preserve both claims rather than tune or select a preferred result.

Evidence: docs/legacy-pareto-figures.md; artifacts/runs/p95-pareto-20261006/{scope,inputs,view,visual-review,readback,static-validation,acceptance,terminal,final-accounting}.json, all-trials.md and three PNG/SVG pairs.

Scope600 aggregate local engineering seconds/60-second hard child/64MiB owned;
fixed160-second manual/discovery/visual/closing reserve plus every captured
attempt/failure; final command reserves its full60-second cap. No whole-session
walltime/processRSS claim. Science350.7925872/360 and runtime168.7993043/180 remain
spent, no reset/rekey. Full pytest/native/Torch/CI/mypy/clean-clone and original
scientific readers skipped for ignored helpers/additive docs; original gates open.
Pillow12.3.0 already installed; no dependencies/install/download/model/dataset/
archive/array/device/CI/sweep/baseline/seed/metric/algorithm/config change or guard
repair/publication/commit/push/merge/delegation/other-chat message.
P9.5/P9.5b/G0/R0.3/fullR3.1 remain open; j6c owning-with repair human-deferred,
R0 publication separate. All unrelated changes preserved.

Plan rationale: known summary mismatch is now checked across all claims, not only a headline. Keep the two source identities/epochs/configs distinct; preserve missing provenance and unknown report-seed association. Boolean source flag is a supported exact type, not numeric zero; malformed text is refused rather than coerced. Only b17 saved scope complete after gates, all parent criteria retained. Legacy-policy-sweep is another independently bound family; no carryover of its source or claims. Full48epoch family still lacks a named original and stays unresolved.

Exact next P9.5b18: freeze complete legacy-policy-sweep publication/coverage metadata and whole benchmark_circadian_policy_sweep_results.json before parsing under a fresh small engineering scope. Inspect all saved configurations/results/seed/resource/failure/environment/unknown limits and registered existing figures; independently validate complete bodies and present only uncovered saved values. Preserve historical test-informed protocol and all parent acceptance. No old semantic reader/model/dataset/CI/scientific dispatch or inferred missing provenance.


## P9.5b18 — historical circadian policy saved reports (2026-10-06)

This completes only P9.5b18 saved policy-report validation/presentation after gates.
The whole 53,335-byte source benchmark_circadian_policy_sweep_results.json was
bound before parsing, SHA256 cb1b0a8ed0d766d2907e44ff2e45763dc5344eda0c4aa02668eeb922f10b2d2f.
Five whole metadata bodies, full original publication record, full parsed source
and all 1,091 typed leaves are retained. All 18 original trials and 24 repeated
ranking/winner records are checked; repeated copies are not additional experiments.
All 396 report/configuration table rows and 108 plotted saved values are retained.
Registered same-family paths contain only this JSON and no preexisting figure.
The Pareto JSON, its mismatched summary, older charts and canonical P6.10 aggregate
are separate sources and were not used to fill missing values.

Original declaration: hard difficulty, noise 0.08, 2,500 training / 700 test
samples, 10 classes, image size 96, CUDA, 14 epochs, ImageNet backbone weights.
These declarations do not establish dataset name, actual archive/weight bytes,
dependency inventory, code/build identity or complete execution environment.
No seed IDs/count or per-seed reports/std/uncertainty are recorded. There is one
report per configuration; this does not prove one seed or fresh independence.
No BP/PC comparator exists here, so no matched-baseline victory is inferred.
Historical test-label-informed stopping/sleep rollback limitations remain.
Actual defaults, complete attempt/failure history, RSS/allocator/resource sampling
and energy units are unknown. No failure field is not a full attempt ledger.
Original publication's Pareto-summary mismatch warning is preserved; it does not
prove that the separate summary belongs to this policy file.

Trial 1 params={} remains empty; no current-default backfill. Trial 7/8 dual-
chemical true and trial 9 dual-chemical false/adaptive-threshold true preserve
boolean type. All saved configurations, integer counts, exact float values and
missing fields remain unchanged. Trial 3 reports hidden 384->376 with 8 splits
and 16 prunes; this contraction is preserved. All recorded rollbacks are 0,
without inferring the absence of every historical failure or rollback opportunity.
The source contains no explicit null numeric values; absent provenance fields
remain absent rather than manufactured nulls, zeros or values from other runs.

Stored accuracy winner is trial 5 (0.93), training-speed winner trial 16
(2899.73672296097 samples/s), inference-speed winner trial 15
(4850.372532536333 samples/s), balanced winner trial 3 (0.8571817530350816).
Every stored top-10 record equals its original primary record; existing rank
metric order/cutoff and winner maxima are checked against all 18 saved records.
Only these existing claims were validated: no new selection, tie rule, balanced
formula, score, mean, uncertainty, confidence claim or experiment. Tie execution
rules and balanced-score execution provenance remain unproved. All trials are
presented in original trial order, including lower-accuracy/faster configurations.
Both complete PNG pages were visually checked; 12 panels have explicit zero
ranges and original labels/units, with energy units explicitly unrecorded.

Evidence: docs/legacy-policy-sweep-figures.md; artifacts/runs/p95-policy-sweep-20261006/{scope,inputs,view,visual-review,readback,static-validation,acceptance,terminal,final-accounting}.json, all-reports.md and two PNG/SVG pairs.

Prospective engineering scope: 600 aggregate seconds, 60-second hard child cap,
64 MiB owned stage; fixed 160-second manual/discovery/visual/closing reserve plus
every captured attempt/failure. Final accounting reserves its own full 60-second
cap. Not whole-session walltime or process RSS. Science 350.7925872/360 and
runtime 168.7993043/180 remain spent without reset/rekey.
Full pytest/native/Torch/CI/mypy/clean-clone and original scientific readers skipped
in this ignored helper/additive-document scope; original full gates remain open.
Pillow 12.3.0 already installed; no dependency/install/download/model/dataset/
archive/array/device/CI/sweep/algorithm/config/baseline/seed/metric change or guard
repair/publication/commit/push/merge/delegation/other-chat message.
P9.5/P9.5b/G0/R0.3/full R3.1 remain open; owning-with j6c repair remains
human-deferred; R0 publication remains separate. Unrelated user changes preserved.

Plan rationale: this independent policy family has no comparator or seed-report data; preserve partial configuration/default/boolean provenance and every stored policy, then validate existing claims across the full file. Presentation completion does not close baseline/isolation/scientific requirements. Next legacy-tuning-hardest is independently registered and requires its own whole source/metadata validation. No parent criteria weakened; unresolved full48epoch/missing-source work preserved.

Exact next P9.5b19: freeze the full legacy-tuning-hardest publication/coverage metadata and whole benchmark_tuning_hardest_results.json before parsing under a fresh small engineering scope. Inspect every saved configuration/result/seed/resource/failure/environment/unknown limit and registered existing figure; independently validate complete bodies and present only uncovered saved values. Preserve original historical test-informed protocol and all parent acceptance. No model, original semantic reader, dataset, CI or scientific dispatch; no source borrowing or inferred independence/provenance.


## P9.5b19 — hardest-tuning original claims and saved figures (2026-10-06)

Only P9.5b19 saved source/claim validation and presentation is complete after gates.
Whole original benchmark_tuning_hardest_results.json: 38,191 bytes, SHA256
0525882a061092d29b84e3c488c5d632930861f2ce06c64f18db4601bee5d905.
Five complete metadata bodies bound before parsing. Full original publication,
whole JSON body and all 794 typed leaves retained; 24 original configurations
(BP6/PC8/CPC10), three family-best copies and four global report copies checked.
All 480 report/configuration table rows preserve original values/types/nulls.
Registered same-family paths contain only this JSON and no preexisting figures.
Pareto/summary/policy and canonical P6.10 aggregate sources remain separate.

Dataset declaration: hard/noise0.08/train2500/test700/classes10/image96/CUDA.
Every saved report records14epochs; requested epoch configuration is absent.
Dataset name, requested weights/defaults, actual archive/code/build/dependency
identity, seed IDs/count, per-seed reports/std/uncertainty and complete attempts/
failure/rollback/resource sampling/environment are unrecorded. Historical
protocol remains test-label informed with unmatched baselines/backbone states.
BP trainable parameters20490; PC527114/790666/1054218; CPC551822–1078926.
BP final metric is loss; PC/CPC final metric is energy. Units/reduction and
comparability are not proved; those scalar values are not renamed cross-entropy.
Loss and energy panels retain distinct source labels/ranges. No corrected-protocol
fairness, independent replication or scientific/source/execution admission.

Family-best copies are accuracy-selected BP trial2 (0.19), PC trial1
(0.11714285714285715), CPC trial10 (0.13428571428571429).
Source global accuracy and accuracy-per-training-second labels hold among all24
stored reports. Source global training speed and inference speed labels do not:
training record BP2=1417.3156399752304 samples/s is exceeded by BP5 and BP6;
inference record PC1=2030.5368152032515 samples/s is exceeded by BP4, PC2, CPC4
and CPC7. Every counterexample/value is retained below. All four stored global
records are maxima within the three accuracy-selected family records. This
observed narrower scope does not prove the original algorithm or tie rule; no
source winner is replaced and no new policy is selected. Negative claim evidence
is accepted for this saved validation scope; parent scientific criteria unchanged.

Existing accuracy/time and accuracy/million-trainable-parameter values match
arithmetic checks for all24 reports (48 boolean checks, relative tolerance1e-12).
Original ratio values unchanged; no new reported score/mean/std/uncertainty.
Primary BP/PC hidden start/end values are null;40 null leaves across whole source
including copies remain null.14 plotted hidden-end nulls have no numeric axis or
bars. CPC trials7/10 retain contractions384->382/512->510 with26splits/28prunes.
Rollback fields are absent, not zero-filled.Three full PNGs visually checked:
18panels,130 saved numeric labels and14 unknown null labels, explicit zero ranges
only for numeric panels. All24 trials shown in original order.

Evidence: docs/legacy-hardest-tuning-figures.md; artifacts/runs/p95-hardest-tuning-20261006/{scope,inputs,view,visual-review,readback,static-validation,acceptance,terminal,final-accounting,diagnostic-claim-control}.json, all-reports.md and three PNG/SVG pairs.

Prospective engineering scope:600 aggregate seconds,60-second hard child cap,
64MiB owned stage; fixed160-second manual/discovery/visual/closing reserve plus
all captured attempts/failures. Final command reserves its whole60-second cap;
not whole-session walltime or processRSS. Science350.7925872/360 and
runtime168.7993043/180 remain spent, no reset/rekey. One failed audit retained.
Full pytest/native/Torch/CI/mypy/clean-clone and original scientific readers skipped
in this ignored helper/additive-doc scope; original full gates remain open.
Pillow12.3.0 already installed; no dependency/install/download/model/dataset/archive/
array/device/CI/sweep/algorithm/config/baseline/seed/metric change or guard repair/
publication/commit/push/merge/delegation/other-chat message.
P9.5/P9.5b/G0/R0.3/full R3.1 remain open; owning-with j6c human-deferred,
R0 publication separate. Source/test/architecture/unrelated changes preserved.

Plan rationale: exhaustive source inspection found broad speed labels that fail over all24 records but hold over accuracy-selected family records. Preserve negative evidence and scope instead of correcting/tuning source. Acceptance is original saved validation/presentation, not a silently weakened scientific gate. Next historical continual-strength text is independently registered and distinct from corrected profile repeats. All parent criteria and unresolved missing full48epoch source retained.

Exact next P9.5b20: freeze full legacy-continual-strength publication/coverage metadata and whole docs/benchmarks/benchmark_continual_shift_strength_case_2026-02-28.txt before parsing under a fresh small engineering scope. Inspect the complete text, exact original configurations/results/protocol/seed/resource/failure/history/environment/unknown limits and registered figures; independently validate all saved values and present only uncovered views. Preserve the historical tuned profile versus corrected profile-repeat distinction. No current-default backfill, other-family borrowing, inferred independence/source/execution admission or original semantic reader/model/dataset/CI/scientific dispatch.


## P9.5b20 — original continual-strength text and saved presentation (2026-10-06)

Only this saved text validation/presentation scope is complete after its gates.
The exact original UTF-8 CRLF text is 735 bytes, SHA-256
`d2411f42266801d22f87ef4442b064c4f0f814460bd98c28091640acd886cdd4`.
Five complete metadata bodies and the original text were bound before parsing.
Nine nonempty lines, seven declared seed IDs, four setup literals, 15 original
center/+/- pairs and four Circadian sleep literals are preserved (45 numeric
literals total). The complete source body and publication record remain in the
view; 27 table rows include all values and explicitly unrecorded BP/PC sleep fields.
One PNG/SVG pair presents six panels; every label, bar, whisker coordinate and
PNG decode/pixel check passed, with the full PNG visually inspected.

The source declares seeds [3,7,11,19,23,31,37], rotation 40.0 degrees,
translation (0.90,-0.70), and Phase B train fraction 0.14. Translation units are
unrecorded. Seed IDs do not prove seven actual independent executions. Original
`+/-` is not defined as SD, SEM, CI or another statistic; figures reproduce the
notation without statistical reinterpretation. Retention and balanced formulas,
aggregation, full tuned configuration/baseline capacities, dataset/build/code/
dependencies/weights, per-seed results, execution environment, timing/memory/work
and attempt/failure history are unknown. No means, spreads, seeds or metrics were
recomputed, selected or substituted. BP/PC sleep fields remain unrecorded;
Circadian preserves sleep_events=5.00, splits=5.00, prunes=0.00, hidden_end=17.00.
The exact zero prune field has no drawn PNG bar; translation sign is retained.
Retention +/- may extend above 1; original center+/- values are retained on a
0–1.05 display range rather than clipped. Statistical meaning remains unknown.

Mixed outcomes are preserved: recorded B_post BP0.933/CPC0.930/PC0.927;
retention PC0.997/CPC0.993/BP0.985; balanced CPC0.949/PC0.947/BP0.946.
These rounded differences support no significance or fair performance ranking.
Registry kind is `historical_tuned`; corrected profile repeats are distinct.
This text does not independently establish exact stopping/test-label use.
Historical protocol limitations remain; no corrected evaluation isolation,
matched capacity, complete provenance, replication or scientific admission claim.
Original registry missing-path lists are empty, not evidence of complete execution
provenance. Only this family is covered; parent P9.5/P9.5b remain unchecked.

Evidence: [saved presentation guide](docs/legacy-continual-strength-figures.md),
`artifacts/runs/p95-continual-strength-20261006/` complete source/metadata copies,
view, all-values table, saved-summary PNG/SVG, visual-review, readback/readback-v2,
static-validation, acceptance, coverage-delta, terminal and final-accounting.
Seven malformed-text refusal controls and a signed-translation positive control
passed. No existing scientific result or publication was rewritten.

Budget: prospectively 600 aggregate engineering seconds, hard 60 seconds per
child, 64 MiB owned artifacts; fixed 160 seconds for manual/discovery/visual/
closing work plus every captured attempt. Final accounting reserves its entire
60-second command cap; this is not whole-session elapsed time or process RSS.
Science350.7925872/360 and runtime168.7993043/180 remain spent without reset.
Full pytest/native/Torch/CI/mypy/clean-clone and original scientific readers were
skipped in this ignored-helper/additive-document scope; their gates remain open.
Existing Pillow used, no new dependency/download/model/dataset/device/sweep.
Whole checkout, HEAD and installed packages preserved except the new guide and
seven reversible additive documentation edits. Owning-with repair j6c remains
human-deferred; G0/R0.3/fullR3.1 and all other unfinished tasks remain open.

Plan rationale: original text has undefined spread notation and incomplete
execution evidence. Present its exact values and unknowns without inventing
uncertainty semantics, favorable winners or corrected-profile equivalence.
This is a separate saved-presentation increment; no scientific acceptance
criterion is weakened. Remaining missing original sources/work is preserved.

Exact next P9.5b21: freeze complete legacy-continual-hardest publication/coverage metadata and whole docs/benchmarks/benchmark_continual_shift_hardest_case_2026-02-28.txt before parsing in a fresh small engineering scope. Inspect complete original text/config/results/protocol/seed/resource/failure/environment/unknowns and registered figures; independently validate saved values and present uncovered views. Preserve historical tuned versus corrected-profile-repeat distinctions; no default backfill, seed selection, source borrowing, inferred independence/admission or original semantic-reader/model/dataset/CI dispatch.


## P9.5b21 — original continual-hardest text and saved presentation (2026-10-06)

Completed scope: saved original text validation and uncovered presentation only.
Full original `docs/benchmarks/benchmark_continual_shift_hardest_case_2026-02-28.txt`
is UTF-8 CRLF, 861 bytes, SHA-256
`6f167aedc3ebbb3762612a127a8ec0341a715c37fbce2a2602fd8f02cda29f1a`.
Five complete metadata bodies and exact source bytes bound before parsing;
checkout reconciled exactly with the P9.5b20 terminal (1023 files/25 packages,
HEAD182077). Whole original publication and text retained. Ten nonempty lines,
seven seed-ID literals, eight partial setup literals, four transform/fraction
literals, 15 center/+/- pairs and four sleep literals yield 53 numeric literals.
All 35 table rows retain recorded notation, list positions and unknown fields.
One six-panel PNG/SVG pair validated by independent original values/full header,
labels, axis extents, all 15 bars/45 whisker lines/four sleep bars and PNG decode/
pixels; full PNG visually inspected for legibility and clipping.

Partial setup records hidden_dim=24, hidden_dims=[24,24,24], Phase A/B epochs
120/180, and noise0.80/1.45. Exact per-method assignment and complete configurations
are unrecorded; these declarations do not prove matched baseline capacities.
Transform rotation68.0deg, translation(1.60,-1.30), train fraction0.05 retained;
translation units unknown. Seed IDs[3,7,11,19,23,31,37] are declarations, not evidence
of seven actually executed independent runs. `+/-` definition is unknown: no SD,
SEM, CI or aggregation formula inferred. Retention/balanced formulas and per-seed
outputs, actual dataset/source/code/build/dependencies/weights/device/environment,
time/memory/work/attempt/failure/rollback history are unknown. Original fractional
Circadian scalars sleep_events24.43/splits48.57/prunes0.00/hidden_end72.57 remain
fractional; neither rounded to event counts nor assigned a new aggregation meaning.
BP/PC sleep fields unrecorded, not zero. The zero prune bar has width0 in SVG and
no colored PNG bar. No source statistic, seed, metric or baseline was changed.

Recorded mixed outcomes preserved: PC A_post0.793 > CPC0.784 > BP0.699;
PC retention0.815 > CPC0.804 > BP0.718; CPC B_post0.841 > PC0.823 > BP0.808;
CPC balanced0.812 > PC0.808 > BP0.753. These rounded summaries/undefined spreads
support no significance, independent replication or fair general ranking.
Registered kind `historical_tuned`, distinct from corrected profile repeats and
the earlier hardest animation. Exact stopping/test-label access is not established
by this text; historical protocol limitations remain. No corrected isolation,
matched capacity, complete execution/source provenance or scientific admission.
Registry empty missing-path lists do not establish execution completeness.

Evidence: `docs/legacy-continual-hardest-figures.md`;
`artifacts/runs/p95-continual-hardest-20261006/` whole source/metadata copies, view,
all-values table, saved-summary PNG/SVG, visual-review/readback/static-validation/
acceptance/coverage-delta/terminal/final-accounting receipts. Nine malformed-text
refusal controls and a signed-translation positive control passed. One failed
audit retained/charged; its helper expected10 lines per panel instead of9 (three
whisker lines times three methods). Small assertion repair changed no source,
view or figures. Failed helper candidate retained; final six-helper AST comparison
uses the complete snapshot after logic repair, not the earlier failed candidate.

Engineering budget: prospective600 aggregate seconds/hard60 per child/64MiB
owned stage, fixed160 manual/discovery/visual/closing reserve plus all captures
including failure; final accounting reserves its full60-second cap. Not entire
session elapsed time or processRSS. Science350.7925872/360 and
runtime168.7993043/180 remain spent, no reset. Full pytest/native/Torch/CI/mypy/
clean-clone and original scientific-reader gates skipped for this ignored-helper/
additive-document scope; original gates stay open. Existing Pillow, no dependency
installation/download/sweep/model/dataset/device dispatch or publication.
Whole checkout/source/test/architecture/unrelated user changes preserved except
new guide and seven reversible additive docs. Parent P9.5/P9.5b/G0/R0.3/fullR3.1
remain open; owning-with j6c human-deferred. Unfinished original-source work stays.

Why this plan increment: the hardest text includes additional partial setup and
fractional sleep fields; exact literals and unknown statistical/configuration
meaning need explicit preservation. No scientific criterion is weakened.
Next historical animation has original registered visuals requiring validation;
illustrative-circadian remains a separate uncompleted family, not experiment data.

Exact next P9.5b22: bind complete legacy-hardest-animation publication/coverage metadata and whole docs/figures/hardest_mode_dynamics.gif plus docs/figures/interactive_hardest_mode_dynamics.html before decoding under a fresh small engineering scope. Inspect every original GIF frame and complete static HTML data/configuration/label payload, exact original source/configuration/seed/protocol/environment/failure/missing/unknown provenance and registered figures. Validate and document original presentation without browser execution or inferred telemetry/current-default reconstruction. Keep earlier test-informed visualization, tuned hardest text, illustrative-circadian and corrected repeats separate; no model/scientific-reader/dataset/CI dispatch or admission.


## P9.5b22 — original earlier hardest visualization validation (2026-10-06)

Completed original visualization validation scope, not scientific execution or
browser functionality certification. Complete sources bound before decoding:
GIF3,530,592 bytes/SHA256 aa338c5051f3cb90447df0b2e972ea6b93a9e08a2e2707d89822e3bb11504ee8;
HTML21,352,286 bytes/SHA256 222abd3f0281ece0b6404fb4ed72afde470c2a055df02b0f6e55edf268d42b7e.
Five whole metadata bodies and exact source copies retained. Reconciled against
P9.5b21 terminal:1024 nonignored files/25 packages/HEAD182077, full AGENTS/plan/log.
Registry `legacy-hardest-animation`, kind `historical_visualization`, earlier
profile/test-informed frames, original invocation/dependencies unknown. Separate
from the tuned hardest text, illustrative-circadian and corrected profile repeats.

All76 GIF frames decoded at1180x680 RGB, all canonical pixel hashes/durations/
disposal values independently checked. Every duration120ms and loop0, recorded
encoding properties rather than training wall time. Four contact sheets cover all
76frames with every thumbnail pixel checked; full original RGB copies at indices
0/30/31/75 (epochs1/120/124/300) match exact decoded pixels. All contact sheets
visually inspected; those four full originals inspected for labels/structure.
Original metric footer extends past right edge and clips Circadian latency in all
four full samples. Original unchanged. Thumbnail overview does not certify every
small numeric GIF label. HTML contains all recorded per-frame scalars separately.

Complete embedded JSON span:21,338,385 bytes; full source retains every array
without copying a second21MB payload. View records exact byte offsets, raw/canonical
payload identity, all root values, all76 scalar records, and identities/shapes/
typed counts for every full frame/array. Independent whole-source parser checked
1,035,935 leaves (13,856ints/1,022,000floats/79strings), no boolean/null coercion.
Every decision map110x110;175 Phase-B test points/labels/predictions per frame,
24 adaptive input values and24-by-hidden weight matrices, hidden-length state/
activation/weight vectors. All912 original scalar cells in76 table rows preserved.
Every accuracy/latency frame field matches its original series position;76 existing
Circadian prediction/label accuracy rounding checks (absolute tolerance0.00005)
and76 hidden=24+splits-prunes checks passed. No new reported score/statistic/winner.

Epoch series[1,4,8,...,300], Phase B begins index31/epoch124. Final saved accuracies
BP0.8286/PC0.7486/CPC0.7771, latency(ms)0.1343/0.1434/0.2127; Circadian hidden74,
splits50,prunes0. These earlier-profile values are not substituted for the tuned
hardest text summaries. Saved normalized objectives are not raw loss/energy;
normalization procedure, comparability and execution provenance unknown.
Source explicitly displays Phase-B test accuracy/labels during Phase-A snapshots;
this is historical test-informed visualization, not corrected arrival/isolation.
No seed identity/count/per-seed independence, full original configuration/baseline
capacity, code/build/dependency/device/weights, original measurement method/hardware/
resources or full stopping/selection/rollback/attempt/failure history established.
No matched fairness, significance, replication or scientific admission asserted.

HTML inspected statically only: declares Plotly2.35.2 CDN URL (not fetched),
260ms playback interval, source numeric axis labels/bounds and all original code.
Its heatmap supplies z but no declared x/y coordinates, while scatter/layout use
physical bounds. Alignment is not certified; original source unmodified. Browser
rendering/controls/network/CDN availability not exercised. Same76frame count and
four sampled matching headers do not prove identical original GIF/HTML backing
execution. Complete HTML shell saved separately; no reconstructed plots/telemetry.
Nine refusal controls passed: nonfinite JSON,duplicate keys,unknown root,epoch
mismatch,boolean hidden size,accuracy-series mismatch,wrong decision map shape,
missing hidden activation values,wrong prediction count. Diagnostic mutations were
restored and whole canonical payload rechecked; original bytes remain unchanged.

Evidence: `docs/legacy-hardest-animation-validation.md`;
`artifacts/runs/p95-hardest-animation-20261006/` whole source/metadata copies,
view/reference/76-row table/complete HTML shell/eight inspection PNGs,
visual-review/readback/static-validation/acceptance/coverage-delta/terminal/
final-accounting. Only saved original coverage complete; P9.5/P9.5b/G0/R0.3/fullR3.1
remain open. Owning-with j6c repair remains human-deferred; missing sources retained.

Budget: prospective600aggregate engineering seconds/hard60 per child/64MiB owned,
fixed160manual/discovery/visual/closing reserve plus every captured attempt; final
accounting reserves entire own60second cap. Not whole-session time or processRSS.
Science350.7925872/360 and runtime168.7993043/180 remain spent without reset.
Full pytest/native/Torch/CI/mypy/clean-clone/scientific-reader gates skipped in this
ignored-helper/additive-doc scope and remain required. Existing Pillow used;
no new dependency/download/model/dataset/device/browser/CI/sweep/publication.
New guide/seven reversible additive docs only; unrelated checkout/source/tests/
architecture and all other task rows preserved.

Why this increment: existing GIF/HTML already present the saved visualization.
Validate originals and document clipping/coordinate/provenance limitations instead
of generating replacement telemetry or favorably selecting snapshots. Keep the
full large arrays in the exact original HTML to stay within64MiB owned storage;
all payload values still independently checked. This changes no scientific gate.

Exact next P9.5b23: freeze complete illustrative-circadian publication/coverage metadata and whole docs/figures/circadian_sleep_dynamics.gif before decoding under a fresh small engineering scope. Inspect all original frames/labels and full illustration provenance/missing/unknown limits; independently validate frame dimensions/durations/pixels and document existing views. Preserve illustration-not-experiment distinction; no invented telemetry, new experiment, scientific source/execution/replication admission or browser/model/original scientific-reader/dataset/CI dispatch.


## P9.5b23 — illustration validation (2026-10-06)

[Guide](circadian-illustration-validation.md): all25 original GIF frames/labels/pixels validated, including two downward width steps. The original remains illustration-not-experiment; constant split/prune labels are not live counters, and chemical telemetry is absent. Plan/log hold complete evidence. P9.5/P9.5b remain open; next P9.5b24 arrived-v1 request/result inspection.


### Saved arrived-v1 confirmation fixture

See [the whole-source validation guide](arrived-confirmation-figures.md) for all seeds/candidates/orders, negative outcomes, figures and explicit checkpoint/environment gaps. P9.5b24 completes saved presentation only; no new scientific or runtime admission. Next: replay-v8 source validation.


### Saved replay-v8 smoke fixture

See [the complete validation and figures](replay-v8-smoke-figures.md) for both replay policies, all seeds/events, retention above 1 and explicit physical memory/runtime gaps. P9.5b25 completes saved presentation only. Next: replay-v9 continuation source validation.


### Saved replay-v9 continuation bodies

See [the complete three-source validation](replay-v9-continuation-figures.md) for exact whole-file equality, all duration differences, figures and the original v8 protocol retained inside v9 filenames. P9.5b26 completes saved presentation only; physical continuation remains unverified. Next: matched-replay schedule/training/outcome source inventory.


### Matched replay planned schedules

See [the complete source validation guide](matched-replay-schedule-figures.md) for all nine originals and five validated planned schedules. P9.5b27a completes inventory/schedules only; inference counts differ and the parent remains open. Next: P9.5b27b applied training and cross-schedule validation.


### Matched replay saved training records

See [the complete training validation guide](matched-replay-training-figures.md) for both whole training bodies, all five schedule comparisons, applied work and clock limits. P9.5b27b completes saved training relations only; unequal inference work is retained. Parent remains open; next P9.5b27c outcome/telemetry validation.


### Matched replay complete saved outcomes

See [the complete outcome and family reconciliation guide](matched-replay-outcome-figures.md) for all scores, telemetry and remaining physical/provenance gaps. P9.5b27c and saved-family P9.5b27 are complete; CPC's negative aggregate result is preserved. Broad scientific parents remain open. Next: P9.5b28 side-effects-v10 originals.


## Saved replay side-effect comparison (P9.5b28)

[Complete guide](replay-side-effect-figures.md): both whole originals and
both conditions validated; scores/work identical, chemistry differs. CPC's
lower aggregate scores and the FIFO seed exception remain. Fourteen figures
and complete evidence are local saved presentation, with physical execution
and independent repetition unproved. Next: P9.5b29 difficulty-v11 full bodies.


## Saved difficulty modulation comparison (P9.5b29)

[Complete guide](difficulty-modulation-figures.md): both whole difficulty-v11
originals, all conditions/backends/seeds/arms and diagnostics validated. Final
modulation/control differences are zero; negative forgetting and failures remain.
Twenty figures preserve every recorded value. Physical execution and independent
repetition are unproved; no superiority claim or new heuristic. Next: P9.5b30
whole structural-ranking-v12 training/outcome originals.


Saved `structural_rank_factors_v12` presentation now has complete four-source readback. See [scope and unproved execution claims](structural-ranking-figures.md); no new isolation or scientific admission evidence.


Saved `sleep_trigger_timing_v13` has complete four-source presentation/readback. [Scope and unproved isolation/execution claims](sleep-trigger-figures.md) retain all null/mixed outcomes and scientific gates.
