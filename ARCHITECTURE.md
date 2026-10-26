# Architecture

`app/clm_probe_inspector_projection.py` validates detached workflow32 CLM review
presentation and copies recorded endpoints/costs into the existing source-cell
contract. The shared app dispatcher selects it; `infra/key_door_inspector_files`
owns bounded JSON admission/exclusive HTML output and the coherent-stream
template owns rendering. No IO/model/metric computation enters the projection.
Legacy game-only numerical imports are deferred to their existing validation
functions, keeping this inspector CLI route independent of model libraries.
See [format, usage and limits](docs/clm-probe-inspector.md).

`build_public_workflow_request` in the existing app module wraps the core request
builder and preserves live query construction. Episodes and an explicit teacher
option share it; the default teacher stays unchanged. The CLM script binds the
selected TRAIN group and offers an explicit learned-only observation mode for
reusing an accepted frozen control. Neither option changes learner math, target
generation, permissions, update authority or the three mandatory scenarios. See
[recipe and observation modes](docs/clm-ticket-workflow.md#public-observation-train-option).

The CLM ticket runner's default-off TRAIN fit option reuses native
`predict_choices` on the original frozen TRAIN features before and after its
fixed eight-pass inbox sequence. The existing app module reduces scalar
probabilities by canonical phase/concept; it imports no backend. The script
guards native state, feature identity/value, readonly work deltas and the two
owned evidence records. No new learner, stream, state format, budget authority
or evaluation path is introduced. See [usage and cost scope](docs/clm-ticket-workflow.md#optional-train-fit-diagnostic).

The existing MiniGrid training arm has a default-off budget-evidence option.
It captures original scalar authority at birth, observes exitcharges before
modelreadback and exports through `finally`; the record grants no authority.
Default schemas/math and native/stream budget guards stay unchanged. See
[failure semantics](docs/train-budget-evidence.md).

`minigrid_public_cue_health` composes existing native activation and pure health
functions on the actual public-selected head. The pilot's explicit opt-in reads
six original TRAIN traces before each ordinary readout, retaining detached
metrics and a bounded paid failure prefix under the same budget and full guards.
No native/core math, evaluator kernel or model update path changes. See
[usage and qualifications](docs/public-cue-activation-health.md).

The existing weighted public-cue adapter computes at most513 immutable four-int
prefix counts at schedule admission, using actual first public observations.
The existing lifetime schedule witnesses and a pinned table identity preserve
invalidation; each fork rebinds its own table. This adds no persistent state
format or history and leaves full native validation in state reads. See
[bounded component evidence](docs/weighted-prefix-validation.md).

The public-cue recurrent pilot's class-balanced opt-in composes the existing
weighted schedule adapter with the existing training arm. Readout binds the
original wrapper and inner native owner under both full guards; evaluator type
admission stays strict. The fixed A64/96 observation plan is explicit, and all
fits precede reads. No runner, optimizer or state format is added. See
[the bounded recurring protocol](docs/weighted-recurrent-context.md).

`PublicCueRecurrentLearner.train_class_balanced_episode` derives fixed weights
from one released TRAIN episode and reuses the existing native recurrent weighted
loss on its public-selected head. The same staging/guard/failure path retains
both full head states and all paid uncertainty. Mean-CE defaults, snapshot format
and session semantics are unchanged; see [usage](docs/public-cue-weighted-training.md).
The existing `TorchTrainWeightedLearner` now binds this family's complete TRAIN
episodes to the normal delayed stream, with per-head prefix validation and a
bounded permanent failure marker that bypasses the old atomic rollback path.
See [stream qualification](docs/public-cue-weighted-stream.md).

`torch_recurrent_learner._sgd_options` binds the existing native optimizer
configuration for both construction and exact snapshot admission. Validation
uses plain group metadata and retains full Tensor/gradient/momentum checks and
owned copies. It creates no temporary optimizer, cache or new state format.
See [qualification and limits](docs/recurrent-snapshot-validation.md).

The existing `ProbeRollouts` has an explicit public-cue recurrent mode for paid
F/A disposable endpoints. `minigrid_policy_evaluation` admits exact native18/34
counts and full two-head policy/gradients/optimizer equality; the shared kernel
retains genuine sessions, target-free predictions and cleanup. The generic
EndpointObserver owns paid-phase read boundaries and each reader keeps its
original zero-work budget. Count/H32 and benchmark source rules remain separate.
No new learner, orchestration, state codec or final-data route is added. See
[usage and the incomplete native result](docs/public-cue-probe-games.md).

`adapters/minigrid_recurrent_replay.py` composes the existing complete-episode
codec, replay banks and application StreamRehearsal for an explicit raw148
two-head recipe. Its finite witness checks causal selections, original owners,
paid counters and preceding evidence; it owns no optimizer or persistence.
The existing pilot fits both fixed controls before any readout. Application
replay has an optional identity-bound boundary callback; defaultNone preserves
existing routes, while externally guarded continuation is explicitly refused.
`infra/local_result_json.py` also provides compact exact-byte bounded encoding
for the existing paired CLI; its original pretty writer stays unchanged.
See [usage and boundaries](docs/recurrent-episode-replay.md).

The existing native evaluator has an explicit final-only affine10/252/head36
binding, separate from FIFO and TRAIN admission. It reuses saved typed data,
original zero-work budget/full-state guards and `_run_checkpoint` for original
Dblue/red. No training, target score, controller or new runner participates.
See [held-out affine games](docs/heldout-affine-games.md) and ADR-0637.

`adapters/minigrid_context_torch_feature_learner.py` owns two full original native
neural snapshots, selecting a head only from the existing public initial cue and
feeding canonical296 inputs. Native mathematics and complete continuation remain
with `torch_feature_learner`; core projection stays pure. The existing control
module composes AttributionStream and the original native evaluator for the fixed
BP known-A gate. App roles/receipts/budgets stay outside the learner, full endpoint
callbacks carry data only, and no new orchestrator or persistence layer is added.
See [usage and limits](docs/public-cue-neural-baseline.md) and ADR-0636.

Original ordinary G TRAIN geometry belongs to existing core/infra key-door ports;
G-only cue placement preserves both keys and role admission precedes physics.
The affine workflow composes those teacher episodes with packagedABC and the
existing delayed stream, then binds its actual endpoint to the original native
G/A loop. Witnesses, paid prefixes and separate TRAIN/readout/whole clocks retain
their ownership; no new orchestration layer. See [usage](docs/ordinary-fourth-task.md)
and ADR-0635. Original presets, teacher, native loop and private body remain.

`scripts/run_minigrid_linear_fit_control.run_context_linear_stream` adds an
explicit recurring catalogue mode through the original AttributionStream and
affine learner: ten delayed TRAIN updates, bounded per-cue retention, original
paid receipts/failure capture and detached scoring after learning. Old callers
and six-update modes retain their defaults. No new orchestration or data-loader
layer. See [API](docs/recurring-train-catalogue.md#ordinary-chronological-train-integration)
and ADR-0634; the original G/native follow-up above has its own scoped evidence.

The existing `adapters/minigrid_train_catalogue.py` adds an explicit ten-arrival
bound permitting three retained TRAIN packets/cue. Original validation, witness,
FIFO routing, spans and transport schema remain; cumulative input bytes are
bounded before each allocation. Fitting and delayed release remain with the
existing learner/stream. See [usage](docs/recurring-train-catalogue.md) and ADR-0633.

`app/minigrid_private_affine_inspector.py` projects the explicit saved private
workflow into the existing inspector. It reuses normal affine TRAIN validation
and native public-view decoding, without reconstructing parameters or invoking
adapters/IO. The original definition, source pointers and private phase clocks
remain separate; the shared file port and HTML render all controls and partial
slots. See [usage](docs/private-affine-inspector.md) and ADR-0632.

Opt-in `scripts/minigrid_affine_workflow.run_workflow(private_adaptation=True)`
composes the existing delayed TRAIN stream, original F collection/release
validation, private two-arm comparison and native F/return-A evaluator. Current-run
full endpoints replace historical file dependencies; no new runner or optimizer.
An optional outer TRAIN boundary and whole-workflow supervisor preserve original
phase budgets, source witnesses and preceding paid evidence. Default workflow
behavior remains separate. See [usage](docs/portable-affine-adaptation.md) and
[ADR-0631](docs/adr/ADR-0631-portable-private-affine-adaptation.md).

The existing native evaluator adds a narrow8/180/head36 F-then-A readout binding
using its original loop and zero-update budget guard. Existing affine game
composition accepts the qualified matched-study endpoint pair, protecting
original clocks/charges/protocol and paid prefixes. No training labels or new
runner. See [native return](docs/affine-rehearsal-return-games.md) and ADR-0630.

`adapters/minigrid_affine_rehearsal.py` composes two fixed private36row fit
recipes with the existing probe helper and affine learner. Original row roles,
detached diagnostics and attempted/committed reuse accounting remain explicit.
Two preallocated arm allowances share the original study clock; source and paid
prefix guards refuse unobserved changes. No benchmark replay bank or new runner.
See [usage/results](docs/affine-private-rehearsal.md) and ADR-0629.

`scripts/minigrid_affine_probe_games.py` composes exact saved before/after F
states with the original policy evaluator. Its narrow adapter binding selects
the original probe_test layouts and admits no fits or labels. Existing state,
budget, RNG, cleanup and paid-prefix guards remain responsible for execution.
See [usage/limits](docs/affine-fresh-task-games.md) and ADR-0628.

## Affine runtime prerequisite

Normal `scripts/minigrid_affine_workflow.run_workflow` reuses the existing infra
`_require_minigrid` before resource decoding or learner creation. The existing
exact version check and failed-report contract remain authoritative; describe
and saved-result inspection keep their base-only behavior. No new checker or
runtime authority is introduced. Fresh cache-only NumPy installation checks are
scoped delivery evidence, not clean-clone/native-install or scientific acceptance.
See [setup](docs/affine-clean-install.md) and ADR-0627.

## Recorded affine learning projection

`app/minigrid_affine_inspector.py` projects one explicit saved TRAIN workflow
format into the existing MiniGrid player. It validates scalar/receipt/role
consistency and reuses the unchanged native episode decoder for actual B/C
public frames. Existing dispatcher/file writer/template own their boundaries.
Affine-only panels display causal receipts, planned versus committed joint
replay and scoped costs. Unknowns have no fabricated pointers or outcomes;
failed receipt capture preserves paid budget counters. No model restoration,
training or native runtime is imported by this projection. See
[usage](docs/affine-workflow-inspector.md) and ADR-0626.

## Portable affine training and native games

The existing linear CLI exposes an explicit bundled-example mode and console
entry point. `scripts/minigrid_affine_workflow.py` composes the existing delayed
stream, arrived-cue catalogue and unchanged native evaluator. Only a complete
receipt-captured stream hands off its actual native endpoint. The existing
paid-report guard seals preceding TRAIN evidence during zero-update games.
`src/infra/minigrid_affine_example.py` locates a small TRAIN-only package resource;
no model, probe/final sources or historical runtime authority is bundled.
Old A/B/A defaults and scientific readback qualification remain intact. See
[usage](docs/portable-affine-workflow.md) and ADR-0625.

## Guarded contextual recorded evidence

app/context_serving_inspector_projection maps one explicit guarded context report
into existing detached source cells/tables. Existing inspector dispatch, bounded
file IO and generic HTML renderer remain owners of their layers. The contextual
title/warning changes conditionally; older schemas retain their routes. No model
imports, state restoration, inferred metrics or source authentication occur.
See [usage](docs/context-serving-inspector.md) and ADR-0607.

## Portable contextual learning example

infra/context_head_example resolves one exact bounded project-generated data
resource; the CLI holds its extraction lifetime with ExitStack. The explicit
example source alternative admits only existing synchronous context serving;
all domain learning/guards/publication/rollback stay in their existing modules.
No authority is deserialized from history. See [usage](docs/context-example.md)
and ADR-0606; ordinary explicit-source behavior remains.

## Complementary context-order validation

context_training_streams holds a separate immutable withheld recipe tuple.
context_stream_comparison reuses the live runner with an explicit option, fresh
applications and frozen classifier fit. Pure metadata admission reconciles FIFO
clocks, cumulative native work and separate zero-mistake acknowledgements before
utility. CLI defaults remain unchanged. See [scope](docs/withheld-context-orders.md)
and ADR-0605; known-template readouts are not sealed final evaluation.

## Causal task feedback

ArrivedTaskFeedback is a separate core scalar unit admitted by ReferenceObservation;
classification-trained contextual selection refuses it. CandidateTaskFeedback
reads only an arrived ordinary TRAIN opportunity's detached candidate through the
existing private six-task reader. Finite paid-record guards, state/actor pins and
original budgets precede decisions. The original reference schedule owns fixed-
drift selection/shared admission; private diagnostics cannot fit, acknowledge or
publish. See [task feedback](docs/task-return-feedback.md) and ADR-0604.

## Controller-development task readouts

context_controller_learning explicitly composes existing private typed actors and
context_head_serving._episodes for before/after six-task development readouts.
Exact controller_train roles, task/phase/version/native/work records precede
measures; existing CounterfactualController binds source/digest/clocks/objective.
The new CLI fits only, with no live application; classifier defaults unchanged.
Its oversized-output failure retains only bounded producer-reported scalar work,
budget/clock and error prefixes, explicitly excluding detailed outcomes/fit/
receipts. This summary grants no authority or evidence acceptance.
See [scope/failure](docs/controller-task-trials.md) and
[ADR0603](docs/adr/ADR-0603-controller-task-trials.md).

## Correct TRAIN acknowledgement

AcknowledgedExperience is a scalar no-update receipt. ActorShadowRuntime verifies
current unmanaged/no-sleep TRAIN under its existing candidate lease; ExperienceInbox
consumes only an arrived FIFO packet, retaining bounded payload/identity history.
The explicit candidate adapter supplies complete current-state correctness proof.
Schedule receipt settlement and optimizer counts stay separate; original guards
decide publication. Unsupported cursor/handoff/erasure paths refuse the history.
See [scope/results](docs/correct-training-acknowledgement.md) and
[ADR0602](docs/adr/ADR-0602-correct-training-acknowledgement.md).

## Contextual stream comparison

ticket_serving_timeline shares pure serving_opportunity/serving_preview between
the worker and CLI protocol view. context_stream_comparison composes existing
live runner, scalar frozen fit, paid-report guards and objective for ten missing
applications plus two limited historical periodics. It opens no final/probe data
and validates all cases before utility. No new runtime or fitting layer.
See [scope/results](docs/context-stream-comparison.md) and
[ADR0601](docs/adr/ADR-0601-context-stream-comparison.md).

## Contextual TRAIN recipe adapter

context_training_streams validates D9 prototypes with original TRAIN trace IDs,
public workflow masking and clean target truth; owned packets carry named noise
provenance. Native D7 recipes stay unchanged. context_head_live composes explicit
24-step recipe plans/budgets with the existing schedule; context_recipe_validation
is a finite two-case helper using existing paid-report guards. No new runtime.
See [scope/limits](docs/context-training-recipes.md) and
[ADR0600](docs/adr/ADR-0600-contextual-training-recipes.md).

## Causal candidate TRAIN diagnostics

candidate_training_feedback reads detached snapshots and only arrived, pending
TRAIN at the existing _reference_schedule observation port. Explicit contextual
drift opt-in reconstructs disposable readouts; metadata binds base/candidate
versions, counters and original/current clocks. Default actor feedback unchanged.
Attempts/returns/repeated reads are charged; no new runtime. See
[scope/results](docs/candidate-training-feedback.md) and
[ADR0599](docs/adr/ADR-0599-candidate-training-feedback.md).

## Contextual controller learning

context_controller_learning composes existing CounterfactualController and
_reference_schedule for four disposable TRAIN trials, then context_head_live for
one frozen application. Contextual policies are pinned; fitted policies bind the
complete qualified source/version before constructors. Classifier estimates and
deployed task outcomes stay separate. No new runtime. See
[scope/results](docs/context-controller-learning.md) and
[ADR0598](docs/adr/ADR-0598-contextual-controller-learning.md).

## Contextual reference comparison

context_head_live accepts exact existing fixed reference modes while retaining its
periodic default. context_reference_comparison composes that runner, existing
paid_report_guard and pure ControllerObjective; common task readouts are readonly
and wait for all cases. Prior periodic bytes are reused, not reconstructed. No
new scheduler, controller algorithm or model framework. See
[scope/results](docs/context-reference-comparison.md) and
[ADR0597](docs/adr/ADR-0597-context-reference-comparison.md).

## Public-context live worker

Existing LiveServingPlan/TicketServingTimeline optionally bind an immutable
context/task sequence to public requests. Context readouts expose the existing
complete-update counter port. context_head_live composes the original reference
schedule, worker, shared gate and drained promotion/rollback; no new runtime.
Main and worker computation share one measured allowance. See
[scope/results](docs/context-head-live.md) and
[ADR0596](docs/adr/ADR-0596-contextual-serving-worker.md).

## Contextual guarded serving

core/context_perceptron now groups/scatters mixed readonly rows; TRAIN remains
homogeneous. adapters/context_serving binds complete state bytes, categorical
guards and one atomic typed actor read to existing generic promotion ports.
scripts/context_head_serving composes AttributionStream/actor/controller through
the existing serving CLI opt-in, without another scheduler. See
[scope/evidence](docs/context-head-serving.md) and
[ADR0595](docs/adr/ADR-0595-contextual-guarded-serving.md).

## Isolated contextual further-learning probe

probe_ticket_policy_learning retains its native default and shares its original
arm/endpoint functions with explicit separate-head mode. context_head_probe owns
known source/data/guards; the existing generic FreshTaskProbe owns disposable
learning, TRAIN monitors, same-fork return and budget/report protection. No core
probe/model changes or new runtime. Reference margin supplementation is local
analysis only, with inference/cost provenance disclosed. See
[usage/evidence](docs/context-head-probe.md) and
[ADR0594](docs/adr/ADR-0594-isolated-context-head-probe.md).

## Conventional separate-context heads

core/context_perceptron composes original pure D7perceptron states/rules with
strict public-onehot routing, atomic updates and complete two-head/globalcounter
validation. adapters/context_perceptron implements the owned learner port and
detached typed readout, delegating original permitted affine decisions. Existing
recurring runner/AttributionStream remain orchestration boundaries. No IO, data
selection or scheduling enters core. See [module/use/evidence](docs/context-head-control.md)
and [ADR0593](docs/adr/ADR-0593-separate-context-perceptron.md).

## Contextual queue replay composition

ticket_workflow_rehearsal certifies two authentic tickets as one ordered queue
with explicit next-ticket advancement; existing episode codec and stream remain
unchanged. D9/six-step helper configuration reuses StreamRehearsal's recent and
coverage policies, original wake authority and separate replay budget. Existing
recurring runner enables it with --context-return --replay recent/coverage;
default D7 and no-replay paths retain their behavior. See
[usage/negative result](docs/context-queue-rehearsal.md) and
[ADR0592](docs/adr/ADR-0592-contextual-queue-rehearsal.md).

## True context return through the existing stream

ticket_controller_streams compiles fixed D9/two-ticket TRAIN phases through the
shared contextual teacher. recurring_ticket_workflow_example composes its existing
AttributionStream/perceptron/typed sandbox with explicit --context-return, fixed
same-policy readouts and conditional recovery measures. Inputs and paid results
are guarded before more work. The core learner, old recipes and final/probe
boundaries retain their behavior. See [usage/result](docs/context-return-stream.md)
and [ADR0591](docs/adr/ADR-0591-context-return-stream.md).

## Shared contextual teacher and local joint control

run_ticket_workflow owns shared fixed joint teacher projection/validation; the
native qualifier delegates its data preparation without an import cycle. Existing
local CLI opts into eight-pass joint D9 training through AttributionStream and
the unchanged perceptron. Separate joint-only readers protect source/predictor,
budget origin/owners/RNG and paid task prefixes; default/single-v2 loops remain
unchanged. TRAIN fit/actual sandbox effects and their costs remain separate.
No new runner, numerical rule or authority. See docs/joint-context-perceptron.md /
ADR-0590.

## Contextual BP fit qualification

The existing qualifier has an explicit D9 joint-four-task mode. It validates
tagged TRAIN teachers, schema/alignment/nonconflicting inputs before creating
the existing native BP stream. It preserves original source/actor/RNG/budgets and
seals paid initial evidence before further work. Existing typed frozen readouts
execute all fixed version-bound scenarios; teacher fit and task truth remain
separate. New role/definition prevent old source-reader admission. D7 defaults,
core/native math and observation semantics are unchanged. See
docs/context-bp-qualification.md / ADR-0589.

## Public ticket policy context

The existing core request can append a validated public version token; the app
episode and TRAIN teacher/example forward the explicit exact-bool option. Script
CONTEXT_FEATURES selects D9; default FEATURES/D7 and old calls/reports are exact.
Tagged source groups and opt-in report markers identify changed observations.
Neither capability masks nor effect/task truth change. The existing example's
model width and nominal work/storage counts follow the selected vocabulary.
No new layer, learner, schema authority or framework. See
docs/ticket-policy-context.md / ADR-0588.

## Isolated ticket-policy learning recipe

probe_ticket_policy_learning composes existing fresh_task_probe/old-TRAIN-return/
endpoint observer with original full native BP state decoding and typed sandbox
readouts. It supplies detached TRAIN-derived probe templates, separate readonly
old monitors, fixed source96/0 arms and finite sessions. Full source/data/RNG and
previous paid results remain guarded; task copies must preserve the disposable
learner even after clock observations. Failed rows stay in the report before
completion checks; later arms refuse incomplete/changed evidence. Existing core,
math, model, defaults and app probe machinery are unchanged. See
docs/ticket-policy-learning-probe.md / ADR-0587.

## Saved policy TRAIN fit

diagnose_ticket_training adds an explicit saved-policy-comparison branch. It
admits the original complete five-case cohort and six fixed source/opportunity4
states, checks qualified96/576 plus six-row update clocks and actual v1 TRAIN
teacher alignment, then checks all paid native v1 scores before interpreting fit.
Only identical parameter decoding/projection is shared with the strict original
reader. No native model, update, probe, final read, restore or runtime authority
is created. The CLI pins original bytes; results are offline diagnostics.
See docs/ticket-train-diagnosis.md / ADR-0586.

## Same-phase controller comparison

The existing comparison driver has an explicit policy-change mode. It validates
each original deferred trial and prior paid evidence, freezes all five candidates,
then runs existing owned development readouts. A pure reducer requires all unique
policy-bound sandbox outcomes and verifies final task truth. Current-task gain and
unchanged-password drop use same-phase no_learning; original cost weights and
unknown all-in overhead remain. Legacy comparison/final modes are unchanged.
See docs/phase-matched-controllers.md / ADR-0585. No new runner or learner is added.

## Native policy-phase trial seam

run_bp_ticket_serving opts the fixed12batch corpus into its original reference
runtime/budget/gate/action envelope. Four phase records retain actual updates on
an opportunity axis. ticket_workflow_native validates finite records and uses
existing full-state readouts/actual sandbox after TRAIN, or defers all task reads.
Original source/fitted binding/actor/RNG stay protected; legacy final/utility paths
refuse new policy markers before access. Existing pure summary supports the new
axis without changing default scalar outputs. See docs/native-policy-trials.md /
ADR-0584. No core learning/model/authority schema or new runner is added.

## Versioned sandbox TRAIN stream

The existing run_ticket_workflow teacher accepts explicit policy versions and
verifies actual transitions/final task success. ticket_controller_streams builds
owned fixed TRAIN packets; recurring_ticket_workflow_example opts them into the
same learner/AttributionStream/budget/checkpoint/typed-sandbox readout path.
Current-policy invoice outcomes, obsolete route errors and unchanged-password
retention remain separate; source/actor/default recipes and final/probe isolation
are preserved. No simulator/learner framework or core policy change is added.
See docs/versioned-policy-stream.md / ADR-0583.

## Fresh TRAIN after publication

The existing ticket runner accepts an explicit `FollowupTrainingPlan` from its
timeline module. One actor/shadow runtime, clock, budget, shared gate and immutable
action envelope span both predeclared windows. Existing first-window guards and
typed receipt authorize fresh observed-version sources/labels; old feedback stays
invalidated. Private native state continues independently of the frozen published
copy. Failure prefixes and full rollback remain. No core inbox/model schema or
new orchestration layer is added. See docs/post-publication-learning.md / ADR-0582.
The optional separate followup recipe port composes owned clean/noisy TRAIN
packets and distinct second-window delay through the same schedule. Immutable
recipe/data pins are checked by owner and worker; defaults retain original
global delays and packets. See ADR-0608.

## Bounded native component sleep

`ResourceSharedRuntime` selects a fixed one-event action and uses the original
`ActorShadowRuntime.consolidate`. The optional `ComponentSleepLearner` port in
core is implemented by the NumPy adapter on detached full native state; app has
no adapter dependency. Native replay/split/prune are disabled and width fixed.
Separate admission/preparation/commit counters and bounded original consolidation
metadata retain failure costs and prior receipt checks. Original model schemas,
Torch no-sleep clocks, budgets and actor publication behavior are unchanged.
See docs/component-sleep-actions.md and ADR-0581.

## Native gate action path

`ResourceSharedRuntime → ActorShadowRuntime → ExperienceInbox → update_learner`
forwards one validated scalar to the optional `GateControlledLearner` port.
`TorchFeatureLearner` stages the complete payload; the native CPC head overrides
only the selected update's plasticity multiplication, with all other adaptive
state native. Config/snapshot identities and durable receipt schemas stay fixed.
Direct inbox callers receive the same pre-payload capability refusal. No adapter
dependency enters app/core; existing ticket work counters cover the new port.
See [gate actions](docs/gate-actions.md) and ADR-0580.

## Final-only FIFO36 game readout

Saved final reports now route through `key_door_inspector.py` to the separate
pure final projection in `minigrid_saved_fifo_inspector.py`, then the existing
exclusive file writer and MiniGrid HTML player. No adapter/model/environment
dependency is added. The shared episode decoder accepts explicit D geometry;
legacy defaults and the F/A probe schema retain their original validation.
See ADR-0579 for final/probe separation and partial-evidence handling.

`minigrid_policy_evaluation.py` shares the unchanged complete FIFO36 state copier
between its original TRAIN API and a separate final-only entry point. The latter
admits only benchmark128/source0, fixed D blue/red and pinned0update/0replay
authority before delegating to the existing raw-action game loop. No training,
source loader, model selection or new runner is added. Final public-frame metadata
stays distinct from TRAIN/probe records. See docs/fifo-heldout-games.md/ADR-0578.

## Public-cue contextual MiniGrid control

The explicit contextual alternative uses `minigrid_context_encoding.py` to append
two initial-public-cue bits to canonical296 history, with encoding-bound owned
state in `minigrid_context_perceptron.py`. Existing two-head unit-mistake math
and delayed `AttributionStream` run the fixed recurrence. A narrow game facade
checks the original fully spent32-update budget through frozen known-A readouts;
there is no new training authority or scheduler. See
docs/contextual-minigrid-control.md and ADR-0609. B/C imitation scores and actual
A gameplay remain separate evidence.

## Frozen TRAIN-standardized MiniGrid control

`standardized_minigrid_learner.py` validates TRAIN sources, reuses core population
statistics and binds an immutable transform to the unchanged native BP port.
Full snapshots include transform and native optimizer/gradient/RNG ownership.
The existing MiniGrid evaluator has a narrow known-A facade; its shared physics,
raw argmax and source/budget guards remain. No model fitting or evaluation enters
the core statistics module. See docs/standardized-minigrid-bp.md and ADR-0577;
the observed normalization variant is negative for complete gameplay.

## Explicit live TRAIN recipes

`scripts/ticket_controller_streams.py` composes LiveTrainingRecipe around the
unchanged comparative recipe schema/builder. An explicit retained-half fixture
modifies only a copied TRAIN label and records its experimental meaning. The
native script accepts a separate pinned live input, preserving comparative
deferred-evaluation gates and original clean guard/task truth. Same causal loop,
worker, hard budgets and original guard authority; no new runner or native math.
[Contract](docs/live-training-recipes.md) ·
[ADR0576](docs/adr/ADR-0576-explicit-live-train-recipes.md).

## Publication at a causal TRAIN drain

The native serving script factors original guards/prepare/commit into
`_publish_candidate`; an optional `_reference_schedule` callback runs it once
after consumed labels/receipts, empty pending predictions and worker/gate quiescence.
`_publish_at_drain` validates typed receipt/generation/version transitions and
records then invalidates old scalar feedback. Native rules, controller/guard
authority, budgets and state formats remain unchanged. Later requests use the
atomic actor publication; original complete rollback remains.
[Contract](docs/drained-ticket-publication.md) ·
[ADR0575](docs/adr/ADR-0575-publish-at-causal-train-drain.md).

## Live typed serving in the existing native schedule

`scripts/ticket_serving_timeline.py` adds a finite one-worker service helper to
`run_bp_ticket_serving._reference_schedule`. Original native-start callbacks and
the original serving gate provide explicit pre-admission contention and admitted
overlap. Each paid episode joins before the next causal tick; request versions,
latencies and failure prefixes remain separate from TRAIN/controller feedback.
Worker source checks do not mutate shared budget clocks. Existing post-stream
guards/publication/rollback retain authority; no new training runner or native
math/state format is added. [Contract](docs/live-ticket-serving.md) ·
[ADR0574](docs/adr/ADR-0574-live-typed-serving-in-existing-schedule.md).

## Per-update native rate effects

`core/learner_ports.py` defines an optional scalar rate port. The Torch adapter
stages one update with an effective parameter rate, restores BP's base optimizer
rate before publication and retains full policy/momentum/gradient validation.
Existing learner_step/inbox/actor/shared methods forward the scalar; finite
rate+wake IDs use original role/label/admission/budget authority. Detached polls
expose the requested multiplier; durable state/receipt schemas remain unchanged.
[Contract](docs/learning-rate-actions.md) ·
[ADR0573](docs/adr/ADR-0573-per-update-native-rate-effect.md).

## Fixed categorical replay action

`app/categorical_replay_action.py` composes existing categorical episode codec,
original exposure budgets, native forks and ActorShadowRuntime consolidation.
ResourceSharedRuntime admits finite IDs through its existing serving gate.
Observed native wake receipts advance one expected candidate; bounded metadata
and original owner/budget/bank checks reject unobserved changes. Actor snapshots
have an additive nonblocking port. Core adds detached counters/replay receipt.
[Contract](docs/categorical-replay-actions.md) ·
[ADR0572](docs/adr/ADR-0572-fixed-categorical-train-replay-action.md).

## Deferred controller comparison

`scripts/ticket_controller_streams.py` constructs bounded declared TRAIN receipt
shifts from existing teacher packets with owned buffers and explicit noise.
The existing native schedule derives readiness from arrived labels and keeps
original FIFO/admission semantics. Its opt-in deferred mode returns only a
complete candidate, without workflow task readouts or promotion.
`scripts/ticket_controller_comparison.py` reuses frozen fitted/reference choices,
validates each predecessor, freezes the finite cohort and only then invokes the
existing detached readonly final evaluator. No final results feed fitting,
choice or promotion; original live serving and full-stream completion remain.
[Contract](docs/withheld-controller-streams.md) ·
[ADR0571](docs/adr/ADR-0571-deferred-withheld-controller-evaluation.md).

## Version-bound contextual controller

`core/contextual_controller.py` fits finite scalar context/arm means and defines
an immutable source-bound policy. `app/counterfactual_controller.py` binds one
complete versioned base snapshot and admits at most four causal disposable
trial results before freeze, rejecting stale source/payload changes before
metrics. Native digest/fork details stay at the adapter/script boundary.
The existing serving CLI delegates only its optional finite study recipe to
`scripts/ticket_controller_learning.py`; all update choices use the original
arrival/shared admission and native task ports. Trial publication is suppressed,
while fresh applications preserve original guards and complete rollback.
[Contract](docs/contextual-controller.md) ·
[ADR0570](docs/adr/ADR-0570-version-bound-contextual-controller.md).

## Prospective controller objective and stages

`core/controller_objective.py` defines normalized scalar utility and immutable
protocol records. `app/controller_chronology.py` admits ordered callbacks, owns
at most three scalar rows and one freeze identity, and exposes stage-appropriate
feedback. The existing serving CLI composes native development/validation owners
and detached readonly sealed readers. No model/data store enters the core;
original native stream, role, action admission, budget and guards retain authority.
The outer trial clock is distinct from each learner's arrival/update clocks.
[Contract](docs/controller-outer-objective.md) ·
[ADR0569](docs/adr/ADR-0569-prospective-controller-chronology.md).

## Causal reference controllers

`core/reference_controller.py` accepts only scalar policy/causal TRAIN feedback
and returns finite wait/wake IDs. The optional existing native serving-demo path
uses the original arrival helper, ActorShadowRuntime and ResourceSharedRuntime.
It ends a finite observation schedule with pending counts explicit, preserving
AttributionStream.finish's full-completion contract. Candidate task results and
guarded deployed actor outcomes remain separate from chooser input. Monitoring,
selection and admission work are recorded; no new orchestration framework or
learned controller is introduced.
[Contract](docs/reference-controllers.md) ·
[ADR0568](docs/adr/ADR-0568-causal-reference-controllers.md).

## Finite adaptation admission

`core/adaptation_action.py` declares scalar action catalogs, hard envelopes and
bounded detached observations. `app/resource_sharing.py` consumes only a catalog
ID through an opt-in method and the original native admission gate. No-update/
bounded-wake effects are executable; fixed replay is opt-in through the executor
above. Per-update rate+wake is supported through the optional native port;
gate/sleep and combined effects remain refused. TRAIN role, budget,
serving priority, clock and receipts remain authoritative. No scheduler,
final/probe data release or growing success history is added.
[Contract](docs/adaptation-actions.md) ·
[ADR0567](docs/adr/ADR-0567-finite-adaptation-admission.md).

## Native typed actor publication

`torch_serving.py` binds existing `PromotableActor.serving_snapshot()` to the
existing raw-logit typed decoder, one detached request at a time. It retains no
native readout history/cache. Native categorical callbacks configure the existing
promotion evaluator/controller; model/cache/configuration/metadata publication
and one-step rollback remain owned by those application ports. AttributionStream
can receive an ordinary TRAIN-only actor and stamps the original actor version
in delayed experience/label receipts; default streams remain a0/c0.
[Contract](docs/native-ticket-serving.md) ·
[ADR0566](docs/adr/ADR-0566-native-typed-serving.md).

## Fresh BP through the existing stream

`run_native_ticket_controls --fresh-bp` explicitly admits BP into the existing
live candidate recipe with its own112-update budget and no historical state read.
Peer invocations/schema/admission remain unchanged. A distinct fresh report role
uses the same pure inspector with honest ownership notes. The stable actor stays
unchanged; this route demonstrates candidate learning, not deployment/promotion.
[Contract](docs/fresh-bp-ticket-demo.md) ·
[ADR0565](docs/adr/ADR-0565-fresh-bp-ticket-demo.md).

## Portable saved-ticket resources

`examples/ticket_workflow` maps to installed `src.infra.ticket_workflow_data` via
existing setuptools package resources. The inspector CLI keeps a selected
resource alive through `as_file` and exclusive publication, with explicit family
choices and no model imports. Four original report bytes, provenance and license
notices are retained. [Contract](docs/portable-ticket-examples.md) ·
[ADR0564](docs/adr/ADR-0564-portable-ticket-examples.md).

## Saved ticket workflow inspection

`app.ticket_workflow_inspector_projection` validates bounded saved ticket JSON
and returns detached source cells. Existing key-door dispatch/exporter and
coherent-stream template own publication, source dialogs and readable scrolling
tables. No experiment import, model state decoding or metric recomputation.
[Usage](docs/ticket-workflow-inspector.md) ·
[ADR0563](docs/adr/ADR-0563-saved-ticket-inspector.md).

## Native ticket controls

`scripts.run_native_ticket_controls` composes the unchanged teacher builders,
AttributionStream, native adapters and frozen task readout in one live stream.
It reuses saved BP evidence without refitting or adding codecs/frameworks.
Equal exposure does not imply equal objectives, computation or runtime lifecycle.
[Contract/results](docs/native-ticket-controls.md) ·
[ADR0562](docs/adr/ADR-0562-native-ticket-controls.md).

## Qualified BP context acquisition and return

Small `adapters.torch_bp_record` reconstructs only complete bounded BP payload
records through existing native validation. `scripts.run_bp_new_contexts` binds
the accepted source and composes fresh local budgets/stream/readout/effects.
Original native96clock and prior costs survive; new16updates are distinct from
old runtime authority. No native rule, general loader or controller framework.
[Usage/evidence](docs/bp-new-contexts.md) ·
[ADR0561](docs/adr/ADR-0561-qualified-bp-new-contexts.md).

## Isolated conventional BP fit recipe

`scripts.qualify_ticket_bp_fit` composes original teacher/margin helpers, native
counted forks, delayed AttributionStream and typed task effects. New isolated
joint TRAIN owners use a fixed96update budget, two endpoint copies and separately
charged teacher reads. No native rule/benchmark/default runner change or second
framework. [Contract/results](docs/ticket-bp-fit-qualification.md) ·
[ADR0560](docs/adr/ADR-0560-isolated-bp-joint-fit.md).

## Offline native BP TRAIN diagnosis

`scripts.diagnose_ticket_training` reads saved scalar BP parameter evidence,
checks the original teacher/public feature binding and validates its reference
logits against paid native scores. Existing core probability margins provide
reductions. No model restoration, optimizer, controller, app/core change or
second runner; historical pending TRAIN rows remain offline only.
[Usage/evidence](docs/ticket-train-diagnosis.md) ·
[ADR0559](docs/adr/ADR-0559-saved-bp-training-diagnosis.md).

## Native families in the chronological recipe

The existing recurring ticket loop selects an explicit native method through
`scripts.ticket_workflow_native`; the helper supplies counted native forks,
complete JSON-safe checkpoint observations and native costs. Original teacher,
delayed AttributionStream, task slots and dispatcher remain shared. Native copied
typed readouts preserve source/seed/full-state/RNG isolation; no core/app changes
or second orchestrator. Default perceptron/replay paths retain their reports.
[Usage/results](docs/recurring-ticket-workflow.md#native-family-option) ·
[ADR0558](docs/adr/ADR-0558-native-chronological-task-option.md).

## Frozen native typed readouts

`adapters.torch_decision` composes a complete owned Torch feature checkpoint with
the existing core feature/catalog and typed result schemas. Native logits remain
in the learner adapter; candidate masking precedes scalar softmax. Full local
optimizer/adaptive/RNG state is pinned around reads; work and faults remain paid.
Existing app stream and dispatcher own training schedules/effects. No core/app
dependency on the new adapter. [ADR0557](docs/adr/ADR-0557-frozen-native-typed-readouts.md)
and [usage/cost contract](docs/torch-typed-decisions.md).

## Live CLM memory/fallback composition

The existing temporal CLI combines its memory and typed-serving options after
TRAIN. `app.clm_train_evidence_memory` creates one explicitly paid bounded copied
store and returns the existing evidence-first wrapper over a frozen predictor.
`app.clm_memory_serving` declares four known requests/one unlabeled control and
admits their scalar source/selection/work records. The outer script keeps native
head binding, encoder and source guards. The old phase record remains sealed;
new view costs are separate. Inner modules import no adapter/backend. No new
runner/dependency or optimizer. [ADR0550](docs/adr/ADR-0550-live-clm-memory-fallback.md) ·
[Usage/tree/checks](docs/clm-catalog-learning.md#live-typed-memory-with-a-frozen-clm-fallback).

## Evidence-first frozen fallback

The existing `app.decision_evidence.EvidenceDecisionPredictor` adds an explicit
fallback mode. It uses existing core causal retrieval, then the existing protected
typed callback path only for empty selections. Original request and candidate
schema remain unchanged; shared finite call limits and separately owned callback
budgets retain costs/failures. Retrieval/context defaults and the CLM stream's
TRAIN decoder functions are unchanged. No backend-specific code or dependency.
[ADR0549](docs/adr/ADR-0549-evidence-first-frozen-fallback.md) ·
[Usage/check commands](docs/decision-evidence.md).

## Optional CLM TRAIN evidence memory

`app.clm_train_evidence_memory` composes existing inner evidence/store/feedback
ports through the CLM stream's synchronous checkpoint callback. The outer temporal
CLI selects the observer and original reuse consent. It validates paid catalog
receipts, records raw TRAIN mirrors at phase cutoffs and collects bounded retrieval
results; scores follow TRAIN completion. It imports no backend or adapter, changes
no learner updates and sees no final/probe labels. Default runs create no memory.
No new runner, dependency or persistent configuration.
[ADR0548](docs/adr/ADR-0548-phase-delayed-clm-train-evidence.md) ·
[Usage and costs](docs/clm-catalog-learning.md#phase-delayed-train-evidence-memory).

```text
scripts/run_clm_temporal_sequence.py  selects the existing stream + memory hook
src/app/clm_train_evidence_memory.py  receipt admission, bounded views, late scores
src/app/decision_evidence.py          existing TRAIN feedback and typed decode
src/core/decision_evidence.py         existing causal bounded storage/retrieval
tests/test_clm_train_evidence_memory.py  matched literal behavior and refusals
```

## Catalog TRAIN support composition

The existing `build_catalog_training_batches` binds selected pure
`temporal_catalog_support` declarations to3/4/4 query packets and each phase's
complete3candidate bank. The existing causal app admits canonical indices,
actual objective-tagged receipts and recipe-specific query/candidate/logit work.
Its mode/owner/feature/clock/consent/budget/paid-prefix guards remain inward.
The outer script composes independent repeat/route catalog owners over one frozen
feature bank, then admits both complete caches and their shared A prefix before
any scalar scoring. Existing TRAIN/development diagnostics stay in the same run.
No new orchestration, dependencies or final/probe access. Paired/none defaults
and the older paired generic-callback API remain. Support plus reserved probes
is refused. [ADR0545](docs/adr/ADR-0545-catalog-train-support-composition.md) ·
[Usage/tree/commands](docs/clm-catalog-learning.md#explicit-train-support-recipes).

## Catalog-objective disposable probe

`core.clm_catalog_probe` declares explicit probe_train catalog inputs, distinct
from benchmark TRAIN records. `adapters.clm_catalog_probe` admits an exact catalog
learner, then reuses the existing disposable probe's protected ownership/policy
lifecycle and catalog learner's explicit-kind tensor validator. The public paired
and catalog ports retain exact admission. Warm weights are copied into independent
native state; only source counters are retained, and no installation/export port
is supplied. Existing inbox/clock/budget ports can execute causal probe updates.
The existing reserved-task app and checkpoint adapter now bind phase-specific
catalog records/indexed targets, actual objective-tagged receipts and permanent
candidate/logit costs. The temporal CLI composes the same stream with four
independent probe owners; cached scoring admits all main/child evidence first.
No inner-to-outer imports, new runner or dependencies.
[Integration ADR0544](docs/adr/ADR-0544-catalog-reserved-task-integration.md). [ADR0543](docs/adr/ADR-0543-explicit-catalog-probe-roles.md) ·
[Usage/module tree](docs/clm-catalog-probe.md).

## Explicit CLM catalog training

`core.clm_catalog` declares TRAIN-only ordered catalog IDs and query target indices.
`adapters.clm_catalog_learner` supplies directional full-catalog cross entropy and
an objective-specific owned learner/snapshot. It reuses the native paired learner's
protected atomic update engine, with permanent candidate/logit exposure counters.
The existing temporal app binds labels/catalog features and causal receipts; the
outer CLI assembles frozen features and scores cached results after all TRAIN.
Default paired math/records remain intact. No new runner or dependencies.
[ADR0542](docs/adr/ADR-0542-explicit-clm-catalog-training.md) ·
[Module tree, objective and commands](docs/clm-catalog-learning.md).

## CLM temporal-rule use case

`app.clm_temporal_sequence` composes the existing learner/inbox/clock/budget ports
for twelve fixed paired updates, validates bound inputs and actual receipt prefixes,
and returns only unscored probability readouts. `run_clm_temporal_sequence` is the
outer local-asset composition: it binds twelve raw feature templates, keeps released
heads frozen, and reuses the existing temporal typed evaluator after TRAIN ends.
A cached frozen-A v2 readout separates learned rule changes from candidate removal.
No inner-to-outer import, new generic runner, projection cache or checkpoint chain.
The opt-in TRAIN catalog diagnostic binds three phase-specific choice batches
back to actual TRAIN states and the same active catalog. It adds three readouts
under exact permanent counters and retains only three unscored matrices; the
outer script scores them after fixed TRAIN work. Disabled defaults remain intact.
`core.temporal_catalog_support` separately declares explicit new TRAIN route
semantics and an invoice-repeat control, preserving the original corpus. The
same app validates declared3/4/4 packets and sums exact row/candidate/query costs.
The outer comparison uses one frozen feature bank and independent head owners;
all fixed training precedes both arms' scoring. Defaults and objective remain.
[ADR0539](docs/adr/ADR-0539-explicit-train-catalog-support.md).
[Usage](docs/clm-temporal-sequence.md).

## Reserved CLM checkpoint probes

`core.temporal_clm_probe` fixes marker supervision and distinct probe roles.
`app.clm_reserved_probe` composes six native updates through the existing inbox,
consent, clock and independent budget, returning unscored scalar facts only.
The existing temporal app invokes an explicit checkpoint operation at baseline/
A/B/return and protects the four completed child owners/values before later work.
`adapters.clm_checkpoint_probes` binds the16-template feature union, creates each
disposable native probe, proves full benchmark state/work preservation and scores
cached rows only after all source/probe TRAIN completes. The CLI composes local
assets. Default off; incompatible support modes fail before loading. No app/core
imports from adapters, no probe export/install or new orchestration framework.
[ADR0541](docs/adr/ADR-0541-fixed-clm-checkpoint-probes.md) ·
[Usage and commands](docs/clm-reserved-probes.md).

## Rectangular CLM candidate ranking

`core.clm_head.FrozenChoiceBatch` separates label-free query/candidate features
from native paired TRAIN batches. `CLMHeadLearner.predict_choices` validates
independent row bounds and reuses owned staging, projections and state/RNG
checks. Aggregate prediction/query costs include both prediction APIs; permanent
choice counters add candidate exposure. Paired training/prediction admission
and snapshot schema remain unchanged. No inner-to-outer dependency or additional
cache/runner. [Usage](docs/clm-choice-readout.md) ·
[ADR0538](docs/adr/ADR-0538-separate-choice-readout-from-paired-clm-learning.md).

## Conventional CLM recurring TRAIN example

The outer `run_clm_head_adaptation` script reuses frozen encoder features,
CLMHeadLearner and ExperienceInbox. Its finite phase tuple supplies native
paired batches and real deferred group-code receipts; no app/core ports change.
Only copied learner heads/scale are trainable. One frozen control and four
unscored learner readouts share exact raw-feature identity; changed-head
projections are recomputed and all diagnostic scores wait until training ends.
The existing categorical metrics retain their clipping/tie rules. No new
orchestration, checkpoint, projection cache or inner-to-outer dependency.
[Usage](docs/clm-head-adaptation.md).

## Released CLM frozen features and decision

`infra.clm_local_loader` performs local pinned-asset IO and lazily loads optional
backends, returning frozen CPU components. Outer scripts inject them into
`adapters.clm_frozen_encoder` and `adapters.clm_decision`; core identities, paired
TRAIN batches and typed result ports remain backend independent. The encoder
owns the large model without copying it, bounds permanent work and keeps only
one detached diagnostic slot. The predictor owns small frozen head copies and
recomputes projections. No app/core import of infrastructure or new orchestration
framework. The CPU plaintext TRAIN recipe is explicitly separate from native
vLLM serving. Future head learning must unfreeze only separate learner-owned
head/scale copies and preserve the released control and encoder.
[Usage](docs/clm-frozen-encoder.md) ·
[ADR0537](docs/adr/ADR-0537-connect-released-clm-features-through-existing-ports.md).

## Frozen-feature CLM training

`core.clm_head` declares exact frozen encoder identity, paired TRAIN batches and
bounded head policy without Torch. `adapters.clm_head_learner` owns native CPU
heads, scale, AdamW, schedule, gradients, modes and private RNG behind the existing
learner port. It stages publication, preserves physical work and recomputes head
projections. The outer fixed example composes ExperienceInbox and compares the
native reference; no new queue/cache/runner framework or inner-to-outer import.
[Usage](docs/clm-head-baseline.md) · [ADR0536](docs/adr/ADR-0536-preserve-native-clm-head-training-before-consolidation.md).

## Causal feedback beside a delayed learner

The existing AttributionStream exposes detached committed receipts through its
quiescent owner guard, without new state or checkpoint fields. The outer online
example observes separately consented raw TRAIN requests at the existing
prearrival hook, then joins row feedback to each actual committed batch receipt.
`app.decision_evidence.apply_decision_feedback` validates source/label bindings
and original consent before the core store admits an addition. Its small
`decision_from_evidence` decoder represents no selection explicitly; existing
wrapper refusal semantics stay unchanged. Weight/memory readouts use live fixed
prefixes; only cached outputs are scored after training finishes. No new runner,
learner objective, model-family or core-to-outer dependency is introduced.
[Usage](docs/online-temporal-feedback.md).

## Editable decision evidence

`core.decision_evidence` owns consented TRAIN records, bounded revisions, exact
canonical UTF-8 budgets, deterministic retrieval and cutoff sealing. It imports
only core schemas and stdlib. `app.decision_evidence` composes that view with typed
retrieval-only or actual context callbacks, retaining paid returns before refusal.
Instructions/candidate definitions stay outside editable JSON; semantic injection
immunity is not claimed. The outer controlled script reuses the accepted frozen
linear snapshot and existing ticket dispatcher without fitting. No learner or
orchestration framework is introduced. Context edits and physical weight readouts
have separate work reports. [Usage](docs/decision-evidence.md) ·
[Decision](docs/adr/ADR-0534-keep-editable-evidence-causal-and-separate-from-weights.md).

The optional native memory comparison composes the same store/wrapper with the
existing ticket TRAIN runner through a default-preserving post-TRAIN evaluator.
Original observation consent explicitly permits evidence reuse; records inherit
paid receipt times. One shared fit/two snapshots provide source0/24 for the four
arms, with permanent global charges and separate per-arm deltas. Context paths
return actual guarded native choices. Evidence-build failure retains its paid
prefix; the aggregate prediction cap binds before readout and cannot renew.
[Native comparison](docs/native-memory-comparison.md).

## Local ticket episodes

`core.ticket_workflow` owns immutable ticket/capability records, observation
construction, pure effect admission and terminal correctness. It depends only
on core typed/temporal rules and stdlib. `app.ticket_workflow` composes the typed
predictor port with bounded episodes; it checks effect-time authority outside
the model and retains paid failures. The outer script uses existing perceptron/
AttributionStream ports and scores only after TRAIN. No external handlers,
new learner framework or core→app dependency are introduced.
The optional native script composes the same demonstrations/dispatcher with the
existing OpenJevTrainingLearner typed port and ExperienceInbox. Its bounded
callback admits exactly five active candidates, counts TRAIN plus readout
forwards, preserves raw returns before postchecks, and checks model/gradient
versions, mode and RNG. Two local source snapshots supply controls without
renewing work; all TRAIN precedes episodes. Batched typed readout semantics differ
from independent-candidate serving; no serving-parity or usage claim is made.
[Usage](docs/local-ticket-workflow.md) ·
[Decision](docs/adr/ADR-0533-enforce-ticket-effects-outside-the-policy.md).

## Temporal tool and policy decisions

`core.temporal_decision_sequence` owns fixed effective rules, active tools,
observation-bound targets and strict TRAIN/development cases. It imports only
core typed inputs/stdlib. The outer temporal reference script composes existing
AttributionStream/PerceptronLearner/evaluator ports and scores owned states after
all TRAIN. `perceptron_decision` adds explicit default-off subset TRAIN admission;
union class indices and the native mistake objective stay fixed. Masked readout
already exists. No new framework or core→app dependency.

The native temporal outer script reuses this core corpus, existing native owner,
delayed-label inbox and probability evaluator. Explicit delivery validation and
readout instructions preserve original routing defaults. Four existing owner
snapshots supply current-catalog controls after all TRAIN; spent work remains
permanent, and no new checkpoint or owner layer is introduced.
[Native usage](docs/native-temporal-workflow.md) ·
[Decision](docs/adr/ADR-0532-use-native-owner-for-temporal-workflow-controls.md).

Old-domain identity does not rewind policy truth. Reports separate obsolete
still-allowed choices, retired-tool masking, stable-rule retention and current-
rule return, with unchanged and frozen-A controls under the same active catalog.
[Usage](docs/temporal-decision-reference.md) ·
[Decision](docs/adr/ADR-0531-bind-temporal-tool-rules-at-observation.md).

## Fixed native recurring-task study

`core.native_choice_sequence` owns a small declarative TRAIN/development corpus,
role/group admission and fixed phase order. `scripts.run_open_jev_sequence`
composes existing NativeLearner/ExperienceInbox/loader ports and the shared
bounded `record_readout` boundary from the one-update CLI. One inbox/budget spans
all18updates; endpoint metrics never choose training work. Core/app acquire no
Torch dependency or new orchestration framework. Phase weight summaries identify
changes separately from native original-load metadata. The unchanged control is
derived baseline reuse, not a concurrent actor. [Protocol and usage](docs/native-choice-sequence.md).

`scripts.run_open_jev_b_control` composes the same helpers for a separate fresh
six-update B learner. It binds the accepted report and initial trainable/RNG/
optimizer state before reusing baseline readouts. No new core/app framework or
reserved-probe role is introduced. Cold-versus-sequential endpoints retain their
different history and total computation; source and data stay fixed.

`scripts.run_open_jev_optimizer_control` uses those same phase/inbox helpers for
the declared A6→reset→B6 intervention. The native adapter alone owns an opt-in
empty AdamW replacement, state-preservation guards and permanent reset charges.
Initial/after-A hashes bind inherited readouts before more work. No app/core
API or dependency changes. [Decision](docs/adr/ADR-0520-declare-bounded-native-optimizer-reset.md).

Its default-off projection option uses the same A6/B6 deliveries. The native
owner projects once and consumes its existing reset authority; the script binds
after-A state and admits the fixed accepted reset-only report before model load.
Endpoint metrics report signed historical contrasts without choosing updates or
rank. Core/app orchestration and default branches remain unchanged.
[Decision](docs/adr/ADR-0527-compare-explicit-native-projection-with-reset.md).

`adapters.open_jev_consolidation` owns one bounded factor-reference record,
identity/version admission, detached receipts and permanent capture charges.
The existing `open_jev_projection` orchestrates capture proofs and two-term
staging/reset/publication using that helper and pure low-rank composition.
Training exposes the explicit opt-in methods; core/app gain no dependency or
selection rule. Dependency direction is training→projection→reference helper;
the reference helper imports only the standard library. No sibling import cycle.
[Decision](docs/adr/ADR-0528-consolidate-owned-native-adapter-operators.md).

The same optimizer-control runner's default-off consolidation mode captures at
bound A6, preserves warm B6, binds the accepted twelve-update weights, and calls
one terminal consolidation/reset before readouts. The reference loader adds an
opt-in after-B checkpoint without changing its default schema. Capture preserves
the entire training boundary; terminal publication permits only LoRA changes.
The warm endpoint is the primary comparator because no TRAIN follows the reset.
[Decision](docs/adr/ADR-0529-evaluate-terminal-native-consolidation.md).

`scripts.summarize_native_consolidation` is a separate saved-report command.
It composes the pure native decoder, fixed corpus and existing app probability
metrics, validating captured-result/metric parity without a model or learner
import. Four bounded accepted inputs produce owned calibration/rank/cost records;
historical model work remains distinct from the scalar derivation. No metric
feeds training or selection. [Usage](docs/native-consolidation-calibration.md).

Its opt-in rehearsal mode reuses the same phase runner: one exposure helper
records delayed labels and paid receipts, and a rehearsal boundary admits only
two prior authorized A TRAIN cases. Full receipt copies match the inbox's own
applied updates before more work. Every replay uses the same budget/clock and
links to the first paid source exposure. Core/app and native adapter math are
unchanged. [Decision](docs/adr/ADR-0521-rehearse-authorized-native-train-cases.md).

## Native Choice TRAIN update

`adapters.open_jev_training.OpenJevTrainingLearner` implements the existing
opaque-input NativeLearner port for typed text requests and hard candidate IDs.
The outer `scripts.run_open_jev_native_update` composes the original loader,
compiler and ExperienceInbox; app/core gain no Torch dependency. The loader's
optional model observer exposes the loaded model explicitly and preserves
original smoke defaults. Local snapshots own LoRA/head/AdamW/gradients/private
CPU RNG; base identity/version guards avoid a full base copy. Permanent attempted
work survives restore; uncertain mutation stops the learner. No actor fork or
portable training resume. [Decision](docs/adr/ADR-0519-train-native-choice-through-existing-inbox.md).

The adapter's optional `open_jev_update_health` helper reads validated trainable
parameters/gradients and produces scalar head/LoRA norms. Temporary pre-step
trainable copies identify actual AdamW deltas; no frozen base or graph survives
in a report. Separate permanent attempted-element charges stay outside native
snapshots. Existing `last_update`/inbox receipt copies transport the record;
the outer CLI captures final costs even on failure. App/core acquire no Torch
dependency. [Decision](docs/adr/ADR-0522-read-native-train-update-health.md).

`scripts.run_open_jev_train_health` composes those same helpers for fixed A6→B1
TRAIN-only diagnosis. Initial/after-A bindings precede B; a loader counter
reporter observes all TRAIN work without making an inference. RMS/clipping
summaries are scalar projections, never a training controller. No app/core API,
objective or dependency changes. [Usage](docs/native-choice-sequence.md#seven-update-train-diagnosis).

`adapters.open_jev_owned_fork` implements the existing ForkableLearner boundary
for opt-in native copies. Memo-shared frozen Parameters avoid base duplication;
modules, buffers, trainables, optimizer/RNG/policy and paid history are owned by
each copy. Separate permanent copy charges and inherited spent limits prevent
recursive allowance renewal. App/core acquire no new dependency or framework.
[Decision](docs/adr/ADR-0523-own-disposable-native-learners.md).

`core.native_choice_probe` owns frozen reserved text declarations; the outer
`scripts.run_open_jev_reserved_probe` composes existing native fork, probe-role
ExperienceInbox, clocks and scalar scores. Source snapshots/buffers/RNG/modes/
work are compared without a base copy or child-to-source restore. Test targets
remain in application scoring. No native objective, app/core API, new runtime
framework or dependency changes. [Decision](docs/adr/ADR-0524-probe-reserved-text-on-a-disposable-native-owner.md).
The same CLI explicitly admits warm A6/cap12 or cold source0/cap6. A cold
control reuses the accepted observed-task warm report; comparison and transfer
interpretation stay at the outer boundary, with native training ports unchanged.

`core.low_rank_composition` owns pure same-base factor sums and optional thin
QR/SVD rank projection through a small core. Explicit orientation/scaling,
owned float64 factors, tail residual and scoped work have no learner or data
authority. Native binding, scalar-head and optimizer interventions remain outer
responsibilities. [Module guide](docs/low-rank-composition.md) ·
[Decision](docs/adr/ADR-0525-compose-low-rank-updates-through-a-small-core.md).

`adapters.open_jev_projection` binds opted-in live Linear adapters and composes
the existing pure rank projection with staged float32 installation and the
owner's existing explicit AdamW reset. The owner retains Parameter identities,
allocated rank, permanent attempt/work limits and one historical receipt;
snapshots cannot refund these and forks inherit them. Default native training
remains unchanged. [Usage](docs/native-choice-sequence.md#explicit-native-adapter-projection) ·
[Decision](docs/adr/ADR-0526-project-native-adapters-with-explicit-reset-authority.md).

## Fixed native decision comparison

`scripts.run_native_decision_comparison` composes the existing raw-score and
probability evaluators with a saved lexical report, final frozen perceptron
snapshot and original native model loader. It owns file IO and fresh output;
core/app gain no model dependency. The loader enforces explicit finite work caps
for this distinct cohort while preserving the original smoke defaults. Completed
linear evidence is saved before optional native loading. No training, new stream
framework or baseline tuning occurs. [Protocol and results](artifacts/runs/native-decision-comparison-20261022/RESULTS.md).

## Native probability scoring

`core.typed_decision.ProbabilityDecisionResult` defines observed probabilities
without unexposed logits. `adapters.open_jev_choice` validates native wire data
and projects that inward type. `app.decision_reference_evaluation` owns role
admission, isolated callbacks and shared descriptive metrics; it imports core,
never the native adapter. The outer `scripts.score_open_jev_smoke` binds saved
wire replies and composes both. No model IO or new runner lives in core/app.
Why this: native Choice has no raw-score field; inventing one would corrupt the
older result contract and imply observations that did not occur. Native margin
confidence stays separate from maximum probability used for ECE. See
[usage and validation](docs/native-choice-bridge.md#probability-only-scoring).
[Decision](docs/adr/ADR-0518-score-native-probabilities-with-existing-metrics.md).

## Native Choice wire boundary

The open_jev_choice adapter maps existing immutable request fields to the
inspected native wire schema and validates a bounded response byte body. Its
frozen observation preserves native probabilities/confidence/request-order ties
and raw metadata separately from the lexical/linear DecisionResult contract.
No transport, model loader, synthetic logits/token counts or provenance authority.
[Decision](docs/adr/ADR-0517-preserve-native-choice-wire-semantics.md).

## Learned typed routing reference

Core decision_features owns fixed vocabulary/catalog and pure query projection.
The perceptron_decision adapter builds TRAIN-only AttributionBatch records and
reads detached native coefficients with allowed-candidate softmax. Existing
PerceptronLearner/AttributionStream own updates, role isolation and scheduling;
the fixed CLI scores development snapshots only after all updates. No old learner
or metric changes, external model loader or new orchestration framework.
[Decision](docs/adr/ADR-0516-bridge-typed-routing-to-native-perceptron.md).

## Typed decision reference

Core typed records and pure lexical retrieval have no IO or labels. App scoring
keeps roles/source groups/targets outside the predictor input, with bounded paid
partial reports. The CLI alone owns exclusive JSON publication. No external-model
loader or new adaptation runtime is introduced. The categorical reference contract
does not replace future released-model native semantics.
[Decision](docs/adr/ADR-0515-type-allowed-candidate-reference-decisions.md).

## Saved continuous comparison boundary

`src/app/continuous_comparison_inspector.py` projects bounded saved JSON into the
existing source-cell tables. The existing infra exporter owns files/HTML and the
CLI owns source/resource selection. The mapped `src.infra.continuous_comparison_data`
package contains exact accepted summary/controls and notice; projection imports no
experiment driver or learner. Unknown slots and unsafe browser integers have
explicit behavior. [Usage and checks](docs/continuous-comparison-inspector.md).

## CPU distribution boundary

Ordinary setuptools packaging preserves the existing src/scripts namespaces.
The original example directory maps to private installed
`src.infra.taskspace_demo_data`; the outer CLI selects it with importlib.resources
and as_file. No data duplicate, source loader, learner or inward dependency change.
Explicit source-file selection and source-checkout defaults remain supported.
Optional SDKs are separate; the alpha CPU package does not qualify every research
entrypoint. [Decision](docs/adr/ADR-0514-package-cpu-demo-resources.md) ·
[Installation and checks](docs/cpu-memory-package.md).

## Portable recorded taskspace replay

`scripts/run_taskspace_memory_demo.py` owns CLI/filesystem output and delegates
fixed JSON admission/reproduction to `adapters/taskspace_memory_demo.py`, which
depends on the existing sequence/nearest adapters and pure episode/history core.
The bundled generated TRAIN data is separate from recorded replay trajectories;
replay labels never enter training, and stored outcomes are not new simulator
results. No SDK or research-archive import is required on this CPU path.
[Decision and boundaries](docs/adr/ADR-0513-package-recorded-taskspace-memory-replay.md).

The explicit `--native` route lazily imports `scripts/taskspace_memory_native.py`.
That outer composition reuses the same admission/stream and existing frozen
Pick/Push rollout ports at one detached final checkpoint. It owns bounded partial
capture and validates prior model/report/file evidence before each next scene.
Native results never feed training or promotion; no inner-layer dependency changes.

## Bounded grouped supervision diagnosis

A new pure core composition calls the unchanged100-row consistency kernel in bounded group blocks and maps all pairs to original global identities. The outer fixed CLI admits accepted TRAIN inputs and checks both byte sources after math before publication. Paid partial work remains visible. Why: test exact supervision ambiguity without changing learner/labels or broadening the existing kernel; no new runner, dependency or inner-layer IO.
[Usage, boundaries and commands](docs/aggregated-supervision-consistency.md) · [Evidence and next action](artifacts/runs/aggregated-supervision-consistency-20261021/RESULTS.md).

## Aggregated corrective fit through existing boundaries

The existing study composes two qualified correction loaders, fixed scalar weights, explicit198 continuation and six-group detached readouts. Exact balanced120 reconstruction validates before any new operation. Source/candidate/inputs/paid reports stay guarded. Why: use observed current-state supervision while retaining all old rows; no new runner, optimizer, collector, dependency or inner-layer IO. Default four/five-group/readout and row scopes remain unchanged.
[Evidence, commands, costs and next action](artifacts/runs/aggregated-correction-study-20261021/RESULTS.md).

## Explicit retained-correction scope (2026-10-21)

Existing mixed facade/continuation carry an immutable selected row cap through
forks, weights, native receipts/aggregation and guarded extended report metadata.
The148 default/head130 remain;198 is explicit mixed-only and grants no update
budget. Existing correction parser validates complete original packets before
deriving a source120-specific target ID; source112 contracts stay exact. Why:
using both50-label batches plus retained98 rows requires198 and distinct IDs.
No new kernel/collector/runtime/source-admission framework.
[ADR-0512](docs/adr/ADR-0512-explicit-198-row-retained-corrections.md) ·
[Modules, API, commands and qualification](docs/push-cube-learned-continuation.md).

## Current-student supervision through existing boundaries (2026-10-21)

The existing Push study reconstructs balanced120 and matches whole continuation/
private readout before using the unchanged finite TRAIN collector. A shared release
helper checks complete causal labels and exact cached trajectory; both routes
validate paid evidence before publishing a packet. Distinct artifact/count
provenance separates112 and120 packets despite shared context IDs. Why: supervision
at new observed states is needed after a failed velocity fit intervention; no new
collector/runner or inner math is needed. No new-label fitting in this scope.
[ADR-0511](docs/adr/ADR-0511-label-current-balanced-student-states.md) ·
[Modules, commands and results](docs/push-cube-corrections.md).

## Private TRAIN velocity intervention (2026-10-21)

Existing continuous_probe_control shares its exact zero-array operation between
role-separated reserved probes and private native TRAIN preparation. Source and
private owner/array/state pins surround copied restoration and boundary callbacks;
paid failures release no candidate. Existing Push study composes preparation,
weighted continuation and detached diagnostics. Why: isolate inherited velocity
with fixed data/dose, without a new learner, runner or role conversion.
[ADR-0510](docs/adr/ADR-0510-private-train-velocity-fit-control.md) ·
[Modules, commands, checks and observed result](docs/push-cube-learned-continuation.md).

## Saved Push D12 diagnosis (2026-10-21)

The existing saved-control CLI admits the bound Push study/correction packet
through an explicit mode, then shares its original exact-input/support numerical
body. Physical7/history5 coverage and identity scaling preserve D12 semantics;
targets never rewrite executed history. Original Pick modes remain exact. Why:
resolve known-input fit versus coverage after a failed learning intervention,
without another learner/runner or simulator call. No new dependency/configuration.
[Modules, commands, checks and results](docs/learned-control-diagnosis.md).

## Fixed prefix/group study through existing boundaries (2026-10-21)

scripts/push_cube_correction_weights owns only a pure scalar fixed recipe and
ordered TRAIN input validation. The existing study composes accepted weighted
continuation, private TRAIN diagnostics and native endpoints, binding uniform120
controls and prior evidence before further work. No new runner or numerical
kernel. Why: test the diagnosed prefix/group tradeoff once with all148 rows and
the same eight-update dose. The cache reader validates explicit semantic group
order independently of JSON mapping key order. Physical outcomes and fitting
remain separate; accepted scoped correctness does not establish useful learning.
[ADR-0509](docs/adr/ADR-0509-fixed-prefix-group-corrective-learning-study.md) ·
[API, file tree, commands and observed results](docs/push-cube-learned-continuation.md).

## Fixed weights through recorded mixed continuation (2026-10-21)

Existing continuous_train_continuation admits/seals fixed positive TRAIN weights,
retaining row order/provenance and actual exposure through ActorShadowRuntime.
Existing RecordedContinuousMixed owns configuration across forks and stores raw
weights with named weighted objectives in bounded native receipts. Original
source/actor/seed, policy/momentum/normalization, clocks, budgets and prior evidence
remain guarded; failures retain native definitions/work and release no candidate.
Default schemas/calls remain unchanged. Selected weight-input counts are distinct
from extra validation/normalization/copy costs. Why: native weighting required a
recorded runtime path before a controlled learning study. No new runner or dependency.
[ADR-0508](docs/adr/ADR-0508-fixed-weights-through-recorded-mixed-continuation.md) ·
[Usage, modules and commands](docs/push-cube-learned-continuation.md).

## Positive row-weighted continuous BC (2026-10-21)

Existing core/continuous_bc owns pure positive-weight admission/normalization and
weighted common/mixed derivatives. Existing adapters/numpy_continuous_bc owns
episode roles, frozen normalization, inherited momentum and atomic publication;
it validates representable mass before admitting gradient work. Weights are
copied batch supervision, excluded from policy/snapshots; exposure counters count
real rows. Default arithmetic, objective IDs and callback shapes remain exact.
Why: supply the conventional capability needed to test the diagnosed first-action
versus mean-fit tradeoff. No new runner, dependency or optimizer. Recorded mixed
continuation carries weights/provenance/costs; the fixed study is reported above.
[ADR-0507](docs/adr/ADR-0507-positive-row-weights-in-continuous-bc.md) ·
[API, files and qualification](docs/continuous-bc.md).

## Detached corrective fit diagnosis (2026-10-21)

The existing TRAIN readout has an explicit five-group148 mode; defaultfour/130,
native math and PhaseReadouts64KiB stay. Existing study binds/reconstructs the
accepted120 scope and measures the same groups at112/120 on isolated forks, with
original models/inputs/prior evidence guarded. No new runner/checkpoint loader/
optimizer/dependency. Why: physical failures left correction versus expert/old
fit unknown; reconstruction is charged and adds no new dose.
[ADR-0506](docs/adr/ADR-0506-measure-corrective-fit-with-detached-readouts.md) ·
[Usage, files and checks](docs/push-cube-learned-continuation.md).

## Retain all released corrective TRAIN rows (2026-10-21)

Explicit mixed admission/recording permits148 rows; head130/default8 and numerical
kernels stay unchanged. The outer push_cube_correction_inputs reader binds the
original hashed release and reuses core state/history/arrival validation. Existing
study composes112 reconstruction → eight corrective updates → frozen120 Push/Picks,
keeping original40/48/112 owners and prior returns exact. No new runner, optimizer
or dependency. Why: retain50actual-state corrections,18later-phase expert rows and
80old rows together. [ADR-0505](docs/adr/ADR-0505-retain-all-released-correction-fit-rows.md) ·
[Usage/tree/checks](docs/push-cube-learned-continuation.md).

## Push student correction supervision (2026-10-21)

New core push_cube_corrections owns immutable physical states/labels, observed and
arrival clocks, actual student history and provisional target export. New outer
push_cube_correction_collection composes existing native BP/Push port/teacher,
guarding frozen source/prior returns/teacher state and paid failures. It executes
student actions only. Existing study provides fixed112 reconstruction, original
trajectory comparison and serialization; no new runtime/loader/optimizer/dependency.
Why this: expert-only data did not cover the student's failed trajectory, and
relative12 features cannot recover unobserved absolute geometry. Expert success-
only release remains separate. [ADR-0504](docs/adr/ADR-0504-push-on-policy-corrections.md) ·
[API, tree, commands and limitations](docs/push-cube-corrections.md).

## Fixed112 physical endpoint composition (2026-10-21)

Only the existing outer study changes: explicit native selection reproduces the
accepted112 continuation/readout before reusing frozen Push/Pick runners. Existing
native ports, inner runtime and numerical kernels remain unchanged. Historical48
Push is distinct from new112 native evidence; fixed cache readers and whole-report
comparisons bind accepted112 and48 controls. All raw42 Pick starts and Push12 starts
match their controls, with scene/controller/history settings exact. Models/cached/
raw/copied prior evidence stay guarded. Why this: physical acquisition/retention
remained unobserved despite improved TRAIN fit. No loader/framework/dependency.
[Usage and scoped negative evidence](docs/push-cube-learned-continuation.md).

## Finite longer mixed TRAIN scope (2026-10-21)

The existing continuation adapter changes only explicit mixed-dose admission to
maximum64; default8/head maximum8 and all native math remain unchanged. Existing
recorders already bound storage to scalar receipts plus one latest model pin.
The outer study composes original48 reconstruction → separate64-update scope →
detached112 TRAIN readout, guarding source40/candidate48/candidate112 and prior
returned evidence. The stage-local pure saved-report comparison computes all
fixed physical scalar deltas against accepted48 diagnostics without model calls.
Historical physical48 evidence remains explicitly distinct from112 TRAIN results.
Why this: an observed fit gap warrants a longer conventional fixed dose before
another algorithm. No new runtime, persistence, dependency or inner-layer IO.
[Scope, commands, evidence and limitations](artifacts/runs/mixed-longer-train-fit-20261021/RESULTS.md).

## Reserved-reset probe composition (2026-10-21)

The existing outer study composes `continuous_probe.run_continuous_probe` before
and after its original continuation. Its opt-in selection requires diagnostics-
only/retention mode; no new native scene or deployed learner update. New outer
`scripts/push_cube_reserved_probe_inputs.py` reuses the strict JSON reader and
causal history validator, binding the full decoded episode to original hashed
23/24 bytes. Source/actor, input roles, exact returned plan/native objective and
existing completed receipt/clock proof are checked before subsequent work. Raw/
copied partial results remain in the study's existing evidence boundary. Why this:
reserved data and private runtime already exist; only their mixed-learner study
composition needed qualification. No new inner runtime, optimizer or dependency.
[Usage, tree, tests and limitations](docs/push-cube-learned-continuation.md).

## Detached per-group TRAIN readouts (2026-10-21)

Outer `scripts/push_cube_training_readouts.py` owns fixed TRAIN group admission,
eight operation receipts and one detached `PhaseReadouts` callback. Existing
conditioning/residual adapters remain the numerical boundary. The study calls it
before/after its existing private continuation; no new inner runtime, objective,
optimizer, persistence or dependency. Original source/deployed state and prior
returned evidence are guarded between operations. Native objective-specific loss
is distinct from comparable physical axis residuals; returned work and unknown
callback/failed work remain separate. Why this: observed mean loss concealed a
nearly maximal initial gripper error and old-task fit damage. Diagnostics-only
retains historical Push provenance and creates no native scene. Extend only under
a separately frozen protocol; final/probe labels cannot enter these TRAIN groups.
[API, tree and checks](docs/push-cube-learned-continuation.md).

## Private mixed-gripper continuation (2026-10-21)

The existing continuation adapter has an explicit `bp_mixed` selection; default
head-GN behavior is preserved. `src/adapters/continuous_mixed_training.py` owns
RNG-free private objective copying, bounded returned SGD receipts and one latest
native state pin. The native core loss/gradient is reused without another readout.
The original source and explicit StableActor retain their original policy/state;
one immutable private training seed makes five bounded named owners. Mixed policy,
all endpoint data and source-derived successful costs are explicit in the report.
The native SGD adapter now validates immutable birth/commit state; gradient/proposal
math is unchanged. Inner core has no new dependency or IO. Why this: an observed
same-input gripper fit error needs a learning path, while serving/source evidence
must remain exact. No new runtime or loader. [ADR-0503](docs/adr/ADR-0503-private-mixed-gripper-continuation.md)
records alternatives, losses and scope; software checks do not imply task success.

## Pick-only retention successor (2026-10-21)

The existing study adds bp_mixed_retention, reusing the accepted complete negative
Push return from the failed prior study. Both caches are immutable throughout;
whole source/data and whole mixed-continuation evidence must replay exactly,
excluding only the two named present descriptive duration values. Only3Pick scenes
are new. Reconstruction remains paid, source policies/counters/clocks/receipts/work
remain exact, and no model snapshot loader/framework is added. Why this: valid
Push evidence need not be reexecuted when the original CLI owners have ended.
The original failed enclosing result remains distinct from this completed successor.

## Selected mixed study profile and saved reset values (2026-10-21)

The existing outer study selects mixed continuation explicitly and hash-binds
accepted control/data reports. Historical before-Push is reused; source remains
common-MSE. `_same_push_reset` compares canonical JSON of all12observation values
and7reset metadata fields, permitting list/tuple serialization only, with no
numeric tolerance or omitted values. Why this: the original native run refused
identical saved/live values after a valid negative Push return. Failed original
producer/results remain separately bound; repair is not a scientific retry.
No new inner interface, dependency, loader or orchestration framework.

## Fixed learned Push continuation study (2026-10-21)

`scripts/push_cube_continuation_study.py` is an outer fixed local protocol, using
existing qualified input readers, joint source/acquisition adapters, private TRAIN
continuation and optional native ports. It directly consumes the live candidate;
no inner learner/runtime API, optimizer, dependency or cold loader was added.
The bounded original returned reports/saved copies and learner owners are checked
before later work. Refused paid returns/corrupted evidence stay distinguishable
from original records. Only the qualified port supplies physical success.
B references project actual history before separate released target substitution;
that derived supervision packet retains original raw evidence and release clocks.
Why this: existing APIs compose; the missing result was learned physical-task
behavior. The frozen study found failed Push acquisition and retained known Pick
successes with a higher context19 control cost. [API/scope](docs/push-cube-learned-continuation.md).

## Fixed PushCube supervision (2026-10-21)

`src/core/push_cube_teacher.py` owns bounded pure geometric decisions from named
physical state and executed action history. It has no SDK/IO/learner dependency.
`src/infra/maniskill_push_teacher.py` composes the accepted port with that core,
retains paid partial evidence and releases owned existing `ContinuousEpisode`
TRAIN records only after complete native success and exact causal/work checks.
Why this: distinct physical tasks require genuine labels; a phase name or queried
closing command is not observed contact or executed history. Existing continuation
can consume these records with explicit old references. No new optimizer, runner,
initialization, promotion or evaluation access is introduced. [API and checks](docs/push-cube-supervision.md).

## Explicit physical PushCube port (2026-10-21)

`src/core/cube_taskspace_state.py` owns observed grasp/world positions and pure
D12 displacement/executed-history projection. `src/infra/maniskill_push_cube.py`
depends inward on that core and existing continuous records/history. It isolates
optional SDK imports, TRAIN admission, live native controller/owner checks,
sparse task outputs, bounded trace and permanent work/cleanup accounting.
Only the outer SDK layer reads physical state or sends controls; core has no IO.
Why this: PushCube has different observation extras/success semantics, so named
physical inputs avoid inheriting PickCube flattened offsets or relabeling data.
The extra grasp sensor is explicit and must be shared by future compared learners.
The explicitly reviewed native admission permits bounded float32 scalar quaternion
norm error with exactly zero vector components; it adds no angular tolerance.
[ADR-0502](docs/adr/ADR-0502-explicit-physical-cube-task-state.md) preserves the
rationale and the failed original strict-predicate attempts.
No new dependency or orchestration framework; Pick defaults remain unchanged.
[API, resource accounting and current verification](docs/push-cube-train-port.md).

## Existing-source TRAIN continuation (2026-10-21)

`src/adapters/continuous_train_continuation.py` composes the existing
`RecordedContinuousHead`, native BP learner and `ActorShadowRuntime`. It owns
already released model-space TRAIN episodes, admits a bounded new/reference
batch before forking, and uses one private delayed-label update scope. The
source and stable actor remain unchanged; only a validated detached candidate
can leave the scope. Original counters and frozen normalization carry forward.

The adapter depends inward on app/core contracts. It does not qualify external
data, initialize models, fit normalization, collect native controls, read held-out
roles or promote a candidate. Row groups retain new/reference exposure costs;
bounded scalar evidence and one latest native state pin avoid a model history.
Why this: reuse the existing runtime without changing the accepted fresh-init
stream or its defaults. [API and validation status](docs/continuous-train-continuation.md).

## Fixed typed joint acquisition workflow (2026-10-21)

The ordinary CLI's explicit `--joint-acquisition` profile uses
`scripts/joint_acquisition_inputs.py` to own six capped JSON receipts. It reuses
the existing readiness teacher reader with already owned byte payloads and core
episode/correction/curation binding and release checks. Hashes identify those
same input bytes; the historical source-path field grants no filesystem access.
The separate original curation source and export must agree before training.
Archived projected features are unused. This outer IO module grants no model,
query, promotion or final-evaluation authority.

Optional receipt metadata is copied and guarded by the existing typed workflow
before paid work, preserving original acquisition costs separately from new work.
Default callers omit it and keep the same report shape. No new model loader,
orchestration framework, optimizer or dependency is introduced.

`scripts/run_continuous_robot_learning.py` re-exports the typed
`run_joint_acquisition_rollouts` helper from
`scripts/joint_acquisition_robot_learning.py`. The helper composes existing
joint data admission, joint stream, retained-reference correction and frozen
robot ports. Dependencies point from this outer script into adapters/core;
there is no new learner, optimizer, runtime or model persistence boundary.

The fixed profile owns qualified input snapshots, bounded training/native
report prefixes and private candidate checks. It returns only a report.
Qualification of external receipts, CLI intake, deployment and comparative
selection are outside this module. Keeping this fixed composition separate
leaves the legacy A/B/A API and CLI defaults intact.
[Interface and remaining gates](docs/learned-continuous-robot.md).

## Borrowed disposable-probe endpoints (2026-10-21)

The existing continuous_probe adapter adds an explicit synchronous observer of
its existing private readout. Before/after observations reuse app paid-report and
original-budget witnesses, plus pure original native source/runtime/readout proofs
after opaque observations. The observer owns separate physical results and may
not retain/train/export the borrowed learner. Returned probe reports contain no
learner. Existing probe_train updates and probe_test imitation scores stay separate.
Default schema/call shape remain, with no new orchestration or dependency.

Existing frozen serving/history explicitly admit reserved probe_test24 with a
distinct rollout identity, exact native BP/PC-family and executedTS12 only. This
boundary grants no teacher, final, training, promotion or correction authority.
Native port already supported the reserved role; no infrastructure change needed.
[API, tree and commands](docs/continuous-probe-endpoints.md).

## Frozen guard scoring (2026-10-21)

The existing outer run_frozen_rollout composes the same native continuous port
with explicit guard31/37. app/action_history_encoding binds original role,
reset/context/episode identity and causal executed history; core geometry and
native learner math are unchanged. Only exact native TRAIN BC/PC-family owners
and executed TS12 are admitted. Guard traces stay inner_guard and cannot return
training corrections. Source/frozen native state checks follow opaque observers
and close. TRAIN defaults remain. No new dependency, IO layer or runner.
[API and limitations](docs/joint-readiness-corrections.md) ·
[Executed evidence](artifacts/runs/corrected-policy-guard-20261021/RESULTS.md).

## Explicit prepared-TRAIN support (2026-10-21)

adapters/joint_readiness_support reads existing JointReadinessData and composes
core executed-history geometry/validation with marginal source/stage summaries.
It validates structural TRAIN bindings, not qualification or release authority;
those stay at the existing input/preparation boundary. No targets/clocks/learner,
IO, selection or promotion. Outputs own containers and report extra projection/
aggregation work separately. Existing prepare/train paths are unchanged. Recorded
stages are metadata; marginal envelopes are not reachability or joint support.
[API/tree](docs/joint-readiness-corrections.md#train-only-feature-support) ·
[Executed scope](artifacts/runs/train-feature-support-20261021/RESULTS.md).

## Optional recorded-stage weighting (2026-10-21)

core/joint_stage_sampling supplies a pure bounded immutable original-index plan;
existing correction and stream adapters explicitly opt in and pre-admit both
plans before initialization. Original per-batch MSE/default arrays/schema remain.
Selected batches gather original row/label/history bindings; guarded original
schedule ownership separates planned/attempted/validated-prefix exposure even
on failure. Runtime paid/native records stay bounded. No new orchestration,
optimizer, dependency or outward core import. Additional planning/gather/preflight
work and unequal per-label/context exposure are reported. Independent tests and
TRAIN/native comparisons are accepted; both negative results are preserved.
[API](docs/joint-readiness-corrections.md#optional-recorded-stage-sampling-control) ·
[Decision](docs/adr/ADR-0501-reusable-simultaneous-readiness-corrections.md).

## Simultaneous released TRAIN corrections (2026-10-21)

adapters/joint_readiness_data owns exact typed source admission, both original
arrival gates for supplied inputs, existing executed-history projections and per-row raw/model
provenance. adapters/joint_readiness_corrections composes existing actor/shadow,
head recording and GN ports; it owns eight private updates, bounded paid records,
residual readouts and detached success. No artifact/script import, source
qualification, IO, simulator, promotion or inner-layer outward dependency.
Explicit required curated=None selects the preceding A+B-only stage with no
C projection, rows, group or clock. Supplied C retains its full release gate.
joint_readiness_stream composes both correction modes at the existing
run_continuous_learnability detached B observer, preserving the original benchmark
and returning the adapted branch only after full success. Its two paid reports
and fixed state/input pins remain bounded; scalar observer summaries stay small.
All source qualification/collection stays at the existing outer input boundaries.
Legacy single-batch and replay semantics remain separate.
[API/tree/commands](docs/joint-readiness-corrections.md) ·
[ADR0501](docs/adr/ADR-0501-reusable-simultaneous-readiness-corrections.md).

## Saved TRAIN return curation (2026-10-21)

core/curated_readiness_corrections owns immutable provisional labels with original
optional observation times and new query/arrival clocks. The narrow adapter
saved_readiness_corrections composes the existing teacher and executed TS12
projection. Core depends only on core; no infra, learner, simulator or IO enters
these modules. Delayed extraction never rewrites source actions or legacy clocks.
Public relative geometry is explicitly distinct from privileged historical labels.
[Contract/tree/commands](docs/saved-readiness-curation.md) ·
[ADR0500](docs/adr/ADR-0500-curate-saved-train-returns.md).


## Arrived head corrections (2026-10-21)

The existing correction adapter owns its eight-update schedule, labels, clocks,
receipts and permanent charges. The existing head recorder supplies bounded native
work and one latest success pin. New continuous_head_corrections summarizes actual
work only; no scheduling, model initialization or solver estimates. Native smoke
composition forwards the explicit BP-only flag to the private update, preserving
its collection/frozen-return boundaries. Core/app have no new infrastructure
imports. [API/tree](docs/bp-head-gauss-newton.md) ·
[ADR0498](docs/adr/ADR-0498-compose-arrived-head-corrections.md).

The same adapter now composes GN through existing ContinuousStreamRehearsal,
using distinct original wake/reference budgets and a bounded recorder capacity.
No new runner, dependency or inner-layer boundary. Native replay returns are
owned before callbacks; final pure evidence checks follow all facade observers.
[Accepted composition](artifacts/runs/continuous-head-correction-replay-20261021/RESULTS.md).

## Detached neural phase observations (2026-10-21)

adapters/continuous_phase_readouts owns exact native forks, bounded metadata/
finiteJSON results and unchanged readout checks. The existing continuous neural
runner owns fixed checkpoint scheduling and guards original benchmark models,
budget, clocks, receipts and runtime/inbox authority. None preserves the default
schema/callshape. Callback costs are separate; callbacks cannot select training.
No simulator or persistence enters these adapters. Native studies compose the
existing frozen serving API outside them. [Contract/tree](docs/continuous-phase-readouts.md) ·
[ADR0497](docs/adr/ADR-0497-detached-neural-phase-readouts.md).

The accepted artifact-local checkpoint probe composition uses that callback to
invoke existing disposable reserved-role probes. DefaultSGD adaptation and probe
work remain outside GN benchmark authority; fulloriginal benchmark is checked
unchanged. No production wrapper or new framework was added. [Evidence](artifacts/runs/gn-checkpoint-probes-20261021/RESULTS.md).

## Original-MSE BP head update (2026-10-20)

core/continuous_head_gauss_newton owns pure bounded linearization, one damped
per-active-axis solve, global head-step cap and original-objective acceptance.
It imports only core/NumPy. Existing numpy_continuous_bc opts in through explicit
TRAIN methods, owns normalization/atomic native state/counters and preserves
hidden momentum while resetting head momentum onaccepted commits. Default SGD,
policy/snapshot schemas and PC/CPC stay unchanged; no new runner/dependency.
adapters/continuous_head_learnability is a small native-port facade for the
existing continuous learnability runner's explicit BP-only head_gauss_newton
option. It records bounded returned work before app callbacks, pins one latest
private success, and separates actual GN work from read-only diagnostics.
Selected ownership/evidence guards surround existing delayed updates and final
observers. Default runner behavior stays unchanged; no new orchestration layer.
[Usage/tree](docs/bp-head-gauss-newton.md) ·
[ADR0496](docs/adr/ADR-0496-bounded-original-mse-head-update.md).

The artifact-local GN native control case composes the existing runner's owned
final BP candidate, generic candidate admission and frozen robot return. It
requires exact whole saved benchmark reproduction before native work. One known
TRAIN17 success is accepted without a new production adapter/orchestration layer;
controlled adaptation and reliability remain distinct unfinished outcomes.


## Disposable frozen-hidden TRAIN readout (2026-10-20)

core/continuous_readout_fit owns bounded fixed-hidden least squares and explicit
original-target physical scoring. adapters/continuous_readout_fit admits released
complete TRAIN episodes against an existing owned stream learner, applies its
frozen normalization and checks actual native source state before/after. The
adapter depends inward on core and reuses existing BP/PC state validators; core
imports no adapter, IO or learner. Returned head arrays are disposable diagnostics,
with no deployment/runner/optimizer changes. Solver work and nonuniqueness remain
explicit. [Usage/tree/checks](docs/frozen-hidden-readout-fit.md) ·
[ADR0495](docs/adr/ADR-0495-disposable-frozen-hidden-readout-fit.md).


## Explicit gripper execution (2026-10-20)

core/pick_cube_action_decoding owns the fixed pure endpoint rule and scalar
validation. app/action_decoding owns bounded pending/paid receipts and source
recipe/native-prefix integrity; it depends inward on core, with no learner,
simulator or IO. Existing maniskill_continuous_smoke composes it after validated
prediction and before the existing native step, using actual returns for history.
The existing learner-to-rollout helper forwards the opt-in recipe. Defaults and
training objectives stay unchanged. No new dependency or orchestration layer.
[API/tree/checks](docs/gripper-action-decoding.md) ·
[ADR0494](docs/adr/ADR-0494-explicit-binary-gripper-execution.md).

## Explicit fixed-input nearest composition (2026-10-20)

The existing numpy_nearest_continuous_bc adapter adds an exact fixed-input policy
variant with identity copies/None statistics; old fitted-mode admission remains
strict. Existing continuous_stream_demo/sequence select it only with strict
fixed_input=True, skip fitting and retain original delayed labels/owned readouts/
receipts/charges. DefaultFalse schema and delegation remain unchanged. Existing
maniskill_continuous_smoke admits the pure fixed-input learner only with executed
TS12 and its full native/prototype/history guards. No core algorithm, simulator
controller, new runner, dependency or training/probe/final boundary is introduced.
[Usage, responsibilities and extension constraints](docs/taskspace-memory-control.md).

## Reserved robot probe inputs (2026-10-20)

`core/pick_cube_probe_context.py` owns the prospective23train/24test allocation
and original role/context/IDs. Existing action-history code exposes a separate
probe encoder/validator sharing pure transformation loops while keeping TRAIN
admission strict. The native port protects final29 and score-only24 before IO;
the existing teacher loop owns original role/reset pairing and full v2 work/
feedback qualification. A distinct typed result releases probe data and refuses
TRAIN/guard extraction. Existing teacher CLI adds explicit reserved selection;
existing generic JSON/probe/stream execution remains unchanged.
Why this: acquire genuinely separated inputs for the implemented disposable
probe without relabeling TRAIN data or creating another runner.
[Contract and usage](docs/reserved-robot-probes.md).

## Readiness label consistency (2026-10-20)

The existing core continuous diagnostic module owns bounded pure pair arithmetic:
exact duplicate inputs, separate raw/native32 target differences, robust norms,
deterministic nearest neighbors and partial work. The new
`adapters/readiness_label_separation.py` owns same-context TRAIN admission and
original query arrival, then composes existing teacher/correction causal history
projections. Actual commands and provisional labels remain distinct. External
teacher qualification remains with the existing reader. Why this: diagnose
supervision before another learner change, without a new runner, model, fitted
scale or evaluation path. [Contract and usage](docs/readiness-label-separation.md).

## Student-prefix recovery collection (2026-10-20)

The existing core readiness selector gains a separate exact completion-plan type;
the old one-field teacher-prefix plan stays intact. The app ledger owns bounded
full raw-observation/control receipts and retains actual paid returns even when
later callbacks fail. The existing robot query loop owns the fixed switch and
actual-command history. Its adaptation API sends the plan only to collection;
private learning and pure frozen return retain their existing boundaries.
Why this: teacher completion can add connected near-goal TRAIN examples missing
from a failing student trajectory. Mixed behavior is never teacher qualification,
probe data or autonomous success. No new runner, loader, learner, objective,
dependency or evaluation path. [Use and evidence](docs/readiness-roll-in.md).

## Stage-balanced acquisition composition (2026-10-20)

`core/readiness_training_packets.py` owns deterministic indices and detached
exposure metadata for an exact original TRAIN stage binding. It has no array,
learner, filesystem or simulation dependency. The existing stream adapter gathers
already encoded rows into its existing packet application; the robot adapter
binds the explicit option and validates original source/paid packet evidence
through frozen return. The source episode is never reordered or synthesized.
Why this: rare recorded stages justify a declared sampling hypothesis while the
existing learner/optimizer interface already accepts arrays. Empirical weighting
changes; original unweighted readouts and default behavior remain. No new runner,
layer or dependency. [Usage](docs/observed-grasp-corrections.md).

Opt-in readiness step diagnostics compose the existing correction adapter with
`adapters/readiness_step_diagnostics.py`, which validates existing model-space
TRAIN bindings and calls `core/continuous_step_diagnostics.py` on detached arrays.
The outer correction adapter owns scheduling, paid receipts and native snapshots;
the leaf owns scalar grouping and attempted/returned diagnostic work. Core math
owns physical-loss gradients, with no learner, optimizer, role conversion or IO.
Why this: stage fit worsened after a native-negative correction trial; observe an
actual unchanged step before choosing another training change. Defaults incur no
diagnostic math, and diagnostic failure releases no learner.

## Goal-consistent student-arrival corrections (2026-10-20)

The existing core correction batch gains a fieldless versioned observed subclass.
Core copy/projection helpers preserve family, feedback, original labels and arrival
ticks; infra alone reads privileged geometry/current feedback and selects the
teacher. The existing adapter applies updates privately and preserves definition/
provenance through paid receipts. Robot orchestration binds the explicit request
and sends no correction flag to pure frozen returns. Defaults/model objectives
stay unchanged; no new layer, runner or dependency.
[Usage and boundaries](docs/observed-grasp-corrections.md).

## Goal-consistent observed teacher (2026-10-20)

A narrow core subclass overrides held-object settling with Euclidean geometry;
the existing infra collector and CLI explicitly select it and bind a distinct
frozen result/definition/episode identity. Defaults and v1 supervision are kept.
Native final success remains an IO qualification; the core cannot observe arm
velocity or certify success. Existing learning readers refuse the new version.
No layer inversion, new runner, dependency or learner objective.
[Decision](docs/adr/ADR-0493-goal-consistent-observed-teacher.md).

## Explicit observed warm source (2026-10-20)

Existing artifact reader and robot CLI explicitly select qualified observed
TRAIN17 data while retaining legacy defaults and read-only B19. Original serialized
flags validate before normalization; existing core stage bindings retain two
fields and derive definitions from known source identities. No new learner/loss/
runner/dependency. [Decision](docs/adr/ADR-0492-explicit-observed-warm-source.md).

## Observed-grasp teacher collector (2026-10-20)

Existing infra collection composes the unchanged core observed teacher with current
raw42 feedback and a distinct frozen result subclass; the base result schema stays.
The existing CLI explicitly selects/binds the family and saves paid evidence.
Strict native state/record/feedback guards retain failed work before cleanup;
qualification requires complete TRAIN final sparse success. No new layer, runner,
model objective or dependency.90 scoped checks/platform and one15-control TRAIN17
teacher qualification passed; learner usefulness remains open.
[Decision](docs/adr/ADR-0491-observed-grasp-teacher-collection.md) · [API/evidence](docs/observed-grasp-teacher.md).

## Observed-grasp TRAIN readiness corrections (2026-10-20)

Opt-in current official grasp feedback selects closed transport/settle supervision
through the existing query, arrived-update and frozen-return path. A separate
teacher and record version retain feedback across raw/history views; geometry-only
v1 defaults, thresholds, actual student commands and all prior negatives remain.
Software146 checks/platform independently accepted. One native question corrected
held-state labels to10ALIGN40LIFT, but return still failed/no grasp; A reference
fit worsened, B slightly improved. All auxiliary readback failures are preserved.
[API, rationale, costs and scoped evidence](docs/observed-grasp-corrections.md).

## Explicit mixed BP private corrections (2026-10-20)

Existing correction adapter and robot runner accept explicit binary_gripper_axis=3
for exact mixed BP. Arrived endpoint labels and optional complete TRAIN replay
retain original clocks, budgets, permanent charges and mixed receipts; soft student
commands/history and frozen source remain unchanged. No new math or native recovery
claim. Software169 checks/platform and one separately budgeted native recovery
question are independently accepted. Native recovery was negative: all50 labels
were ALIGN despite source grasp, updated student never grasped, and both TRAIN
reference fits worsened. [Usage, costs and next dependency](docs/bp-binary-gripper-robot.md).

## Explicit BP binary-gripper robot objective (2026-10-20)

The robot learning API/CLI can select the existing BP mixed binary-gripper loss,
with explicit candidate/receipt binding and unchanged soft tanh serving. MSE
remains the default; PC/CPC objectives are unchanged.79 focused checks per platform
and independent review passed. A native run grasped/lifted the cube but missed the
goal; task success remains zero, and readback-budget conformance is unverified.
[Usage and qualified evidence](docs/bp-binary-gripper-robot.md).

## Optional reference rehearsal during corrections (2026-10-20)

The existing private correction and robot return APIs can compose bounded TRAIN
reference replay for all four neural families. Eight wake updates remain; replay
adds separately charged consolidations and exposure. Defaults are unchanged.
[API, costs and boundaries](docs/correction-reference-rehearsal.md). Final196
focused checks per platform and independent review passed. A subsequent native
trial remained negative: extra replay modestly reduced reference-fit damage but
delivered no autonomous success. [Evidence](artifacts/runs/taskspace12-reference-rehearsal-acquisition-20261020/RESULTS.md).

## Optional fixed physical taskspace12 (2026-10-20)

The existing fourfamily learner and robot/correction runner accepts a separate
12-input official-state recipe with fixed0.1m displacement scaling and causal
actual-command history. No empiricalfit; raw42 and default47 paths retained.
197focusedchecks/platform and independentreview pass, including a preserved
admissionfailure and repair. This establishes scoped softwarecorrectness.
[Usage and boundaries](docs/official-state-taskspace12.md) ·
[Evidence](artifacts/runs/official-state-taskspace12-20261020/RESULTS.md).

## History-consistent readiness corrections (2026-10-20)

The existing correction runner now supports47D history BP/PC/neutralCPC/activeCPC:
raw arrived labels, optional bounded teacher roll-in,8privateupdates and pure
studentreturn. History follows actual commands; source/statistics stay frozen.
62focusedchecks/platform and independent review pass. Native improvement remains
unverified. [Usage](docs/causal-action-history.md#readiness-corrections) ·
[Scoped evidence](artifacts/runs/action-history-corrections-20261020/RESULTS.md).

## Causal executed-action history (2026-10-20)

The existing learner/robot path supports47D causal executed-action inputs and an
equal-width zero control, with fixed history scaling and original raw traces.
137 focused checks per platform, scoped statics and independent review pass.
The first native zero arm failed during final capture after50zero-reward controls;
the originally unlaunched executed arm subsequently completed with zero success.
The comparison remains incomplete; no comparative benefit is established.
[Usage](docs/causal-action-history.md) · [Evidence](artifacts/runs/causal-action-history-20261020/RESULTS.md).

## Recorded readiness phase diagnostics (2026-10-20)

Pure core readiness_stage_diagnostics owns original TRAIN episode/stage bindings;
existing adapter sends selected phases to its existing residual helper. Robot
CLI binds original raw inputs before explicit projection and validates returned
sources/partitions before native work. No stage features, objectives, qualification
or outer-layer dependencies enter core. Why: aggregate fit hid phase failure;
reuse existing predictions and preserve the original saved evidence.
[Contracts](docs/readiness-stage-residuals.md), ADR0486.

## Explicit frozen scale floors (2026-10-20)

Core observation_standardization owns an exact frozen subtype with separate
empirical/declared/effective metadata. Existing TRAIN adapter/stream supplies one
fit; native request binding and savedraw42 diagnosis preserve selected semantics.
Private BP/PC/CPC updates and frozen serving use the same transform, with truthful
numeric statistics accounting. Why: tiny empirical spreads exposed conditioning
risk, but olddefaults/evidence require an explicit opt-in. No new framework or
outer dependency enters core. [Contracts](docs/observation-scale-floors.md), ADR0485.

## Bounded teacher prefix and student handback (2026-10-20)

Core pick_cube_readiness_roll_in validates both original commands and selects an
explicit native32 action; app readiness_roll_in owns at most50 attempted/returned
receipts. Existing infra readiness query supplies its last observed label without
extra geometry; scripts compose mixed collection and pure frozen candidate return.
Dependency: script→app/core, infra→core; no simulator/IO/model enters core/app.
Why: later teaching coverage was missing, and original action precision differs
from the actual native control boundary. [Contracts](docs/readiness-roll-in.md),
ADR0484. Existing defaults and paid failure/source/clock guards remain in place.

## Canonical readiness correction view (2026-10-20)

Pure core state_corrections owns detached raw/model readiness batches, retaining
original arrival authority and explicit projection/pairing work. Existing adapter
admits raw arrivals before deriving model features, reuses private updates/frozen
statistics, and validates native state/birthbudget/receipt/clock continuity across
callbacks. Existing smoke composes bounded17 acquisition or default19→17→19.
Why: the missing canonical correction view prevented student-trajectory teaching;
no new orchestration layer is needed. Scripts/adapters → pure core; no IO/model
enters core. [Contracts](docs/state42-canonical-corrections.md), ADR-0483.

## Explicit state42 feature view (2026-10-20)

Pure core `pick_cube_state_quaternions` validates owned42 vectors/episodes and
canonicalizes the two unit-quaternion signs. Existing robot CLI composes the
same explicit recipe before fit and frozen native inference; smoke retains raw
measurements and separate projection work. Dependency: scripts → pure core;
no SDK/IO/model dependency enters core. Why: antipodal features need a consistent
training/serving view without changing raw evidence or fitting statistics again.
Typed metadata/full-state/late-callback guards preserve paid failure evidence.
[Contracts and checks](docs/state42-quaternion-features.md), ADR-0482.

## Recorded paired inspector (2026-10-20)

`src/app/paired_quota_inspector.py` is a pure bounded JSON projection. The existing
`key_door_inspector` dispatcher injects its episode playback function; the paired
module never imports the dispatcher, adapters, learner or environment. The
existing infra writer/template consume this projection through the unchanged CLI.
Dependency direction: CLI/adapters → infra/app → pure core playback rules.
Why: separate source-stage binding from existing episode reconstruction without
a new framework or circular import. All callbacks receive owned source copies;
complete/partial bindings and displayed cost types fail before rendering.

Saved quota ordinals bind physical/native stages separately from raw episodes;
absent physical fields are not repaired. Action mapping affects legal/apply
paths, while observation channels stay spatial. Raw and legal-conditioned laws
are separately validated/displayed. Comparison and inclusive timing are copied,
not reevaluated. This is offline presentation, with no policy/model execution
or final-data feedback to training. [Usage and limits](docs/paired-inspector.md).

Recorded-state follow-up: the same pure binder validates optional observation
arrays/counters and selects by physical ordinal before native-index validation.
Unbound raw observations remain preserved but unavailable. The existing template
clears stale DOM before selection changes and renders actual shape/source pairs;
no model reconstruction, byte inference, new layer or dependency. Why: all five
frozen native indices are0, so matching solely by update count selects wrong data.

## Paired physical quota control (2026-10-20)

key_door_quota_comparison composes the existing key_door_reward_pilot arm and
finish-only evaluation. A private optional initialized-state list permits paired
quota deferral while preserving old routes. Two own-policy budgets/quotas/RNGs
share exact initial native values, not mutable owners. Bounded TRAIN/completed
row seals and eight final packets guard costs/results across peer callbacks.
Why: equal on-policy exposure cannot be represented by common teacher packets
or the existing demonstration warm-start comparison. No new native objective,
environment, runtime or dependency. Existing forks' initialization cost remains
visible. [Contracts/checks](docs/paired-quota-control.md) and ADR-0479.

Timing follow-up: seven fixed scalar records measure TRAIN/evaluation callbacks
and inclusive pair/predecessor/pending guards. They stay outside the paid row
seals and native state; a separate perf_counter never replaces the monotonic
budget. Ordinary clock errors preserve primary paid failures. No call history,
new collector or timing-driven training/evaluation decision is introduced.

## Camera stream admission (2026-10-20)

Only the existing run_native_stream_demo CLI production boundary changes. Fixed
camera TRAIN receipts reuse TaskTeacherResult qualification, bounded episode
parsing and RGB lifecycle validation. The default A recipe and shifted camera
recipe stay explicit through the existing public21 projection and result metadata.
The stage-only native case composes existing strict readers, sequence runtime and
frozen rollout; no new learner, runtime, codec or dependency. Why: the qualified
camera teacher existed but lacked the ordinary saved-stream admission route.
[Boundaries, failure behavior and checks](docs/camera-stream-input.md).

## Cumulative continuous contexts (2026-10-20)

The existing continuous_stream_demo entrypoint delegates explicit sequence options
to continuous_stream_sequence. This adapter owns bounded TRAIN admission and
readout/report composition around the existing ActorShadowRuntime; no runtime,
learner, core or simulator boundary changes. Fresh readout forks expose no parent
authority or future labels. Why: preserving the legacy default while composing
several contexts does not require another orchestration framework.
Explicit fixed-input5/6-release scopes preserve complete packet identities and
the256-row capacity; default3 remains. The B22 coverage case appends its separate
22-row episode after198 original rows, with six original per-invocation charges.
[API, costs and failure behavior](docs/continuous-context-sequence.md).

## Nearest corrective composition (2026-10-20)

The existing correction adapter dispatches eight MLP batches or one nearest append
through ActorShadowRuntime. Existing raw paired TRAIN corrections/arrival clocks
remain authoritative; native-owner checks follow public getters before work and
release. The existing smoke pipeline composes collection and frozen readouts,
including nearest distance/prototype costs. Core learners, generic runtime and
controller are unchanged. Why: repeated prototype appends consume capacity without
new information. [Interfaces and boundaries](docs/nearest-corrective-adaptation.md).


## Saved control diagnosis (2026-10-20)

`scripts/diagnose_learned_robot_control.py` composes existing core episode records
and frozen standardization with saved TRAIN JSON. Only exact full observations
admit supervised errors; prefix/range summaries describe support. It uses existing
controller unit constants, with no learner, optimizer, simulator, framework or
new dependency. Why: teacher actions at corresponding times are not expert labels
for a diverged student state. [Boundaries and checks](docs/learned-control-diagnosis.md).
Explicit state42 mode validates the official noRGB TRAIN contract and summarizes
all42 coordinates; default RGB789/prefix outputs remain unchanged. Both modes
share exact-input label consistency and frozen statistics. Why: public21 is an
RGB-prefix contract and cannot describe a full-state trajectory by truncation.


## Replay comparison composition (2026-10-20)

`scripts/continuous_replay_example.py` reads the accepted saved control cohort,
composes the existing continuous runner/replay/disposable probes, and publishes
bounded scalar comparisons. It reuses native-state and completed-probe contracts;
no new learner, runtime, optimizer or dependency. Current owners are checked after
all public result observers; paid failed reports remain separate from validated
success. [Protocol, costs, interpretation](docs/continuous-replay.md).


## Continuous replay composition (2026-10-20)

`continuous_learnability_demo` -> `app.continuous_stream_rehearsal` -> existing
`ActorShadowRuntime.consolidate`, `ToyBudgetSession`, complete continuous records
and injected native full-state equality. The app owns bounded fixed TRAIN input,
separate permanent replay charges and publication checks; adapters own model
selection, native diagnostics and work reporting. App imports no adapters/infra.
Why: categorical replay cannot safely encode continuous target vectors. No new
runtime/optimizer or dependency. [Contract/status](docs/continuous-replay.md).


## Current local integration: unexecuted matched probe arms

scripts.continue_continuous_probe_control composes existing continuous runner and
accepted JSONnormalizer, reusing paidBP scalar evidence and calling only missing
PC/neutral/active. No new inner layer/optimizer/runtime framework or hash loader.
Return reports are owned before later observations; captured native PC fullstates
finish AFTER all public getters. Existing neutral-array/commonreport gate precedes
active; failure retains paid reports and unknown observations remain unknown.
Only fixed bounded scalar cohort reports, no old parameter archive or model chain.
[Recipe, source boundaries and limitations](docs/continuous-fresh-task-probes.md).

## Current boundary: reserved continuous probe JSON and CLI

scripts.run_native_stream_demo -> infra.continuous_probe_json -> core continuous
records; CLI -> existing adapters.continuous_probe plan -> existing continuous
runner/ActorShadowRuntime. Inner layers do not import IO. One strict outer reader
bounds2MiB+1 bytes/schema/native scalar/list/role conversion into owned records.
CLI binds final model spec/context/IDs and fullRGB suffix before fitting, adds
caller-reservation metadata only on opt-in; it grants no collection/release.
Default dispatch/output and mathematical/runtime components remain unchanged.
No new scheduler, optimizer, hashing/provenance framework or environment variable.
[Boundary contract/tests](docs/continuous-fresh-task-probes.md).

## Current adapter addition: paired disposable continuous probe

continuous_learnability_demo -> continuous_probe -> continuous_probe_control leaf
-> supplied unchanged single-probe callback -> existing ActorShadowRuntime/native
learner. No cycle: leaf never imports continuous_probe. No new core/runtime layer.
Only detached source velocity changes; parameters/fullpolicy/globalcounter/fullwake
remain exact. Private expected/restoreargument copies and retained paidreports,
checked after callbacks, prevent evidence replacement. Two separate fixed local
probe_train allowances; probe_test score-only. Partial costs/failures bounded and
preserved. DefaultFalse unchanged, no probe learner returns to benchmark/deployed
state. [Contract and tests](docs/continuous-fresh-task-probes.md).

## Current integration: learned continuous robot command

scripts/run_continuous_robot_learning composes existing qualified input readers
-> continuous_learnability_demo/ActorShadowRuntime -> detached native candidate
-> maniskill_continuous_smoke frozen return. Core/learner/optimizer/runtime layers
are unchanged. The outer script admits owned TRAIN inputs and complete handoff
metadata, preserves returned reports before later assertions and uses exact
native-owned snapshot validation after public observers to catch late callbacks.
It retains one expected state, not a success/model chain. The native API alone
owns scene/cleanup/trace/control/joint accounting; no promotion/model IO/new runner.
Native zero reward is a complete runtime outcome and supplies no useful-skill
or comparative-benefit proof. Original source/private owners remain frozen.
[Boundary, costs and extension contract](docs/learned-continuous-robot.md) ·
[ADR0463](docs/adr/ADR-0463-run-fixed-continuous-learning-through-existing-stream.md).

## Current evidence: existing stream, four matched lighting arms

The accepted CLI -> continuous_learnability_demo -> ActorShadowRuntime/native
learner path required no production change for the cached lighting comparison.
An artifact-local environment shim checks Linux thread-setting strings before
CLI imports; stdlib saved-report validation/presentation runs no model math.
Existing role/owner/native-receipt boundaries remain unchanged. Same update and
information exposure does not imply equal BP/PC computation. The next runnable
integration can compose the already returned detached candidate with the existing
frozen robot API; no additional orchestration or checkpoint layer is needed.
[Evidence and limits](artifacts/runs/matched-continuous-lighting-20261020/RESULTS.md).

## Current integration: exact PC/CPC frozen robot caller

Existing maniskill_continuous_smoke unwraps PC basepolicy only for controller/input
geometry, retaining full PCconfig/wake/counters and reusing accepted continuous
state validation. Exact source/private owners remain frozen before predictions/
controls and after callbacks/cleanup; final native-owned state validation avoids
callback races without another runner or audit framework. Prediction is feedforward
only, no latent/optimizer/wake updates. BC/nearest serving and corrector remain
unchanged; exact PC corrective queries now compose through the existing private runtime. Separate numeric/derived costs
exclude validation/copy/process overhead. Native smoke establishes no learned skill.
[Contract](docs/frozen-bc-robot-rollout.md) ·
[ADR-0463](docs/adr/ADR-0463-run-fixed-continuous-learning-through-existing-stream.md) ·
[Executed evidence](artifacts/runs/continuous-pc-rollout-20261020/RESULTS.md).

## Current integration: opt-in physical probe direction

Pure continuous_step_diagnostics depends inward on the existing task-gradient
kernel/standardization and returns scalar groups/work only. The continuous_probe
adapter alone binds released PRE/AFTER states, native paid receipts and original
probe authority. Common physical BP derivatives remain distinct from PC/CPC
inferred/gated energy partials. Diagnostics cannot update/promote benchmark or
probe owners; failures retain charges/work and stop the stream. DefaultFalse
preserves existing reports/calls; no native optimizer API or new runner.
[Contract and extension](docs/continuous-fresh-task-probes.md) ·
[ADR-0463](docs/adr/ADR-0463-run-fixed-continuous-learning-through-existing-stream.md) ·
[Accepted evidence](artifacts/runs/continuous-probe-direction-20261020/RESULTS.md).

## Current integration: disposable continuous probes

continuous_probe.py is a focused adapter leaf composing the existing probe_train
ActorShadowRuntime. The existing continuous runner invokes four fixed independent
scopes. Reserved-role ownership/geometry admission precedes fit, probe_test labels
enter only physical scores, and scalar results return without adapted models.
Full native/wake state and original benchmark authority stay separate. Bounded
native/receipt reports survive late failures; numeric costs exclude validation/
metadata/process overhead explicitly. No classifier API or core layer is widened.
[Contract](docs/continuous-fresh-task-probes.md) · [ADR-0463 followup](docs/adr/ADR-0463-run-fixed-continuous-learning-through-existing-stream.md) ·
[Evidence and open direction diagnostic](artifacts/runs/continuous-isolated-probe-20261020/RESULTS.md).

## Current integration: matched continuous stream families

The existing continuous_learnability_demo composes BP or accepted continuous PC/
CPC in the same ActorShadowRuntime. continuous_learnability_modes owns a bounded
native-port scalar-report facade and common pure TRAIN task diagnostics; raw
source/readout/final models remain detached native learners. Complete valid
adaptive equality plus one expected candidate protects preceding updates/final
observations without retaining a model history. Native event costs, diagnostic
work and excluded validation/copy/process overhead remain explicit; no BP training
formula is assigned to PC/CPC. No new runner, data release or serving promotion.
[Contract](docs/continuous-mlp-learnability.md) · [ADR-0463 followup](docs/adr/ADR-0463-run-fixed-continuous-learning-through-existing-stream.md) ·
[Evidence and open probe dependency](artifacts/runs/continuous-learnability-modes-20261020/RESULTS.md).

## Current composition: optional continuous CPC state

core/continuous_cpc.py owns validated wake/config copies and composes existing
coupled chemistry/gate equations. The same adapters/numpy_continuous_pc.py optimizer
uses optional final policy/snapshot fields, gates current partial gradients and
atomically publishes full momentum/parameter/wake state. No IO, data selection,
runner or history chain is added. NativeLearner/ActorShadow boundaries remain;
existing continuous runner integration is next. Four transition counters exclude
additional validation/copy overhead. [Contract](docs/continuous-cpc.md) ·
[ADR-0478 followup](docs/adr/ADR-0478-own-continuous-pc-native-updates.md) ·
[Scoped evidence](artifacts/runs/continuous-cpc-bridge-20261020/RESULTS.md).

## Current adapter: owned continuous PC updates

adapters/numpy_continuous_pc.py composes the existing BC initializer/validated
vector-bias state with continuous PC row-bias energy and shared guarded inference.
The one-hidden architecture matches the current robot BP control. NativeLearner
and existing ActorShadowRuntime provide delayed training/stable serving without
another runner. Parameter/momentum/counters publish in one assignment only after
candidate validation; errors retain copied actual work. Core's private per-call
progress record adds failure accounting without public API/guard-math changes.
No IO, evaluation choice, replay or promotion belongs to this adapter.
[Contract](docs/continuous-pc-learner.md) · [ADR-0478](docs/adr/ADR-0478-own-continuous-pc-native-updates.md) ·
[Scoped evidence/limits](artifacts/runs/continuous-pc-learner-20261020/RESULTS.md).

## Current composition: bounded continuous latent inference

core/continuous_pc_inference.py owns the continuous problem/policy and composes
accepted continuous_pc energy with coupled_inference._run_inference. The existing
binary guard loop is one private typed shared function; binary API/math/ordering
and update behavior are preserved. Core depends only on core. No IO, role release,
parameter optimizer, trajectory, new runner or public callback framework is added.
Target-conditioned inference outputs remain separate from target-free prediction;
existing NativeLearner/stream/state boundaries are the next integration point.
[Contract/cost boundaries](docs/continuous-pc-inference.md) ·
[Scoped gates and example](artifacts/runs/continuous-pc-inference-20261020/RESULTS.md).

## Current addition: continuous PC task energy

src/core/continuous_pc.py composes existing CoupledParameters and continuous BC
geometry/validation into a separate pure multi-output likelihood. Hidden penalties
sum neurons/average rows; task loss averages rows/active action axes. Owned inputs
and outputs, exact partial derivatives, finite preactivations; no IO, optimizer,
learner state or role authority. Binary energy/BC sources remain unchanged.
The runnable script only illustrates fixed mathematical inputs. Existing binary
inference is a concrete next dependency before the continuous learner adapter;
reuse current stream/state boundaries rather than adding another runner.
[Contract](docs/continuous-pc-energy.md) · [ADR-0477](docs/adr/ADR-0477-separate-continuous-pc-task-energy.md) ·
[Scoped evidence](artifacts/runs/continuous-pc-energy-20261020/RESULTS.md).

## Current composition: frozen rollout joint telemetry

The existing smoke API/CLI conditionally composes the port's owned joint monitor;
core owns strict summary/sign/count validation and bounded raw-position checks,
infra owns paid native-work validation. Final reporting uses existing raw traces,
without model projections, added native calls or learning decisions. Independent
work access and final owner/context checks preserve failed paid work and withhold
usable corrections after reporting failure. No new runner/layer/dependency.
[Interface and limits](docs/joint-position-telemetry.md) ·
[Scoped evidence](artifacts/runs/joint-rollout-integration-20261019/RESULTS.md).

## Current addition: owned physical joint-position observations

`src/core/joint_position_limits.py` owns finite immutable limit metadata and an
O(joints) signed-extrema/count monitor, <=51samples, no IO/NumPy dependency or
control/learning decisions. `src/infra/maniskill_continuous.py` composes it via
explicit monitor_joint_positions=False, one public native limit snapshot and
existing encoded qpos9. Dependency remains infra -> core. Defaults/features/
actions unchanged; pure report/work snapshots survive close/failure. Changed
robot/helper/admission binding permanently refuses further controls/episode export.

[Rationale/alternatives](docs/adr/ADR-0476-observe-signed-physical-joint-limits.md) ·
[API, sampled-time/reset-limit scope and extension](docs/joint-position-telemetry.md).
Scoped365Windows/Linux checks plus one independently accepted native29control
case do not establish collision/velocity/substep safety or full R4.5 readiness.

## Existing MLP comparison and WSL resource boundary (2026-10-19)

The existing lighting reader/stream runner produced a jointly frozen A/B/A vsA-only
appearance comparison, without production changes. Source/actor isolation, delayed
receipts and diagnostic copies remain in their existing layers. Both raw outcomes
and exact shared-prefix evidence are retained; this is not a new learner design.

Resource boundary correction: setting thread variables only on Windows wsl.exe
parent does not establish Linux child settings. Future launches pass explicit
/usr/bin/env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 insideWSL.
A zero-model configuration check passed; prior arms' actual thread caps remain
unverified, not requalified. No observer framework or experiment retry added.
Next physical joint-limit telemetry belongs in pure core summaries +the existing
infra episode boundary, outside learner features/control feedback by default.
[Evidence and limits](artifacts/runs/lighting-mlp-joint-20261019/RESULTS.md).

## Qualified saved lighting inputs in existing stream CLI (2026-10-19)

Bounded owned JSON -> exact declaration/readback/work + current TaskTeacherResult
qualification -> ContinuousEpisode -> existing nearest/MLP stream runner. The
outer CLI validates original acquisition evidence before any fit/runtime creation;
inner learners and objectives remain unchanged. Default/mass routes refuse light
markers, while compatible None fields/default report keys remain supported.

Why this: explicit lighting ingress preserves TRAIN17 identity and isolates heldout
roles without another runner. Shared complete-field checking prevents Python context
dataclass defaults from backfilling missing serialized light/mass evidence. Metadata
comes from the same parsed payload, with historical acquisition separate from new
compute. [Usage/boundaries](docs/pick-cube-lighting-context.md#saved-lighting-teachers-in-the-existing-stream).

## Fixed lighting acquisition and return composition (2026-10-19)

Saved RGB nearest state/statistics -> existing qualified teacher -> existing
StableActor/ActorShadowRuntime -> existing frozen rollout. The stage-only case
restores A without fitting, withholds B labels until2, appends once privately and
reads B then fresh A. Frozen owners/prefix/stats, original light readback and
preceding JSON contents/receipts bracket dependent work; paid failures remain.

Why this: demonstrate usable adaptation through accepted boundaries. No production
layer/objective/runner change. Same-seed lookup recovery has explicit prototype/
distance costs; it is not neural generalization or PC/CPC evidence. Next the
existing stream CLI needs strict lighting-teacher ingress for MLP diagnostics.
[Case/limits](artifacts/runs/lighting-context-return-20261019/RESULTS.md).

## Lighting metadata through existing teacher/rollout boundaries (2026-10-19)

Fixed core declaration → existing episode port → existing teacher or frozen
rollout, with owned typed original readback/paid-work metadata. Qualification
requires complete TRAIN17/baseRGB/controller/light/visual lifecycles. Pure final
port checks follow external owner/getter callbacks without another live light
read. Shared visual validation separates partial paid failures from successful
n+1/zero-export accounting; cumulative counters cannot decrease.

Why this: accepted native lighting could not enter these existing interfaces.
Keep orchestration and model objectives unchanged; no second runner/framework.
Default CLI fields stay exact, raw Python result appends three None fields.
[API and next extension](docs/pick-cube-lighting-context.md#teacher-and-frozen-rollout-integration).

## Ambient-light binding before visual encoding (2026-10-19)

Pure core declaration → existing episode port → focused infra native light helper
→ existing frozen RGB encoder. The helper owns original measurements and constant
read/write/refresh/discard counters; the port supplies binding/identity checks and
keeps ordinary control/endflag/work semantics. Defaults never enter lighting IO.

Why this: reset pixels precede a post-reset setter. Discard that packet and pay
for one explicit refresh before admitting changed-context features. Capture
expected values/IO identities across callbacks, retain original paid errors and
permanently refuse corrupted evidence. No learner/optimizer/runner change.
[ADR0475](docs/adr/ADR-0475-bind-ambient-light-before-visual-encoding.md) ·
[API and extension boundary](docs/pick-cube-lighting-context.md).

## A-only information control in the existing runner (2026-10-19)

CLI opt-in → existing learnability runner → existing delayed ActorShadowRuntime
updates on warm A only. Actual A/B arrays still enter the unchanged four read-only
checkpoints; both inputs require released TRAIN admission before fit. Actual
context/inbox/receipt identities and permanent costs remain explicit. Work derives
training counts from selected sources; control-only metadata separates per-source
training and diagnostic exposure. Default False preserves reports/call shapes.

Why this: distinguish B-label contribution from ordinary continued optimization
without a second orchestration framework or learner/math change. Unequal sources,
B-target perturbation and direct full momentum tests verify isolation/accounting.
A frozen saved-trajectory comparison was negative on all non-tied aggregate
metrics, with a gripper tradeoff; no native/generalization claim follows.
[Guide and extension](docs/pick-cube-mass-context.md#matched-a-only-control).

## Qualified mass input in the existing stream CLI (2026-10-19)

Bounded owned JSON → shared private episode decoder → existing typed mass/teacher
qualification → existing nearest or MLP stream runner. Explicit mass selection
binds new TRAIN17, retaining context/readback/work and historical acquisition in
owned report metadata. Defaults keep TRAIN17/19; non-None guard/camera/mass markers
cannot be silently discarded. Complete RGB counters are checked when opted in.

Why this: the prior hardcoded new seed19 prevented usable mass diagnostics. IO
stays at the CLI boundary; core/learners/runtime/optimizer and dependencies are
unchanged. Public-vector projection remains limited to its declared contexts.
One separate fixed MSE A/B/A run used existing ownership/label-delay/diagnostic
checks; no new orchestration or persistence framework. A-only equal-exposure
control is needed before causal attribution. [Guide](docs/pick-cube-mass-context.md).

## Fixed mass acquisition/return composition (2026-10-19)

Saved A-only nearest snapshot and frozen statistics → existing qualified mass
teacher → ActorShadowRuntime with labels at logical2 → one private append →
existing frozen B and A-return rollouts. Serving remains A-only. Artifact scripts
compose existing ports; no production orchestration, learner, dependency or
persistence framework is introduced. Full owner/receipt/prefix/retained-trace
checks precede dependent work; paid failures retain charges.

Why this: validate the new physical-context interface with an operational memory
control. A narrow legacy-source view binds the accepted older inner context and
absent recipe; current native receipts stay strict. Original failed launch is
preserved, corrected allocation reviewed prospectively. This is no neural/CPC/
generalization or full safety proof. [Guide](docs/pick-cube-mass-context.md) ·
[outcomes and source boundaries](artifacts/runs/mass-context-return-legacy-20261019/RESULTS.md).

## Mass context across existing use boundaries (2026-10-19)

Declared context → existing episode port → qualified teacher or frozen rollout →
opt-in CLI/report fields. Common typed readback/work validators and a trusted pure
final port check follow external owner/getter callbacks; no hidden native read.
Original measurement/full source state/paid counters retained. Default CLI schema
unchanged; raw result dataclass adds optional fields. No new runner/learner/loss.
Why this: make already verified physical setting usable in existing adaptation
paths without admitting mismatched teacher or stale owner evidence.210focused
platform checks/scoped statics/review pass; native adaptation remains untested.
[ADR0474 follow-up](docs/adr/ADR-0474-declare-bounded-mass-only-context.md) ·
[tree/API/limits](docs/pick-cube-mass-context.md).

## Explicit object-property context (2026-10-19)

Pure fixed TRAIN17 mass declaration → existing episode port → one after-reset
public SDK mass setter/readback and live identity guards. Constant original
measurement/work snapshots are owned; inactive default calls never resolve mass.
No new orchestration/learner/dependency.216focused platform checks and real CPU
API restoration pass; inertia/COM and adaptation remain unmeasured.
Why this: qualify a concrete missing object-property boundary before extending
existing teacher/rollout reports. [ADR0474](docs/adr/ADR-0474-declare-bounded-mass-only-context.md) ·
[structure/API/checks](docs/pick-cube-mass-context.md).

## Saved mixed native composition and next boundary (2026-10-19)

Qualified sources/public projection → explicit axis3 policy + bounded8array
snapshot restore → existing physical readouts/frozen native rollout. MSE
references retain their own axisNone policy. Full owner/preceding trace guards
admit second controls only after validation; paid failures remain recorded.
No production learner/runner/loss or dependency changes.41tests perplatform and
scoped statics/independent acceptance; native benefit remains negative.
Why this: directly test offline-to-native transfer without retraining baselines.
Next object-mass context belongs in pure declared metadata + existing infra
port with observed readback/default guards, not a new orchestration framework.
[Tree/API/cost/limitations](docs/public-vector-mixed-native.md).

## Matched public21 objective control (2026-10-19)

Qualified sources/public projection + read-only p32MSE → existing initialization
parity/owned-result runner with explicitaxis3 → fullphysical comparison and typed
8array finalstate. Explicit mixed receipt validation leaves acceptedMSE contracts
unchanged. Why this: test an existing objective hypothesis at matched exposure,
without new learner/math/fitter/runner or baseline retraining. Mean gains retain
axis/retention harms; [composition and limits](docs/public-vector-mixed-control.md).

## Saved-public21 corrective native composition (2026-10-19)

Existing saved-source loader/readouts → existing corrective pipeline → fixed
collection19/eight private updates/frozen17then19. Validation-only wrappers
bind retainedraw trace and paidupdate after owner getters before morework and
restore originalcallables finally. No production architecture/math/runner change.
Why this: synthetic correction correctness does not prove native utility. The
single actualcase remains negative; [structure and limits](docs/public-vector-corrective-native.md).

## Owned public21 correction views (2026-10-19)

Existing raw789 collector → pure core raw/model correction view → existing
private eight-update adapter/runtime → existing frozen21D rollouts. Core stays
inward-only; the updater derives the view after original arrival admission.
Raw actions/labels/clocks remain exact; final release rechecks evidence and
owners after the last getter. Why this: preserve raw query pairing while giving
the learner its existing21D specification, without a new fitter or framework.
Default paths remain; [ADR0473](docs/adr/ADR-0473-own-public-vector-correction-views.md)
and [structure/costs/limits](docs/public-vector-corrections.md).

## Saved p32 native validation through existing paths (2026-10-19)

Accepted96state → existing bounded array/typed snapshot/readout helpers →
existing public21 frozen rollout17then19. Explicit historical24/704 reference
counters remain distinct; full-owner/preceding-evidence checks retained. No
production learner/runner/math change. Why this: test native utility without
retraining after lower offline mean error. Both task outcomes remain negative.
[Structure and limits](docs/public-vector-mlp-depth-native.md).

## Fixed public21 p32 study through existing runner (2026-10-19)

Qualified existing TRAIN sources/public projection + accepted p8 report → bounded
stage comparison → existing A/B/A owned-result runner → existing typed snapshot.
No production source/learner/math/newrunner change. Why this: use the accepted
explicit count option for a prospectively fixed optimization-depth question.
All new96paid updates remain visible; improved mean fit and mixed worst-error/
retention outcomes do not establish native skill. [Composition](docs/public-vector-mlp-depth.md).

## Bounded fixed A/B/A count in the existing adapter (2026-10-19)

Existing stream CLI → count admission → existing continuous learnability adapter
→ existing actor/shadow runtime and NumPy learner. Exact-int1..128/default8;
derive finite packet budgets and work without changing core math, boundaries,
source/actor isolation, endpoint diagnostics or historical experiment readers.
Why this: explicit fixed work enables a prospective optimization-depth question
without a new runner or outcome-dependent schedule.103scoped checks perplatform
and independent review pass; scientific benefit is untested. [ADR0472](docs/adr/ADR-0472-bounded-continuous-phases.md).

## Saved public21 MLP native composition (2026-10-19)

Accepted report/eight-array snapshot → bounded stage restoration → existing
TRAIN residual diagnostics → existing public21 frozen rollouts17then19.
Production learner/runtime/math/defaults unchanged. Full owner checks and
preceding saved-trace validation preserve the existing boundaries. Why this:
saved state allows native validation without spending another training budget.
Scoped checks and independent review pass; both native task outcomes are
negative. [Structure, ownership and limits](docs/public-vector-mlp-native.md).

## Explicit public21 learnability through existing CLI/runtime (2026-10-19)

Qualified raw TRAIN789 readers → pure owned public21 views → existing generic
continuous learnability runner/owned result. Explicit CLI metadata/costs and
pre-IO incompatible-option refusal; defaults/source records unchanged. Stage
comparison reuses accepted RGB report and saves complete final8arrays under
existing snapshot semantics. Why this: make the sufficient simpler input usable
by the neural baseline without inventing a learner/runner or retraining merely
for a later native check. Final mean errors worsen despite lower arithmetic;
offline/model-layout limits explicit. [Contract/API](docs/public-vector-mlp-learnability.md), ADR0471.

## A-only public21 native control reuses the existing rollout (2026-10-19)

Stage-only accepted A teacher/full789 statistics+warm prototypes → exact21prefix
restore → same actual StableActor B→A through existing frozen rollout. Full
source/actor and returned/retained/preceding trace checks guard each episode.
No production change/new learner/runner/fit/append/SGD. Why this: test the missing
simpler input control with acquisition/render costs preserved. Both succeed;
this known-fixture sufficiency does not prove neural/visual/generalization gain.
[Composition/costs/extensions](docs/public-vector-camera-control.md). Next use
existing generic continuous learnability runner for a frozen21D MLP ablation.

## Explicit public21 input at the existing rollout boundary (2026-10-19)

Pure known789→21 projection → matching existing learner/actual actor → existing
frozen rollout. Core validates all source coordinates and recipe, owns TRAIN
views and preserves controller/role contracts. Adapter retains raw789 traces,
renderer/camera cost and defaults, accounting projection attempts/returns against
prediction progress. Existing full-owner checks run after metadata reads before
control and at cleanup. Unsupported clock/correction combinations refuse.
Why this: isolate the missing simpler matched input control without changing
the encoder, privileged state, learner, runner or dependencies. Scope accepted
under fake IO; native matched control and scientific benefit remain untested.
[Boundaries/API/extensions](docs/pick-cube-public-vectors.md), ADR0470.

## Native camera acquisition reuses the existing stream (2026-10-19)

Stage-only qualified camera teacher → one A fit/append → actual StableActor
camera baseline → existing delayed0/1→2 private append → detached camera/default
return rollouts. Source/actor unchanged, one permanent charge, frozen stats/A
prefix exact. Receipt/full-owner validation precedes empty polling and repeats
at later seams; paid evidence survives postcommit failure. No new production
interface, kernel, schema, dependency, observer or runner. Why this: execute the
new camera capability with a working simple control before broader drift.
[Source/API/limits](docs/camera-context-acquisition.md). Observed memory acquisition
and return do not prove adaptation necessary; next add explicit public21 input
projection/control while preserving rawRGB/defaults and rendering costs.

## Explicit named camera context through existing native paths (2026-10-19)

Pure immutable camera context/known recipe → existing port factory/public RGB →
existing TRAIN teacher or frozen rollout. Default strict RGB validator unchanged;
explicit opt-in binds TRAIN17/visual/context before creation. Contexts are copied;
port corruption permanently stops control/record admission with paid work kept.
Teacher/rollout validate typed metadata/recipe at seams and cleanup; qualification
binds camera identity/context/spec/sparse success, even when other markers are
removed. No origin attestation against rewriting all metadata. Geometry stays
label-side; incompatible corrective queries refuse. Default CLI JSON/calls exact.
Why this: add the missing appearance-context capability with a working simple
control, keeping neural R4.4/full R4.5 open. No new runner/schema/dependency.
[API/boundaries](docs/pick-cube-camera-context.md), ADR0469; fake IO verified,
actual camera effect and adaptation unexecuted.

## Range learner native validation reuses existing boundaries (2026-10-19)

Stage-only composition → existing continuous learnability result API → private
range24 candidate → existing frozen native rollout. Typed recipe/settings/source/
receipt/work gates precede paid reconstruction;38 quantitative checks precede
controls. Full original/private state stays exact, each completed trace validates
before the next. No production interface, kernel, schema or dependency change.
Why this: validate offline improvements in the actual environment with one bounded
case. Native outcomes are negative. Next missing capability is an explicit known
camera context through existing RGB port/teacher/rollout while preserving default
recipe binding; neural R4.4 and full R4.5 stay open.
[Source tree, checks, resources and limitations](artifacts/runs/rgb-range-native-20261019/RESULTS.md).

## Explicit sensor-range RGB normalization through existing policy (2026-10-19)

ExistingCLI/recipe → TRAIN fit adapter → pure pick_cube_rgb_standardization →
existing populationfit(prefix21) +fixed RGB768 means/scales.5. Frozen RGBrecipe/
completeTRAINrange admission precedes fit/model; no B fit or terminalfit. Policy
keeps existing flat owned stats, so fork/restore/actor/native transforms need no
schema/kernel change. None preserves empiricaldefault calls/report/work.
Whythis: bound a measured RGBinput excursion using declaredsensorsemantics,
without tuning a variancefloor or clipping. Metadata distinguishes empirical/
fixedcoords and validationcosts; vectorprefix remainsunbounded. No newdependency.
[API/limits](docs/pick-cube-rgb-standardization.md), ADR0468. Offline finalmeans
improve withaxis/forgettingregressions; native skill remains unproven.

## Corrective learning composes existing rollout and private stream (2026-10-19)

Existing robot smoke exposes FrozenRolloutResult/run_frozen_rollout from the same
CLI body. continuous_correction_demo validates owned arrived TRAIN pairs before
forks, then uses existing ActorShadowRuntime for8 fullbatch exposures. Original
source/actor/policy/stats/momentum are preserved; full candidate continuation and
original labeltimes/paid receipts checked beforemorework and after detach. Existing
script composes collection19→privateupdate→frozen17/19, stoppingfailedstages and
preserving paidreports before IO. Whythis: connect accepted ports without another
simulatorloop/modelarchive/framework. No new dependency or serving publication.
[API/boundaries](docs/continuous-corrective-adaptation.md), ADR0467. Native result
negative; taskbenefit/fullR4.4/R4.5 remain open, defaults unchanged.

## Separate provisional corrections from executed robot controls (2026-10-19)

Existing smoke → infra PickCubeCorrectionQuery → existing fixed teacher/geometry;
core PickCubeCorrectionBatch owns public step/observation pairing and arrived
training pairs. Geometry stays infra label-side. Actual student trajectory/actions/
rewards remain separate and unchanged. Port retains admitted seed; query binds
exact first owner/17-or19 seed, and existing full original/private learner guards
run before controls and after cleanup. Why this: collect supervision on visited
states without inventing an executed expert trajectory or another runner.
Default unchanged; failed work retained/no usable batch. No new dependency.
[API/boundaries](docs/pick-cube-corrections.md), ADR0466. Physical quality/learning
benefit remain unexecuted; nominal off-reference targets explicitly provisional.

## Explicit existing objective in the fixed stream recipe (2026-10-19)

Existing CLI → continuous_learnability_demo → existing policy/core kernels and
ActorShadowRuntime. Optional binary axis validates both TRAIN episodes before
fit/model, and actual policy objective gates receipts. None preserves original
MSE delegation/report/work. Common physical residuals remain comparable; unlike
training losses remain separately named and CE row costs explicit. Why this:
expose already implemented mathematics without changing defaults or building a
runner. No new dependency/layer bypass/optimizer. [API](docs/continuous-learnability-objective.md),
ADR0465. Matched visual result has old/new-task and forgetting tradeoffs; no repair.

## Detached continuous learner connects to existing robot caller (2026-10-19)

Existing fixed recipe returns its report and already detached readout owner via
ContinuousLearnabilityResult; JSON wrapper/CLI still return only the report.
Complete-state/receipt/finalpoll checks gate release; errors release no model.
Existing frozen robot smoke consumes the exact MLP through its original/fork
inference guards. No new initialization/update/fork/runner/codec/dependency or
serving mutation. Why this: reuse an existing owned object to close the concrete
training-to-native connection without storing a model archive. [API/limits](docs/continuous-learnability-result.md),
ADR0464. Both native task outcomes negative; useful robot learning remains open.

## Fixed continuous learnability through existing stream (2026-10-19)

Existing CLI → focused continuous_learnability_demo adapter → existing continuous
MLP/diagnostics and ActorShadowRuntime. Fixed TRAIN A/B/A recipe, one frozen A fit
and original24-update budget; internal source/actor stay exact. Endpoint readout
uses owned full native restore, no model archive or serving publication. Partial
failures retain bounded paid receipts; final empty poll verifies full state.
Why this: demonstrate real fitting/retention/recovery using the existing runtime
before another model change or native validation. No new framework/dependency/
optimizer/codec. [Tree/interfaces/limits](docs/continuous-mlp-learnability.md),
ADR0463. TRAIN fit evidence accepted; physical skill/generalization remain open.

## TRAIN-only conditioning in existing learner (2026-10-19)

Existing continuous adapter → existing pure diagnostic module → actual core
forward/loss/gradient kernels. One frozen standardization,2forwards1backward;
scalar-only bounded outputs and explicit work, no optimizer/default/RNG changes.
Complete owned snapshot and TRAIN gates preserve isolation and next-update
behavior. No new runner/dependency/serialization or publication architecture.
Why this: reuse known kernels and residual API while exposing missing conditioning
signals before a separate learning study. [API/tree/limits](docs/continuous-bc-conditioning.md),
ADR0462. Scoped behavior and initialization readout accepted; actual useful
learning/generalization and full research/release requirements remain open.

## Frozen native guards through existing promotion app (2026-10-19)

Stage-only fixed composition: accepted model snapshots → existing actual actor/
runtime + isolated teacher guards → existing evaluator/controller → conditional
existing smoke. No production architecture change/new runner/dependency/codec.
Source binding and full public bundle checks preserve owners; complete rejected
report leaves serving unchanged. Actual guard rejection accepted, committed
native serving remains open. [Tree/rationale/limitations](docs/native-guard-promotion.md).
Next learner work reuses existing TRAIN residual diagnostics; final/guard labels
remain unavailable to new training/tuning.

## Guard-only collection in existing teacher path (2026-10-19)

CLI→existing infra teacher/port→pure guard allocation + continuous episode core.
31/37 allocated prospectively; port binds role/seed/context before creation.
Shared teacher loop preserves TRAIN API; separate collector and owned arrived
guard getter refuse training conversion. Existing app/evaluator/controller/
runtime/learners unchanged. No new runner/dependency/publication mechanism.
Qualified collection is distinct from arrived/disjoint/promotable evidence.
[API/tree/rationale](docs/guard-only-teacher.md), ADR0461. Fake IO isolation
accepted; native data quality/publication and scientific benefit remain open.

## Actual promotable actor at existing robot boundary (2026-10-19)

Existing smoke → new adapters/promotable_robot_binding → existing app serving
and core interfaces/nearest adapter. Public locked predict/serving_snapshot
provide actual controls and full generation/version/model/config/cache/meta/tick
observations. Private inspection admits exact slot/bundle/backend identity only.
App/core/runtime/learner unchanged; controller retains publication authority.
No copied serving model/new runner/codec/dependency. Local auxiliary bounds do
not bound existing trusted snapshot deepcopy; checks do not hold a lock across
native commands. [Tree/rationale/limits](docs/promotable-actor-robot-boundary.md),
ADR0460. Fake IO/controller correctness accepted; native publication and genuine
guard-only collection remain open. Historical sections below retain prior state.

## Native visual acquisition on existing runtime (2026-10-19)

Stage-only fixed case composes existing teacher, nearest learner, actor/shadow
runtime and native frozen smoke. Production layers/dependencies unchanged; no
new orchestration framework. Actual actor supplies B baseline controls; one
delayed private append feeds detached candidate B/A rollouts. Full source/actor/
candidate ownership and A stats/prefix checked before more work. [Tree and
rationale](docs/visual-context-acquisition.md). Native candidate acquisition/return
accepted; serving publication is still missing. Existing promotion app exposes
generation/config/cache/metadata snapshots, but smoke currently refuses it.

## Explicit visual input on existing delayed stream (2026-10-19)

The existing stream CLI owns bounded receipt IO and complete frozen JSON recipe
admission. Explicit visual selection uses789D/4MiB; defaultstate stays42D/1MiB.
Validated ContinuousEpisode records flow inward to the unchanged generic
continuous adapter/app runtime/nearest learner. No new module or dependency.
Source acquisition costs remain separate from saved runtime work. [API and
rationale](docs/continuous-delayed-stream.md), ADR0459. Synthetic integration
accepted; native visual acquisition/retention/domain drift remain unfinished.

## Public visual features in existing PickCube boundary (2026-10-19)

New core/pick_cube_rgb_features is pure NumPy fixed encoding/immutable recipe,
with no IO/training authority. Existing infra native port selects public SDK
RGB/qpos/qvel/goal, owns current CPUframe and separate visual work; imports point
inward. Existing teacher/CLIs opt into789D mode while default42D state remains.
No learner/app/runtime rewrite. Frozen analytic pooling avoids new encoder
dependencies; exact recipe checks prevent observed drift. Teacher privileged
geometry generates TRAIN labels only. [API/tree/rationale](docs/pick-cube-rgb-features.md),
ADR0458. One native reconstruction accepted; visual generalization/adaptation,
domain recurrence, actualactor native and release gates remain unfinished.

## Actual stable serving at existing robot boundary (2026-10-19)

Existing maniskill_continuous_smoke depends inward on StableActor and typed NN
state. A private bounded binding retains actor/backendidentity/version/fullstate;
backendinspection istype/identityonly. Publiclocked actor.predict supplies every
control; publicsnapshots are fullyvalidated eachobservation. No learnercopy/fork
substitutes; no genericruntime/core/port rewrite. Derived actor work is separate
from returnedNearestPrediction fields. [API/tree/rationale](docs/stable-actor-robot-boundary.md),
ADR0457. Fake integration accepted; nativeexecution/promotion/visual stillopen.

## Continuous delayed TRAIN recipe (2026-10-19)

New adapters/continuous_stream_demo composes existing core continuous records/NN
learner and app ActorShadowRuntime/ExperienceInbox/ToyBudgetSession. The existing
native-stream CLI owns bounded qualifiedTRAIN receipt IO. No inward layer imports
outer code; no runtime/model/codec/dependency rewrite. Source/actor staywarm;
candidate receives one delayed episode packet, with append/SGD and derivedactor/
returnedcandidate work distinguished. Actual stable-actor robot controls remain
unfinished. [API/tree/rationale](docs/continuous-delayed-stream.md), ADR0456.

## Explicit TRAIN reset-context boundary (2026-10-19)

Pure core/pick_cube_train_context maps exact17/19 to fixed TRAIN context IDs;
infra/maniskill_task_teacher and existing teacher/frozen actor CLIs depend inward
on it and bind actual reset/episode/role metadata. The low-level port, default17,
learner algorithms and frozen-state guards remain unchanged. No context label
becomes an input feature. [Rationale/API/tree](docs/pick-cube-train-contexts.md),
ADR0455. The generic runtime is not yet wired to this continuous recipe; stage
acquisition evidence does not imply deployed online adaptation or visual support.

## Bounded continuous nearest-observation control

core/nearest_continuous_bc depends inward on continuous data/array contracts and
owns pure retrieval/work counts. adapters/numpy_nearest_continuous_bc owns frozen
statistics, capacity, two prototype arrays/counters and atomic state/append ports.
Existing smoke explicitly admits this exact second learner/state type while
retaining MLP guards; no new runner/simulator/codec or inward dependency. Why this:
test executable data cloning independently of the current optimizer with explicit
memory/distance costs. [ADR0454](docs/adr/ADR-0454-bounded-nearest-continuous-bc-control.md).

## Explicit binary-gripper objective in the existing BC learner

core/continuous_bc owns exact actuator/label validation, mixed logit2z math and
analytic derivatives. adapters/numpy_continuous_bc owns optionalLASTaxis config,
full snapshot/fork/restore state and named atomic updates. Common diagnostics and
physical predictions remain unchanged. Existing stage readers default only the
missing optional field toNone in copied metadata, with before-sources preserved.
Why this: test endpoint-label semantics without new output parameters, action
thresholds, weights, runner or codec. [ADR0453](docs/adr/ADR-0453-explicit-binary-gripper-objective-in-continuous-bc.md).

## Explicit saved clock policy integration

Stage native_case composes the existing exactTRAIN record, clock recipe, frozen
normalizer/full model snapshot and smoke entry point. Stage native_guard reuses
the accepted Linux guard with explicit method forwarding and separate bounded
outputs/cache. Production layers/policy schema remain unchanged. Why this: test
actual fixed-policy commands and task outcomes before another learner change.
[Accepted integration, negative task results and costs](artifacts/runs/continuous-bc-clock-20261019/NATIVE_RESULTS.md).

## Public episode clock at the existing observation boundary

core/continuous_episode_clock depends inward on continuous_demonstration only.
It owns bounded pure clock configuration and causal current/next TRAIN projection.
The existing outer robot smoke composes that feature with the existing NumPy
learner and native port; raw native records remain unchanged. Clock recipe is
explicit caller-owned configuration, copied and guarded separately from the full
frozen model/normalizer state. No runner, policy schema, codec or dependency is
added. Why this: test a public phase input with an equal-width zero control.
[ADR0452](docs/adr/ADR-0452-explicit-public-episode-clock-at-observation-boundary.md).

## Frozen BC observation preprocessing

core/observation_standardization owns pure population statistics/application and
immutable means/scales/fitrows, depending inward on existing BC array/spec helpers.
The existing NumPy adapter owns TRAIN-only fitting and optionalLASTpolicyfield,
copied through full snapshots/forks and applied identically to train/predict.
No runner/codec/dependency or actionobjective change; defaultNone preservesraw.
Why this: test input conditioning with state/role isolation and matched budgets.
[ADR0451](docs/adr/ADR-0451-freeze-train-observation-statistics-in-bc-policy.md).

## Frozen BC in the existing robot entry point

The outer smoke script composes existing NumPy learner and ManiSkill port,
validates TRAIN/controller/spec, forks owned state, and checks frozen inference
before controls and after cleanup. Core/adapter/infra contracts remain unchanged.
Prediction/native work and task-success fields stay distinct. No new runner,
checkpoint codec or inward dependency is added. Why this: execute the actual
fixed policy through the already qualified simulator boundary. See
[ADR0450](docs/adr/ADR-0450-frozen-bc-in-existing-robot-smoke.md).

## TRAIN continuous-action diagnostics boundary

core/continuous_bc_diagnostics depends inward on continuous_bc math and the
continuous action spec; it owns pure bounded immutable axis/stage summaries.
The existing NumPy adapter requires matching TRAIN policy/episode/spec/labels
before one prediction and reports its work. No optimizer, simulator, IO, probe
selection, role conversion, new runner, or persistence authority is introduced.
Why this: measure actual TRAIN underfit before changing learner or data.
[ADR0449](docs/adr/ADR-0449-train-continuous-action-residual-diagnostics.md).

## Fixed robot teacher boundaries

`core/pick_cube_teacher.py` is pure root-frame geometry and scalar schedule.
`infra/maniskill_task_teacher.py` reads privileged TRAIN actor poses and routes
every action through the existing continuous port. The CLI owns evidence/IO.
Qualified ContinuousEpisode data reuses the existing NativeLearner/BC API; no
new application runner or evaluation/training authority is added. See ADR0448.

## R4.4 official expert conversion boundary

`src/infra/maniskill_expert.py` owns two bounded native scenes and adapts the
official expert/converter to existing `ManiSkillPickCubeEpisode` records. Core
continuous data/learner contracts remain independent of simulator/planner IO.
The CLI reserves exclusive evidence and enables flushed phase/fatal diagnostics.
No app runner, evaluation or learner authority is added. See ADR0447 and
[module API/costs](docs/maniskill-expert.md).

## Continuous BC learner boundary

core/continuous_bc owns pure bounded MLP/loss/analytic derivatives; adapters/
numpy_continuous_bc owns policy, local RNG initialization, momentum, atomic state
and NativeLearner ports. Both depend inward on existing continuous data/role and
learner contracts. No new runner, simulator, release or persistence layer.
[ADR0445](docs/adr/ADR-0445-continuous-bc-native-learner.md).

## Continuous-action data boundary

`src/core/continuous_demonstration.py` owns immutable units/bounds, complete episode
validation and matching-role training extraction. It depends inward on the existing
core role helper only; no adapter/IO/backend dependency. Future continuous BC and
ManiSkill adapters depend on this contract and the existing generic NativeLearner.
Categorical episode encodings remain separate. [ADR-0444](docs/adr/ADR-0444-continuous-action-demonstrations.md).

## Saved FIFO36 evidence presentation

`src/app/minigrid_saved_fifo_inspector.py` validates and detaches the one explicit
saved F16/A16 JSON contract using existing pure app helpers. It emits bounded
curves, exact source pointers, preservation flags, nonnegative costs and separate
signed outcomes; it does not execute learners, environments, snapshots or IO.
Existing `key_door_inspector` dispatches; infra owns bounded exclusive publication
and the shared MiniGrid template. Why this: reuse the runnable inspector without
introducing another orchestration path or treating fit as autonomous success.

## Explicit F48 through the bounded fresh-task loop

## Explicit H32 fixed64 through existing learner ports

The relaxation adapter selects complete H8/H32 policies via exactbool width;
shared snapshot/train admission binds full registered policy and54*point clocks.
The learnability facade/CLI composes existing stream/F16/A16 with distinct outer
control IDs, while native base methods remain unchanged. Costs use64*rows;
unknown work remains unknown. No core/app/objective/runner change. Why this:
one predeclared finite-relaxation sensitivity endpoint with original defaults
preserved. See ADR0443 and docs/fixed-relaxation-controls.md.

## H32 diagnostics through the existing TRAIN checkpoint reader

The learnability facade and CLI admit the original H32 PC/neutral recipe to the
existing shape-independent realized-update adapter at TRAIN checkpoints0/128.
Full policy/clock/source/evidence checks and original budgets remain in force.
Recipe metadata declares actual dimensions; diagnostic work stays separate from
learner updates. No core/app/algorithm/new runner changes. Why this: inspect the
observed plateau using an already validated readout without changing the learner.
See docs/pc-update-diagnostics.md and the accepted H32 readout report.

The app validates/copies original16input batches, then repeats owned references
three times in its existing loop only for explicit extra_f_exposure. Fresh F48
and original A16 budgets remain separate; return/observer witnesses bind exactly
48 from the flag. Default F16 and original65536entry input bound stay unchanged.
The pure minigrid_f_exposure adapter declares public H32 recipe/checkpoint/cost
metadata; the existing facade/CLI selects it. No additional runner or objective.
Why this: match old-A replay native exposure without expanding retained input
arrays or hiding wake counts. See docs/f48-exposure-control.md and ADR0442.

## Explicit H32 through existing fixed TRAIN rehearsal

The fixed-rehearsal adapter admits the original full H32 policy only under an
explicit exact-bool flag. The learnability facade/CLI composes existing
StreamRehearsal/fresh-task/return boundaries and reports actual replay work.
Wake16, separate replay288rows, and return16 keep original distinct budgets;
full native clocks include replay. No inner layer, objective or runner changed.
Why this: the existing composition already enforces bank ownership, roles and
permanent charges. See docs/minigrid-probe-rehearsal.md and ADR0430; games plus
replay remain unsupported until an explicit recipe is verified.


## H32 through the existing deferred game reader

The adapter `minigrid_deferred_probe_rollout` optionally owns full typed H32
states for four fixed endpoints; the existing learnability facade/CLI selects
the explicit recipe. The same inline observer/evaluator reads two independently
restored frozen readers after all training. Source/cohort proofs bracket the
phase, with original budget/resource/native proofs at each game port. This
changes ordering and has its own protocol; it adds no runner or domain objective.
Why this: actual game control must be evaluated without repeating the full
training-cohort proof inside every prediction. Original budgets and evidence
remain binding; bounded capture storage and charged copies are explicit.
See docs/wide-neural-probe-games.md and the stage RESULTS.md.

## Exact packet comparison in the existing adapter

MiniGrid packet signatures retain original immutable bytes and owner/metadata
bindings. A private adapter helper compares large exact native float64 arrays via
uint64 views, preserving every bit; other inputs retain byte-comparison behavior.
It keeps no validity cache or extra authority and adds no layer or public port.
See artifacts/runs/h32-preservation-components-20261018/OUTER_RESULTS.md.

## Existing H32 endpoint elapsed phases

`minigrid_endpoint_timing` is a bounded adapter diagnostic used by the existing
`minigrid_policy_evaluation` and `minigrid_probe_rollout` observer. Four inclusive
span aggregates are sealed with paid reports; no model, IO, budget authority or
per-call history is introduced. Production's exact native recorder reads the
system clock directly; injected clocks are isolated helper/test capability.
[ADR0441](docs/adr/ADR-0441-measure-existing-h32-endpoint-phases.md) explains the
callback-boundary rationale and cost limits. Native attribution remains unmeasured.

## Explicit H32 readout resource selection

Existing adapter facade, observer and CLI select a fixed resource envelope at birth; opt-in60s endpoints/240s body/270s outer, defaults40/180 retained. Configuration joins sealed observer state, all paid/clock/native guards stay. No new runner or domain objective. See docs/wide-neural-probe-games.md and ADR0439.


## Shared experience validation

Core constructors and app inbox share unchanged metadata rules. Complete exact builtin records avoid reconstruction; opaque/missing metadata keeps original capture and error order. No validation cache or payload/provenance authority is added. See docs/experience-contracts.md.


## Owned native checkpoint binding

Shared recipe helper returns the same complete owned state it validates; legacy counter retains None. External-port fences remain. Public failure tests now close input/native proofs in finally and verify global isolation before completion refusal. Fullchain remains unqualified. See ADR0439 and docs/wide-neural-probe-games.md.


## Exact paid-report identity reads

Existing app helper proves builtin equality at binding, then rereads every owner/key/shape/leaf identity each call. Any drift uses the original dynamic proof; no cached validity or external-port fence removal.491scoped passes; fullchain remains failed. See ADR0439 and docs/wide-neural-probe-games.md.


## H32 known-TRAIN live composition

## Qualified endpoint read groups

Existing app enclosure optionally rechecks native state after original clocks.
The evaluator avoids duplicate feedforward reads; paid reports retain exact
owner/type/value proof with a scalar fast path and original fallback. FutureH32
children40s/count20s/parent180s; fullchain still unqualified, prior failures kept.
[ADR0439](docs/adr/ADR-0439-inline-wide-probe-games.md),
[contract/costs](docs/wide-neural-probe-games.md).

## H32 disposable endpoint games

Existing evaluator/observer binds complete mixed-exposure H32 endpoints to fixed
A/F games behind explicit API/CLI admission. No new runner/native objective.
Full public qualification remains incomplete because original20s budget expires;
see [ADR0439](docs/adr/ADR-0439-inline-wide-probe-games.md) and
[contract/failures](docs/wide-neural-probe-games.md).

## H32 isolated diagnostic composition

The existing MiniGrid history-learnability adapter explicitly binds the fixed
H32 policy to the original shared stream, frozen readers and disposable F16/A16
phases. Source/actor/budgets and H8 defaults remain owned; no new runner or native
update objective. See [ADR0438](docs/adr/ADR-0438-wide-neural-isolated-probe.md)
and [module/CLI contract](docs/wide-neural-learnability.md).

The existing evaluator explicitly admits the fixed H32 policy, source0 and
128/6912 endpoint for both original A cues. Existing frozen-reader/game/history
logic and readonly budget guards remain. The rollout helper forwards the explicit
family; constructive-fit retains its genuine stream checkpoint and original
parent clock/charges while adding the independent zero-update live allowance.
Returned game work precedes later publication guards; its budget binding remains
active through final readback. No new runner, learner law or checkpoint format.
[Usage/checks](docs/wide-neural-train-rollout.md), ADR0437.

## Fixed width-32 neural TRAIN composition

The existing constructive-fit adapter selects one already supported native
TorchFeaturePolicy at D296/H32/K7/seed17. The original invariant wrapper owns
projection and complete state; the existing stream owns TRAIN scheduling,
budgets, receipts and detached scoring. Wide checkpoints undergo exact native
copy/policy/counter validation. CLI selects one explicit family, with no width or
optimizer search. Native algorithms and default branches are unchanged. Shared
parameter payload excludes adaptive/optimizer/RNG state; derived work is labelled.
[Usage and checks](docs/wide-neural-train-fit.md), ADR0436.

## Bounded polynomial-kernel TRAIN control

core/categorical_kernel_perceptron owns the fixed quadratic kernel, unit mistake
rule and complete immutable bounded dictionary/counters. The native adapter owns
atomic pass commit/fork/restore; the invariant wrapper binds raw296 projection.
Existing constructive-fit/CLI reuse TRAIN admission, stream, budget and scorer.
Capacity failure retains observed uncommitted work through the existing core
failure contract. Committed native training pairs and derived prediction pairs
are separate. Dependencies point inward; no new runner or IO in core. ADR0435,
[usage and verification](docs/kernel-perceptron-train-fit.md).

## Separate multiclass perceptron TRAIN control

core/categorical_perceptron owns the deterministic unit mistake rule and complete
immutable policy/state. adapters/perceptron_learner implements the native port;
invariant_perceptron_learner binds raw296 histories to the declared projection.
The existing constructive-fit adapter/CLI composes this explicit family with the
ordinary TRAIN-only stream/budget/checkpoint/report boundaries. Dependencies point
inward; no new runner, IO in core or baseline recipe changes. Array-equivalent
coefficients, online rows, mistakes and prediction work stay separate.
[Usage and checks](docs/perceptron-train-fit.md), ADR0434.


## Optional bounded ordinary stream diagnostics

core/categorical_training_diagnostic defines the exact owned bounded subtype and
separate native/scheduled failure work. The existing core cursor validates that
exact subtype alongside the unchanged base. TorchTrainWeightedLearner owns the
explicit stopping configuration and reuses existing native/outer publication
boundaries. Existing app stream/inbox serializes richer dataclasses without
adapter imports or new orchestration. Native snapshots keep their prior format;
wrapper configuration is explicit. Probe/managed-capture consumers require
separate qualification. [ADR0419](docs/adr/ADR-0419-owned-bounded-stream-diagnostics.md).

## Explicit bounded categorical TRAIN episodes

core/categorical_pc_inference proposes owned bounded latent/parameter updates.
adapters/torch_pc_inference composes that numerical port with the existing
weighted episode's full native transition and atomic donor/publication boundary.
Its scalar DTO retains stopping and core/native work; no core import points to
adapters. Fixed weighted/categorical catalogue callers share the private native
transition helpers and retain their public signatures and snapshot formats.
Scheduled bounded stream diagnostics remain a separate unfinished integration.
See [ADR0418](docs/adr/ADR-0418-atomic-bounded-categorical-episodes.md).

## Weighted fixed64 composition

The existing minigrid_weighted_catalogue_route selects the existing named fixed64
policy and derives benchmark computation from it. It proves complete zero-state
ordinary S64 probe templates before checkpoint restoration on disposable copies;
S2 delegates its original online helper unchanged. CLI admission extends only
these two recipes. Existing stream/solver/budget/source/actor/receipt/scorer guards
remain, with final evaluation unavailable. No new orchestration or core law.
See ADR-0414; fixed64 readout diagnostics remain unsupported.

## Weighted checkpoint readout composition

Existing minigrid_gradient_readouts derives fixed weights from original TRAIN
catalogues and calls the exact detached API. Existing minigrid_learnability owns
source/actor/checkpoint/report/work/budget guards and keeps ordinary probes
isolated. The CLI combines existing flags for PC/neutral only. No core update law,
outer framework, final-data path or budget authority changes. See ADR-0413.

## Weighted PC diagnostic boundary

Existing torch_pc_update_diagnostics composes exact core weighted proposals on
owned TRAIN snapshots with explicitly named CE/energy/geometry measurements.
No learner publication, optimizer, stream, source collection or final access is
added. Native proposal work is charged separately from actual oracle updates;
original unweighted behavior remains unchanged. See ADR-0412 and scoped evidence.

## Public weighted catalogue composition

`minigrid_weighted_catalogue_route.py` validates original six-source/128transport
binding, names the aggregate objective, builds the existing weighted catalogue
learner and verifies receipts. `minigrid_learnability`/CLI expose a separate flag;
original online flags/defaults retain their meanings. Ordinary probe templates
prove full zero-state identity and receive checkpoints only on disposable forks.
App budgets/release/guards own execution; no new math/orchestrator/final access.
The transient comparison plan adds ~16.4MB plus time/RSS. ADR-0411 documents
coverage, optional ports and scientific boundaries.

## Catalogue binding in the weighted schedule

`adapters/torch_train_weighted_learner.py` now composes the accepted catalogue
native update through from_catalogue. One owned complete catalogue/witness/span
seal binds repeated independent row arrays; fixed count*rows prefixes and catalogue
objective identity share the existing candidate/publication/fork/restore guards.
Original episode/recurrent paths stay available. Existing app/runtime retains
release authority, permanent budgets, source/actor/checkpoints and evaluation;
no new runner or inward dependency violation. See ADR-0410.

## Weighted TRAIN catalogue boundary

`adapters/torch_catalogue_weighted.py` composes existing source-span validation
and native BP/PC/neutral owners. Typed complete TRAIN sources remain separate
from independent catalogue row arrays; aggregate weighting has distinct receipt
IDs. It reuses the accepted numerical PC proposal and native complete transition,
then validates candidate/donor/source/RNG/modes before one payload write. It owns
no stream, budget, evaluation or promotion. Dependency direction stays inside
adapters toward existing app/core ports. See ADR-0409; future public composition
must use existing schedule/probe boundaries rather than inventing a trajectory.

The existing weighted feature route accepts explicit nativePC/neutral policies
through weighted_pc/--weighted-pc. It seals complete online TRAIN schedules,
validates distinct objective receipts and constructs identical ordinary unweighted
probe templates. Existing stream/F/oldA/return helpers retain state/source/budget
guards; API/CLI add no learner framework. [Decision](docs/adr/ADR-0408-explicit-online-weighted-pc.md).

`adapters/torch_train_weighted_learner` now admits BP/PC/neutral bound policies and
dispatches each guarded candidate to its explicit weighted episode update/ID.
Original recurrent/BP paths, sealed schedule prefixes, detached publication and
source/actor isolation remain in the same existing adapter. No new runner/core
objective/state format. [Decision](docs/adr/ADR-0407-weighted-pc-schedule.md).

`adapters/torch_pc_weighted` admits complete TRAIN296/K7 episodes for invariant
PC/neutral owners, stages the accepted core proposal and complete native state,
then validates unchanged donor/source/RNG/modes before one payload write. Native
PC/CPC share their extracted private commit, preserving original adaptive/legacy
state and active/default updates. Existing schedule composition remains open.
[Decision](docs/adr/ADR-0406-atomic-weighted-pc-episodes.md).

`core/categorical_pc_weighted` accepts explicit CPU TRAIN-role tensors/weights
and returns owned detached weighted PC parameters, partial gradients and separate
actual/legacy diagnostics. It depends on core weight validation and shared lazy
Torch loading only. It has no learner publication, source admission, adaptive
state/counter or evaluation authority; complete runtime composition remains open.
[Decision](docs/adr/ADR-0405-weighted-categorical-pc.md).

The existing TRAIN A game helper/evaluator admits explicit joint checkpoint
recipes through the shared complete-policy/54-row counter. Catalogue readout
requires source0 and endpoint128, fixed A-only game roles and the original
zero-update budget. Native source/scorer/checkpoint guards and paid-prefix handling
remain in the existing runner; no additional orchestration layer is introduced.
[Decision](docs/adr/ADR-0404-joint-train-a-live-readout.md).

`adapters/minigrid_relaxation_control` owns exact named64-step PC/neutral policies,
protocol metadata and nominal returned inference costs. Existing learnability,
learningpilot, joint-route and CLI adapters admit the fixed recipes through the
same stream/disposable-probe orchestration. Complete policy/54-row clocks and
source/actor/state/budget guards remain authoritative. Native algorithm identity
and original S2 defaults do not change; no new core objective or dependency.
[Decision](docs/adr/ADR-0403-fixed-relaxation-acquisition-controls.md).

The existing MiniGrid learnability adapter routes optional PC/neutral diagnostics
only at owned joint checkpoints0/128. The existing gradient-readout helper gains
optional catalogue grouping and exact evidence guards, with legacy defaults
unchanged. New DTO/work costs remain separate; original stream/probe budgets,
state/row/source checks and source/actor isolation remain authoritative.
[Decision](docs/adr/ADR-0402-joint-pc-update-checkpoints.md).

`adapters/torch_pc_update_diagnostics` composes existing invariant snapshot/episode
validation, public encoding and core gradient reduction with detached native PC
formulas. Its output contains scalar TRAIN-only directions/fit and paid costs.
It creates no learner/optimizer, runner, role authority or published candidate.
The legacy diagnostic and source state remain unchanged; existing MiniGrid route
composition is now an explicit opt-in as described above.
[Decision](docs/adr/ADR-0401-realized-pc-update-readout.md).

`adapters/minigrid_joint_catalogue_route` supplies strict option validation,
protocol description and six-source composition. Existing MiniGrid pilot owns
collection/stream/checkpoint/probe orchestration; `minigrid_probe_support` adds an
explicit complete six-source admission mode with unchanged default128 admission.
Sources bind every54-row wake; support sets and repeated exposure costs remain
distinct. Existing native learners own objectives and projection. No new runner,
state authority or final-data access is introduced.
[Decision](docs/adr/ADR-0400-public-joint-catalogue-route.md).

Existing MiniGrid counter/snapshot/TRAINscore/probe adapters share an explicit
optional54-row initial checkpoint binding. Complete native clocks/policy/state
remain checked; omitted mode preserves the online nine-row law. Disposable F and
A-return retain their original additional row budgets and owned-fork app helpers.
No new learner, state format, runner or final-evaluation path is introduced.
[Decision](docs/adr/ADR-0399-explicit-catalogue-checkpoint-rows.md).


`adapters/minigrid_train_catalogue` composes complete causal TRAIN sources into
bounded owned independent-row batches with original source spans and repeated
exposure/array accounting. It validates schema but transports raw296; existing
invariant learners own projection/objectives, and AttributionStream owns delayed
labels/candidate isolation/budgets. No joined episode, evaluator or new runner.
[Decision](docs/adr/ADR-0398-joint-train-catalogue.md).


`core/categorical_linear_fit` owns bounded immutable one-hot least-squares math
and stable softmax prediction. `adapters/minigrid_linear_fit_control` composes
existing causal/canonical encoders, TRAIN source witnesses, detached operations
and paid reports. `scripts/run_minigrid_linear_fit_control` owns bounded saved-JSON
IO and exclusive output creation. Dependencies point inward; no deployed learner,
new stream runner, evaluation selector or dependency is introduced.
[Decision](docs/adr/ADR-0397-isolated-linear-train-fit.md).


`core/categorical_probability_margins` reduces already-paid categorical rows
into all-row/class signed target gaps. The thin `minigrid_action_margin_readout`
adapter describes/totals costs; existing TRAIN checkpoint orchestration owns role
admission, detached copies, private charges, state guards and optional publication.
Explicit API/CLI flag leaves default reports unchanged and touches no learner,
probe or final-evaluation path.
[Decision](docs/adr/ADR-0396-paid-train-action-margins.md).

`core/demonstration_schedule_witness` proves complete owned TRAIN schedules:
exact mutable record fields plus immutable leaf identity, full semantics/copy
comparison at owner creation. The existing weighted array adapter composes fresh
own/seal witnesses only for bound BP; the RNN path retains its prior checks.
Lifetime storage is bounded, with no history or TTL. Native publication, release,
budgets and evaluation remain in their existing layers.
[Decision](docs/adr/ADR-0395-immutable-schedule-witness.md).

`minigrid_weighted_feature_route` composes the opt-in bound BP schedule factory,
objective receipts and ordinary isolated probe templates into existing learnability
and CLI paths. Original runner owns release/budget/source/actor separation; native
ports own objective/state. Full checkpoint/optimizer transfer stays on disposable
probe forks. [Decision](docs/adr/ADR-0394-weighted-learnability-route.md).

The existing `torch_train_weighted_learner` array boundary also accepts bound BP.
Its sealed complete TRAIN schedule chooses the next episode and fixed weights;
the existing native weighted port trains a detached candidate. Full state and
source/input proof precede publication. The existing attribution stream still
owns label release, budget and actor separation. RNN sessions/rollback remain
separate existing behavior. [Decision](docs/adr/ADR-0393-weighted-feature-stream.md).

`adapters/torch_feature_weighted.py` adds an explicit weighted TRAIN-episode BP
function over the existing invariant/native feature owner. Core class-weight and
public-projection rules point inward; native head/optimizer restoration supplies
an owned candidate. Full candidate/donor/input/RNG proof precedes atomic native
publication. No scheduler, state format, PC/CPC law or original training-method
change. [Decision](docs/adr/ADR-0392-explicit-weighted-feature-bp.md).

The opt-in bound prior-gradient diagnostic stays in adapters:
`torch_feature_gradient_diagnostics` validates full bound snapshots, projects
public rows, reads detached native prediction-CE gradients and delegates scalar
geometry to the existing core reducer. `minigrid_gradient_readouts` owns paid
TRAIN evidence; existing learnability/CLI compose it without new orchestration.
No learner update or probe/final access. [Decision](docs/adr/ADR-0391-bound-prior-gradient-readout.md).

The optional recurring FIFO36 route composes the existing invariant wrapper/FIFO
backend through the same count pilot and delayed stream. Its distinct typed TRAIN
evaluator and explicit private training-counter option admit only 36/87264 while
old defaults retain64-entry admission. Five snapshots preserve complete queue,
retained/retired counts and lifetime clocks; paid receipt clocks remain separate
from observed native commits on failure. No core law, dependency or runner changes.
[Decision](docs/adr/ADR-0390-recurring-fifo-capacity-control.md).

The optional recurring invariant count control composes the existing
`InvariantActionMemoryLearner` into `minigrid_action_memory_pilot` and its current
delayed stream. A separate typed TRAIN entry point in `minigrid_policy_evaluation`
validates bound policy/encoding/state before the shared evaluator; original raw
admission/defaults remain. Paid private reports and final sampler/clock proofs
stay at the adapter boundary. Core count/projection laws and app ports are
unchanged; no new runner/dependency. [Decision](docs/adr/ADR-0389-recurring-invariant-memory-control.md).

The optional deferred count-game recipe stays in adapters: the existing
learnability stream invokes the app endpoint observer, and
`minigrid_deferred_probe_rollout` owns four bounded typed in-memory captures.
After all fixed learning it restores two existing count readers and invokes
`minigrid_probe_rollout` and the existing native MiniGrid evaluator. Full original
source/cohort/report proofs enclose the phase; per-game private/native/budget
proofs remain inside. This explicit timing contract preserves inline/default
routes. Core encoding and app learning ports are unchanged. Capture/decode/
restore/storage costs are separate; no checkpoint export or new runner.
[Decision and boundaries](docs/adr/ADR-0388-deferred-known-task-games.md).

The existing learnability recipe selects one bound native head by exact method
name. Selection changes factory/descriptor scope, without changing stream, probe,
objective or encoding ports. Native neutral/active CPC configurations remain their
original values; the legacy PC alias/default descriptors and omitted kwargs are
preserved. Singleton reports do not fabricate cross-method comparisons.

Encoding-bound activation health reuses the existing native diagnostic adapter:
complete bound-state admission and pure public projection precede the shared
float32 hidden-prior read. Existing learnability helpers admit TRAIN data and own
paid scores/diagnostic costs; generic app ports and probe orchestration are unchanged.
A reader return validates preceding evidence before publishing health. This keeps
canonical representation at the adapter boundary without another runner.

The existing minigrid_learnability adapter and attribution CLI expose an explicit
single canonical-PC qualification. This is a recipe selection over the same stream,
owned checkpoints, TRAIN-fit reader and disposable-probe ports; no new orchestration
layer or core dependency. Singleton reports omit cross-method comparisons while
retaining original acquisition gates and native policy/objective.

The invariant_torch_feature_learner adapter composes the existing native heads,
binding public representation identity to complete owned native policy/state.
Core projection remains pure; generic app stream/probe ports require no model-
specific branches. Existing MiniGrid TRAIN admission validates the declared bound
neural recipe, while a separate fixed A/E evaluator entry point supplies a native
snapshot copier and tensor-aware comparison to the shared runner. Original raw
and count API admission remain separate. No new runner or persistence codec.


The saved-A32 invariant recipe composes the existing delayed TRAIN arm and frozen
MiniGrid evaluator via explicit count-family/representation admission. Its CLI
reuses bounded saved-source IO once; original raw states are never relabeled or
retrained. Original training clocks guard all later zero-update readouts. Four
fixed private paid captures bound storage and survive refused callback mutations.
No new core/app dependency on adapters or new orchestration framework is introduced.


An opt-in public representation control lives in core/minigrid_invariant_encoding:
pure discrete public projection and causal first/current history only. The native
invariant_action_memory_learner adapter composes existing exact/FIFO count ports
and binds its encoding ID/native policy to owned complete snapshots. It accepts
raw296 history arrays, projects every row before native work and rejects raw or
noncanonical checkpoint states. The generic app probe/delayed stream accept that
opaque owned state without imports into core/app or a new runner. Role release,
budgets, actual game recipes/evaluation and persistence stay outside the transform.
[Usage and source tree](docs/minigrid-invariant-encoding.md),ADR0387.

The existing frozen MiniGrid evaluator also has a distinct saved-memory-return
entry point. It admits only recorded exact/FIFO native endpoints and four known
A/E target-free game slots; the original evaluator admission stays separate.
The new readout CLI owns bounded source IO, eight saved inputs/results and a
zero-update game budget. Original training budgets/serving authority are never
restored. Guards check parent evidence after native snapshot/resource callbacks;
private bounded paid captures survive refusal, with lost reports explicitly unknown.
[Usage](docs/minigrid-action-memory.md#frozen-saved-return-game-readout).

The saved-memory adapter's opt-in E8/A8 comparison reuses the existing generic
probe helper and two native count learners. The CLI binds typed source states and
original known-TRAIN A transitions to one parsed report; the adapter owns role/
phase copies, separate budgets and four bounded results. Stored local validators
are checked in flat passes before more work, including accepted atomic failed
prefixes. Exact core validation precedes native full-state equality. No new
orchestrator, final access or serving update is introduced.
[Usage/boundaries](docs/minigrid-action-memory.md#opt-in-matched-new-task-and-return-comparison).

## Separate bounded FIFO control

`src/core/fifo_action_memory.py` defines insertion-order counts and bounded
retirement totals; it depends only on existing core validators/types and NumPy.
`src/adapters/fifo_action_memory_learner.py` owns native snapshots, fork/restore
and atomic whole-batch publication. Existing stream/probe helpers compose these
ports without new orchestration or changes to the refusal learner. Callers own
roles, budgets, IO and promotion. Ordered entries encode the queue; two bounded
retirement counters replace any history chain. The16-byte extra logical metadata
is separate from the retained-table cap. [ADR-0386](docs/adr/ADR-0386-bounded-fifo-action-memory.md),
[usage and boundaries](docs/fifo-action-memory.md).

## Saved MiniGrid memory probes

`src/adapters/minigrid_action_memory_probes.py` composes the existing reserved
MiniGrid source/history and `src/app/fresh_task_probe.py` ports on owned0/end128
count-memory snapshots. `scripts/run_minigrid_memory_probe.py` owns bounded JSON
input, existing typed restore and exclusive output publication. Core count laws,
the generic probe helper and benchmark/evaluator algorithms stay unchanged.
Separate disposable native copies and original16-update budgets isolate probes
from serving/benchmark learners; failed capacity prefixes retain charges and
missing metrics. Source and final-data boundaries are documented in
[the MiniGrid guide](docs/minigrid-action-memory.md#isolated-saved-state-e16-capacity-assay).

The MiniGrid count control composes existing collection, public history, delayed
stream and frozen evaluator ports. Its explicit TRAIN-only policy admission and
backend-neutral state comparator preserve neural defaults. Exactly five private
readouts retain prior success; known receipts survive unavailable model reads.
[Usage and boundaries](docs/minigrid-action-memory.md).

Count memory admits only16/5 and296/7 geometries in the existing pure core.
Policy-derived key/count payload costs and snapshot geometry binding preserve
the unchanged native adapter; caller/stream still own roles and evaluation.
See [the geometry decision](docs/adr/ADR-0384-bounded-supervised-action-memory.md#explicit-minigrid-geometry-extension).

The public-history inspector projection validates three saved protocols and
reuses recorded geometry through an explicit raw-TRAIN selector. Existing infra
owns bounded input, inert JSON escaping and exclusive output; one static template
owns presentation. Source JSON/pointers are retained without learner/physics/IO
in the app projection. [ADR-0385](docs/adr/ADR-0385-public-history-inspector.md).

The memory recipe's opt-in E16 mode composes existing checkpoint/fresh-task helpers
on disposable native0/end8 copies after TRAIN scores. Separate budgets share the
original deadline; source, serving actor, candidate, receipts and prior reports
remain sealed. Accepted live evidence is reused. No new orchestration framework
or learner algorithm is introduced. [Boundary/usage](docs/key-door-action-memory.md#isolated-e16-capacity-assay).

`core/action_memory.py` supplies pure bounded count transitions/predictions;
`adapters/action_memory_learner.py` owns full validated native snapshots/forks.
The fixed memory recipe composes existing AttributionStream/history/live helpers
under an explicit method identity. Backend-neutral comparison is injected locally;
neural defaults remain unchanged. [ADR-0384](docs/adr/ADR-0384-bounded-supervised-action-memory.md),
[usage/boundaries](docs/key-door-action-memory.md).

The prediction-only `key_door_public_history_session.py` adapter composes the
existing pure history encoder and owned native learner port. Each episode owns
a frozen full-state fork, one public initial frame and bounded counters. Existing
`_live`/episode execution and history TRAIN composition publish only one private
episode at a time, checking prior evidence and original budget bindings.
[ADR-0383](docs/adr/ADR-0383-owned-local-history-live-sessions.md),
[usage/costs](docs/key-door-public-history.md#closed-loop-train-layout-history).

The existing local history adapter's opt-in capacity mode composes
`run_checkpoint_probe` and `run_fresh_task_probe` after all four benchmark streams
and TRAIN readouts finish. Shared reserved packets are encoded independently;
full native0/end8 snapshots feed disposable forks with separate16-update budgets.
Pure boundary checks preserve benchmark source/actor/candidate/receipts/budgets/
clocks, data and preceding results. Private active output publishes only its
selected slot; other evidence remains sealed, including object replacement.
No new execution layer, native learner or final data access is introduced.
[Inputs, outputs, costs and extension](docs/key-door-public-history.md#matched-fresh-task-capacity).

The local matched-history route adds a pure8→16 episode transform in
`core/key_door_public_history.py`. Its adapter composes the existing four feature
learners, AttributionStream and TRAIN metric projection; the CLI only dispatches
the fixed recipe. Existing8-input routes and official MiniGrid148→296 behavior
are preserved. Source/actor/receipt/budget/data guards retain prior success and
failed-work evidence; no final/probe/live data are created. See
[boundaries and extension](docs/key-door-public-history.md) and
[ADR-0382](docs/adr/ADR-0382-local-matched-public-history.md).

The quota learning-capacity route composes existing checkpoint/probe helpers.
`recurrent_probe_source.py` restores a complete owned snapshot into the existing
array-native learner; `key_door_quota_probes.py` binds the fixed E16 diagnostic.
The original reward runner supplies pure initial/end callbacks and a terminal
readback check, then returns before final construction. Typed benchmark training
roles and default routes remain unchanged. See
[decision](docs/adr/ADR-0381-quota-reserved-probes.md) and
[usage/boundaries](docs/key-door-quota-probes.md).

The existing exact state comparator checks builtin scalar/container records
before Tensor/dataclass inspection. Every value is still compared, with original
type, NaN, mutation, dependency, fallback and exception behavior. This avoids
repeated inspection on prefix histories without caches or history exemptions.
The frozen reference, tests and saved-record timings are in
[repair evidence](artifacts/runs/r43-exact-comparison-20261017/RESULTS.md).

`src/adapters/key_door_acquired_quota.py` composes two fixed TRAIN-prefix arms
through the existing reward recipe, actor/inbox and quota collector. The original
runtime owns all192prefix receipts, subsequent native updates and logicalclock.
Serving source state stays initialized; the isolated frozen candidate retains its
learned prefix and optimizer state. The physical evaluator accepts an explicit
native origin from the completed prefix; omitted origin preserves native-zero
semantics. No new learner, objective, artifact loader or execution framework.
[Decision](docs/adr/ADR-0380-acquired-prefix-mapped-reward.md).

The fixed repeated-instance game comparison is a thin adapter over the existing
reward/quota recipe. `src/app/key_door_reward_comparison.py` validates detached
scalar reports and pairs physical stages; `src/adapters/key_door_comparison_pilot.py`
composes four declared arms through existing ports. The CLI exposes the recipe.
Final values do not enter subsequent training or case selection; first failure
stops further admission. No new learner, objective or orchestration framework.
[Decision](docs/adr/ADR-0379-repeated-game-reward-control.md).

The existing reward runner has an opt-in frozen recurrent collection path with
a zero-update budget and no inbox submission. Complete source/actor/candidate
state remains fixed; collections have separate validation from native receipts.
`src/adapters/key_door_physical_evaluation.py` composes the existing scorer/live
evaluator per physical stage, preserving duplicate native0 indices and partial
paid results. No generic runner or native objective changes.
[Decision](docs/adr/ADR-0378-frozen-recurrent-physical-control.md).

`src/core/key_door_instances.py` maps explicit finite catalogue indices to frozen
layout configuration, without RNG/IO/learners. The existing reward adapter/CLI
validate an optional train/final-only pair, bind provenance across runtime
callbacks and use the existing96-step quota/final access boundaries. Model17 and
sampler29 remain separate. No new orchestration or dependency.
[Decision](docs/adr/ADR-0377-explicit-key-door-instances.md).

`src/core/physical_interaction_quota.py` owns bounded numeric attempted/completed/
failed physical-step accounting, with no history, IO or learner imports. The
existing app collector optionally charges it around physical calls. The existing
reward loop fills phases using fresh single-episode updates and original actor,
clock, budget and final-isolation guards. Explicit physical/native checkpoint
mapping prevents variable episode counts from changing metric meanings. No new
orchestrator or native objective. [Decision](docs/adr/ADR-0376-shared-game-interaction-quota.md).

KeyDoorLayout now carries an optional identity-default movement permutation.
The existing core maps controls/legality independently of spatial observations;
the app collector/evaluator retain frozen-session/role/work boundaries and mapped
provenance. Existing reward adapter/CLI compose the opt-in scenario. Explicit
key_door_layout_record preserves legacy identity wire fields; direct Python
asdict includes the additive map field. No new production layer or dependency.
[Decision](docs/adr/ADR-0375-explicit-game-action-mapping.md).

The shared inspector now dispatches the exact coherent-stream removal protocol
to `src/app/coherent_stream_inspector_projection.py` (pure saved-data validation
and detached source-addressed cells). Existing
`src/infra/key_door_inspector_files.py` selects the standalone
`src/infra/coherent_stream_inspector.html`; the existing CLI owns invocation.
Dependency direction remains CLI → infra → app. No learner imports or scientific
metric reconstruction occur in this projection. Unsupported topology/lineage/
sleep data stay unknown. [Decision](docs/adr/ADR-0374-coherent-stream-inspector.md).

## Arrived-incident replay admission

Optional recurring learnability in the same shift adapter composes the existing
`learnability_pilot.create_sixteen_probe_observer`; its optional prepared-budget
port spends four original setup sessions through existing `fresh_task_probe`.
Benchmark and independent probe templates remain separate. No new runner/core
algorithm was added. Semantic PC-neutral parity validates native snapshot storage
separately (840B vs968B), preserving the explicit128B auxiliary-state cost.
See [ADR-0368](docs/adr/ADR-0368-isolated-probes-after-incident-replay.md).

The existing `adapters/shift_signal_pilot.py` also composes this gate behind explicit
calibrated incident replay. Its callback routes completed copied observations and
records the preceding snapshot's replay count. The existing stream/rehearsal and
budget layers own training and permanent charges; the observer remains passive.
Final scores are released only after the per-case stream and all observers finish.
See [ADR-0367](docs/adr/ADR-0367-compose-incident-feedback-in-shift-pilot.md).

`core/incident_replay.py` owns only bounded scalar transition/admission state. The
existing app rehearsal helper optionally owns one gate and validates incoming
incidents against actual wake receipts. It consumes an eligible onset before a
native consolidation; skipped event slots retain zero work and explicit reason.
The existing stream calls rehearsal before checkpoint observers, so boundaryN
uses observations throughN-1. Its original-state guard also refuses another wake
when the preceding observation is missing. Why: reuse actual ordering instead of
reordering callbacks or inventing same-wake knowledge. Complete gate data follows
existing owned pause/copy/unchanged-state checks; budgets/clocks remain original.
The sensor adapter routes only copied arrived-monitoring incident records; final
data stays in finish-only scoring. Default periodic helper reports remain fixed.

## Shared recorded sensor inspection

The existing app dispatcher routes three sensor protocol families to the pure
`sensor_inspector_projection` module. It validates bounded JSON/visible domains,
copies saved evidence and produces source-addressed cells, keeping missing and
null distinct. It neither reconstructs a model nor recomputes scientific metrics.
The existing infra file adapter applies the same loader to optional separately
supplied resources and publishes a new standalone `sensor_inspector.html` using
existing inert-JSON escaping/output limits. The CLI remains the shared entry
point. Why: recorded evidence already exists; reuse the established file boundary.
Resource files lack a unique shared report identity, so association remains
explicitly unverified. Planned counts never become measured costs. No dependency,
configuration/environment variable, learner or stream algorithm changes.

## Finish-only sensor context readout

`core/sensor_retention.py` owns fixed independent noisy A/B arrays and pure signed
summary validation; it performs no inference or selection. The sensor adapter
assigns final-test roles and invokes the existing `AttributionStream.finish` only
after training and requested probes complete. Its optional fresh `score_progress`
dict records attempted/returned prediction work and partial metrics using existing
state guards; default callers/wire remain unchanged. A final resource guard failure
marks overall progress failed while retaining paid metrics and training receipts.
Scheduled BP prediction admits the existing four-row stream or 32-row final set;
training, buffering and full snapshots stay unchanged. Why: existing scorer/state
checks provide the required boundary without another evaluation runner. No final
labels enter learner updates, observers, promotion, tuning or probe selection.

## Feature-only prearrival shadow observations

`adapters/sensor_learnability.py` composes the existing application fresh-task
probe and core activation diagnostics at four fixed checkpoints. It owns separate
reserved arrays, an independent native template and four setup-born probe budget
sessions sharing the original deadline/RSS sampler. It receives copied snapshots/
arrived training metadata, with no live benchmark/actor/final-data handle.
Disposable forks restore full state; boundary checks refuse template/policy,
checkpoint/metadata or reserved-input mutation. Bounded four-slot reports retain
partial work and measured cost. Sensor adapter joins paid forecasts before calling
this optional observer. Why: existing probe/fork/stream mechanisms already provide
the required isolation; no new orchestration or learner rule is needed.

`adapters/sensor_learners.py` composes existing BP/PC/neutral/active CPC with the
fixed16→8→1 sensor geometry and common owned initial parameters. Native imports
are lazy; method validation is lightweight. CoupledLearner retains original
fixed4step/.2 inference/precision/chemistry state on forks. Sensor prearrival
prediction accepts full CoupledSnapshot, requires PC without chemistry or CPC
with wake count equal to stream receipts, and uses only detached parameters for
target-free prediction. The existing two-hidden-layer pilot factory stays fixed.
No new core rule or orchestration layer; native coupled-energy costs and BCE
scores remain distinct. Scheduled/replay sensor controls currently require BP.

The sensor adapter's explicit --replay composition passes the existing
StreamRehearsal directly to AttributionStream, with existing RecentStreamReplayPolicy
and a separate original replay budget using the shared monotonic clock/RSS sampler.
Only committed arrived wake rows enter retention; content dedup yields2/4/4
individual-row calls. Prediction rows record the preceding replay clock separately
from wake receipts. Failure output retains permanent charges/attempts and all
paid targetless pending forecasts. Retention8rows/1088numericB is bounded; process
RSS also covers metadata, selections, native state and report copies. Why this:
the existing helper already owns these semantics; adapter composition avoids
another replay framework. --scheduled and --replay are separate explicit modes.

`adapters/scheduled_backprop_learner.py` adds a sensor-specific four-arrival
accumulator around ordinary native BP. Its complete snapshot owns native model,
fixed rate, pending arrays, receipt/optimizer counters and stopped flag; no budget
authority is restored. The existing stream owns all12delayed arrival receipts,
while scheduled native optimizer count is3. Buffered diagnostics are definition-
tagged no-loss observations. CLI composes this adapter via --scheduled, unwraps
only its native state for prearrival predictions and reports both clocks. No
change to default BP or monitoring policy; no scheduler or model math duplication.

The explicit stationary sensor control passes one validated boolean through the
existing CLI/adapter/fixture/monitor. Twelve owned baseline-A batches give eight
postprefix opportunities without changing the detector or stream. Absent phases
report not_applicable; original False calls/wire remain unchanged. No new model
forwards, runner or dependency; see ADR-0370.

Opt-in sensor monitoring composes existing core shift calibration/incidents in
`app/sensor_monitoring.py`; pure `core/sensor_recovery.py` owns phase timing and
recovery statuses. Adapter retains bounded prearrival feature copies until labels
arrive and records the paid forecast/score before invoking diagnostics. Monitor
has no learner/runtime handle, cannot feed training, and retains at most12scalar
rows. Prefix calibration fits only4A; schedule truth labels alarms in reports,
not model inputs. No new orchestration layer. Default demo/stream stay unchanged.

Dependency direction: CLI→sensor adapter→existing AttributionStream/core learner
and core sensor fixture. Core remains independent of app/adapters. Existing stream
owns label release/tick/continuation; optional observe_features receives copied
pixels/rowIDs and current CandidateState before _record_tick. Existing candidate
lease plus original-state/authority checks protect subsequent work. Observer
results never drive training; stable actor remainsfrozen. Adapter joins retained
probabilities to arrived TRAIN targets. This adds no scheduler/orchestrator or
promotion path. Callback counts and actual native work have separate accounting;
see ADR-0364 and docs/sensor-drift-demo.md for boundaries and limitations.

## Paired-episode recurrent learner

The existing paired pilot/CLI's explicit public-history condition reuses the
shared training arm with a narrow296 counter/transport lane. Raw148 validation
is unchanged. Cold-law diagnostics run before SGD and preserve paid partial
predictions in a private buffer; retained raw paired endpoints remain a historical
comparison, not another baseline run. Only fixed0/64 readouts are declared for
this separate condition; old full-five protocols remain unchanged. No new
orchestration or native objective is introduced. See ADR-0350 and the recorded
negative acquisition report.

`adapters/torch_recurrent_episode_batch.py` reuses the native recurrent math and
SGD validation behind a separately chosen learner and strict snapshot. It owns
two independent episode graphs and one atomic update, with explicit optimizer-
step/episode/row clocks. Legacy snapshot fields and raw defaults remain unchanged.
`torch_recurrent_public_history` extends only a cold148-input paired snapshot to
296 public first/current features: zero first-frame weights and copied raw law,
with all other parameters and cold optimizer/clocks preserved. Native policy
explicitly admits296; raw experiment validators remain148-only. Existing core
history encoder and paired array owner compose at the ordinary delayed stream,
with no new app layer. The existing MiniGrid evaluator's explicit public_history
branch now accepts296 paired snapshots and original148 TRAIN-A packets; raw
scores/live inputs use each episode's own first/current public frames with fresh
recurrent sessions. Default raw validators/roles stay fixed; trained conversion
remains unsupported. Added1184parameters and memory/compute prevent a claim of
matched capacity or scientific benefit. See ADR-0350's October17 extension.
`torch_train_episode_batch` binds owned complete TRAIN episode pairs to the
existing array stream without joining their recurrent paths. Explicit paired
branches admit actual new-type step/episode/row counters in stream, raw scorer
and TRAIN-A-only live evaluator. `minigrid_paired_bc_pilot` composes those existing
boundaries and retained reference; its CLI owns output and resource setup.
Serialization codecs and benchmark dispatch remain separate unfinished
boundaries. No private validation bridge enters benchmark state. See ADR-0363.
`torch_paired_probe` binds a warm full paired snapshot to original probe_train
episodes at the existing `app/fresh_task_probe` array port. Native absolute
base-plus-probe clocks remain separate from the app's local probe budget. The
explicit native probe entrypoint reuses paired math without relabeling roles.
`core/learner_ports.TrainingUpdateFailed` carries bounded scalar failed-work
facts inward; the app captures them only from its update call and imports no
adapter. Evaluation labels remain in app scoring on a disposable fork.
The existing `minigrid_learnability._run_probe` selects this adapter only with
explicit paired=True, admitting original16+2 E packets and local8step/144row
transport while preserving legacy16step defaults. Frozen original stream source
and supplied trained checkpoint are independently guarded; they need not have
equal weights. Paid reports/charges and original metadata remain separately bound.
Measured guard overhead is reduced by C-level field collection/identity comparison
with the original full proof cadence and strong references. There is no cross-call
proof cache or exemption for trained/read-only states.
The opt-in endpoint-first pilot changes observation order only. It privately
captures validated fixed endpoints and distinguishes them from full five-readout
completion. Final integrity proof still enforces the original arm clock/state;
the expired diagnostic clock never admits more work.

The actual reserved-E measurement composes existing training, collection and
probe ports in a stage-local driver. It adds no production layer or promotion
authority. Both probe learners are disposable; recorded gains remain separate
from old-task retention/recovery and learner live success.

## RNN-only warm gradient use case

`minigrid_warm_gradient_pilot` composes the existing warm-start stream/parity
helper and checkpoint scorer. Its thin CLI configures local resource controls
and publishes a new JSON file. The original returned warm source, guard, packets
and training charges stay authoritative; diagnostics use a distinct max0-update
budget. No learner math, release authority or new orchestration framework lives
here. See [usage and limits](docs/minigrid-warm-gradients.md) and ADR-0362.

## Frozen TRAIN gradient diagnostic

The warm-gradient CLI also has an explicit isolated cue-update option. Its small
adapter uses the existing typed native demonstration port and checkpoint scorer
on two disposable full-state forks. It preserves the original source and clocks,
charges both steps separately, and publishes both arms/all TRAIN effects. No
new trainer, deployment path, task selector or data collector is introduced.

`torch_recurrent_gradient_diagnostics` depends on the existing native recurrent
snapshot/features/forward ports and the pure `core.gradient_geometry` reducer.
The core imports no adapters, Torch or I/O. The adapter validates a TRAIN pair,
uses detached clones with zero recurrent activation, and returns scalar signed
geometry and paid work. It owns no deployment, stream, probe or final authority.
See [usage and trust boundaries](docs/recurrent-gradient-geometry.md) and ADR-0362.
The existing `minigrid_learnability` scorer optionally delegates three original
TRAIN pairs to `minigrid_gradient_readouts` after raw scoring. The helper owns
bounded scalar evidence and separate cost counts; the existing caller retains
source/actor/checkpoint/probe/final authority. No additional runner is introduced.

## Variable-row TRAIN readout port

The existing MiniGrid evaluator adds a distinct recurrent_warm_reward method and
explicit native row metadata for128/136 checkpoints. Exact TRAIN triple/source
admission and the caller's completed-work guard preserve release/state authority;
metadata only checks consistency. Existing BC methods keep original clocks,
default output and final gates. Reuse the same owned scorer/session/cleanup loops.
The existing reward recipe's explicit train_readouts option retains owned128/136
snapshots and passes the original guarded TRAIN sources to a small readout helper
after all training. The helper owns only two diagnostic records and a zero-update
budget. Completed training clocks/charges/resources remain exact; new readout time
is bounded separately within the original whole deadline. Caller guards retain
full receipt/source/actor/candidate authority. See ADR-0359/0360 and
docs/minigrid-warm-readout.md. Native and scientific qualification stay separate.

## MiniGrid BC-to-reward continuation

The opt-in warm helper reuses accepted TRAIN reference admission/source DTOs and
the existing guarded stream to regenerate a necessary native checkpoint. Exact
raw/native prefix parity gates a complete-state restore into the existing typed
demonstration/reward port. The unchanged actor/inbox/collector interfaces then
carry optimizer state while reporting warm and reward phase clocks/charges
separately. The original whole deadline bounds both phases. No new trainer,
checkpoint format, objective or evaluation authority is added. See ADR-0358.

## Reference-based MiniGrid weighted control

The outer weightedBC recipe admits the fixed accepted TRAIN artifact once,
reconstructs its core DTOs and reuses the original unweighted control/readouts.
Only the distinct weighted arm uses existing _train_arm and native weighted CE;
new readouts follow completed training. Evaluator admission adds weighted148
only for predefined TRAIN A. Metadata byte guards retain exact records/seals,
native owner/state and objective laws. Historical/new costs stay separate.
See ADR-0357 and docs/minigrid-weighted-bc.md.

## MiniGrid delayed reward recipe

The outer minigrid_reward_pilot composes the accepted native collector with
ActorShadowRuntime/Experience/LabelArrival/RewardUpdate, adding no runtime layer.
Collection binds the current candidate checkpoint; matching-ID release is delayed
one tick and each original commit validates before more collection. Source and
serving actor remain unchanged. Existing inbox retention is bounded to8payloads;
JSON excludes opaque checkpoints. See ADR-0356 and MiniGrid reward usage.

## Sampled MiniGrid reward boundary

Outer adapters/minigrid_reward_collection composes existing native environment
IO, recurrent episode sessions and core PolicyGradientEpisode. It records raw
seven-action samples under an owned frozen checkpoint and caller budget/RNG.
No SGD or final/probe data enters collection; existing RewardUpdate and native
reward objective remain the update boundary. Why: truncated sampled trajectories
cannot use the terminal-only demonstration conversion. See ADR-0355.

## Recorded MiniGrid public views

A pure app projection validates the fixed-A saved protocol and derives public7x7
frames with exact evidence pointers. Existing inspector dispatch/file IO selects
an explicit MiniGrid template, preserving custom paths and8MiB/16MiB bounds.
Why: actual148/296/K7 observations differ from the old customD8/K5 map. The
browser renders recorded inputs/actions and unknowns, without native physics or
model execution. See ADR-0354 and docs/key-door-inspector.md.

## Fixed-task MiniGrid acquisition

An explicit option in the existing history CLI composes the original guarded128
stream, midpoint raw scorer and endpoint live evaluator. Exact A-only TRAIN
admission avoids unrelated B/C rollouts; original default/final gates remain.
Why: establish an acquisition premise with original learner objectives before
interpreting retention. Outcomes/gains remain unavailable on refused cases.
See ADR-0353 and the existing history-evaluation guide for scope and costs.

## Public-history live port

The existing minigrid_policy_evaluation adapter resets the pure public encoder
for each opt-in296 feedforward episode and uses its offline transform for raw
TRAIN scores. Recurrent148/default148 behavior remains. A fixed CLI reuses the
existing AttributionStream through the guarded32/128 training helper. Why: run
accepted history learners without another runner or privileged policy inputs.
See ADR-0352 and docs/minigrid-history-evaluation.md for costs and boundaries.

## Native activation readout

The torch_activation_diagnostics adapter validates detached native snapshots and
computes actual float32 hidden states; the existing pure diagnose_activations
reducer computes health. Existing MiniGrid TRAIN checkpoint readouts opt in via
one CLI flag. Native recurrent forward adds an optional pre-tanh finite check,
preserving default arithmetic. Why: observe missing native states without adding
another metric/runner or exposing final evaluation. All extra work and partial
costs are explicit; see ADR-0351 and docs/native-activation-health.md.

## Shared public initial-frame memory

Pure core/minigrid_public_history transforms only public first/current frames.
The existing episode/native/probe ports admit exact296; the opt-in MiniGrid
diagnostic selects296 for all4feedforward methods and preserves148 for RNN.
Existing AttributionStream and disposable probes own updates/budgets; final D
stays unavailable. Why: resolve an observed TRAIN input ambiguity with one bounded
causal state, preserving original recipes. Extra parameters/storage/computation
are explicit; see ADR-0350 and docs/minigrid-public-history.md.

## Actual MiniGrid learner composition

The minigrid_learning_pilot adapter composes the existing AttributionStream,
six TRAIN teacher packets and five native learner implementations. Its separate
minigrid_policy_evaluation adapter restores detached checkpoints and performs
raw scoring/target-free live episodes, including fresh recurrent sessions.
All learners finish before any readout; final-only D never supplies labels.
Original budgets, clocks, native states and completed receipts remain guarded.
Why: measure useful adaptation with existing algorithms and runner, preserving
the distinction between demonstration fit and autonomous success. See ADR-0348
and docs/minigrid-learning.md.

Completed MiniGrid readouts are private immutable payloads until the last callback
and guard. Public slot identity rejects replacements; only private originals are
decoded into the original result schema. Tensor state comparison remains exact,
with one Torch dependency resolution per comparison. Why: target measured repeat
work without accepting mutable historical evidence. Whole-native cost remains
unresolved after the preserved second budget failure.

## Actual MiniGrid cue/key/door game

Pure core layouts and teacher accept only explicit configuration or full public
observations. Infra lazily constructs native MiniGrid objects and an environment
with a transient cue/neutral lock. Its owned adapter and original Memory adapter
share a private lifecycle for roles, flags, detached evidence and failed work.
The CLI collects fixed complete demonstrations and validates existing records/
packet codecs; it adds no learner or runner. Why: advance R4.2 without leaking
the required key through door color or creating divergent episode accounting.
See ADR-0347 and docs/minigrid-key-door.md.

## Optional actual MiniGrid boundary

`infra/minigrid_memory` lazily owns one pinned Gym environment and normalizes its
full public image/direction into148 coordinates. It depends inward on existing
core episode records; app/stream code has no Gym dependency. Native categorical
and recurrent ports admit this exact additional geometry, preserving old widths,
losses, state formats and other bounds. Native action IDs/step costs/end flags
remain explicit. Why: preserve the full partial observation instead of cropping
the cue or introducing another runner. See ADR-0346 and docs/minigrid-memory.md.

## Acquired-task recorded presentation

The existing key/door app projection explicitly admits single acquired-task
TRAIN diagnostics through its own recorded key and original JSON pointers.
A focused pure app validator binds protocol/control/roles/clocks and readouts;
existing geometry playback validates the conditional legal law on the TRAIN
branch. Legacy final schemas stay distinct. Infra still performs bounded JSON
IO and exclusive safe HTML publication; the CLI never constructs a learner or
environment. Why: make verified adaptation traces inspectable without running
another experiment or relabeling training readouts as final evaluation.

## Acquired-task replay composition

The existing train_acquisition adapter composes StreamRehearsal with its unchanged
recent48/coverage policies. Wake budgets/receipts stay320; a separate original
replay session charges1280rows/160commits. return_replay_accounting accepts320
wake updates and binds native counters to wake+replay; acquired_task_return binds
owned wake192/256/320 to native288/384/480 with explicit checkpoint clocks.
No core/native objective or runner changes. All diagnostics remain TRAIN-only
owned forks after finish; defaults preserve their schema. Why: measure retention
failure using existing policies without confusing replay work with wake exposure.

## Acquired-task retention composition

The existing train_acquisition stream composes A64/B64/A64/C_train64/A64.
Core context_return_metrics adds strict pure three-endpoint arithmetic; the
acquired_task_return adapter owns twelve train-only matching-law live slots after
all training. Source/state/receipt/budget authority stays with the caller and
existing evaluator. One completed-result seal protects derived acquisition through
final caller hooks. Why: distinguish acquired-skill retention from refitting an
unacquired task. See docs/acquired-task-return.md and ADR-0344; no new runner or
native objective is introduced.

## Fixed-A exposure control

train_acquisition_pilot composes A128 through the same delayed training-only
stream/native mean objective; it collects only the two original A train traces.
train_policy_diagnostics admits exact64(default)or128 endpoints with unchanged
six-slot authority/zero-mass refusal behavior. Historical comparison is saved-only
raw A arithmetic; runtime never reads that report or final/probe data. Why: fill
the missing equal-A-exposure control without another runner. See
docs/train-acquisition.md and ADR-0343; total128vs192 work stays explicit.

## Existing weighted objective in the train-only stream

TorchTrainWeightedLearner is an adapter over the existing recurrent native owner.
It binds ordered array releases to complete owned TRAIN episodes, derives current
episode weights and delegates existing weighted CE. The native counter supplies
the schedule position; forks retain independent schedules/state, without changing
native snapshot format. Original stream/caller retain authority; the adapter
does not attest physical provenance. train_acquisition selects this explicit
option and reuses its raw scoring/legal diagnostic. See ADR-0335 and
docs/train-policy-diagnostics.md; no new runner or core objective is added.

## Train-only operational readout

The existing train_acquisition adapter finishes64 updates before handing owned
0/64 states and original authority to train_policy_diagnostics. That adapter
reuses the live evaluator, preserves six bounded slots and validates actual
support for exceptional zero-mass prior refusals. No training/final/probe access
or new runner lives in the helper. Why: distinguish teacher fit from operational
control without inventing completed baseline scores. See
docs/train-policy-diagnostics.md and ADR-0343.

## Matched existing replay controls

The small four_phase_replay adapter composes fixed existing policy/protocol
descriptors; train_acquisition remains the runner. return_replay_accounting
explicitly admits192 or256 original wake limits with full event/source/charge
checks. Separate CLI processes isolate comparative final data. Why: compare
equal update/storage ceilings while reporting differing content/cue coverage.
See docs/matched-four-phase-replay.md and ADR-0339; no new algorithm/orchestrator.

## Five-point final policy readout

The existing return_policy_evaluation adapter accepts an explicit four-phase
route; train_acquisition keeps final data unavailable until256updates/finish
and guards complete state/receipts/budgets around every call. key_door_pilot owns
the shared C_train descriptor and validated optional layout tuple on its live
runner. Old defaults are unchanged. Why: measure actual policy behavior rather
than equate trace fit with useful skill. See docs/four-phase-policy-evaluation.md
and ADR-0338 for boundaries, costs and observed forgetting.

## Four-phase training composition

The existing train_acquisition adapter adds one explicit A/B/C_train/A route;
AttributionStream and the recurrent optimizer continue across all four phases.
Core context_return_metrics validates five fixed endpoints and computes signed
differences; CLI dispatch is explicit. No final/probe/live/replay work enters
this train-only route. Why: exercise the missing third context while preserving
historical final C/D evidence and old defaults. See docs/four-phase-training.md
and ADR-0337 for costs, boundaries and limitations.

## Recorded game inspection

The pure app key_door_inspector validates saved reports and derives frames from
recorded layout/applied actions. Infra key_door_inspector_files bounds JSON IO,
escapes data into a standalone HTML template and exclusively creates a new file;
build_key_door_inspector is the thin CLI. No environment/learner runs here.
Why: preserve scientific records and the existing v14 schema while making actual
episodes inspectable. Unknown structure/sleep remains null. See
docs/key-door-inspector.md for inputs, outputs, limits and extension guidance.

## Return-state disposable probes

The train_acquisition adapter composes the existing checkpoint/probe helpers.
return_task_probes binds selected authoritative states and original benchmark
budgets/evidence around optional per-native callbacks; the app probe runner owns
disposable learning and independent charges. Outer clocks remain unchanged during
a probe. Why: measure remaining learning ability while preserving the benchmark.
Old helper defaults remain; no new runner or data-selection path. See
docs/return-fresh-task-probes.md for boundaries, costs and observed limitations.

## Context-return replay composition

The existing train_acquisition adapter composes StreamRehearsal only for an
explicit replay route. return_replay_accounting verifies wake/native progress
from existing event/commit receipts and released train content; it does not train
or choose policy. Separate original wake/replay budgets retain costs on failure.
Existing post-training evaluator guards bind both sessions and complete evidence.
Why: reuse the existing service and expose extra work without changing defaults.
See docs/return-episode-replay.md for source boundaries and measured limits.

### Final cached/live readout after return training

The existing train-acquisition adapter seals training, retains four bounded owned
checkpoints and invokes `return_policy_evaluation` for final collection/accounting
glue. Existing app cached scoring and key/door live sessions perform evaluation;
opt-in complete-state/callback guards bind the completed candidate, receipts,
original budget, source/actor and saved checkpoints. Explicit A/B/A identity marks
C/D held out. Why this: verify actual closed-loop behavior without final feedback
or another training/checkpoint framework. [Usage and structure](docs/return-policy-evaluation.md).

### Fixed train-only context return

The existing train-acquisition adapter composes A64/B64/A64 on the same native
optimizer/stream and isolated learners. Pure core `context_return_metrics` owns
strict fixed endpoint validation and signed gain/loss/gap arithmetic. All values,
negative curves and incomplete initial fitting remain explicit; metrics cannot
change training or budget. Why this: distinguish acquisition, interference and
return refitting using existing runtime mechanisms. No new checkpoint framework.
[Protocol and usage](docs/train-acquisition.md).

### Explicit training-only acquisition

`AttributionStream` admits exact `training_only=True` with empty heldout data,
preserving native release/receipts/budgets/observers and binding mode across
continuation. Finish seals training with no heldout scoring; default evaluation
remains unchanged. Adapter `train_acquisition_pilot` composes this existing stream
on an isolated learner, scores train packets on owned checkpoints and records
costs plus a fixed train-label prior/majority control. Why this: train-fit data
must retain their role and resubstitution meaning. No alternative training loop.
[ADR-0336](docs/adr/ADR-0336-keep-task-fit-training-only.md),
[usage and structure](docs/train-acquisition.md).

### Explicit BC-only recipe conditions

The existing key/door reward adapter composes its typed demonstration owner and
frozen-control runtime for standalone mean or weighted BC. CLI dispatch declares
the condition; bounded train-derived weights stay per-update inputs. Objective
identity is bound across collection/commits/final scoring, preserving preceding
success evidence, original budgets and source/serving isolation. Historical
matched reference analysis is saved-only, with no scored model restoration.
Why this: expose the accepted learner through existing orchestration and measure
a small hypothesis without another runner. [Usage](docs/weighted-bc.md).

### Explicit per-episode class-weighted demonstrations

Core `demonstration_class_weights` derives bounded inverse-frequency weights from
complete train-role labels. Typed `ClassWeightedDemonstrationUpdate` owns episode
and derived tuple, revalidates canonical weights and matching release at dispatch.
Adapter `torch_recurrent_weighted` reuses recurrent BPTT/optimizer/capture with
target-weight-mass normalization and one atomic payload commit. The existing
native public method and shadow runtime compose it; policy/snapshot fields,
ordinary mean CE/reward behavior and serving activation stay unchanged.
Why this: weighting is an explicit experimental objective; retained baseline and
role semantics must remain visible. [ADR-0335](docs/adr/ADR-0335-keep-class-weighted-demonstrations-explicit.md).

### Bounded diagnostics from existing reserved probe predictions

Pure core `probe_decision_diagnostics` owns binary/categorical decision semantics,
bounded class/confusion/row reports and detached primitive outputs. Existing app
`fresh_task_probe` owns the exact-bool opt-in, reserved row/episode provenance,
original callbacks/budgets and complete-checkpoint publication. No new native
calls or ranking metrics are introduced. Additional report/NumPy work is measured
in total invocation time/process RSS and excluded from input-array byte counts.
The existing key/door helper/runner/CLI forwards the explicit opt-in only for its
learnability scenario. Default callback kwargs and protocol fields remain stable.
The helper checks enabled/context-map presence before dependent work, preserving
failed reports and charges; accepted core arithmetic remains the single numeric
implementation. No second stream or scoring framework is introduced.
Why this: aggregate accuracy/CE alone did not establish useful task learning.
[ADR-0334](docs/adr/ADR-0334-reuse-reserved-predictions-for-decision-diagnostics.md).

### Matched frozen warm-start control through the existing recipe

The existing reward recipe builds separate fresh BC-only and BC→reward owners,
checks complete native state equality at8 before sampled reward work, and retains
original8/16update budgets. At most two owned completion closures hold final work
until both train. Each rechecks captured records/state/RNG and budget bindings;
source/actor mismatch or a preceding final failure refuses dependent work.
The control stays native8. Endpoint reporting requires every unique final pair
and retains both values/all signed metrics with unequal costs and momentum caveats.
Why this: parameter-only archived JSON cannot establish matched optimizer state,
and a new runner would duplicate the existing runtime. [ADR-0333](docs/adr/ADR-0333-match-warm-start-state-before-reward-comparison.md).

### Typed demonstration-to-reward phases on one native owner

Adapter `torch_demonstration_reward_learner` owns one existing recurrent learner
and accepts exact train-role DemonstrationUpdate or RewardUpdate packets plus
matching ID releases. It delegates unchanged CE/reward objectives and the same
full native snapshot/fork format; gradients and momentum survive the boundary.
The existing reward recipe owns the opt-in eight-demo/eight-reward schedule and
one original sixteen-update budget. Shared budget defaults remain eight; final
scoring waits for all sixteen commits. Bounded evidence guards and separate
phase-cost errors preserve the primary refusal and unknown partial work.
Why this: phase order belongs to orchestration, while objective types prevent
teacher labels from acquiring reward semantics or resetting optimizer state.
[ADR-0332](docs/adr/ADR-0332-preserve-native-state-across-demonstration-and-reward-phases.md).

### Reward episodes through the existing adaptation runtime

Adapter `torch_reward_update_learner` implements the existing opaque NativeLearner
port with an exact owned RewardUpdate and matching episode-ID release. It delegates
the unchanged explicit native reward objective and complete snapshot continuation.
Adapter `key_door_reward_pilot` composes the existing ActorShadowRuntime/inbox,
collector, budgets, checkpoint scoring and fresh live sessions; final scoring is
deferred until all training commits. Bounded previous-record/state/clock/RNG checks
precede further collection. The app checkpoint scorer depends on a prediction-only
structural port and explicit training contexts, without requiring supervised train
signatures. Core/app import no adapter. Why this: opaque runtime ports already
support reward records; separate objective types prevent teacher-label confusion.
[ADR-0331](docs/adr/ADR-0331-route-on-policy-rewards-through-existing-runtime.md).

### Explicit recurrent reward trajectories

Core policy_gradient_episode owns separate immutable physical/behavior records
and scalar discounted weights. Adapter torch_recurrent_reward reuses the existing
recurrent family math/SGD/snapshot internals, checks full on-policy behavior state,
and publishes only a validated staged update. Existing learner adds an explicit
reward entry point; supervised array/episode methods remain CE. App
key_door_training composes existing environment and frozen sessions with explicit
caller-owned PCG64, honest endings and partial-cost reports. Core/app import no
adapter. Why this: reward targets cannot share supervised label semantics, while
another stream framework is unnecessary. Native admission protects against stale
source state; collector relies on the owned-session port contract. [ADR-0330](docs/adr/ADR-0330-separate-recurrent-reward-trajectories-from-demonstrations.md).

### Reserved probes at owned stream checkpoints

The explicit learnability CLI scenario uses the existing key/door stream observer
and app probe. Adapter key_door_learnability owns bounded planned slots, separate
original probe budgets and full source/checkpoint/budget readback. It does not
run another benchmark stream or select/promote a model. Whole owned checkpoints
restore on disposable forks; source/actor/benchmark budgets remain outside updates.
Existing default routes add no probes. Why this: optimizer state cannot be rebuilt
from old parameter-only reports. Per-slot timing includes detachment/execution;
final readback is covered by full invocation time. [ADR-0329](docs/adr/ADR-0329-probe-owned-key-door-checkpoints.md).

### Reserved categorical complete-episode probes

Core complete records explicitly allow reserved probe roles; existing exacttrain
replay gates remain. App fresh_task_probe binds one class count and all complete
metadata, dispatches categorical scoring while preserving binary semantics, then
uses existing owned fork/budgets/charges/checkpoints. Each call is one packet;
record IDs/counts and memory exclusions are detached before callbacks. Benchmark
roles are never mapped to probes, and source/serving state remain outside updates.
Why this: recurrent whole-episode outputs require explicit class/reset/role
boundaries. [ADR-0328](docs/adr/ADR-0328-preserve-reserved-roles-in-complete-episode-probes.md).

### Recurrent control through the existing key/door recipe

The existing adapter/CLI select either original feedforward arms or two recurrent
BC replay modes through one private execution function. Complete-episode scoring
resets recurrent history; every live slot creates a fresh frozen session while
full learned-state checks remain. Explicit memory/unrolled work and partial-failure
reporting augment only the recurrent route. Original controls/authority/metrics
stay in their existing layers; accepted baselines are reused posthoc. Why this:
repeating baseline training or adding another runner would not change the learner
question. [ADR-0327](docs/adr/ADR-0327-reuse-key-door-runner-for-recurrent-control.md).

### Recurrent behavior cloning and isolated live activation

Adapter `torch_recurrent_learner` implements the existing opaque NativeLearner
port with one ordered complete episode per array call, zero initial hidden state,
mean CE/BPTT and nativeSGD. The existing complete-record helper adds train-role/
class validation. Staged owned parameters/gradients/optimizer/counters publish
atomically. Separate bounded live sessions own frozen learner copies and staged
activation; full learned-state guards remain, while common identity explicitly
admits recurrent_bc. The existing attribution stream/shared packet buffer/live
callback composition requires no orchestration changes. Why this: hidden cue
aliasing requires observed history and activation changes are distinct from
learned-state mutation. [ADR-0326](docs/adr/ADR-0326-separate-recurrent-activation-from-frozen-learned-state.md).

### Actual local key/door episodes through existing stream ports

Core `key_door_navigation` owns deterministic partial-observation physics and
scripted complete demonstrations. App `key_door_policy_evaluation` consumes only
a predictor callback and common evaluation identity; it validates target-free
probabilities with the existing core categorical validator and records actual
bounded transitions/partial failures. Adapter `key_door_pilot` collects real
packets, composes the existing attribution/replay/native ports, then evaluates
owned complete native checkpoints without changing source/deployed actors.
The existing CLI selects the recipe. Dependency direction remains inward;
there is no new runner, training objective or final-feedback path. Why this:
actual legal actions and returns cannot be inferred from classification scores.
Episode-local recurrent activation state is a future learner boundary; arbitrary
native state mutation is still refused. [ADR-0325](docs/adr/ADR-0325-evaluate-actual-hidden-cue-key-door-episodes.md).

### Categorical packets through existing attribution and replay

Core packets/schemas bind explicit class counts while preserving original binary
wire bytes; a pure metric module validates categorical targets/probabilities.
App ingress enforces one declared count, helper identity binds it, and final-only
scoring dispatches to unchanged binary or explicit categorical metrics. Existing
shared buffer, permanent whole-packet charges, native objectives and continuation
remain. The new fixed CLI recipe labels its synthetic records explicitly; no new
trainer or environment claim. See [ADR-0324](docs/adr/ADR-0324-carry-declared-action-count-through-existing-rehearsal.md).

### Categorical native feature policy

`TorchFeaturePolicy.num_classes` binds native constructors, complete state and
parameter/gradient/optimizer geometry in the existing adapter. Hard class-index
labels reach unchanged native objectives. Count2 preserves the original scalar
probability contract; explicit larger counts return full class distributions.
No new layer, learner family or runner; categorical stream/episode support is
now implemented by the recipes above. Recurrent learning remains open. See [ADR-0323](docs/adr/ADR-0323-bind-native-feature-actions-to-explicit-class-count.md).

### Live episodes from explicitly recorded native parameters

App `line_policy_evaluation` consumes a predictor callback over actual core
LineNavigation, with bounded complete/partial reports and explicit evaluation
identity. Adapters `frozen_torch_predictor` and `line_rollout_pilot` validate
recorded weights, reproduce every cached score before live work, then compose
native prediction, existing execution budget and RSS sampler through the existing
CLI. Dependency direction remains inward. No training, final feedback or new
runner; fresh local state is not historical optimizer/RNG/publication recovery.
See [ADR-0322](docs/adr/ADR-0322-evaluate-recorded-weights-through-live-policy-episodes.md).

### Complete demonstration episodes in existing rehearsal

Core records validate actual contiguous transitions and encode one bounded
complete episode per existing replay row. LineNavigation supplies real legal
steps. App attribution validates exact metadata/arrays and carries episode
identity through arrival; optional rehearsal floors native-transition quotas
to whole packets, charges transitions and trains ordered action batches.
Original shared selection, runtime authority, native objectives and owned
continuation remain. No RL/reward loss or new runner. See
[ADR-0321](docs/adr/ADR-0321-replay-complete-demonstration-packets-in-the-existing-buffer.md).

### Shift incidents over unchanged detector decisions

Core ShiftIncidentTracker consumes exact ordered decisions and retains three
active onsets, count and monitoring flag. App observer optionally records its
bounded scalar outputs within existing timing/failure boundaries. The shift
adapter adds onset-only outcomes after observation while preserving old level
annotations; contexts and final data never enter tracking. No learner/policy
feedback or calibration change. See [ADR-0320](docs/adr/ADR-0320-distinguish-shift-incidents-from-alarm-levels.md).

### Full capacity comparison and one guarded prune

The capacity recipe composes existing raw/ordinary learner ports through the
same attribution stream. App StreamCapacityPrune owns one original validation
policy and consumes the existing structural gate. Capacity observes actual wake
usage and published changes; rejected unchanged state continues with a spent
attempt. Default-off retained gate evidence holds only one owned validated
decision and clears before each call. Why this: full successful diagnostics
previously disappeared, and rerunning a gate would alter work/evidence. Native
math and runtime publication authority stay in existing boundaries. See
[guide](docs/capacity-comparison.md) and ADR-0319 for scientific distinctions.

### Observe bounded renewal without another runner

Attribution pairs observation-only capacity with existing unit renewal, measures
wakes before renewal and candidate usage afterward. The helpers retain their
own scalar work/usage reports and original authority. Original renewal settings
are checked after opaque measurement and before schedule arithmetic; a pinned
event receives any secondary readback error, preserving the primary failure.
Why this: the NumPy measurer already understands renewable snapshots, so a
boundary-ordering change supplies required cost evidence without a second
orchestrator or native state conversion. See [guide](docs/renewal-capacity.md)
and ADR-0318; full cohort comparison remains unfinished.

### Structural matched BP

The structural BP adapter composes the existing strict raw BP port with core's
rooted structural state. It validates the complete original configuration/history
and constructs an owned replacement before publication. Native math and the
fixed-config port remain unchanged. Existing capacity per-arm orchestration
serves explicit BP controls; no second stream framework is introduced. Why this:
width-changing matched controls need owned reconstruction, not relaxed native
restore. See [guide](docs/structural-matched-bp.md) and ADR-0317.

### Committed capacity on the existing stream

`core/capacity_cost` defines pure count units; `adapters/numpy_capacity` reads
validated native arrays without model operations. App `stream_capacity` receives
that measurement port and uses the existing structural consolidator/runtime.
Attribution binds the original helper/budgets, observes wakes before checkpoint
capture and leaves default execution unchanged. Candidate-byte means/peaks and
completed work differ from temporary/process memory; RSS is separate. The shared
binary scorer now lives in core: attribution and pruning gate depend inward,
removing their demonstrated import backedge without changing metric arithmetic.
See [guide](docs/stream-capacity.md) and ADR-0316 for scope and next controls.

### Optional slow-teacher replay

Core slow-teacher records validate bounded owned target arrays, versions and
the affine BCE mixture. The adapter owns complete ordinary NumPy student/teacher
state and stages native wake/replay updates before publication. App rehearsal
uses only the core policy and teacher port, with arrived rows and permanent
forward/replay charges; the pilot composes the existing runner. Why this: reuse
verified native soft-label objectives and evaluation isolation without a second
orchestrator or silently changing Torch semantics. Default paths remain intact.
See [guide](docs/slow-teacher-replay.md) and ADR-0315 for limits and extensions.

### Existing attribution loop continuation

`AttributionStream` exposes the existing delayed-label loop; the original function
is a completion facade. An owner-local token holds complete candidate/inbox/
checkpoint/replay data. Resume stages independent payloads using existing inbox
materialization, validates before/after opaque operations and publishes under
the original runtime lease. Actor, budgets, clocks and receipts stay original;
the old inbox retires and the consumed token clears. Why this: model-only copies
cannot preserve delayed labels or spent replay authority. No disk/capture
framework is added. See ADR-0314 and attribution-stream-continuation.md.

### Owned CPU Torch feature heads

The opt-in `torch_feature_learner` adapter owns existing native PC/CPC/matched BP
heads behind the generic inward learner port. Complete snapshots include native
adaptive/RNG fields or BP SGD/gradients; training/restore publish validated staged
payloads. `torch_feature_rehearsal_pilot` composes the existing attribution app and
shared rehearsal. Why this: preserve native objectives while reusing isolation
and permanent admission. Component continuation does not restore runtime
authority/whole-stream state. See ADR-0313 and `docs/torch-feature-rehearsal.md`.

### Long-stream rehearsal evidence

The small `long_stream_rehearsal_pilot` adapter composes existing learners, replay
policy and attribution application on fixed64-batch data. Existing event snapshots
already expose bounded retained IDs. A read-only helper property returns current
detached retention for final64; old report schemas/method bodies persist. The new
adapter's train-only content manifest classifies retained/exposed IDs for reporting
after execution; it never feeds context, future labels or metrics to a selector.
Why this: measure representation and exposure separately without another runner.

### Bounded candidate unit renewal

Core `coupled_unit_renewal` defines heterogeneous histories and deterministic
mature/low-gradient selection; explicit renewable PC/CPC classes atomically reset
selected parameters/history/chemistry. Complete format3 snapshots bind policy and
state, while old formats keep their meanings. The existing owned learner adapter
copies exact renewable classes. App `stream_unit_renewal` owns finite permanent
charges and uses the existing actor/shadow consolidation from the attribution
runner. App depends on core interfaces/state, not the concrete adapter. Final
scoring follows training; simultaneous rehearsal/renewal is initially unsupported.
Why this: restoration must preserve continuation without refunding authority.
See ADR-0312 and unit-renewal.md.

The renewable diagnostic reader explicitly accepts snapshot3 and validates history,
policy, geometry and clock before feature inference. Existing learnability probes
restore complete renewable state into independent templates without a renewal
helper. The small `unit_renewal_pilot` adapter composes these existing applications;
the attribution CLI selects it explicitly. Historical control scores are external
post-run analysis inputs and never reach training or probe selection.

### Opt-in coupled native histories

The existing isolated observer reads exact tracked snapshots before feature work.
An explicit rehearsal scenario binds independent tracked benchmark/template
learners to the same runner and records full history bytes/time; no new training
orchestration is introduced.

Core `coupled_unit_history` owns a bounded fixed-width record and pure gradient-EMA
transition. Explicit tracked native subclasses prepare history at existing commit
hooks; default models gain no fields. Format-2 snapshots retain full tracked state.
The existing owned adapter copies exact tracked types and the diagnostics adapter
reads detached summaries. Existing exact format-1 codecs/capture refuse unsupported
tracked state. Why this: measure genuine native histories without changing default
math or silently dropping state. See ADR-0311 and learnability-diagnostics.md.

### Coherent stream rehearsal

App `stream_rehearsal` composes the existing core shared buffer and actor/shadow
consolidation hook. The existing attribution runner supplies only the just-
committed arrived training batch and captures post-rehearsal checkpoints; final
scoring remains deferred. The thin adapter/CLI binds four native methods to a
fixed local case. Separate initially planned wake/replay sessions preserve the
runtime's replay-limit refusal. Why this: execute existing proposals without
another runner or changing native objectives. See ADR-0310/stream-rehearsal.md.
A fixed paired-initialization adapter scenario reuses that same single-row
execution helper. It adds bounded reporting and pairing, with one global deadline
and sampler; it introduces no alternate training orchestration.
The optional learnability mode uses a small public facade on the existing
learnability adapter. It binds an independent template and fixed reserved data
to the existing checkpoint callback; no new runner or probe algorithm is added.
An explicit recent-only policy reuses the same rehearsal helper with the existing
FIFO buffer/selector and applicable controls only. It shares charge/failure and
isolation paths, while preserving the rotating policy's original report schema.

### Partitioned shared replay

Core `shared_replay_schedule` owns opt-in recent/reservoir quotas, one unique
array store and detached interleaved selection. Existing app schedule owns
arrived-training role access and common per-method work proposals. Existing CLI
owns exclusive JSON output. Why this: improve bounded memory coverage without
another runner. The legacy native runner's CPC-buffer equality gate remains;
schedule proposals are not executed learning. See ADR-0309.
Optional reservoir rotation retains only the next turn and one scanned ID,
publishing that cursor after detached selection construction. The schedule
binds a separate sampler identity while preserving the rank-first route.

### Arrived-only shift observations

Core `shift_signals` owns bounded measurements/reference/window/persistence.
App `arrived_shift_observer` validates arrived training metadata, retains one
previous copied state and sends a separate copy to a target-free predictor.
Adapter `shift_signal_pilot` binds matched native snapshots and composes the
existing stream runner. Signals are computed after update k using parameters
from k-1; they do not control training or access final evaluation. Why this:
observe operational data without adding a runner or contaminating evaluation.
Storage, failures, timing and descriptive statistical limits are in ADR-0308.
Optional core `shift_calibration` fits from a bounded arrived-training prefix
and composes the same detector; app/adapter opt in explicitly. Fitted thresholds
freeze before monitoring. Replay of scalar summaries is measured separately
from training, and reference-dependent unreachable channels remain explicit.

### Learnability diagnostics and disposable fresh-task learning

Core `learnability_diagnostics` summarizes bounded tanh activations without labels
or model mutation. App `fresh_task_probe` owns fixed reserved-data roles, copies,
disposable forks and separate budgets. Adapter `learnability_pilot` composes these
with the existing attribution stream observer; its closure contains a separate
probe template and no benchmark final data. Final scoring stays after observers.
Why this: measure learning capacity without contaminating benchmark/deployed
learners or adding another orchestration framework. Coupled native importance/age
remain unavailable rather than inferred from chemistry. The separate
`adapters/native_unit_diagnostics` binds exact legacy CPC stored vectors to pure
bounded unit summaries, using current adaptive width and detached immutable
results. It calls no model method; callers keep the model quiescent while read.
The existing native stream demo can opt in to initial/final observations without
changing training or its default report. See ADR-0307.

The learnability adapter also selects a fixed sixteen-update probe scenario,
passing its checkpoints and separate budgets through the same observer and probe
app. Defaults remain the original four-update protocol. No new runner or native
learning rule is introduced; benchmark final scoring remains after every probe.

### Successive current updates after retained publication

`app/replay_publication_updates.py` observes the original ledger update when both
birth history flags are enabled. Existing publication installs a paid weak
baseline. Existing nonblocking copies/manager gates cover the subsequent poll;
the first completion is provisional until original ledger checks and final pure
revalidation succeed. Each next poll pins and verifies the latest paid success;
replacement releases that weak record without refunding its payment. Complete
capture combines the unchanged retired proof with
the paid current success and complete original inbox history, including native
eviction. Why this: replay fingerprints do not attest the whole native state.
No native work or retrospective authority is introduced by capture. Manager
observation limits are capped separately from lifetime admission. See ADR0306.

### Runnable successive replay composition

The existing restored example adds a fixed continuation mode composed from the
accepted API: original birth enrollments and resources, real restore, two more
ledger updates and an observed paused poll. The same capture/report helper
validates complete components and live authority; current history is three and
retired history remains one. Existing codecs and joint detachment checks remain
separate from provenance. Why this: expose useful runtime continuation through
one CLI without another proof, graph format or algorithm. Existing mode defaults
remain unchanged; no native operations occur during capture/report.

### Runnable restored replay composition

`adapters/numpy_restored_capture_example.py` reuses the issued-snapshot setup with
history retention enabled at original manager birth, then performs one real
restore before complete capture. It uses the accepted publication proof and
existing native/inbox codecs. Complete report components remain data; the joint
captured graph independently preserves aliases and detachment. Capture/report
invokes no native methods. The existing CLI selects this composition explicitly.
Why this: expose the accepted runtime branch without another proof or codec.
Default and issued-snapshot routes retain their original behavior.

### Restored replay capture

`app/replay_restored_capture.py` retains a bounded weak publication witness only
when the original manager opts in at birth. The validated handoff prepares all
metadata and charges the original admission before publication; trusted commit
only assigns prepared sealed storage. Complete capture borrows that proof under
existing leases and verifies old/current native rows, inbox history and every
holder role. No native operations or restore permissions occur during capture.
See ADR-0305. Later work, chained handoffs and other-holder branches remain open.

### Runnable issued snapshot capture

`numpy_snapshot_capture_example` composes the existing managed replay ledger,
issuing checkpoint manager and complete capture API for one fixed trained CPC
graph. Existing native/inbox component codecs render complete data observations;
joint graph aliases and original authority stay separate. The existing NumPy
capture CLI selects this composition explicitly; default cases stay unchanged.
No new core/app policy, serializer or restoration capability is introduced.

### Legacy populated-holder capture example

The optional complete-capture `replay_copies` boundary uses an inward app reader
under original leases. Existing synchronous model-copy observations expose a
borrowed constructor recipient; original copied-row witnesses retain it weakly.
The reader proves original checkpoint role/recipient/rows at each bounded round,
spends the existing single ledger admission and issues no authority. Native state
fields and default copy behavior remain unchanged. See ADR-0303 for why original
recipient identity must be observed before policy callbacks. Optional
`replay_checkpoints` uses existing issued token/state/inbox witnesses through a
second app reader under the original manager gate (ADR-0304). It reuses transient
operation validators without public-method lock reentry or native work. Only the
actual issued view in its original snapshots role qualifies. A separate opt-in
publication reader supports one unchanged restored pair (ADR-0305); later work
and other-holder lineage remain unfinished under original d2/d2b.

`numpy_capture_example` composes original managed ownership, checkpoint handoff,
promotion preparation and `capture_numpy_managed_composite`. Existing native codecs
and inspectable NumPy builder bind snapshots and policy; no new storage or source
schema is needed. The CLI exposes two bounded zero-training cases. Naturally empty
replay is supported; copied nonempty rows retain the original lineage refusal.
Full managed reconstruction and parent all-variant qualification remain open.

### Live coupled complete checkpoint composition

The opt-in live checkpoint adapter composes original paused-owner capture with
the complete data codec and existing bounded store. A narrow observer seam in the
managed-serving example runs at B and committed C. Its additional raw-copy budget
is declared at owner creation, never renewed or refunded. The same original owner
continues and expires; no decoded object becomes runtime authority. See
`docs/coupled-composite-files.md` for the live command and caller-copy limits.

### Complete coupled checkpoint file composition

`scripts/run_coupled_composite_files.py` composes the existing generic
`BoundedCheckpointFileStore` and `CoupledCompositeCheckpointCodec` through their
typed port. Input expectations are explicit caller controls, never derived from
wire content. Load/save/load produces a verified data-only copy in a new directory;
the existing infrastructure alone owns staging, bounded file reads and replacement.
No inner-layer or native algorithm changes are needed. See
`docs/coupled-composite-files.md` for failure and authority limits.

### Coupled composite data bytes

A separate adapter encodes/decodes the complete detached coupled graph using
explicit record/container nodes and alias references. Existing pure graph,
lifecycle and record checks validate data without original live ports. Raw wire
and aggregate array admission precede array materialization. Live authority stays
outside bytes, so decoded state cannot relax recovery completion or publish owners.
See ADR-0302 and `docs/coupled-composite-codec.md`.

### Complete coupled managed capture

The coupled capture adapter supplies exact learner/model schemas and bounded
native validation to the existing app capture coordinator. A narrow protected
extension of the NumPy graph walker reuses alias-preserving copying without
widening legacy admission. The native helper borrows codec structural validation;
it does not encode or call models. Original leases, copy accounting and authority
remain in app/core. See ADR-0301 and `docs/coupled-managed-capture.md`.


### Failed coupled cleanup

A dedicated one-shot trusted adapter composes the exact coupled erasure port at
original lifecycle birth. Shared typed composition preserves default factory
behavior. The manual failure control uses existing app revocation, invalidation,
candidate fencing and terminal retry; successful erasure does not reopen driver
or training authority. See ADR-0300 and the coupled handoff guide.


### Busy terminal cleanup

The managed-serving adapter also exercises original holder leases during bounded
stop. A joined driver retains a separate pending-cleanup state; existing
finish_cleanup clears owned generations after holder release. This adds no app/core
scheduler or cleanup implementation. Public access stays closed throughout.
See ADR-0299 and the coupled handoff guide.

### Managed coupled cached-serving expiry

The same fixed adapter also exposes an explicit committed mode. It retains both
serving generations and the original rollback receipt until managed expiry, then
checks that cleanup clears their raw dictionaries and revokes rollback authority.
Default pending behavior remains separate; no app/core cleanup change. ADR-0298.

A separate adapter composes the fixed coupled learner with existing cached serving,
promotion guards/controller and managed expiry. It adds no app/core dependency
on adapters or new cleanup mechanism. Actor/pending metadata and cache references
join the same original ownership registry and byte ledger across handoff. See
ADR-0297 and the coupled handoff guide for the fixed protocol and limitations.

### Coupled elapsed retention

A separate adapter supplies accepted coupled measurement ports to the existing
ManagedDataLifecycle. Callers explicitly own the RetentionExpiryDriver lifetime.
The example preserves original elapsed authority and copy accounting across
handoff, then exercises automatic purge and bounded stop. Both direct actor
prediction paths check lifecycle access before copying and recheck after trusted
callbacks before publication. Dependencies point from adapters to app/core; the
existing lifecycle owns cleanup and scheduling. See ADR-0296.

### Coupled candidate checkpoint callbacks

The adapter binds a detached complete update policy and codec limits to the
existing candidate controller's state/build/policy callbacks. Dependencies point
from adapter to existing app/core and codec; no app/core behavior changes. The
controller retains preparation charging, cursor/resource checks and single-point
owner publication. The fixed integration retains the surviving actor, budget,
clock and sharing gate; disk/process/lifecycle authority is outside this slice.
See ADR-0289 and [coupled handoff](docs/coupled-candidate-handoff.md).

An opt-in repeated integration reuses this same composition and original live
objects across three transfers, ending in spent preparation/update refusals.
It adds no app/core contract or recovery authority; see ADR-0290.

The resource-control mode retains the existing budget clock and RSS sampler/
reader across owners. Stable fields remain equal while legitimate cumulative
telemetry advances; deterministic injected limits are separate from actual host
RSS boundary telemetry. Existing refusal/accounting behavior is preserved;
see ADR-0291.

The repeated adapter also exposes a fixed16-cycle endurance mode through the
same internal loop, retaining original monotonic/RSS resources and compacting
records only after complete state/cursor checks. Existing modes keep their
schemas; no new controller or recovery framework is introduced. See ADR-0292.

The managed-consent adapter installs the existing ManagedExperienceOwner before
registration. Its original catalog, opt-out sets and bound admission/eligibility
guards survive candidate replacement. A consent change invalidates a captured
checkpoint before preparation; fresh transfer preserves authorized continuation.
This composes existing app/core behavior, retains raw records and makes no
deletion/unlearning/process-authority claim. See ADR-0293.

The coupled lifecycle adapter supplies exact-learner zero-replay callbacks and
real numeric NumPy byte measurement to existing ManagedDataLifecycle. Coupled
native models keep no raw replay batches, so erasure leaves learned state intact.
The cleanup integration exercises registry-owned retired/current inbox erasure
and pending-token invalidation with original resource/ingress/holder accounting.
This basic factory refuses owned-copy/elapsed policies; the separate factories
above supply those ports explicitly. No app/core dependency points outward.
See ADR-0294.

The opt-in coupled copy adapter supplies bounded typed graph/checkpoint/growth/
preparation/prediction measurements to the existing lifecycle copy ledger. An
exact inspectable builder delegates complete-policy reconstruction to existing
checkpoint callbacks. The basic lifecycle factory stays unchanged. The shared
cleanup loop adds a separate fixed byte-limit variant and retains its original
default interface/schema; both modes are exercised after this change. Raw array
copy bytes exclude learned parameters, caller copies and OS memory. See ADR-0295.
The controller's integrity check uses a private borrowed validated inbox cursor
under its existing exclusive owner/gate leases. Public captures remain detached;
this avoids a raw inbox clone before byte admission while retaining full hashing.
Temporary serialization bytes remain outside the retained-array ledger.

### Complete coupled learner checkpoints

Core records describe native state plus fixed learning/inference policy without
IO or array copying. The owned adapter exposes a detached policy getter while
preserving its native snapshot API. A separate envelope codec deducts exact
metadata overhead before existing native codec work and validates all bytes
before explicit model reconstruction. The file store is unchanged; no runtime
authority is reconstructed. See ADR-0288 and
[learner checkpoints](docs/coupled-learner-checkpoint.md).

### Native continuation through coordinator loss

The native transfer session composes existing coordinator reservations, complete
native codec/store and acknowledged SQLite completion. Its strict wire preserves
actual completion proofs; the fixed worker/example retains independently known B
and live physical handles before coordinator loss, transfers explicitly, charges
full B restoration and completes B → C under the original allowance. Existing
core transition rules remain pure and unchanged. No journal bootstrap, actor
publication or full managed lifecycle restore is introduced. See ADR-0286 and
[native transfer](docs/native-coordinator-transfer.md).

The optional interruption wrapper pauses after delegated native training and
cannot return its result to the session. A separate adapter validates full C
telemetry against the independent reference, terminates only its owned peer and
uses existing terminal reconciliation for exact pending authority. Completed B
and spent uncertainty remain; later supervisor observations are recorded
separately from the last child stamp. See ADR-0287.

### Retained physical transfer supervision

The Windows supervisor retains original live registrations and independently
known authority across a coordinator crash. A directly owned replacement child
observes its own registered worker over the bounded private channel; the final
supervisor sample records later time/RSS separately. Existing SQLite CAS precedes
child adoption and existing coordinator continuation. All process lifetimes stay
with the outer example, all pure transition rules in core. The native adapter
above adds bounded checkpoint restoration. See ADR0285 and
[physical transfer](docs/windows-coordinator-transfer.md).

### Explicit coordinator transfer

The pure `core/coordinator_transfer` module validates a separately witnessed
anchor/worker transition without IO or physical capabilities. The SQLite adapter
adds an exact-floor CAS and terminal-only ambiguity reconciliation; existing
generic authority paths stay strict. Original absolute start, full manifest and
cumulative usage survive the transition. Physical supervisor binding belongs in
the outer Windows layer described above. See ADR0284 and
[coordinator transfer](docs/coordinator-transfer.md).

### Native failure accounting control

The native-failure adapter composes existing structural native ports and actual
inbox observer stages. A local wrapper injects a failure after delegate mutation;
an observer injects the separate post-receipt failure. Original runtime/inbox
stop and accounting rules remain authoritative. Complete private forensic reads
are local observations only; actor cache serving remains independent. See
ADR0283 and [native failures](docs/coupled-native-failures.md).

### Delayed feedback through serving transitions

The fixed delayed-feedback adapter composes existing inbox producer-version
pairing, raw structural native ports and serving transactions. Candidate learning
after promotion stays independent of the published bundle; rollback restores
the prior bundle without renewing training allowance. Refused registration's
spent revision is tracked separately from preserved native/inbox state. No new
feedback or authority layer is introduced. See ADR0282 and
[delayed feedback](docs/coupled-delayed-feedback.md).

### Native cache transactions

The fixed cache-transaction adapter composes existing raw structural native
ports, ActorShadowRuntime, PromotionGuardEvaluator and ServingPromotionController.
It exercises cache expiry/eviction/refusal and complete bundle publication/rollback
without adding a transaction layer or changing native learning rules. CLI output
is bounded; full native and serving observations stay local. See ADR0281 and
[native cache transactions](docs/coupled-cache-transactions.md).

### Finite growth endurance

The endurance adapter composes existing growth/native/runtime/cache policies and
streams fixed counters, blocks and scalar samples. Complete states and final
cursors are retained separately; per-request native payloads are not accumulated.
The aggregate summary and original attempt quota remain bounded. A tiny test
prefix is separate from the single budgeted full run. See ADR0280 and
[finite endurance](docs/coupled-growth-endurance.md).

### Coupled split width ceiling

The existing split consolidator optionally owns a narrower max_width. Current
state and target metadata validation precede the check; expanded kernel work
follows only if admitted. Defaults and inner split mathematics remain unchanged.
An outer integration demonstrates accepted growth, retained failed attempts and
native continuation with a stable actor/cache. See ADR0279 and
[bounded split growth](docs/coupled-growth-limit.md).

### Repeated rejected native consolidations

The outer rejected-consolidations adapter composes existing native learner ports,
actor/shadow runtime, serving cache, consolidation cursor and boundary RSS sampler.
Failed detached transforms retain original consumed IDs and quota while native
candidate state and actor/cache stay unchanged. Attempt/revision changes are
reported separately; no inner runtime or model change. See ADR0278 and
[bounded refusals](docs/rejected-consolidations.md).

### Native candidate checkpoint corruption

The existing outer completion adapter/CLI adds explicit owned-file corruption
after verifying and saving a complete healthy B, before bounded reload with the
original expected digest. The unchanged store refuses damaged bytes and the
unchanged coordinator records terminal uncertainty with all costs retained.
Default healthy completion remains separate from the forensic damaged file;
actor A stays unchanged. No inner-layer or dependency change. See ADR0277 and
the native completion guide's corruption control.

### Native update interruption

The outer update worker exposes an explicit engineering pause after computation
and bounded encoding, before full B response. The interruption adapter terminates
its owned registered child inside preparation, exercising existing coordinator
live-process guards before validation. Actor A remains usable; all admission,
copy and attempt charges remain terminal and uncertain. No inner-layer or journal
rule changes. Separate ended observations do not invent a fresh terminal stamp.
See ADR0276 and [native interruption](docs/native-update-interruption.md).

### Explicit native-only completion

`core/recovery_completion.py` binds one original admission, live registered host
facts and an exact candidate-only completion transition. The coordinator's new
opt-in method calls trusted complete-checkpoint validation before a fresh sample
and explicit journal CAS. Reporting witnesses include pending/completed ambiguity
for terminal-only reconciliation. Generic floor validation, legacy execute and
authority wire fields remain unchanged. New outer native worker/demo compose
complete model bytes and Windows authority while actor A remains unchanged; full
managed-bundle/actor promotion/coordinator-loss recovery remain separate. See
ADR0275 and [native completion](docs/native-update-completion.md).

### Supervised checkpoint writer interruption

`infra/bounded_json_child.py` owns bounded JSON pipe/process IO without native or
authority imports. `adapters/coupled_checkpoint_worker.py` stages bytes without
publication or restores/predicts a complete model without training. The outer
interruption adapter composes unchanged codec/file storage, Windows retained
registrations, SQLite journal and surviving coordinator. Native A/B preparation
precedes authority creation; spent checkpoint/copy charges survive writer exit
and handoff. Uncertain native completion and coordinator-loss/full lifecycle
recovery remain separate. See ADR0274 and the writer interruption guide.

### Bounded checkpoint files

`infra/checkpoint_files.py` composes the existing core codec port with bounded
regular-file reads and caller-known digest checks. Save completes a same-directory
temporary write, flush, fsync and close before atomic replacement; failure cleans
only owned resources. The outer CLI selects the native codec and verifies complete
restored continuation. Trusted parent directories and one cooperating writer are
required. Runtime ownership, process recovery and power-loss durability remain
separate. See ADR0273 and [checkpoint files](docs/checkpoint-files.md).

### Complete coupled native snapshot bytes

`adapters/coupled_checkpoint_codec.py` implements the existing inward
`CheckpointCodec[CoupledSnapshot]` port with explicit canonical JSON and existing
NumPy frame helpers. Aggregate topology/encoded/array bounds precede scans or
payload decoding; native policy/gates and every complete field are validated.
Returned arrays are detached. The outer CLI composes unchanged native models
for round-trip and continued learning. No model construction, IO, runtime
authority or adapter settings enter the codec. See ADR0272 and the codec guide;
full disk/crash and lifecycle recovery remain separate.

### Optional whole-iteration timing qualification

`app/timed_coupled_inference.py` accepts an optional iteration ceiling, validates
granularity before external work, and observes full intervals inside the existing
allocated wall window. Known records survive failure; scores are suppressed.
`app/coupled_inference_summary.py` checks explicit window/ceiling metadata before
complete-seed aggregation. The outer adapter/CLI alone choose the frozen v2
30ms/3ms protocol; default v1/native inference remains. No calibration or score
feedback enters timing admission. See ADR0271 and the timing comparison guide.

### Observed linear merge damage

`app/linear_merge_gate.py` composes the existing linear admission/scoring helpers
with the unchanged core projected merge. Owned full configurations and lineage
bind the candidate; every declared train/validation probe must satisfy inclusive
prediction, half-MSE and full-equilibrium energy limits. Rejected or failed work
returns the owned source and completed diagnostic prefix. This application
boundary grants no serving authority or global damage guarantee. The outer CLI
demonstrates both families without introducing an inward adapter dependency.
See ADR0270 and docs/linear-merge-gate.md.

### Mixed raw structural runtime

`core/coupled_structural_state.py` owns full origin/raw/CPC state and a bounded
joined tagged configuration chain. `adapters/structural_coupled_scaling_learner.py`
composes strict raw ports with full-origin binding and one private payload swap.
`app/coupled_structural_consolidation.py` uses accepted split/merge kernels and
delegates pruning through an internal temporary prune envelope, preserving the
full mixed history. The outer demo composes unchanged candidate/serving ports,
whole-envelope digests, declared selection exclusions and independent guards.
No app/adapter dependency enters core; prior strict precision bindings remain.
See ADR0269 and docs/coupled-structural-runtime.md for sequence/refusal semantics.

### Raw coupled dyadic structure

`core/coupled_scaling_structure.py` owns uniform factor2 replication, compensated
scalar precisions, exact compatible merge and detached full raw/CPC state. It
reuses the core raw pruning encoding guard; no app/adapter or IO dependency enters
core. Target configuration/caps validate before expansion, reversible effective
halves before raw encoding, and complete target/pair compatibility before merge.
The outer CLI composes existing raw learner/fork ports for continuation. Existing
runtime precision binding remains strict; mixed-history publication needs a
separate configuration-transfer contract. See ADR0268 and
docs/coupled-scaling-structure.md for exact-arithmetic identities and limits.

### Raw coupled structural transfer

`core/coupled_scaling_pruning.py` selects old effective coordinates, rebinds the
new uniform width's raw weight/bias factors, validates representation error and
owns complete chemical/coordinate state. It imports core only.
`app/coupled_scaling_pruning_gate.py` owns probe admission, independent native
scoring copies, observed limits and complete raw fallback. It reuses existing
native score/diagnostic helpers; no adapter is imported by the application.
The CLI fixes engineering fixtures and constructs new raw learners for accepted
continuation. Why this: effective-only geometry changes cannot preserve raw
configuration or width-dependent factors. Publication/rollback remains a separate
runtime boundary; strict existing raw restore remains unchanged. See ADR-0266.

### Isolated nonlinear scaling measurements

`app/nonlinear_scaling_measurement.py` composes existing owned raw ports, copied
checkpoint observers and the delayed actor/shadow stream. `app/coupled_attribution.py`
keeps all final scoring after training. The application owns role/shape/storage
validation, target-free initial and arrived-training diagnostic probes, and
failure suppression of partial final scores. `app/nonlinear_scaling_summary.py`
retains exact complete seed vectors. The adapter fixes data/factories/protocol;
the CLI owns numerical threads, encoding and command resource telemetry. Why this:
new measurements need copied observation, not a second training runtime or access
to final labels during updates. Dependencies point inward; see ADR-0265.
The adapter/CLI select either the original protocol or a separate fixed legacy-only
extension. Summary axes admit only three canonical groups; saved-data aggregation
checks each source's declared cases before union validation. No experiment settings
are inferred from prior scores and the original 96 cells are not rerun.

### Scaled matched BP

`core/scaled_matched_backprop.py` composes the existing full feedforward BCE
derivative with shared raw config/state/chain/update helpers. Precision remains
matching metadata; CPC state is refused. `adapters/scaled_matched_backprop_learner.py`
owns fixed base rate, strict raw restore and forks. The CLI fixes matching data
and checks continuation/native agreement. Why this: initialization and per-layer
optimizer factors both need matching before a scaling comparison. Algorithms and
native objective/work diagnostics stay distinct; see ADR-0264 and the guide.

### Nonlinear coupled raw coordinates

`core/coupled_pc_scaling.py` owns explicit raw config/state, effective mapping,
held-latent partials and stable functional updates using the existing objective/
inference/CPC policy. `adapters/coupled_scaling_learner.py` owns fixed training
settings, raw snapshots and forks. The CLI supplies fixed equation controls.
Why this: initialization and optimizer factors must agree, while raw state cannot
be replaced by an effective-only snapshot. Existing native/topology ports and
defaults stay separate. Bias/precision/objective choices are local extensions;
see ADR-0263 and docs/coupled-pc-scaling.md.

### Coupled topology runtime ports

`core/coupled_topology.py` owns complete geometry/native state/coordinate metadata.
`adapters/topology_coupled_learner.py` composes native ports with explicit bounded
width reconstruction and a single replacement payload. Fixed bindings/settings
remain adapter-owned. `app/coupled_prune_consolidation.py` translates an accepted
isolated gate into existing consolidation state; refusals retain diagnostic
evidence. The demo adapter/CLI uses existing actor/shadow, promotion guards and
serving commit/rollback. Why this: changed geometry needs an explicit constructor
boundary without duplicating publication or changing strict adapter defaults.
Managed retention still refuses arbitrary transforms; see ADR-0262 and
docs/coupled-topology-runtime.md.

### Native coupled structural transfer

`core/coupled_width_pruning.py` copies validated native PC/CPC state and constructs
owned one-unit prune candidates, preserving precision and adaptive association.
`app/coupled_pruning_gate.py` owns declared probes, bounded scoring, damage decisions
and retained source on failure. The CLI composes fixed engineering scenarios.
Why this: topology and wake state must be complete before selecting a replacement.
Core grants no runtime authority; shared-holder publication and nonlinear scaling
are separate integrations. Existing native/adapter restore behavior is unchanged.
See ADR-0261 and docs/coupled-width-pruning.md.

### Linear pruning gate

`core/linear_width_pruning.py` constructs an owned raw-axis candidate and coordinate
lineage using the structural numerical helpers. `app/linear_pruning_gate.py` owns
declared train/validation probes, damage limits, budget checks and complete
replacement selection. It retains the source on rejection/fault. The CLI supplies
fixed engineering examples and JSON. Why this: proposing a structure must not
authorize its publication or let final-test data choose it. No shared runtime
holder is mutated by this functional API; see ADR-0260.

### Linear width structure

`core/linear_width_structure.py` composes owned linear scaling/residual states,
axis isometries, raw-factor conversion and coordinate-lineage validation. It
imports core only and performs no IO or training. The standalone CLI composes
equilibrium diagnostics and existing SGD continuation. Why this: raw width changes
must preserve the complete effective objective, including residual identities,
without embedding runtime consent, data-role or publication authority in lineage.
See ADR-0259 and docs/linear-width-structure.md for merge/error/rate limitations.

### Allocated-time coupled inference comparison

`app/timed_coupled_inference.py` copies native PC state and declared role batches,
times whole training windows, then scores detached states. It depends on core
models and the existing application budget/metric helpers. Pure
`app/coupled_inference_summary.py` retains complete ordered seed vectors.
`adapters/coupled_inference_comparison.py` composes fixed data/resources; the script
owns thread settings and JSON. Why this: all policy/check/bookkeeping cost must
be timed together while final labels stay outside training/stopping decisions.
The application does not import its adapter. See ADR-0258.

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

`src/core/linear_pc_reference.py` is a pure scalar-output clamped linear
equilibrium and local-gradient reference. The outer script owns its fixed
synthetic inputs and JSON presentation. It depends only inward on core; there
is no learner/runtime state, dataset access or change to binary objectives.
See [the reference map](docs/linear-pc-reference.md) and ADR-0250.

`src/core/linear_pc_scaling.py` composes that pure reference with validated raw
weights, forward factors and raw-coordinate SGD. Configuration and tensor state
remain separate from effective gradients; the outer script owns fixed example
inputs and JSON. ADR-0253 records the squared effective-rate mapping and stable
product repair. This does not modify binary or legacy learner defaults.

`app/linear_scaling_measurement.py` owns copied regression roles, train-only
diagnostics and deferred target-free checkpoint scoring. Its summary module
uses the existing core seed statistics. The outer `linear_scaling_pilot.py`
owns the fixed grid/data/initialization and wall/RSS composition; the script
owns CLI/JSON. See [the measurement guide](docs/linear-scaling-measurement.md)
and ADR-0254. Dependencies point inward; no tuning enters this fixed protocol.
`LinearMeasurementPorts` now supplies copied state/effective maps/advance through
trusted native MLP and residual wrappers. `residual_scaling_pilot.py` composes
the same outer case/grid boundary and seed summaries with the residual factory.
See [residual measurements](docs/residual-scaling-measurement.md) and ADR-0256;
the public MLP API/protocol remain compatible, with focused regression coverage.

`core/residual_linear_pc.py` composes the unchanged MLP domain/draw/delta helpers
and equilibrium kernel. Raw branch state is separate from fixed middle identity
maps, so complete effective matrices enter the equilibrium while only branch
weights receive updates. The outer `run_residual_linear_pc.py` owns fixed data
and JSON. See [the residual guide](docs/residual-linear-pc.md) and ADR-0255;
residual measurement and nonlinear/CPC applicability remain separate.

`src/core/coupled_energy.py` supplies a pure binary energy/gradient kernel;
`src/core/coupled_predictive_coding.py` owns the opt-in alternating learner.
`src/core/coupled_inference.py` adds a pure guarded policy/kernel on that same
energy. Per-call native PC/CPC policies run before existing atomic publication;
the default fixed loop and native snapshot formats stay compatible. Adapter
fixed policies are independently copied through forks.
`adapters/coupled_attribution_pilot.build_learner` binds optional owned policies
for ordinary/tracked PC/CPC; None keeps old fixed controls. The small
`adapters/coupled_inference_stream.py` composes that factory with the existing
`app/coupled_attribution.AttributionStream` and shared wall/RSS budget sessions.
It adds no runner or timing framework. The existing bounded-inference CLI's
`--stream` flag selects this engineering demo; final readouts remain after wakes.

See [bounded inference](docs/coupled-bounded-inference.md) and ADR-0257 for fallback,
counter scope and the unfinished matched work/time comparison.
Both depend only on NumPy and the standard library. They neither alter legacy
learners nor import orchestration, adapters or storage. See
[the objective specification](docs/coupled-predictive-coding.md) and ADR-0247.
The separate `coupled_circadian_predictive_coding.py` shares its inference and
candidate checks, adding only the declared final-layer wake trace/gate. Its
[neutral control](docs/coupled-neutral-control.md) tests genuine unit-gate parity;
ADR-0248 records supported scope and failure semantics.
`CoupledSnapshot` and `CoupledWakeState` define complete compatible local native
state; `src/adapters/coupled_learners.py` reconstructs owned forks through those
ports and composes the existing actor/shadow runtime. ADR-0249 separates model
continuation from budgets, permissions and durable recovery.

`src/core/matched_backprop.py` consumes owned `CoupledParameters` and differentiates
the entire shared tanh/BCE feedforward graph. Its family-specific state contains
only model parameters. The outer `matched_backprop_learner.py` adapter supplies
owned forks with fixed rate. ADR-0251 and the [baseline guide](docs/matched-backprop.md)
separate likelihood/tensor matching from total PC energy and runtime work.

`app/coupled_attribution.py` validates disjoint role batches and orchestrates
delayed training before scoring detached fixed checkpoints. Its summary module
reuses pure `core/seed_statistics.py`, preserving all planned failures. The outer
`coupled_attribution_pilot.py` owns fixed synthetic data, model composition and
existing wall/RSS budget integration. ADR-0252 separates scoring from training;
the [pilot guide](docs/coupled-attribution.md) documents scientific limits.

The [native stream example](docs/native-stream-demo.md) composes the existing
NumPy learner adapters and actor/shadow application runtime from the outer
adapter layer. Its script only parses arguments and presents JSON. It adds no
core learning rule, persistence mechanism or dependency from app/core to adapters.

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
## Raw topology runtime extension

`core/coupled_scaling_topology.py` owns full raw config/parameters/wake state and
bounded joined prune history. `adapters/topology_coupled_scaling_learner.py`
composes existing strict raw ports, validating complete private replacements
before one swap and pinning every non-width field. The app raw prune consolidator
checks owned, coherent accepted evidence and returns candidate state only.
The adapter demo composes existing runtime and independent serving guards with
whole-envelope digests, disjoint selection IDs and complete-bundle rollback.
No runtime dependency enters core; defaults/managed retention remain unchanged.
See docs/coupled-raw-topology-runtime.md and ADR0267 for usage and limitations.

Opt-in declared-context coverage uses core `coverage_replay` through existing app
`StreamRehearsal`; adapter `four_phase_replay` declares matched48 controls. The core
has no IO/evaluation dependency. See [coverage replay](docs/coverage-replay.md).

Four-phase fresh-task probes compose existing stream and disposable cohort
boundaries; `adapters/four_phase_probes` describes fixed costs without IO.
One preceding probe record is retained by the opt-in caller; shared complete-
cohort validation preserves old two-cohort behavior. See [guide](docs/four-phase-fresh-task-probes.md).

Opt-in legal reward conditioning adds a pure owned support wrapper in core,
uses the existing train-only collector, and validates conditional policy likelihood
at native SGD admission. Existing reward/demo ports accept distinct LegalRewardUpdate;
old episode schemas/snapshot formats remain stable. [Details](docs/legal-reward-trajectories.md).

R4.2 legal game integration: existing reward pilot/CLI composes owned legal
trajectories through the same runtime; final packets follow all training commits.
Existing live evaluator uses the declared conditional action law; cached teacher
CE remains raw-policy diagnostic. [Boundaries and usage](docs/legal-reward-trajectories.md).

R4.2 paired legal recipe reuses the existing deferred-arm closures and runtime;
both live policies receive the same support. Shared final data follow24commits,
with at most2completed-arm and2final-result guards; no new runner/math module.
[Usage/evidence](docs/legal-reward-trajectories.md).

The MiniGrid learnability adapter composes existing guarded `_train_arm` owned
checkpoints, raw categorical scoring and `run_fresh_task_probe`. The existing CLI
owns scenario selection; inner app/core ports keep update/role/metric semantics.
This fixed diagnostic recipe has no final episode, replay or promotion authority.
Its costs and raw-fit scope are explicit. [Rationale](docs/adr/ADR-0349-compose-minigrid-checkpoint-learnability.md).

R2.1 full-width probes use the existing app/fresh_task_probe envelope, exact148
feature geometry and original aggregate bound. Core declares separate cue-specific
E layouts; infra admits them only with matching cue and reserved roles before
native creation. App owns disposable-fork training and test-only metrics. Context
metadata adds no policy feature; each complete episode remains a separate native
call. No new runner or dependency. [Usage](docs/categorical-fresh-task-probes.md#full-minigrid-observations).


## Explicit observed-episode momentum control

`TorchDemonstrationRewardLearner` dispatches the distinct `MomentumOnlyUpdate`
to `TorchRecurrentLearner.train_zero_loss_episode`, backed by
`src/adapters/torch_recurrent_zero_loss.py`. The adapter copies and validates
original TRAIN evidence; only observations enter the dense-zero objective.
Existing native staged restore/SGD/capture validates full state before atomic
publication. Core policy-episode validation remains inward; no core module
imports this adapter. The caller owns release, budgets and exposure accounting.
[Usage, current qualification and open paired runtime](docs/minigrid-momentum-control.md),
[ADR-0361](docs/adr/ADR-0361-explicit-original-episode-zero-loss-control.md).


`src/adapters/minigrid_momentum_control.py` composes the accepted zero-loss port
through the existing ActorShadowRuntime, with independent warm forks, delayed
matching IDs and bounded8slot original-observation exposure/receipt accounting.
The existing reward CLI runs it only after reward budgets freeze, then passes
owned endpoints to the existing TRAIN evaluator with a distinct control method.
Default routes remain unchanged. Native update composition is accepted; the
three-readout native case is incomplete at its diagnostic cap. See the linked
momentum-control guide for exact qualification and the remaining comparison gate.


The explicit `--extended-control-readouts` selects a separately versioned paired
engineering allocation: actual whole300/parent330/readout210 seconds, retaining
original45 training phases and all data/receipt/state guards. V1/default caps
remain unchanged. The native v2 case completes all three owned TRAIN readouts;
its mixed negative comparison and paid diagnostic costs are documented in the
momentum-control report. No final/probe authority or new learner objective added.


### Explicit neural representation comparison

The existing history-smoke recipe composes original TRAIN streams and the frozen
MiniGrid evaluator through separate raw/bound neural admissions. The CLI alone
reads saved source evidence through the existing bounded loader; native learner
ports remain IO-free. Full paired initial states and original source/actor/
training charges are guarded before target-free readout. Sixteen fixed private
paid captures bound storage; invalid receipts stop further work and publish
partial evidence. Native objectives and default routes remain unchanged.
See docs/minigrid-invariant-encoding.md#matched-neural-a32-comparison and ADR0387.


### Bound learnability diagnostics

Existing minigrid_learnability composes explicit bound neural factories, owned
TRAIN-fit snapshots and the existing disposable probe use case. Adapters validate
encoding/native policy/role/checkpoint contracts; the inner probe owns independent
updates/budgets. Parent orchestrator checks all completed arms/probes in one flat
loop; completed bound guards only validate local source/data/paid-report state.
Why: invoking parent from published completed guards recurses. Original raw and
recurrent interfaces remain unchanged. CLI selection is explicit; no new runner
or persistence codec. See docs/minigrid-learnability.md and ADR0387.


Canonical BP A-only acquisition adds an explicit recipe to the existing
minigrid_learnability adapter/CLI. minigrid_learning_pilot._packets validates either
the original phase tuple or four A phases before creating owned episode copies.
AttributionStream, complete native state, paid readback proofs and disposable E
probe use-cases stay shared. A-only publication names A fit and untrained B/C
fits; it cannot publish the original continual forgetting/return summaries.


The distinct public F task uses existing MiniGrid layout/episode/teacher ports.
minigrid_probe_support is an adapter-level pure cohort description over core
history/encoding and app batch validation, with no infra/model/update authority.
The existing learnability route checks actual exposed TRAIN support before probe
updates and publishes one bounded protected diagnostic; failed prior-proof checks
retain its measured cost separately. No orchestration or persistence layer added.

Optional old TRAIN readouts compose inside app.fresh_task_probe on the same owned
disposable learner after its original F checkpoints. app.probe_training_readouts
validates the combined bounded array envelope and records separate partial score
work; it has no update, selection, persistence or final-data authority. The
learnability adapter supplies two original A TRAIN packets, protects completed
reports and publishes per-cue fit/gains plus aggregate costs. The support gate
continues to describe only the original eighteen F packets.


Fixed A return composes inside the existing fresh-task use-case: two bounded
owned TRAIN templates reuse its _train/_score loop on the same disposable learner.
app.probe_return_phase supplies whole-envelope admission and protection of the
preceding bounded paid report, not another learner runner or checkpoint system.
Separate return budgets/progress never renew original F charges. The adapter
minigrid_return_fit supplies original A template copies, frozen descriptor/report
validation and aggregate costs; minigrid_learnability retains source/actor/previous
result guards and public dispatch. Readonly monitor packets have no update role.

## Disposable probe endpoint readouts

`app.fresh_task_probe -> app.probe_endpoint_observer` protects completed report
containers and original budgets before an optional endpoint callback. The app
imports no evaluator or environment. `adapters.minigrid_probe_rollout` binds the
actual count learner to `adapters.minigrid_policy_evaluation`, using separate
zero-update budgets and private typed state/report captures. The evaluator alone
uses existing MiniGrid infra. Only fixed after-F16/after-A16 A/F readouts are
admitted; no generic checkpoint export or final-data release is supplied. Partial
work and unknown costs remain visible; the original native case is incomplete
at its20s readout deadline, so full live validation remains open.


### Explicit weighted MiniGrid probe recipes

`minigrid_bounded_probe` admits fixed/bounded weighted recipe configuration, binds
owned reserved sources and reduces observed scalar work. Existing MiniGrid pilot
orchestrates; existing fresh-task app owns fork-only F/readout/A-return execution.
Both depend on the existing domain DTOs and atomic native adapter; no core-to-
adapter dependency or new runtime/budget authority. See ADR-0421 and the bounded
inference guide for recipe and scientific scope.

### Prospective paired wall cap

The existing quota-comparison adapter validates its optional wall allowance
before runtime construction, captures one local deadline for both arms and their
deferred final readouts, and checks completion against that original cap. The CLI
accepts the option only on that scenario; defaults and native session ownership
remain unchanged. A descriptor is reporting metadata, never budget authority.

## PC corrective composition (2026-10-20)

Existing robot collection → continuous_correction_demo → RecordedContinuousPC
→ ActorShadowRuntime → frozen17/19 rollout. Exact native PC/neutral/active owners
retain full momentum/wake/config state; serving is feedforward and frozen.
Bounded eight-return native evidence is captured before outer receipt checks;
original budget/receipt identities and actual native state validate before more
work. Costs use native events, not BP dense formulas. No new inner-layer rule,
runner, dependency or promotion. [API](docs/continuous-corrective-adaptation.md).

## Geometry-readiness teaching (2026-10-20)

core/pick_cube_readiness_teacher owns bounded scalar phase/closing decisions from
exact copied privileged geometry. infra/maniskill_task_teacher reuses its one
collector loop and existing episode port, with a separate TRAIN17/19 entry point;
the existing CLI explicitly selects it. Returned decision carries its actual
command stage. New definition/episode IDs and native42/baseRGB789 qualification
prevent silently using old fixed-schedule evidence. Geometry stays teacher-side;
no learner receives privileged coordinates through these public observations.
Original fixed teacher/signatures/default JSON remain. No new runner, dependency,
configuration or environment variables. [API and limits](docs/pick-cube-readiness-teacher.md).

## Distinct readiness corrections (2026-10-20)

ReadinessCorrectionLabel/Batch → existing raw/public21 view → existing correction
adapter/runtime → private candidate/frozen17/19. Existing query reads teacher-side
geometry and previous executed student gripper; no privileged geometry in public
features and no student-action substitution. Exact types/provenance/arrival gate
separate geometric labels from original elapsed schedule. Query pins mode/teacher/
port/state/paidlabels, validates nativeTRAIN authority after public observers,
and retains poisoned bounded evidence. Core optionalfeedbackNone preserves
selfexecuting teacher. No new runner, kernel, objective or dependency.
[Usage and costs](docs/readiness-corrections.md).

## Qualified readiness inputs (2026-10-20)

Existing stream CLI bounded decoder → TaskTeacherResult readiness qualification →
owned TRAIN17/19 episodes → existing learnability/runtime → existing frozen robot
composition. Explicit mode preserves definition/IDs and retained acquisition work;
old reader/defaults refuse readiness declarations. No new domain rule, orchestration
framework, optimizer or dependency. [Usage](docs/readiness-demonstration-learning.md).

## Existing A-only control at the robot boundary (2026-10-20)

The robot script passes explicit A-only mode through the existing continuous
runner and validates actual warm-only receipt history before frozen native serving.
A-only changes optimizer exposure while B diagnostics retain their separate cost.
Default call/report schemas and original full-owner/native guard helpers remain;
no inner-layer or optimizer change. [Usage](docs/continuous-robot-a-only.md).

## Taskspace memory inspector

`src/app/taskspace_memory_inspector_projection.py` validates bounded public CPU/
native memory reports and returns detached source cells. It imports only pure app
presentation helpers. The existing dispatcher routes before the legacy status
field check because memory reports expose a boolean `complete` instead. Infra
selects the shared coherent-stream table template; its memory branch hides
inapplicable comparison controls and curves. No inference, evaluation, learning
or IO occurs in the app projection. [Usage](docs/taskspace-memory-inspector.md).

The existing native memory runner also accepts bounded frozen TRAIN scene
descriptors for development resets absent from its admitted bank. Outer script
admission validates reset authority and snapshots primitive descriptors before
constructing a learner; the existing stream owns learning and detached callbacks.
Native ports retain their original TRAIN authority. Development classification
and paid returns are sealed separately from the legacy four-scene report.
[Usage and scientific scope](docs/memory-development-contexts.md).

The existing continuous replay CLI selects explicit/default/bundled control JSON
at its IO boundary. Preflight delegates original saved-control validation and
returns before runtime learning; normal execution retains its producer predicate.
No new experiment runner or model-state reconstruction is introduced.


## Disposable CLM probes

`core.clm_probe` declares reserved paired/readout roles without importing a backend.
`adapters.clm_head_probe` owns warm projection weights with a fresh, finite native
optimizer/schedule and a separate readout allowance. A protected validator helper
in the existing CLM learner takes an exact record kind; normal TRAIN paths bind
the old types, while the private probe engine binds the new types directly.
No role conversion, duplicated loss, source budget reset or public probe snapshot
is involved. Inputs are bound CPU float32 features; outputs are detached choice
probabilities, native diagnostics and permanent work counters. The probe neither
scores labels nor loads an encoder, selects data or installs a learner.

Dependency: role declarations → existing CLM feature declarations; probe adapter →
role declarations + existing native CLM learner. Future app composition belongs
in the existing temporal runner. A warm-weight/fresh-optimizer probe is a distinct
protocol from source continuation. [Rationale](docs/adr/ADR-0540-owned-clm-probe-roles.md)
and [usage/verification status](docs/clm-owned-probe.md).

## Catalog learned inference binding

`adapters.clm_catalog_decision` admits an exact trained catalog learner and its
matching frozen encoder, takes and validates a fresh current native snapshot,
then delegates head copying to the existing typed `CLMDecisionPredictor`.
Inputs are current owners and bounded caller metadata; output is the existing
frozen predictor plus a copied historical work receipt. One binding attempt is
permanent, and native references are released after it. No inner layer imports
the adapter. It performs no training, evaluation, promotion, file IO or optimizer
construction. [Contract, rationale and commands](docs/clm-catalog-learning.md#bind-a-trained-catalog-learner-for-typed-prediction).

The existing CLM temporal script can opt into a post-TRAIN typed development
readout using that same adapter. It declares encoder work before loading, keeps
the trained source live for binding, and admits the additional snapshot only
with a complete canonical typed work record. No app/core dependency changes.
[CLI and accounting](docs/clm-catalog-learning.md#actual-typed-choices-after-a-stream).


The optional CLM catalog-probe rehearsal path uses the existing layers:
`core/clm_catalog_probe.CatalogProbeRehearsal` declares role-bound history and six
packets; `adapters/clm_checkpoint_probes` concatenates bounded native features and
binds checkpoint availability; `app/clm_reserved_probe` guards consent, source,
original update and replay budgets, paid attempts and receipts. The temporal app
creates replay-consented TRAIN experiences only in explicit new executions. The
CLI remains the existing temporal runner. No inner module loads a backend, scores
probe labels to select work, or exports a probe to the benchmark learner.


The extra-marker catalog probe control uses the same existing boundaries as
rehearsal. `CatalogProbeMarkerControl` binds six probe_train packets; the outer
checkpoint adapter materializes original markers twice, and the app validates
exact roles, owners, features, budgets and paid extra-TRAIN counters. It shares no
probe weights with main/deployed learners and never reclassifies markers as replay.
The existing CLI exposes mutually exclusive control/rehearsal modes; equal dose
is distinct from equal information or retained storage.

The catalog reserved probe can optionally observe the exact arrived TRAIN triple through a prediction-only role. The checkpoint adapter binds the original history; the app guards owners and work, and cached scoring adds a separate `arrived_train` group after all work. The original suffix evaluation, learner updates, encoder and default orchestration remain unchanged. This separates training-point retention from template transfer without granting replay or evaluation access.

The disposable catalog probe exposes explicit all/state_only parameter scope. Protected optimizer-selection and staged-validation hooks in the existing native engine preserve all-parameter defaults; the private state-only subclass pins fixed action/temperature values and admits only state optimizer owners. App/hook/CLI bind scope/count metadata before cached scoring. See [ADR-0546](docs/adr/ADR-0546-explicit-state-only-catalog-probes.md).

The separate `CLMStateOnlyCatalogLearner` adapter exposes the same constraint for
ordinary TRAIN records through the existing temporal app. Its exact snapshot
family and fixed-birth restore guard preserve objective/scope admission without
migrating all-head records. Reserved probes still own independent optimizers and
data roles; the runner declares TRAIN and probe scopes separately. Typed frozen
binding remains limited to its original family. See [ADR-0547](docs/adr/ADR-0547-state-only-train-catalog-scope.md).

## Paired CLM TRAIN replay candidate

`adapters.clm_replay_candidate` wraps an exact paired TRAIN source with separate
warm heads, fresh native optimizer/schedule and bounded source-state protection.
`app.clm_replay_candidate` admits causal consented original records and composes
the existing inbox with separate update/replay sessions. It imports no backend.
The temporal CLI opts in through the existing native observer, retaining two
actual current-policy origins and running a finite child only after source
completion. Default behavior, source authority and private probe roles remain
separate. [Contract and costs](docs/clm-paired-replay.md), ADR-0552.

## CLM workflow TRAIN-to-effects composition

`app.clm_workflow_training` owns fixed delayed catalog TRAIN batches using core
records and the existing inbox/budget. It imports no backend. The outer
`scripts.run_clm_ticket_workflow` binds raw features/native heads and separate
unchanged/learned typed predictors, then uses the existing ticket workflow app
and effect-time dispatcher after training. No new handler/runner framework or
external authority. Preserve one full trained-source snapshot and bounded scalar
prior episodes with separately charged binding/serving work.
[Contract and commands](docs/clm-ticket-workflow.md); ADR-0553.

Optional `core.clm_action_features.FrozenActionFeatures` declares bounded raw
inference rows without a backend dependency. Concrete predictor/binding adapters
validate and own immutable copies; the workflow can reuse its five TRAIN-extracted
public action embeddings. Exact catalog/encoder lifetime, fail-stop invalidation,
separate copy/read work and fresh head projections are described in ADR-0554.

The outer `scripts.recurring_ticket_workflow_example` composes the same existing
ports with fixed chronological TRAIN phases and complete task readouts between
phases. It reads the stream's candidate snapshot, preserves seed/budget ownership,
and adds no core learning rule/controller/promotion authority. See ADR-0555.

`scripts.ticket_workflow_rehearsal` compiles authentic fixed teacher episodes for
the existing core codec and StreamRehearsal. The stream's additive detached
publication receipt getter binds observed native counter additions; wake/replay
budgets remain separate. No new replay framework or algorithm. See ADR-0556.

## Optional installed game environment

The `minigrid` packaging extra mirrors the existing five optional pins. Base
installation remains NumPy-only. `infra.minigrid_memory` retains lazy imports and
direct version checks; its failure message supports both local-wheel and checkout
setup. The existing coupled-attribution CLI describes the fixed FIFO36 protocol
without optional imports or model/environment work. Actual execution uses the
existing collector, stream, learner and game ports; no second runner or bundled
TRAIN archive is introduced. See [installation commands](docs/minigrid-cpu-install.md)
and [ADR-0610](docs/adr/ADR-0610-declare-optional-game-installation.md).


## Categorical arrived shift observations

`core.shift_signals` reuses categorical validators for bounded simplex/hard-label
summaries; it has no data-role or clock authority. The existing
`app.arrived_shift_observer` binds an opt-in class count to ordinary TRAIN batches
and copied preceding states. The delayed stream owns arrival authority; diagnostic
results do not feed model/controller changes. Default binary reports and original
detector/calibrator/incident laws remain. Actual delegates own model/copy costs.
[Contract and scope](docs/categorical-shift-observer.md), ADR-0612.


## Optional observations in guarded context serving

The existing outer context_head_serving composition binds ArrivedShiftObserver
to the guarded stream only and supplies the pure owned-context prediction port.
Relative observer clocks map to the captured source native-update origin.
Diagnostics are detached in finally and never feed training/promotion decisions.
Default call kwargs/reports, guards/rollback and lifetime budgets are unchanged.
[Contract/costs](docs/context-shift-serving.md), ADR-0613.

The outer `scripts.context_shift_study` composition uses the same stream and
observer on three fixed private TRAIN conditions. A phase barrier completes all
training before known-task readouts from bounded selected state copies. It adds
no inner-layer dependency, scheduler, controller or final/probe data access.
[Fixed protocol and observed limits](docs/context-shift-study.md).

`scripts.arrived_candidate_shift` separately observes each newly arrived TRAIN
packet before reference wake selection, using a detached current candidate.
Its bounded reader owns causal cursor/state/report checks; pure core
ArrivedShiftFeedback carries scalar alarms to the existing drift choice.
Original FIFO/gate/credit admission retains authority. Defaults and after-update
observer semantics are unchanged. See ADR-0614 and
[executed costs/limitations](docs/arrived-candidate-shift.md).

The fixed `scripts.context_fifo_reference_study` uses existing live execution,
declared-length receipt accounting, paid scalar-report guards and core utility.
It reuses accepted shift evidence and admits scoring only after all missing
controls complete. It introduces no model/runtime/controller dependency.
[Reference outcomes and unequal costs](docs/context-fifo-reference-study.md).

The outer `scripts.context_reference_validation` binds policy plus feedback/ack
flags before IO and reuses existing ControllerChronology roles. Known102 metadata
is training; one new stream is validation; reserved sealed source/callback absent.
Core freeze follows validation and remains unused. See ADR-0615 and
[stress results/isolation](docs/context-reference-validation.md).

The existing minigrid_probe_support adapter also exposes contextual TRAIN bounds
through the existing diagnostic CLI. It composes admitted raw/canonical bounds,
public-history current/action checks and per-cue296feature groups. No learner,
model snapshot or game is constructed. [Contract](docs/context-game-train-diagnosis.md).

The explicit context-linear mode composes the existing disposable TRAIN fitter
once per initial public cue after complete source admission. Paid head reports
are checked before further work; aggregate scores restore source order only
after both complete. No deployed snapshot/optimizer/clock authority is created.
See ADR-0616 and [executed fit](docs/context-linear-train-fit.md).

The separate minigrid_context_linear_learner owns untrained/arrived-fit state,
bounded failure work and detached forks. It composes fixed core least squares
with the original AttributionStream via the existing CLI. App roles, receipts
and lifetime budgets remain authoritative; no replay/history bank is added.
See ADR-0618 and [causal TRAIN observation](docs/context-linear-stream.md).

The affine reserved-task adapter reuses ORIGINAL Fcollector/history encoder and
run_fresh_task_probe with a single owned fork of the saved after-C model. F and
two27row cue-grouped A/B/C readonly monitors use independent-row transport;
source episodes/spans and existing row diagnostics preserve task interpretation.
The original two-monitor/65536entry limit remains. No benchmark/deployed updates,
return, promotion, new runner or math. Source/native-state equality and the
explicit0→1fork event are separate; original budgets/clocks/native charges and
bounded collection prefixes remain guarded. See ADR-0624 and
[usage/qualification](docs/affine-fresh-task-probe.md).

The after-C native mode binds only qualified111capacity2state6/108/two27-row
heads to the ORIGINAL game loop for B/C_train. The existing saved-state adapter
shares plain dataclass conversion with older modes; no new schema or runner.
Original source0/owned scorer/full guards remain. The native command opens only
111JSON; paid109+111histories10fits180rows, selected6/108 and new0readoutfits
are distinct. See ADR-0623 and [usage/evidence](docs/after-c-linear-games.md).

The optional two-packet public-cue FIFO capacity extends the same arrived
catalogue composer; default1 remains. Complete retained-ID/arrival tuples and
source spans account for joint replay, finite banks and detached inputs.
`minigrid_catalogue_capacity` binds two C_train-only study forks of the saved
after-B model to the ORIGINAL delayed AttributionStream. The CLI owns qualified
file access; this adapter owns finite study order/diagnostics, not native math,
games or live authority. Both arms finish before any fixed TRAIN scores.
Shared source6fit90row costs, selected4/54 prefix and new4fit90row work are
distinct; new local clocks do not claim resumed authority. See ADR-0622 and
[usage/qualification](docs/catalogue-capacity.md).

The retained after-B game mode adds an exact state4/54row/two18-row-head binding
to the existing affine evaluator and saved-source CLI. It reuses the original
game loop, full copier, public-history features and source/budget guards;
original defaults remain. One owned scorer plays four known A/B games with no
refit, while the untrained source remains unchanged. Full source6fit90row sunk
costs, selected4fit54row prefix and new readout work are distinct. All four
games succeeded; no isolated retention comparison, reliability, generalization,
CPC or release claim follows. See ADR-0621 and
[usage/evidence](docs/retained-context-linear-games.md).

The optional arrived-cue catalogue recipe reuses existing independent-row
composition and AttributionStream. It combines the current packet with one
earlier, label-arrived TRAIN episode selected by public cue. Sources/spans and
unique exposure IDs remain explicit; the native affine owner is unchanged.
The application owns its finite private plan; bank/materialized bytes and extra
joint-wake replay exposures are reported separately from consolidation events.
See ADR-0620 and [retention evidence](docs/arrived-cue-catalogue.md).

A narrow exact causal-state binding in the original MiniGrid evaluator restores
genuine2/4/6-fit states with their full counters. Existing CLI mode owns one
readout-only lifetime budget, checks preceding phase reports and preserves normal
failure prefixes. A140second work deadline reserves cleanup within150seconds;
hard process loss is not durable capture. First native attempt failed at the
outer cap with complete outcomes unavailable; see ADR-0619 and
[evidence/limitations](docs/context-linear-causal-games.md).

The readonly minigrid_context_linear_policy adapter owns detached fitted-model
wrappers/bindings with immutable coefficient leaves. A narrow evaluator wrapper
reuses the original game loop with a zero-update readout budget. The existing
CLI binds the accepted fit bytes before inference; it creates no native
optimizer/history authority. See ADR-0617 and [game evidence](docs/context-linear-games.md).
