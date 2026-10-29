# A small trained adaptation chooser

The existing native serving CLI can fit a bounded contextual chooser and apply
its frozen choices to fresh local learner owners:

```console
python -m scripts.run_bp_ticket_serving --source artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json --learned-controller --output learned.json
```

This requires the existing qualified96-update/576-row native BP source and native
dependencies; see [native serving](native-ticket-serving.md). Output must be
fresh. `--learned-controller`, `--outer-study` and `--reference` are exclusive;
omitting them preserves the original serving path. The learned report limit is
4MiB; original/default/reference1MiB and outer-study2MiB limits are unchanged.
The saved-ticket HTML inspector does not support this report schema.

## Opt-in replay/wake catalog API

The default CLI and its two-arm/four-trial recipe remain unchanged. A separate
explicit API now lets the same `CounterfactualController` owner fit a bounded
three-action catalog. The existing private ticket driver and learning recipe
also accept it through explicit API arguments. The native study and its negative
results are recorded below. Saved catalog comparisons have an explicit CLI opt-in;
fitting a new catalog remains an API operation.

```python
from src.core.adaptation_action import AdaptationAction, AdaptationEnvelope
from src.core.catalog_contextual_controller import ContextualArmCatalog
from src.app.counterfactual_controller import CounterfactualController

envelope = AdaptationEnvelope(
    (AdaptationAction("wait"), AdaptationAction("wake1", 1),
     AdaptationAction("replay_wake", 1, 3)),
    max_decisions=6, max_wake_updates=4, max_replay_examples=6,
)
catalog = ContextualArmCatalog(envelope)
trainer = CounterfactualController(
    source_snapshot, complete_state_digest, objective, controller_clock,
    catalog=catalog,
)
```

Use a real versioned source snapshot and its complete native-state digest.
Six disposable controller-TRAIN trials cover each original context (`clean`,
`mistakes`) and each catalog action once. The callback receives an exact
`CatalogCounterfactualRequest` and detached base state. It returns
`CatalogCounterfactualReturn` with original causal label/return clocks,
`ControllerMeasures` and `CatalogTrialWork`. Native updates include replay and
wakes; attempted/returned rows include their full training exposures. Replay
charges are reported separately as a subset. Prediction calls/rows include
diagnostics/readouts. Time/RSS/violation costs retain the original objective;
detailed gradients/latent/FLOP/initialization/allocation accounting remains the
execution driver's responsibility. Unknown work cannot receive utility.

Freeze only after all six returns. The table maximizes the unchanged utility;
ties prefer wait, then declared order (wake1 before replay/wake). Negative
utilities and selecting wait are valid. The owner pins source/version, catalog,
envelope and all prior trial values/identities before admitting another callback;
known partial failed costs remain inspectable and stop further trials. It keeps
one source state and at most six scalar trial rows, without a model-state chain.

Given an existing shared owner configured with this exact envelope and its
original `CategoricalReplayAction`, choose and execute through the existing ports:

```python
trainer.freeze()
fit_view = trainer.application_snapshot()
# Verify fit_view's base/current source against the actual learner, as in v1.
view = shared.adaptation_snapshot()
remaining = min(
    replay_budget.remaining_replay_examples,
    envelope.max_replay_examples - view.progress.replay_examples_charged,
)
decision = trainer.choose(
    arrived_observation, envelope=view.envelope,
    remaining_replay_examples=remaining,
)
poll = shared.train_action(decision.action_id)
```

The chooser returns only an ID; the original shared runtime owns causal labels,
actual admissions, budgets and native state. Insufficient replay credit returns
wait rather than silently changing dose. A completed replay followed by a deferred
wake keeps its real charge/receipt under the [composite contract](categorical-replay-actions.md#replay-followed-by-fresh-learning).
No final/probe source, permission, budget renewal, callback or target can be
supplied as a selected action.

[Qualification report and commands](reports/catalog-contextual-controller-20261027.md)
separate literal utility fixtures and scripted learner calls from native task
adaptation. No new comparative/native study, retention/recovery, efficiency,
generalization or circadian benefit is established by this API work.

### Ticket trial and private application APIs

`scripts.run_bp_ticket_serving.run_case` accepts mutually exclusive
`catalog_trial=<CatalogCounterfactualRequest>` (with its detached `base_snapshot`)
or `catalog_policy=<FittedCatalogController>`. They use the original qualified
96-update/576-row BP source, eight TRAIN batches and nine causal opportunities.
The matching catalog envelope must declare that horizon/wake count, a three-row
replay choice and a finite multiple of three replay exposures, at most24.

The new path compiles the original twelve complete TRAIN teacher trajectories
and retains the first three-row invoice episode. It uses a separate original
replay budget, original wake budget and separately bounded shared admissions.
Native clock/history includes both replay and wake updates; wake receipts,
replay receipts/rows and pending fresh packets remain distinct. Insufficient
replay credit waits without forcing a drain. Trials with live service, and all
policy-change/in-loop publication combinations, refuse before source IO/models. Frozen applications additionally
support the explicit deferred TRAIN stream described below; trials stay separate.
The existing paid replay-success guard validates candidate, source, actor and
receipts before loop predictions and private task readouts, as well as before
another native admission. Corrupted or replaced evidence stops the case.

The default catalog application is private: it retains the existing four known development
task readouts and has no publication, rollback, sealed evaluation or probe work.
`scripts.ticket_controller_learning.run_study(source_path, catalog=catalog)`
uses the same owner for six complete trials followed by two frozen private
applications, with unchanged utility and all discarded costs. Source, teacher
preparation, monitoring, replay, initialization, task readouts and evidence-guard
overhead must all be included in a native study's declared resources. The current
CLI and default four-trial study remain unchanged.

The study checks all preceding paid case values and container identities before
admitting another case, using the existing bounded evidence guard. Unknown
native return/publication or prediction work prevents utility/selection.
[Driver qualification and commands](reports/ticket-catalog-control-20261027.md)
record the scripted checks and their limits. The
[sole native catalog study](reports/ticket-catalog-native-20261027.md) completed:
wake-only acquired legacy and retained both old tasks, while the fixed invoice
replay choice left legacy unsolved despite twice the updates. The frozen chooser
selected wait for clean feedback and wake-only for mistakes. All six training
utilities were negative; no replay/controller advantage follows. These
private fresh-source applications do not demonstrate sequential loss/relearning
or broader deployed recovery.

### Learn privately while the original actor serves

A frozen catalog policy can use the existing live-service API with its original
nine-opportunity plan:

```python
from scripts.run_bp_ticket_serving import run_case
from scripts.ticket_serving_timeline import LiveServingPlan

result = run_case(
    source_path, catalog_policy=frozen_catalog_policy,
    serving_plan=LiveServingPlan(opportunities=9, contention_ticks=(2, 6)),
)
```

The policy must bind the same qualified96/576 source and original eight-packet
envelope. This mode uses the original six-row new-context TRAIN batches and
delay1. Trials, longer/shorter live horizons, custom/deferred streams, live TRAIN
recipes, other controllers, in-loop publication and followup are refused before IO.
The separate terminal publication option below is explicit; it does not change
the default private path or permit publication inside the live TRAIN loop.
This is an API opt-in; `--catalog-fit` remains confined to saved comparisons.

The original actor serves throughout; the candidate stays private. Actual live
workers reserve the same sharing gate at contention ticks, so denied admissions
consume no native/replay work. On other native admissions, service starts while
the admission is active. Replay may finish while that service lease is still
held: the following wake can defer, preserving replay charges and pending fresh
work. There is no automatic wake retry, forced drain or renewed decision cap.
The actual returned receipts, rather than a promised number of updates, define
completed work. Old private/reference modes keep their original behavior.

`live_catalog_bp_ticket_private_v1` reports the existing bounded live requests,
versions, latency, native/replay history and private candidate task readout.
Completion requires a closed, complete, known service capture. Source, original
actor publication, frozen policy and plan are checked before more work; worker
checks never mutate the main budget clock. Changed evidence and unknown work
stop the run with paid prefixes retained. Service outcomes never become labels,
controller feedback, promotion guards or final/probe inputs. `_case_measures`
refuses this report: live-service outcomes/violations require separate reporting,
and the old comparison utility is not silently extended.

[Scoped live integration checks](reports/catalog-live-serving-20261027.md)
exercise real workers/gates with scripted learner and public prediction ports.
They establish no native live-catalog qualification, deployed adaptation,
long-duration reliability or scientific advantage. Freeze a distinct small native
protocol before any new experiment; the failed/spent #248 comparison remains
unexecuted in this scope and cannot be retried through this API.

The [subsequent native live qualification](reports/catalog-live-native-20261027.md)
completed6updates36rows: private new-task success1/2→2/2, old2/2→2/2. All9live
episodes completed, with6successes and29requests; the original96state actor still
failed both legacy requests, and one intended parcel commit was denied. Two
packets remain unapplied. No publication or chooser/CPC advantage follows;
the report preserves all fit, native, service, teacher and monitoring costs.

### Mixed or corrupted feedback during live service

An explicit `LiveTrainingRecipe` can also be used with the frozen catalog's
private live application. It reuses the existing TRAIN builder, shared runtime
and original nine-opportunity envelope:

```python
from scripts.ticket_controller_streams import LIVE_TRAINING_RECIPES

result = run_case(
    source_path,
    catalog_policy=frozen,
    serving_plan=LiveServingPlan(9, (2, 6)),
    live_training_recipe=LIVE_TRAINING_RECIPES[1],
)
```

The existing recipes have eight mixed twelve-row packets and delay2. Tick8 ends
the window; the last label is due at9 and remains unarrived. `stream` separates
`unarrived_labels`, `arrived_unapplied` and total `pending_unapplied`. A complete
service window does not mean all TRAIN work was consumed. No implicit drain,
shorter delay, extended horizon or renewed budget is permitted. Delay1 recipes
retain their declared chronology. Source/recipe/packet and success-receipt
identities and exact values are checked before more work; worker checks do not
mutate the main budget clock.

`training_corruption` identifies the experimental row1 or row7 label change;
those targets are not described as verified teacher truth. Original frozen
source/policy, separate replay charges, original actor publication and caller RNG
remain bound. Service outcomes never feed learning or chooser selection. Trial,
deferred, overlapping stream, drain, followup and terminal-publication modes are
refused for this combination before source IO. Keep the ordinary clean
publication API separate until these stricter prerequisites are qualified.

[Software checks and costs](reports/catalog-live-recipes-20261027.md) cover real
builders/workers/admission with literal learners. A subsequent
[native clean/retained-corruption observation](reports/catalog-live-noise-native-20261027.md)
completed one frozen pair.
Both arms used5updates60rows and retained both old tasks; clean still failed
legacy, while the corrupted-feedback candidate succeeded on all4private tasks.
Both serving actors remained unchanged. This unexpected endpoint difference is
not evidence that wrong supervision is beneficial or safe; no controller/CPC or
general robustness claim follows. That allowance is spent. No experiment budget
or noisy-comparison retry is granted; all earlier negative/unknown results remain.

### Publish after the live window ends

Use the existing inner guards and promotion controller with an explicit API flag:

```python
result = run_case(
    source_path, catalog_policy=frozen_catalog_policy,
    serving_plan=LiveServingPlan(9, (2, 6)), publish_after_window=True,
)
```

This option requires the same frozen catalog live application and a known,
complete, closed serving window. It pauses the original sharing gate, preserving
limits, admissions, deferrals, clocks and all wake/replay charges, then evaluates
the original independent `inner_guard` batches. It uses the original
`ServingPromotionController.prepare/commit`; service success and private task
readouts are not promotion inputs. Guard rejection is valid and is not retried.

All original experiences, labels and receipts remain in the old ledger. Pending
old-version packets are reported explicitly and remain unapplied; there is no
extra TRAIN decision, new owner, budget renewal or label rebinding. The old
chooser and runtime retire whether publication is accepted, rejected or unknown.
Runtime retirement precedes inbox-ledger retirement, so a ledger-cleanup failure
still stops old training. Such a case is incomplete, keeps the gate paused and
preserves the known commit separately; it never retries or performs post-readout.

A known commit must match the exact original controller receipt, its preserved
values and the full new actor publication. Known rejection must leave the full
actor unchanged. Four ordinary typed task episodes then read that exact actor
with checks before and after predictions. There is no automatic rollback in this
opt-in; original default/drain/followup behavior stays unchanged. Lost, forged,
changed or replaced receipts/evidence stop readout without inventing success.

`live_catalog_bp_ticket_publication_v1` keeps the training-window
`sharing_snapshot` and records `final_sharing_snapshot` separately. Its
`window_publication` records the observed pause, pending IDs/versions, commit
status and retirement. `catalog_actor_unchanged_before_publication` preserves
the preceding source proof. Unknown publication has a null promotion count;
a verified commit followed by cleanup failure retains its known count with
overall completion false. Post-window episodes have role `serving_after_window`.
The old `_case_measures` and `case_work` refuse this format, including guard
rejection: report guard, service and all native costs separately, without
silently dropping guard computation or claiming all-in utility.

[Software qualification](reports/catalog-window-publication-20261027.md) and
[decision rationale](adr/ADR-0641-retire-catalog-window-before-serving-published-policy.md)
use the actual controller/evaluator with literal learners/public ports.
[Native qualification](reports/catalog-window-native-20261027.md) executed one
frozen clean live window: six updates/36rows, original guards accept, and all
four ordinary post-window tasks succeed using generation1/c0/native102. The two
old-version packets remain unapplied and both old owner APIs refuse work. This
is known development grammar, not broad robustness or controller/CPC advantage.
That allowance,251private native success and248failed comparison remain spent;
no automatic rerun. This flag adds no CLI mode and grants no experiment budget.

### Recurring packets with deferred evaluation

Pass the existing `TicketStreamRecipe` with a frozen `catalog_policy` and
`deferred_evaluation=True` to `run_case`. This uses the original TRAIN builder,
source/digest, delayed FIFO and shared executor. Each mixed packet has12fresh
rows; replay still uses the original complete3row TRAIN episode and separate
budget. The fitted record is neither rebased nor refit. Recipe and owned packets
are checked by value and identity before admitting more work.

```python
from scripts.run_bp_ticket_serving import run_case
from scripts.ticket_controller_streams import TicketStreamRecipe

recipe = TicketStreamRecipe(
    "recurring-example", ("new_contexts", "old_return") * 4, label_delay=2
)
result = run_case(
    source_path, catalog_policy=frozen_catalog_policy,
    stream_recipe=recipe, deferred_evaluation=True,
)
```

The original fitted decision cap remains9. With8packets/delay2, the natural
horizon10 is observed only through tick8: seven labels arrive and the last label
remains unarrived. The result reports natural/observed horizons, unarrived labels,
arrived-but-unapplied packets and total pending work. There is no forced drain or
extra decision. With delay1 all eight labels arrive; shorter declared streams
can finish within the original cap. Comparisons must match actual opportunities.

`deferred_catalog_stream_v1` completion means the bounded TRAIN window completed;
pending packets may remain. Its native history derives fresh row counts from
the original FIFO receipts and adds paid replay. It contains a private checkpoint
and no task readouts, utility, promotion or final/probe access. Both `case_work`
and `_case_measures` refuse to admit this TRAIN-only result as task utility.
The existing outer comparison/readout consumer must finish and freeze all
candidates before opening readonly evaluation; final outcomes cannot refit the
policy or select training. Explicit catalog comparison support is described below.

[Scoped software evidence](reports/catalog-ticket-stream-20261027.md) covers
scripted ports and isolation. No native mixed-stream catalog experiment has run;
previous catalog results remain scoped to their original homogeneous contexts.

### Comparison with a saved catalog fit

The existing comparison API accepts `catalog_fit=True` explicitly:

```python
from scripts.ticket_controller_comparison import run_comparison

result = run_comparison(source_path, saved_catalog_study_path, catalog_fit=True)
```

The current ticket-serving command exposes the same operation:

```console
python -m scripts.run_bp_ticket_serving --source qualified-source.json --controller-comparison --fitted-report complete-catalog-study.json --catalog-fit --output fresh-comparison.json
```

`--catalog-fit` requires both comparison and fitted-report options. The saved fit
must be the complete six-trial catalog study bound to that qualified source; it
is validated by the same API before candidates run. Omitting the flag preserves
the original four-trial fit format. Other modes reject the flag before IO. Output
must be fresh; the original 8MiB comparison ceiling and failure exit status1 stay
unchanged. This command executes native comparisons: freeze a separate small
protocol and resource allowance before running it. The failed/spent #248 study
below is not authorized for retry by this software integration.

[CLI software checks and limitations](reports/catalog-comparison-cli-20261027.md)
cover literal dispatch and serialization only; they execute no native comparison.

It reuses a complete six-trial catalog fit and its original objective/source,
without fitting again. Admission checks every original context/arm receipt,
arrived TRAIN monitoring, native/replay/prediction/preparation work, task counts,
utility and frozen policy consistency. Failed, changed or incomplete source work
is refused. Discarded fit costs, whole source-study/application/bootstrap costs
and source pretraining are carried separately; inseparable all-in utility stays
unavailable. The old four-trial path and current CLI defaults remain unchanged.

Each recipe gets the same actual horizon for every control: the lesser of its
natural packet-count-plus-delay and the frozen catalog decision cap. The runner
passes that explicit `decision_limit` to every existing reference and fitted
application. It cannot renew the fitted envelope. `decision_limit` is available
only on declared deferred TRAIN streams, must observe all packet arrivals and
cannot exceed natural opportunities (or the catalog cap). Delayed labels may
remain unarrived, with charges and pending work reported. Old default reference
horizons stay unchanged when the limit is omitted.

Before more cases or readouts, bounded paid-evidence guards validate original
case/result containers and their values. All candidates finish and freeze before
the existing readonly consumer opens. Deferred cost admission reports native
work only; it does not turn the TRAIN-only result into task utility. Final
outcomes cannot change policy choices, training or promotion.

[Comparison software qualification](reports/catalog-ticket-comparison-20261027.md)
records focused checks and one metadata-only admission of the accepted saved
fit. The [first native comparison attempt](reports/catalog-delayed-noise-native-20261027.md)
failed before model work because of a Windows path in the experiment's source-pin
manifest. Its allowance is spent; comparison outcomes remain unknown and the
portable-path proposal has not been applied or rerun. The existing recipe
names refer to previously observed development schedules, not independent fresh
validation. Reuse old controls only when source, data, actions and actual
opportunities match; freeze a small protocol before any new experiment.

## What the legacy command learns

`core/contextual_controller.py` accepts the existing causal TRAIN observation.
The latest arrived original-actor mistake count defines two contexts: zero is
`clean`, positive is `mistakes`. No feedback means wait. Two finite IDs, `wait`
and `wake1`, remain subject to the original readiness, credits and admission.
No targets, native arrays, guard results, final/probe data or permission changes
reach this chooser.

Four disposable full-horizon trials cover new-context/wait, new-context/wake1,
old-return/wait, old-return/wake1 in that order. Utility is the unchanged
[prospective objective](controller-outer-objective.md), including full-task gain,
old-task degradation and fixed cost penalties. Each context-arm receives one
sample; its empirical mean is that sample. The better mean wins, ties wait.
No threshold, epsilon, seed, objective-weight or dose search occurs.

This is a deliberately small contextual selection model, not a claim about
one-step causal rewards. A wake choice is repeatedly applied over a complete
finite horizon. Hidden candidate state, delayed effects and noisy wall/RSS costs
are not modeled. One sample gives no uncertainty estimate. Actor feedback stays
at the original96-version while the shadow improves, so it can recommend extra
work after the candidate has learned. Tiny/cost-noisy fixtures may choose wait
for both contexts; correctness does not require a favorable learned choice.

## Full learner version and ownership

`app/counterfactual_controller.py` holds one detached complete original versioned
snapshot, at most four scalar estimates and one frozen fit. It uses existing
versioned-state ports and the native full-state digest callback, including policy,
parameters, optimizer, gradients and native clocks. The original source version
and payload must still match before each disposable callback and before its
utility is ingested. A stale/changed source, copied or modified request, future
return or unknown/incomplete cost is refused. Paid attempts remain charged and
stop further trials; there is no retry/reset authority. Locks reject reentrancy.

The native boundary additionally requires the exact qualified source and matching
complete base state. A trial's actual candidate is forked by the original
ActorShadowRuntime and receives delayed ordinary TRAIN labels through the same
ResourceSharedRuntime wait/wake admission. Trial mode does not prepare/commit
promotion. Original source/actor state must remain unchanged. Artifacts may hold
the four bounded observation records; the controller keeps no growing sequence
of model states or successful trials. This is trusted local Python, not a hostile
code sandbox or an authorization/persistence format.

The frozen fitted record contains the original learner version, full-state
digest, objective, fit clock, four estimates and derived choices. Inconsistent
choices or a different learner base are refused. Each application gets that
unchanged record before it opens its fresh native stream. Original publication
guards and complete rollback operate normally; application outcomes cannot refit
the table. This recipe supports the original96-start lane, not arbitrary future
model rebasing. The counterfactual outer clock and each application's fresh
learner clock have distinct scopes.

## Work and observed result

The sole frozen engineering study used sixteen discarded counterfactual updates/
96rows and eight application updates/48rows:24new updates/144rows total, plus
shared historical96/576source work. All common monitoring, task readouts,
constructors, snapshots, guards and application effects remain paid. Callback
wall/RSS utility is a partial proxy; source binding, detached-copy/fit/outer
bookkeeping and serialization costs also appear in the overall execution/work
ledger rather than being zeroed or silently treated as trial training cost.
Process RSS includes framework residency, and sampling may miss peaks. Fixed arm
order and instrumented first-use effects prevent a clean latency comparison.

| Context | Wait utility | Wake utility | Frozen choice |
| --- | --- | --- | --- |
| mistakes |-0.620708|-0.129246|wake1|
| clean |-0.629798|-0.631041|wait|

All training utilities were negative. The clean preference differs by only0.00124
utility units, dominated by measured cost proxies; it is not statistically
qualified. New-context application acquired legacy (new1/2 ->2/2), retained
old2/2 and passed the unchanged guard. Published serving solved all four ordinary
tasks, recovered two stale cases and refused both revoked commits. Complete
rollback restored the original legacy failure. Clean application used0updates,
retained old2/2 and did not publish. It starts a separate original96owner, so
this is not recovery from forgetting on the new application's learner.

Both application action traces equal the existing frozen drift reference on
these observations. The existing fixed-eight-update result already solves the
same tasks. Training added16discarded updates; no amortized compute advantage,
isolated controller benefit or CPC advantage is demonstrated. Shared grammar,
features, pretraining and one sample/context-arm do not establish statistical
independence or generalization. No final/probe source was opened by this recipe;
application checks are development evidence, not held-out confirmation.

A posthoc pure structural check covered32six-row conditions (missing feedback
or0..6mistakes, ready/not-ready, credit/no-credit). Choices match drift throughout:
positive integer mistakes/6 always exceeds0.1. At1mistake/12examples they differ
(learned wakes, drift waits); this is an algebraic counterexample, not a task
outcome or evidence for either policy. Repeating the same six-row cohort would
not resolve generalization. See the local `equivalence.json` diagnostic.

## Files and checks

```text
src/core/contextual_controller.py       causal scalar choices, estimates, frozen fit
src/app/counterfactual_controller.py    version-bound disposable trial admission
scripts/ticket_controller_learning.py  finite recipe behind existing CLI
tests/test_contextual_controller.py    causal/stale/mutation/failed-work behaviors
tests/test_ticket_controller_learning.py native forks, fitting, application and CLI
```

```console
python -m pytest -q tests/test_contextual_controller.py tests/test_ticket_controller_learning.py tests/test_controller_objective.py tests/test_controller_outer_study.py tests/test_reference_controller.py tests/test_torch_serving.py
python -m ruff check src/core/contextual_controller.py src/app/counterfactual_controller.py scripts/ticket_controller_learning.py scripts/run_bp_ticket_serving.py tests/test_contextual_controller.py tests/test_ticket_controller_learning.py
python -m ruff format --check src/core/contextual_controller.py src/app/counterfactual_controller.py scripts/ticket_controller_learning.py scripts/run_bp_ticket_serving.py tests/test_contextual_controller.py tests/test_ticket_controller_learning.py
python -m mypy --no-incremental --explicit-package-bases --follow-imports=silent src/core/contextual_controller.py src/app/counterfactual_controller.py scripts/ticket_controller_learning.py scripts/run_bp_ticket_serving.py tests/test_contextual_controller.py tests/test_ticket_controller_learning.py
```

131focused cases/platform and scoped statics passed. Source/procedure/results
self-reviewed, not independently accepted. Exact protocol, commands, profiles,
raw fits/applications and interpretation are local under
`artifacts/runs/learned-contextual-controller-20261023/`.
Live handoff: [issue53](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/53).

Why this: establish a trained finite chooser with real discarded native work
before claiming it improves training efficiency. Next assess a frozen chooser
against existing controls on genuinely withheld recurring schedules, carrying
the fit costs and preserving final isolation. Broader empirical chronology,
meta-generalization, advanced actions, long-duration behavior and release gates
remain open; these outcomes cannot select another objective or favorable run.
