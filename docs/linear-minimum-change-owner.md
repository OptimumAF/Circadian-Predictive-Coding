# Explicit minimum-change updates on the generic linear owner

```python
from src.adapters.linear_fit_learner import LinearFitLearner
from src.core.categorical_linear_state import LinearFitPolicy

source = LinearFitLearner(LinearFitPolicy(feature_dim=7, num_classes=6))
private = source.fork()
diagnostic = private.train_batch_minimum_change(released_features, released_targets)
```

The caller still owns admitted TRAIN or isolated probe roles, label release,
exclusive runtime lease, budgets and permanent attempted-work accounting. Arrays
alone confer no role/replay/promotion authority. This operation changes only the
owned learner; keep probes separate from deployed/benchmark learners and final
data. Existing `AttributionStream` can use an explicitly composed learner port;
its role/label/counter behavior is unchanged. No new production stream mode here.

Why: the generic default cold-refits each packet and can discard prior coefficient
directions invisible to that packet. The wide categorical core already implements
minimum-change residual fitting. The shared arithmetic now also supports the
bounded generic model and its existing atomic learner, without widening the wide
or CLM role interfaces or creating another update loop.

With augmented design Z and current one-hot targets Y:

`W_next = W_prior + pinv(Z) @ (Y - Z @ W_prior)`

The fixed intercept/RCOND1e-12/one-hot equal-row objective are unchanged. This is
current-batch least squares with minimum displacement from prior coefficients,
NOT cross-entropy optimization. It preserves prior components invisible to Z;
old predictions can still change when their directions overlap the new batch.
Conflicting old/current labels cannot both be satisfied by this rule.

An unfitted owner makes exactly the ordinary cold fit. `train_batch` ALWAYS remains
the original minimum-norm refit; the new operation is explicit and NONSTICKY.
Forks/restores/snapshots do not set an update mode. Existing policy/snapshot fields,
prediction format, cumulative committed counters and no-history storage stay.
Caller must choose the new port on every intended update. Diagnostic definition
is `minimum_change_one_hot_least_squares_sum_squared_residual_v1`; the value is
current-packet residual squared sum, not lifetime fit/retention or CE.

Pure `core.categorical_linear_fit.update_categorical_linear(prior, x, y)` requires
an exact valid fitted CategoricalLinearFit. Generic current rows1..128/features
1..296 remain; a prior whose last fit used128rows is valid. Wide current rows
1..16/features1..4096 remain separately enforced. Shared private residual arithmetic
does not grant wider public geometry or CLM encoding/role admission.

Warm owner updates detach a full prior snapshot separately from the unchanged
comparison pin, plus feature/target arrays. Success publishes one validated model/
counter replacement. Source/prior/input/RNG or owner replacement refuses publication;
foreign mutations are not silently restored. Failed/lost fits expose the existing
TrainingUpdateFailed scalar work/unknown flags under the new definition. Budget
successful-update counters are not rewritten into attempt counters: the caller
preserves attempted work and a failed existing stream remains stopped.

This is a conventional software update capability, not evidence that native task
retention is repaired. Earlier minimum-norm native outcomes/negative results stay
frozen. A separate prospectively budgeted protocol is required before measuring
the new rule in a native task or comparing model families. No scientific allowance,
CPC/generalization/deployment/release claim or probe-to-replay conversion here.

## Contextual owner

```python
from src.adapters.context_linear_fit_learner import ContextLinearFitLearner
from src.core.context_linear_state import ContextLinearFitPolicy

source = ContextLinearFitLearner(ContextLinearFitPolicy(4, 2, (2, 3)))
private = source.fork()
diagnostic = private.train_batch_minimum_change(released_features, released_targets)
```

The current TRAIN packet must carry one homogeneous existing public context tag.
The ordinary router removes both tag columns, and the selected private head uses
the generic operation above. One validated whole-context state then replaces the
owner; the inactive head and existing per-head/global counter contracts persist.
Mixed contexts remain valid for inference, not one training packet. The diagnostic
is `public_context_minimum_change_one_hot_least_squares_sum_squared_residual_v1`.

Both cold and explicit paths retain raw policy/state/head/model references and
compare identity, in addition to value/input/RNG checks. Why: an equal-valued
replacement is an unobserved ownership change; retaining references also prevents
object-ID recycling during the operation. Foreign changes remain after refusal.
Failures retain inner attempted/returned/lost work, selected `context_index`,
`private_candidate_commits` and zero whole `candidate_commits`. No stored update
mode, observations, task-label authority, probe promotion or new stream runtime.
Default `train_batch` remains cold even after an explicit residual update.

The existing delayed TRAIN stream was exercised through a test-only subclass.
The retained-task probe now has an explicit production composition described below.
Software orthogonal fixtures prove scoped behavior, not native retention or a
circadian advantage.
[Commands, failures, costs and review](reports/context-linear-minimum-change-owner-20261029.md).

## Isolated retained-task probe

```powershell
python -B scripts/recurring_ticket_workflow_example.py --method linear_fit --context-return --separate-context-heads --paired-serving retained_joint_probe --probe-update minimum_change --output <D-path>
```

Use a new prospectively budgeted local run; this command performs source, retained
and disposable probe work. Omit `--probe-update` for the original cold probe. The
option is restricted to this sequential ordinary LS retained probe and refuses
joint rehearsal and other methods/modes before source work. Source/retained updates
remain cold. Only detached `MinimumChangeProbeLearner` descendants select the named
operation through the existing FreshTaskProbe F/return sequence. There is no new training loop or
snapshot mode. Original16F/16return updates, native counters, clocks,120s source
lifetime at birth/60s child/30s independent probe sessions and permanent charges
remain; no budget renewal. Probes never update deployed/benchmark learners or feed
probe/final labels into TRAIN/replay/promotion.

API equivalents are `run_example(probe_update="minimum_change", ...)`, existing
`context_retained_probe.run_probe(..., probe_update="minimum_change")` and fixed
transfer `probe_ticket_policy_learning.run_arm(..., linear_probe_update="minimum_change")`.
The paid report pins `linear_probe_update` in outer/arm headers and requires16
minimum-change diagnostic records per successful phase. Missing/changed evidence
refuses further work. Earlier cold reports keep their original shape.

Executed literal-source retained/probe fixtures kept old4/4 and selected2/2 after
return. Those fixtures do not establish native retention; earlier native cold
results stay frozen. [Evidence and costs](reports/context-minimum-change-probe-20261029.md).

One subsequent native sequential case also kept old4/4 and selected2/2 after return,
with55solves288rows and31exact saved-control applicability checks. Raw-cohort identity
has an additional deterministic-builder/admission assumption; equal fits do not
match total compute. This fixed-case conventional result stays separate from
deployed behavior, unseen transfer, CPC benefit and release readiness.
[Native result and limitations](reports/context-minimum-change-native-20261029.md).

## Ordinary TRAIN selection

For a new prospectively budgeted run through the existing ordinary workflow:

```powershell
.venv/Scripts/python.exe -B -m scripts.recurring_ticket_workflow_example --method linear_fit --context-return --separate-context-heads --train-update minimum_change --paired-serving retained_joint --output NEW.json
```

Omit `--paired-serving retained_joint` for the source-only ordinary sequence.
Omit `--train-update` for the original cold fit. API keyword `train_update` selects
the operation through the shared counted helper for source and child TRAIN
updates. Its forks keep that invocation's choice; snapshots do not carry a mode.
The original counters/budgets/roles/receipts/clocks remain. Original source30s and
retained child60s limits are not extended. No probe data or probe-state promotion.
The existing `--paired-serving joint` path also accepts this selection. It uses
the existing paired TRAIN packets, inner guard, atomic publication and rollback.
Probe, retained-publication, catalog, reference and adaptive combinations are refused.

```powershell
.venv/Scripts/python.exe -B -m scripts.recurring_ticket_workflow_example --method linear_fit --context-return --separate-context-heads --train-update minimum_change --paired-serving joint --output NEW-PAIRED.json
```

Why this: the paired recipe explicitly teaches both v1 parcel and v2 password;
the retained recipe teaches v1 legacy only. Paired readouts cover those two new
tasks and the four old tasks, using `PAIRED_TASKS`. These are different task sets.
Paired source12/72 plus child6/48 completes at18fits120rows; original source30s,
child60s, six-update limit, labels and receipts remain. Publication still requires
the existing guard; a selection does not promise acceptance or renew a budget.
[Paired software evidence, costs and limitations](reports/context-minimum-change-paired-train-20261030.md).

Paid root/child `linear_train_update` intent and cost objective reflect the selected
rule. The omitted cold choice also stays pinned: inserting a different report tag
refuses further work. Only paid method metadata is restored on refusal, never
foreign learner/input changes. Source-only and retained task readouts use four
known tasks and two legacy requests, a different pair from paired serving and
the isolated transfer probe.

Both bounded ordinary software fixtures completed at23fits144rows, with old4/4
and legacy1/2 successes. This partial result does not establish ordinary benefit;
earlier cold legacy2/2 and private transfer2/2 remain separate evidence. Capture a
separately frozen native ordinary case and inspect the failed trace before changing
the learning rule further. [Checks, costs and handoff](reports/context-minimum-change-train-20261029.md).

The subsequent native ordinary case also returned known4/4, legacy1/2 at23fits144rows.
Saved cold returned legacy2/2. Source/child coverage shows v2legacy was untrained
and the v2head unchanged during retained updates; a tiny `inspect`/`defer` margin
explains the observed action sensitivity, without proving a retention failure.
Keep all outcomes and cold defaults. Use the existing paired task/teaching path
for any separately qualified two-head acquisition demonstration; do not change
tie rules to repair this capture. [Native diagnosis](reports/context-minimum-change-ordinary-native-20261029.md).

The separately frozen native paired case learned the previously failing v2
password task, retained all four old tasks, and kept the new task on private
old-context return: source old4/new1 → published old4/new2 → return old4/new2.
Rollback restored old4/new1. The v1 parcel task already succeeded before learning.
It used18fits120rows including old TRAIN rehearsal exposure; all diagnostic and
task reads were counted. Published16 and private return18 are different states.
This is one conventional known-task result; unseen transfer, CPC advantage and
recovery after observed forgetting remain open.
[Native tasks, failures, costs and scope](reports/context-minimum-change-paired-native-20261030.md).
