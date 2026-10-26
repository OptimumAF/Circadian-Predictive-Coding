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

## What it learns

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
