# Controller learning on the contextual source

```powershell
$env:PYTHONPATH = (Get-Location).Path
.venv\Scripts\python.exe -m scripts.run_bp_ticket_serving --source artifacts/runs/context-head-control-20261023/actual.json --context-controller-learning --output local-context-fit.json
```

Output must not exist. This bounded development recipe composes the existing
`CounterfactualController` and `_reference_schedule` with the conventional context
learner. It does not transplant the older native BP fit or use probe weights.

## Procedure and boundaries

The qualified source supplies two ordinary TRAIN packets: v1legacy (three rows,
mistake feedback) and v1invoice/password (six rows, clean feedback). Each packet
is predicted before labels arrive at logical+1. The existing controller tries
wait then wake1 on separate copies of the complete source for each category.
Each trial has one delayed packet, two opportunities and at most one update.

Trial utility uses before/after classifier accuracy on detached TRAIN-derived
new6/old12 arrays, with the unchanged objective and full trial time/sampled RSS.
These are classifier diagnostics, not closed-loop task success. Costs outside
the trials, including source monitoring, ownership checks, copies and fit, appear
in the outer study measurements. No external actions occur in these trials;
zero violations is not a safety result.

The fit freezes after four complete scalar returns, before one fourteen-opportunity
live application. That application uses the existing worker, gate, original
budgets, guarded publication and rollback. Its task outcomes cannot refit or
select the controller. Pending TRAIN remains explicit. A policy bound to another
source/version is rejected before learner construction; mutation stops work.
Publication can make a fitted policy stale, which is refused rather than rebound.

Failed or malformed paid trial results are retained before further validation.
The existing paid-report guard validates preceding evidence before admitting more
work. Final evaluation and private probes are unavailable to the fit/application.
No new runtime, controller algorithm, dependency or checkpoint authority is added.

## Interpretation and extension

The sole frozen development run completed. All four trial utilities were negative;
the fit selected wake on mistakes and wait on clean feedback. The live application
applied one three-row update, left repair pending and did not publish. It passed
5/6 common tasks, with actions, candidate state and task outcomes exactly matching
the saved drift control. No task benefit from fitting was demonstrated.

Discarded fit trials cost two passes/nine rows/one mistake; application cost one
pass/three rows/one mistake. Total three passes/twelve rows/two mistakes, plus
monitoring/readouts/copies. Full study1.54147s, sampled peak50,380,800B. These are
instrumented development observations, not amortized or paired efficiency evidence.

Clean/mistakes feedback is a weak observation: it does not expose pending repair,
shadow-learner regression or delayed publication effects. A favorable classifier
estimate therefore need not yield useful deployed adaptation. One development
application cannot establish meta-generalization, efficiency, concurrent inference
or CPC benefit. Preserve all failed and inconclusive runs.

Broader R7.5 distinct recurrence/noise validation, R7.7 long deployment and release
remain open. Extend through the existing stream/controller ports only after a
separate prospective protocol; do not tune from application outcomes or silently
change the existing fixed controls. Evidence and exact commands are recorded in
`artifacts/runs/context-controller-learning-20261023/` and
[issue82](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/82).
