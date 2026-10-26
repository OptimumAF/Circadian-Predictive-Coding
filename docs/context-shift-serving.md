# Shift diagnostics during guarded contextual learning

The existing CPU ticket command has an explicit diagnostic option:

```powershell
python -m scripts.run_bp_ticket_serving --example-context-source --context-head-serving --context-shift-diagnostics --output fresh-context-shifts.json
```

Choose a fresh output path. Explicit `--source PATH` also works in this mode.
The flag requires synchronous `--context-head-serving`; unsupported modes refuse
before source lookup or output creation. The default command preserves its prior
calls, report schema, guarded learning and rollback behavior.

## What it observes

Only the guarded arm attaches the existing categorical ArrivedShiftObserver.
It scores each ordinary arrived TRAIN batch using the preceding copied native
state through `predict_context_perceptron`, without constructing another learner.
There are two callbacks over3then6rows: preceding native updates12and13,
relative stream indices0and1. The qualified source's12updates/72rows/16mistakes
are history, not new runtime authority. New training remains2updates/9rows.

The added `shift_observations` record identifies the native clock origin and
contains a detached observer report: normalized K6 entropy, feature means,
mean1-p_true residual, attempted/returned diagnostic input rows and measured
durations. Admitted failures retain paid report prefixes in the arm's `finally`.
Signals never select training, guards, publication, replay, or task outcomes.
The frozen arm does no diagnostic prediction.

The original reference requires three batches. This two-update slice therefore
has two reference rows and no monitoring alarms. Changing that policy to obtain
an alarm would change the question; no threshold, reference length, window or
persistence setting is tuned here. A longer frozen protocol is required for
false-alarm/useful-trigger evidence.

## Executed result and costs

The one bounded real-learner invocation passed. All42workflow episodes complete;
guarded task success is5/6initial,5/6after-new,6/6after repair/publication, then
5/6after exact rollback. Original retention rejection precedes final acceptance.
Candidate/source/actor states, complete task traces and rollback equal accepted
issue90evidence. Only the guards' measured prediction-time fields and runtime
budget clocks differ, as expected for separate executions. Initial/rollback
failures and the already known second-context legacy task remain visible; no
new forgetting/recovery claim is inferred.

Two extra full-class predictions over9rows used no extra learner constructors.
Five observer-owned state deepcopy boundaries account for480logical coefficient
slots; three stream-delivery state copies account for288more. Existing native
helper copy counters are separate. Logical slots describe visited coefficient
payloads, not actual Python allocation/RSS.

Observer durations total0.0289833s, including its own copying, validation and
instrumentation, excluding upstream stream-copy/other case work. The whole case
took17.846341s with sampled peak49,795,072B. Profiler overhead and setup differ
from historical runs, so subtracting their wall times would not isolate the
observer's cost. Original45sarm/120swhole/512MiB/1MiBoutput limits were unchanged.

Native training added2passes/9rows/2mistakes. Other registered work:10context
constructors/20zero heads,820context and1914head copies/91872helper coefficient
slots,81full-class rows,126context+126inner typed wrapper entries,2teachers/
3packets/282encodes/150dispatches/128routes. Typed wrappers are nested; their
counts are not additive unique task calls. No Torch/PC/gradient/probe/final/
replay/random initialization. All costs and original history remain charged.

[Exact commands, failures, checks and handoff](../artifacts/runs/context-shift-serving-20261023/RESULTS.md),
issue98. Thirty focused cases qualify on Windows/Linux; scoped statics and lead
self-review pass. Self-review is not independent review. No package reinstall,
clean environment, broad reliability, CPC/generalization/efficiency or release
claim follows. Original failed studies, including the spent game96case, remain.

Next freeze a small longer stationary/noise/recurrence observation through the
existing stream with fixed packets, detector settings, task readouts and complete
prediction/copy/training costs. This command remains an optional diagnostic
composition and is not another controller or a new scheduler.
