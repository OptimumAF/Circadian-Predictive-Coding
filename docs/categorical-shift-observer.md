# Categorical arrived-only shift diagnostics

`ArrivedShiftObserver(..., num_classes=6)` now supports the six-action ticket
learner. It reuses the existing detector, calibration prefix, incident tracker
and delayed stream callback. The default `num_classes=2` keeps the binary
probability/soft-target contract and report schema unchanged.

## Metrics and inputs

`core.shift_signals.summarize_categorical_shift_batch(x, p, y, K)` accepts exact
float64 arrays: features B×D in[-1,1], probabilities B×K, hard class indices B×1.
Bounds are1..128rows,1..16features and2..16classes. Existing categorical
validators enforce finite simplex rows (absolute row-sum tolerance1e-6), native
class counts and hard labels. No coercion, probability repair or normalization.

The returned bounded `ShiftMetrics` contains:

- Uncertainty: mean `-sum(p*log(p))/log(K)`, with0log0=0; only roundoff-range
  clamping to[0,1]. Uniform predictions have uncertainty1; exact ones0.
- Feature means: the existing raw per-column arithmetic means.
- Residual: mean `1-p[true class]`, in[0,1]. This describes target probability;
  it is neither misclassification rate nor cross-entropy.

For two classes and hard labels the categorical metrics agree with the binary
formula within floating arithmetic. The separate binary function still accepts
soft targets and retains its original numerical implementation.

Why separate: changing a binary probability column into a categorical simplex
silently would change existing recipes and metrics. Explicit mode binds the
report and each arrived batch to its declared class count.

## Compose through the existing stream

Supply the existing owned-state prediction callback when creating the observer:

```python
observer = ArrivedShiftObserver(predict_state, num_classes=6, max_updates=3)
# Pass observe_checkpoint=observer to the existing AttributionStream or
# run_attribution_stream, with the ordinary TRAIN batches and original budget.
```

The callback primes state0 without labels/predictions. After update k, it scores
the just-arrived ordinary TRAIN batch using a copied checkpoint k-1. Current
state, delegate features, input batches and returned report are detached. A batch
with a foreign class count, final/probe role, reused row ID, malformed label or
nonconsecutive update refuses before state copying or prediction. Admitted
prediction failures retain attempted work, mark that row failed and stop later
diagnostics. The original max31update/report-slot limit remains.

Arrival authority comes from the trusted existing stream callback. A direct
call with a fabricated ordinary batch cannot authenticate real label arrival.
The observer does not receive prearrival labels or feed decisions into training,
promotion, replay or a controller. It remains retrospective descriptive telemetry.
Features/uncertainty are also reported only after arrival in this interface.

Categorical reports add `prediction_schema`: class count, uncertainty/residual
definitions and attempted/completed diagnostic input rows. A normal callback
return counts before probability validation; a malformed return remains paid
and then fails. These counts do not measure a delegate's hidden model/copy work.
Callers must account for actual forwards/state copies/arrays/time/RSS separately.
Existing measured observer durations include copying and validation. Batches
retain equal detector weight irrespective of row count, as before.

Optional calibration retains its original fixed prefix-fit procedure and
caller-declared identity; incidents remain onset/continued/cleared descriptions.
No threshold is tuned or relaxed by this extension. Old binary false alerts,
missed shifts, unreachable uncertainty alarms and negative studies remain.

## Validation and limits

174 focused cases pass on Windows and Linux, including scalar entropy/residual
oracles, malformed simplex/labels, owned state, paid failures, unchanged binary
reports, fixed calibration/incidents and the actual delayed stream with a
scripted K6 delegate. That fixture has no learned model or final data; candidate,
actor/source checks, original budgets and receipts are unchanged by observation.
Scoped Ruff/format/mypy pass; lead self-review is not independent review.

[Exact commands, resource receipts and handoff](../artifacts/runs/categorical-shift-observer-20261023/RESULTS.md),
issue97. No actual trained ticket/simulator/comparative run occurred here.
Useful triggers, false-alarm rates, model costs, retention/recovery and scientific
benefit remain unestablished. Full R2.7 and the wider repository/release goal stay
open; the earlier failed installed-game case96 remains spent.

The subsequent [guarded serving composition](context-shift-serving.md) now passes
its separately budgeted real-learner check, with two extra predictions/nine rows
and unchanged task/guard/rollback behavior. That short reference-only composition
does not establish useful alarms. A separate fixed longer stationary-noise/
recurrence study remains necessary; no threshold tuning or new runner is needed.
