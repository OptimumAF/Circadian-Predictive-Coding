# Same-recipe FIFO reference study

This fixed study compares existing references on the qualified contextual ticket
learner. Three controls call the original live API; the accepted arrived-shift
case is reused unchanged. No new learning policy or controller is fitted.

All methods receive the same eight packets, delayed labels, 24 live opportunities
and eight-update ceiling. Actual training and diagnostic work differ.

| Reference | Live successes | Updates / training rows | Correct acknowledgements | Publication tick | Frozen utility |
|---|---:|---:|---:|---:|---:|
| No learning |15/24|0 / 0|0|—|−0.2220|
| Periodic |17/24|8 / 39|0|18|0.2200|
| Candidate feedback + acknowledgement |19/24|2 / 9|6|9|0.1866|
| Arrived shift (reused) |15/24|2 / 12|0|—|−0.2473|

The original common six-task window contains four old-task successes for every
method. Periodic and acknowledgement obtain one of two new-task successes; the
second task has a predeclared revoked commit and remains a failure. Both publish
only after the original guards pass, then fully restore the source on rollback.
Periodic and acknowledgement finish with identical candidate coefficients.

Acknowledgement verifies correct ready TRAIN receipts and consumes them without
inventing model updates. It reaches useful new-task data sooner here. Its eight
diagnostic forwards add39rows, and measured time/RSS are higher than periodic.
Periodic has the higher original utility score. These observations support a
tradeoff within this known-template development case, not a general winner or
an efficiency claim. The shifted-signal reference still fails to train new data.

Utility retains the original weights/scales and includes full invocation costs,
including conditional rollback. Serial order, carried reports and the historical
shift run make time/RSS unpaired proxies. All planned failures, diagnostics,
acknowledgements, native updates and pending work remain in the reports. Final
evaluation, probes, replay and policy fitting are absent.

## Evidence and reproduction

`scripts/context_fifo_reference_study.py` is a fixed outer composition and readonly
scorer using the existing live API, receipt helpers and core objective. It validates
all four complete reports before scoring; a failed case stops subsequent work.
Existing shorter comparison protocols retain their original scope.

The spent local study is under `artifacts/runs/context-fifo-reference-20261024/`:
PROTOCOL.md, actual.json, readback.json, RESULTS.md and all cost/test receipts.
Inspect the accepted result without running models:

```powershell
python -B artifacts/runs/context-fifo-reference-20261024/readback.py
```

Its original native launch is permanently marked CASE_SPENT. Do not repeat it to
select outcomes. New scientific work needs a separately frozen question/data/
cost protocol; reuse accepted evidence where its assumptions remain unchanged.
[Issue102](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/102)
contains the live acceptance and handoff.
