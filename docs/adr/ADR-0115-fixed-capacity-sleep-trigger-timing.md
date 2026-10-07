# ADR-0115: Compare existing sleep triggers with fixed capacity first

## Context

The current adaptive trigger requires enough wake batches since sleep,
an eight-energy plateau window, and chemical variance above a fixed
threshold. Periodic forcing can execute sleep without those conditions.
Comparing them under the full sleep stack would also vary topology,
replay exposure, homeostasis, and guard decisions, making trigger
timing hard to attribute. P4.8 requires periodic, current adaptive, and
no-sleep controls on stationary noise and a distribution shift.

## Decision

Use a prospectively frozen v13 NumPy study with three arms, two new
seeds, stationary noisy phases, and an axis-shifted B phase. Every arm
receives the same role-separated train stream, 32 wake updates, and
fixed eight-neuron capacity. Sleep retains its existing chemical reset
and disables structure, replay, and homeostasis for this timing control.
Use the existing scheduler and core trigger; keep the adaptive defaults
unchanged. Train and preflight all twelve cells before releasing common
final roles once per seed/condition. Record every decision, actual
performed event, matched wake/inference work, accuracy and BCE, and
negative as well as positive paired outcomes. The full sleep-component
and broader shift test remains P4.8b; v13 does not close P4.8.

## Alternatives

- Lower the chemical-variance or plateau threshold when adaptive sleep
  fails to fire. This would use v13 train/final results to choose the
  comparator and would no longer test the current adaptive trigger.
- Compare only forced sleep against no sleep. This would omit the
  adaptive control required by P4.8.
- Change sleep topology/replay at the same time as timing. That would
  confound this first attribution; P4.8b retains that separate test.

## Consequences

The fixed v13 train-only gate passed in all twelve cells. Periodic
performed four chemical-reset-only events; adaptive and no-sleep each
performed zero. Adaptive never passed the unchanged variance threshold
in any of its 23 spacing/window-eligible opportunities per trial.
Adaptive and no-sleep saved states and held-out metrics are identical.
Periodic produced mixed BCE changes and one lower shifted B accuracy.
This is a narrow, valid negative for default adaptive timing on the
fixed streams. It motivates a prospective full-component/shift study,
but does not by itself justify a tuned threshold or new trigger rule.
The exact protocol, all cells, and limitations are in
`docs/sleep-trigger-comparison.md`.
