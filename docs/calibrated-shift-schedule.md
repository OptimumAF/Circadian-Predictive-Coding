# Calibrated incidents in the existing TRAIN schedule

The optional Python composition in `scripts.run_bp_ticket_serving._reference_schedule`
connects the accepted calibrated observer to ordinary FIFO learning. It adds no
runner or CLI flag. Native learned-model usefulness remains unqualified here.

Pass all of these together, alongside the original source, actor, batches, budget,
clock, sharing gate and finite serving horizon:

```python
arrived_shift_predict=predict_probabilities,
candidate_predict=predict_probabilities,
acknowledge_correct=True,
shift_calibration=ShiftCalibration(),
shift_model_id="caller-declared-initial-model",
shift_prefix_rows=ordered_stationary_train_row_ids,
```

This requires drift mode, an actual monitoring batch after the declared prefix,
and the existing separate-mode restrictions. Defaults retain the uncalibrated
observer and supervised error/ACK paths. Prefix IDs are checked only when their
TRAIN labels arrive; the declaration does not attest statistical stationarity.
No task readouts, private probes or final evaluation enter selection/calibration.

The frozen initial diagnostic model observes each newly arrived packet once,
including while fitting waits. The CURRENT learner separately verifies the
FIFO-front packet. A correct verification may ACK its original receipt without
an optimizer update. A frozen-model alarm alone cannot establish competence.

Why this: on an incident onset, the scheduler records the already-arrived packet
count. Fresh current errors can request wake1 only below that fixed FIFO boundary,
using the unchanged drift error rule and original ready work/credit. Continued
alarms do not extend the boundary. Another observed onset may open a later one.
No new label is needed to finish already-authorized incident work, but each fit
still needs a fresh current diagnostic. Correct ACKs may occur outside an episode.
Neither the horizon nor wake credits are extended; work can remain incomplete.

`calibrated_shift_schedule` reports the declared horizon and at most eight onset
records. The existing shift report retains reference/calibration/replay costs;
the current diagnostic report owns its costs privately and exports detached
copies. Repeated, failed and deferred reads remain paid. Callback replacement of
exported prior evidence is refused before ACK. The optional `expected_candidate`
argument on `read_candidate_feedback` verifies the schedule's pin inside the
original ACK lease, including after its first input hook. It grants no authority
outside ordinary arrived TRAIN. State pins remain bounded; this is not an absolute
native-byte attestation and excludes live/transient copies.

The original runtime still validates source/label receipts, causal clocks,
candidate/native origin, sharing admission and update receipts. Changes outside
observed admission are refused; a no-update/deferred poll must leave weights
unchanged. Publication uses the existing all-consumed/labels-arrived/idle gate.
This increment tests a declined drained callback, not an accepted promotion.
Existing accepted supervised native serving evidence remains in its own report.

See [executed checks, failures, commands and costs](reports/calibrated-shift-schedule-20261029.md).
Core detector thresholds, persistence, incidents and unreachable-score handling
are unchanged. Native stationary/shift behavior, useful selection, matched
comparative claims, circadian benefit and release readiness remain open.
