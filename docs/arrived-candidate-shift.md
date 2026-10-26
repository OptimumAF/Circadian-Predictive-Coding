# Arrived shift feedback before wake selection

The contextual live command has an explicit diagnostic-driven reference option:

```powershell
python -m scripts.run_bp_ticket_serving --example-context-source --context-head-serving-live --context-live-reference drift --context-arrived-shift --output NEW.json
```

The output must be new. This is separate from candidate-mistake feedback, task
feedback and correct-example acknowledgement. Those options cannot be combined.
The short default contains only two packets, so it remains reference-only and
does not establish alarm detection or useful adaptation.

`context_head_live.run_case(..., reference_mode="drift", arrived_shift=True)`
binds the existing finite schedule to `ArrivedCandidateShift`. Each NEW arrived
ordinary TRAIN packet is scored once on a detached current candidate before
choosing wake/wait. A waiting learner still produces observations; repeated ticks
without new labels do not rescore data or reuse a stale alarm.

This is different from the after-update observer, which scores the preceding
parameter state after a training update. It preserves the original three-batch
reference, two-batch window/persistence, thresholds and categorical metrics.
Reports contain full bounded metrics, incidents, native/wake clocks and paid
attempt/normal-return counts, including failures. At most8packets48rows are read.

Any current alarm can request the existing `wake1` action. Readiness, hard credit,
serving priority and FIFO admission still decide whether it executes. No examples
are skipped, reordered or acknowledged. An alarm may therefore spend work on
older pending data rather than the data that triggered it. Scheduling usefulness
must be measured separately.

The outer reader owns arrival/state/report checks and failure latching; pure
`ArrivedShiftFeedback` carries scalar metadata to the existing reference choice.
It checks actor/candidate/native origin across arrivals and binds the full current
candidate through admission. The calling delayed schedule owns arrival authority;
direct callers do not authenticate labels by supplying integers. No final/probe
source, promotion permission, new budget, scheduler or learned chooser is added.
Default modes, call kwargs and report schemas remain unchanged.

## Executed integration and limitation

One frozen A3/B3/A2 case with delayed labels completed24live workflows,15successes
and9failures. Uncertainty/residual onset at arrived batch6 requested two updates
at ticks7/8. Both trained old FIFO packets, with zero mistakes; six packets,
including all new-task data, remained pending. No publication occurred and the
source actor stayed deployed. Training was incomplete although the finite
observation run completed. This mode did not demonstrate useful adaptation.

Eight diagnostics cost39prediction rows; two updates cost12training rows.
The case took18.32214seconds, sampled peak50,782,208bytes. Diagnostic seconds
(.3436197) cover the delegate and subsequent validation/metrics/incidents,
excluding earlier acquisition/pinning/input copies; separate profile counters
retain that copy work. No efficiency or comparative benefit is established.

Local evidence/protocol: `artifacts/runs/arrived-candidate-shift-20261023/`.
[Issue100](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/100) is
the live scope/handoff; [ADR-0614](adr/ADR-0614-arrived-before-wake-signals.md)
records the design. Extend through the existing ports with separately frozen
data/action/credit/readout costs. Do not tune from retained outcomes or retry
spent experiments under another name.
