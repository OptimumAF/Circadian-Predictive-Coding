# ADR-0578 — Separate final D readout from FIFO36 TRAIN admission

## Context

The accepted FIFO36 control learns known cue games and recovers A after return.
Its existing evaluator admits only six TRAIN A/B/C pairs. Probe-trained160 A/F
results do not establish held-out behavior of benchmark128. Full R4.2 retains
held-out-layout and recurrent-RL requirements. The public D geometry was inspected
in older work, so it is not a globally untouched benchmark.

## Decision

Factor the unchanged FIFO36 full typed snapshot copier and add one explicit final
readout using the existing shared evaluator. Admit only benchmark128/1152 with
the matching original source0, fixed36-entry canonical policy and immutable
zero-update/zero-replay budget. The plan is exactly D blue/red, role final_test,
seed17 and16steps, raw seven-action first argmax without masks/teacher/repair.
No TRAIN scores or target arrays enter the final readout. Both successful and
failed report prefixes label the observed initial frames FINAL. Original TRAIN
admission, defaults, selectors and reports keep their contracts.

Freeze the model/protocol before observing D. Final outcomes cannot train, tune,
promote, choose replay/probes/controllers, or select a favorable checkpoint.
Use the original saved endpoint and strict decoder without recreating training
or reviving its spent180-second execution budget. Readonly evaluation has its
own explicit zero-training allowance and records every partial/failure cost.

## Alternatives

Broadening the old TRAIN API risks accidental role relabelling. Probe160 or
best-prefix selection would change the frozen learner. A new runner/loader would
duplicate owned readouts and existing saved-state validation. Retraining a
successful control would add work without answering the held-out question.

## Consequences

The sole frozen case observes0/2Dsuccesses,0returns and32uniform predictions on
unmatched keys; both16-step games truncate. Known recall remains accepted while
this held-out result is negative. No model/probe selection or tuning uses it.

The same bounded source/state/budget/cleanup checks now support a final-only
question. A two-cue deterministic result cannot establish broad statistical
generalization or CPC advantage. Scripted boundary checks are not real game
evidence. Source/procedure/result self-review is not independent acceptance.
Any sealed outcome remains available for readonly reporting, never automatic
learner changes. Full R4.2 and release/scientific requirements remain separate.
