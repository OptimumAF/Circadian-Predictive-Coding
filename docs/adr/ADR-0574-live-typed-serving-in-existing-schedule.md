# ADR-0574 — Serve typed workflows inside the existing causal schedule

## Context

The native BP demo learns/serves known tasks, but typed requests occur only at
phase readouts outside the reference schedule's original serving gate. R7.7
requires behavior and cost while serving and updating under contention.

## Decision

Add an optional finite live plan/helper, not a new training runner. Reuse the
existing schedule, source, actor/predictor, admission, budgets, feedback and
promotion controller. One real worker holds the original serving lease; reserved
ticks defer native admission, other native-start callbacks launch service under
the existing admission. Join each prefix before the next tick. Record versions,
actual gate state and instrumented request/episode time separately. Worker checks
source/plan without changing the main thread's budget clock. Outcomes stay out
of TRAIN/controller/guards/final/probe boundaries. Defaults remain unchanged.

## Alternatives

Another standalone runner duplicates causal accounting. Phase-only readouts miss
contention. Treating live success as feedback leaks outcome truth into control.
Serializing all requests after training hides deployed failures while learning.
Reporting native admission overlap as simultaneous optimizer computation or a
tiny instrumented p95 as a deployment SLO overstates the evidence.

## Consequences

Serving-priority deferral, bounded request histories, native-admitted overlap and
actual actor versions are observable. Existing publication is still post-stream;
all live legacy attempts therefore fail until later publication, despite a learned
candidate. This informs the next useful change. The24tick trace is not long-duration
validation, statistical benefit, physical containment or CPC advantage. Full R7.7
and broader requirements remain open; review is self-review.
