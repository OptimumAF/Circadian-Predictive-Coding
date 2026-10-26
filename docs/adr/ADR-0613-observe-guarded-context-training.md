# ADR-0613: Observe guarded TRAIN through the existing checkpoint callback

## Context

Issue97added categorical summaries and an explicit class-bound arrived observer.
The existing contextual ticket slice already performs2delayed updates, fixed
guards/publication and complete rollback. Its working source and task traces are
accepted; diagnostics must not create a new learner or change that workflow.

## Decision

Add an exact-bool opt-in and matching existing CLI flag. Attach the observer only
to the guarded AttributionStream and use the pure owned-context prediction port
on the preceding full state. Preserve default call kwargs/report shape. Store a
detached observer report in finally, with relative/native clock mapping and paid
failure prefixes. Diagnostic data remains informational and ordinary TRAIN-only.

Why this: the stream already owns label arrival, state copying and lifetime
budgets. A second loop or speculative controller would duplicate those boundaries.
Pure snapshot prediction adds forwards/copies without extra model construction.

## Alternatives

Scoring current states after their updates would contaminate pre-update residuals.
Observing guards/final labels would change data authority. Reducing reference
length merely to produce an alarm would tune the diagnostic to this short slice.
All are unnecessary: two reference rows suffice to qualify composition and costs.

## Consequences

The default source,2update/9row dose, original guards/rollback and resource limits
remain. The real opt-in case has identical model/task/untimed guard evidence to
accepted90and adds exactly2diagnostic calls/9rows plus separately counted copies.
Guard timing fields and runtime costs differ; no isolated efficiency claim.
The three-batch reference never enters monitoring here, so trigger/false-alarm
usefulness remains a separate longer experiment. Native, software and historical
costs/failures are retained; self-review is not independent acceptance.
