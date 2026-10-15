# ADR-0252: Defer fixed-checkpoint attribution scoring

## Context

R1.7 needs stationary and recurring comparisons with evaluation isolation,
matched initial tensors and explicit computational differences. The native stream
demo uses unscored future probes and cannot serve as this evaluation protocol.

## Decision

Add a separate bounded application runner using the existing delayed-label
runtime. Validate roles, row IDs and cross-role array separation before work.
Capture detached checkpoints on a fixed schedule and complete the learner stream
before any held-out metric calculation. Only a separate scorer receives held-out
features; labels stay in application metrics. Fixed outcomes cannot alter later
run settings.

Use one small prospectively specified synthetic pilot with exact shared model
tensors, all planned failure rows, existing seed summaries and original first-pilot
resource ceilings. Report both context metrics, signed paired differences and
nominal work. Keep native objectives distinct and intervals explicitly exploratory.

## Alternatives

Scoring during training allows accidental feedback into scheduling. Reusing
native energy as the shared metric compares different quantities. Dropping failed
seeds changes the declared experiment. A new admission or evidence framework
would not improve these implementation boundaries.

## Consequences

The code supports a reproducible local attribution pilot without promotion or
adaptive tuning. Fixed-data three-seed results cannot establish broad superiority,
biological plausibility, scalable parameterization or equal computation. Resource
checks occur at safe boundaries and are not native-call preemption.
