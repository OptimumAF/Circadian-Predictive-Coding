# ADR-0464: Return the existing owned continuous learner

## Context

The fixed learning recipe demonstrates offline fitting/forgetting/recovery and
has a detached final readout learner, but discards it after reporting. The frozen
robot caller already supports this exact MLP and full-state inference checks.
Native skill remains unproven. Reconstructing from the JSON report is impossible
because it intentionally retains no raw model archive.

## Decision

Expose a small result API with the existing report and detached final learner.
Return no model on incomplete execution. Preserve the JSON-only wrapper, CLI,
recipe/defaults, costs and all previous evidence. Any caught execution error
explicitly clears success before returning. The result owns mutable report/model
objects; freezing the container does not freeze their contents.

## Alternatives

Adding a callback/output sink or new checkpoint framework would broaden an
ordinary return-value requirement. Changing the robot binding is unnecessary.
Replaying the unchanged recipe once under a new frozen native protocol is
necessary work when no raw model was retained; count it separately and never
claim it is free reuse of old training.

## Consequences

No extra initialization, optimizer step, fork, dependency or publication API is
introduced. Callers can validate the local learner through existing ports and
retain responsibility for new experiment budgets and inference authority.
Offline fit and native TRAIN returns are separate from generalization, actual
domain drift, guard promotion and circadian benefit; all negatives remain.
