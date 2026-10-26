# ADR-0557 — Complete native checkpoints for typed categorical readouts

## Context

R4.W.T / #40 follows #39's qualified complete-task replay result. Existing native
four-family learners already train categorical heads. Their probability-only
port cannot safely normalize candidate subsets after global float32 softmax;
all permitted mass can underflow. The older frozen adapter is fixed2→3→2 and
reconstructs fresh nonparameter state, so it is unsuitable for this contract.

## Decision

Add an owned raw-logit method while preserving existing predict arithmetic.
Create a small outer adapter over a full validated native checkpoint and the
existing core feature/typed schema. Reuse the native constructor+restore ports,
validate complete frozen state before/after inference, mask logits before
normalization, latch faults and retain call counts. Existing stream and ticket
dispatcher remain the orchestrators. No training or promotion methods exposed.

Why this: explicit full copies preserve optimizer/adaptive/RNG provenance and
separate probe/readout owners without another inference/training framework.

## Alternatives

- Global-probability renormalization: loses permitted mass under extreme scores.
- Parameter-only reconstruction: discards original optimizer and native history.
- Change existing predict to return logits: breaks binary/categorical callers.
- Generic runner rewrite: not necessary to verify this missing boundary.

## Consequences

Copy/initialization/check overhead is real and must be budgeted. Full-state
value checks are synchronous trusted-process guards, not a private-object
identity or concurrency guarantee. Literal dispatcher success verifies wiring;
learned acquisition/retention/recovery needs a separate frozen observation.
No comparative/CPC advantage or repository release readiness follows.
