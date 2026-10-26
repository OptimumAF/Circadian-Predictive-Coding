# ADR-0546: Explicit state-only catalog probes

## Context

The fixed rehearsal probe loses correct predictions on already-rehearsed TRAIN
points at B and return. It updates both state and action projections plus
temperature, so that observation does not identify the cause. The native engine
requires complete optimized gradients and strict optimizer/scheduler ownership.
Silent requires_grad mutation would violate that contract.

## Decision

Expose all/state_only as an exact constructor argument on disposable catalog
probes, default all. State-only uses a private learner subclass with the same
numeric directional catalog cross-entropy, detached action/temperature computation
and a state-parameter optimizer. Small protected engine hooks select optimizer
parameters and validate staged parameters; defaults retain all existing behavior.
Fixed action/scale finite birth values, None gradients and scoped optimizer/state
owners are checked on original and staged bundles before publication. Keep one
bounded fixed-value pin per probe, no per-success chain. Explicit app/CLI/scalar
metadata binds this mode; main/source learners and public policy/snapshot formats
remain unchanged. Metrics remain cached until all training completes.

## Alternatives

Changing global CLMHeadPolicy would expand unaffected public continuation formats.
Silent gradient disabling would bypass native owner/gradient invariants. Copying
the full update engine would duplicate its atomicity and paid-failure rules.
Increasing replay dose or choosing learning rates does not isolate this hypothesis.

## Consequences

The control can be tested without a new orchestration framework or encoder. Same
updates/forwards have different optimized-parameter and memory costs, which must
be reported. No retention or generalization benefit is assumed; freeze and review
a bounded native comparison after software gates. Broader task/release parents
and all historical negative results remain open/preserved.
