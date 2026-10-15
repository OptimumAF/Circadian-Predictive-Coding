# ADR-0260: linear pruning with an observed damage gate

## Context

R1.6a supplied isometric split/compatible merge. Arbitrary pruning cannot promise
the same objective or function, and prediction-only checks miss changes in the
scalar equilibrium denominator. R1.6 requires bounded observed damage and recovery
for non-preserving structural operations.

## Decision

Separate a pure owned raw-axis prune candidate from an application gate. Rebind
configuration/scales and retain residual identity. Require explicit prediction,
half-MSE and full-energy limits on all declared train/validation probes, excluding
final-test roles. Copy all state/data before callbacks and each evaluation; return
only a complete accepted candidate or the retained source. Distinguish exceeded
limits from execution failure and keep known diagnostics with planned count.

Why this: a structural proposal and the decision to use it have different
responsibilities. Immutable replacement selection makes rejection/fault recovery
simple without mutating a published model.

## Alternatives

Prediction-only gating accepts some energy-damaging operations. Exposing a rejected
candidate invites accidental publication. Reusing evaluation buffers lets faulty
code contaminate the comparison or fallback. Final-test gating would compromise
evaluation isolation. A shared runtime holder needs a separate atomic publication
boundary; it is not implied by this functional gate.

## Consequences

Limits are local observations, not global error or future learning guarantees.
Width-dependent rates change. Invalid inputs raise; execution failures return the
owned source without retry. There is no adaptive optimizer/trace state in the
linear references. Nonlinear/CPC state integration, shared-holder publication,
broader recovery and parent R1.6 remain separate unfinished work. No scientific
study or baseline tuning is required for this engineering equation check.
