# ADR-0500: Curate saved TRAIN returns with new clocks

## Context

The returning A learner can grasp but fails placement despite lower original
imitation error. Its50 saved learner states have no correction labels. Existing
readiness batches derive label times from control steps; reusing those types for
post-hoc annotation would backdate labels. Native teacher geometry uses privileged
poses unavailable from saved traces, while raw42 contains aligned task displacements.

## Decision

Add distinct immutable curated-label/batch records with optional ORIGINAL
observed ticks and explicit NEW query/arrival ticks. Keep legacy batches intact.
A narrow adapter reuses the existing goal-consistent observed-grasp teacher on
public relative geometry, with distinct geometry/provisional definitions and the
same declared default Panda setup as TS12. Validate source/roles/clocks first,
count paid attempts/returns, retain partial failures, release after all arrivals,
and derive existing TS12 history solely from executed commands.

## Alternatives

Backdating legacy labels obscures causality. Recollecting the same return spends
unneeded simulator work. Reconstructing privileged root poses invents unavailable
information. New orchestration or automatic learning/promotion would mix data
curation with a separate scientific decision.

## Consequences

Saved TRAIN states support bounded later correction with no model or simulator
work. New records require explicit handling; they are not silently interchangeable
with legacy batches. Labels remain provisional and setup-bound, without historical
bitidentity, native recovery or generalization guarantees. A later fixed recurrence
study must separately account query, reconstruction, optimization and native work.
