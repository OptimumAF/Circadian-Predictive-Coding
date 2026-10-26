# ADR-0417: Bound categorical PC proposals with existing stopping semantics

## Context

Binary coupled inference already has accepted bounded stopping. The inspected
categorical weighted and native PC paths still run fixed iterations. Existing
weighted S64 results remain negative; adding a cost/safety control does not
establish useful adaptation or justify tuning from evaluation scores.

## Decision

Add a detached TRAIN-only core proposal using the existing weighted objective
and B*dE/dz convention, with validated CoupledInferencePolicy semantics. Admit
finite nonincreasing candidate energy and finite direction/norm before commit;
return the last accepted state on unsafe attempts without retrying. Compute one
parameter proposal at that state. Reuse mechanical weighted finalization so the
fixed API retains its arithmetic order and outputs.

Why: a bounded proposal is the smallest mathematical prerequisite for an explicit
owned adapter route. Counting rejected attempts and finalization separately
prevents early stopping or numeric failures from appearing free. Every returned
counter dictionary is owned; unexpected failures explicitly flag unknown work.

## Alternatives

Changing existing fixed defaults would confound accepted matched controls.
Reimplementing binary inference is redundant. Publishing only bounded parameter
tensors would omit native state transitions; the adapter increment must reuse
the existing complete atomic boundary before deployment.

## Consequences

The kernel passes 43 new cases, 39 existing primitive cases and five targeted
atomic-adapter regressions, scoped statics and independent source/actual review.
Exact extraction of the original fixed finalization was independently checked.
R1.5-T adapter/work DTO/stream integration and full scientific/release goals
remain open. No comparison, tolerance tuning or game experiment ran.

[Usage and accounting](../categorical-bounded-inference.md),
[executed checks and preserved failure](../../artifacts/runs/categorical-bounded-inference-20261018/RESULTS.md).
