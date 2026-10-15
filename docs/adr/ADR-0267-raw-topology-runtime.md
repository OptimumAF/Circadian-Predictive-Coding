# ADR-0267: Complete raw topology through candidate and serving ports

## Context

R1.6e provides raw uniform-width prune candidates and isolated observed-damage
gates. Existing raw learners bind fixed configuration; effective topology wrappers
omit raw factor/configuration state. Runtime consolidation and promotion already
have owned candidate and independent serving boundaries.

## Decision

Add an explicit format1 raw envelope, bounded joined full-config prune history,
and opt-in adapter composing the strict raw learner. Pin all non-width fields
and training settings; validate a complete private payload before one assignment.
Adapt only fully coherent accepted gate evidence into a candidate replacement.
Preserve full-source refusals and spent runtime attempts. Reuse the existing
promotion evaluator/controller with whole-envelope digests and disjoint guards.

## Alternatives

Changing default raw restore would weaken existing fixed-configuration semantics.
Using effective tensors would lose raw scaling and continued-update behavior.
Changing the runtime or adding a codec is unnecessary for the current ports.

## Consequences

Complete raw/CPC state survives prune, continuation, commit and whole-bundle
rollback. Lineage grants no provenance or serving authority. The bounded example
demonstrates transactions, not scientific benefit or deployment quality. Existing
managed-retention refusal and strict ports remain; raw split/merge needs its own
derived mapping, precision and lineage contract before implementation.
