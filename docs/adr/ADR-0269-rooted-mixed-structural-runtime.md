# ADR-0269: Rooted mixed raw structural runtime

## Context

Raw pruning and nonlinear dyadic split/compatible merge are verified separately.
The existing prune topology adapter pins precision, while split changes it.
Existing generic candidate and independent serving ports already support owned
payload replacement, guards and rollback.

## Decision

Add a separate full-origin-bound structural envelope/adapter with at most64
exact tagged full-configuration transitions. Split and merge use their accepted
core maps; pruning delegates to the existing gate through an internal temporary
envelope, retaining the full mixed history on acceptance. Validate complete private
payloads before publication and reuse existing runtime/serving boundaries.

## Alternatives

Relaxing the prune-only adapter's precision pin would admit unbound configurations.
Erasing history when crossing operation types loses state context. Treating every
merge as a historical inverse incorrectly admits post-prune geometry or implies
recovery of old weights. A new runtime/controller or hashing system is unnecessary.

## Consequences

Mixed supported config moves are explicit and owned, with complete CPC state and
fixed learning policy. Invalid inverse attempts refuse before kernels, preserve
history and spend attempt IDs. Metadata is not execution proof or serving
authority. Mathematical/local damage guarantees and scientific limitations remain
those of the primitive operations. Existing defaults and managed-retention refusal
remain; serving still requires independent guards and complete-bundle rollback.
