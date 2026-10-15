# ADR-0293: Preserve live managed consent through coupled candidate handoff

Date: 2026-10-15. Status: accepted design; validation tracked in R3.7w.

## Context

Complete native learner handoffs and original resource limits are covered. The
existing managed owner already enforces declarations, consent and opt-out, while
candidate replacement retains its bound inbox guards. A native integration can
demonstrate these contracts together without changing authority or learner rules.

## Decision

Add a fixed six-case adapter and mutually exclusive `--managed-consent` CLI mode.
Install one managed owner before registration. Declare three synthetic local
records for subjects A/B/A, apply the first, capture, then opt out A. Require the
stale token to fail before preparation. Discard it and transfer a fresh checkpoint
with the same consent owner, catalog, sets, guards, actor and resources. Train B
only; retain A's future raw pair but leave it ineligible with capacity remaining.

Why this: composing the existing admission owner proves continuity using the
actual runtime transaction and avoids a duplicate consent or recovery framework.
Compare full native state, policy, diagnostics and predictions to an independent
uninterrupted reference. Preserve PC-neutral controls and genuine CPC wake state.

## Alternatives

A new managed owner would reset authority and lifetime quotas. Exhausting update
capacity would conceal whether consent filtering works. Deleting raw records or
adding lifecycle-copy ports would change the selected admission-only contract.

## Consequences

Held-out, unmanaged and opted-out ingress must refuse before opaque payload copy.
Failed runtime registration legitimately increments the runtime revision; other
state remains equal. Opt-out also changes the permanent subject set and revision.
No native algorithm, dependency or app/core API changes. Raw retained records and
checkpoint copies are not erased; this is future-use blocking, not deletion,
unlearning, attested provenance, human consent or managed process recovery.
