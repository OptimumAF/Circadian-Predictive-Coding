# ADR-0282 — Delayed feedback through serving transitions

Date: 2026-10-15

## Context

R3.7 includes stale labels. The accepted cache transaction control applies its
only label before promotion; native label chronology across actor transitions
still lacks integrated evidence. Existing inbox pairing binds a label to its
producing experience version, while registration spends runtime revision before
validation. Promotion and rollback publish complete independent serving bundles.

## Decision

Add one finite native integration for PC/neutral/active CPC. Promote a split
candidate before its delayed label arrives; register valid original-version
feedback while the new actor serves; wait until causal arrival and apply once
under the original allowance. Verify that candidate learning preserves the
promoted actor's full bundle, then rollback restores the exact previous bundle
without undoing candidate work or renewing quota.

Use poison targets to prove metadata rejection precedes copying/native work.
Track the spent registration revision separately from preserved native/inbox/
serving/budget state. Reuse existing app/core APIs and fixed disjoint inner guards;
keep all data synthetic and observations local and bounded.

## Alternatives

Rejecting every old-version label would discard valid delayed feedback. Rewriting
producer version to the current actor would corrupt provenance. Replacing inbox
or transaction APIs would duplicate working behavior. A unit-only scalar fixture
would omit complete CPC state and structural history across real publication.

## Consequences

Native causal pairing and serving isolation become runnable and observable. The
integration makes no comparative learning, external provenance, durable recovery
or indefinite service claim. Future scenarios must preserve original identity
and resource semantics rather than treating rollback as a fresh training budget.
