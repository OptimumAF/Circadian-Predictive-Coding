# ADR-0210: Coordinate conservative payload cleanup with original authority

Date: 2026-10-07. Status: implemented integration; full retention acceptance open.

## Context

The consent admission owner and replay/inbox erasure primitives are independent.
Current-candidate deletion misses retired candidates, prepared/failed checkpoint
models, pending observations and promotion/current/rollback bundles. Native
replay snapshots have no managed sample identities. Recreated managers, clocks
or budgets could renew grants and resurrect deleted copies.

## Decision

Install one opt-in coordinator on the fresh original manager/actor registry.
Inject trusted core-result ports from the outer NumPy composition factory.
Acquire the original manager, all-holder and paused sharing leases, prevalidate
all references and histories, revoke requested plus every delivered identity,
invalidate checkpoint/promotion/rollback authority, and clear whole raw replay
and inbox buffers plus auxiliary cache/metadata. Deduplicate native models and
inboxes by identity. Preserve parameters, policy, RNG, clocks, gates, work and
consumed IDs. Retired inbox ledgers freeze historical completed work at handoff.

Why this: whole-buffer conservative deletion is explicit and testable without
inventing unsupported replay provenance. On partial failure keep revocation,
invalidate tokens, stop candidates and require cleanup retry. Retry never
reopens training or refunds quotas. Logical age blocks retained-state access;
explicit `expire()` performs purge. Lifetime authorized ingress bytes and holder/
controller capacities do not claim total retained-byte or automatic elapsed-time
enforcement. Full original parent acceptance stays unchecked.

## Alternatives

- Erase only the current candidate: misses owned checkpoint/retired/rollback data.
- Rebuild owners after deletion: can renew original consent, IDs, ages and budgets.
- Add per-record native provenance immediately: larger model-state/interface change;
  unnecessary for conservative cleanup and deferred until a scoped need.
- Treat ingress bytes or holder counts as retained-memory bytes: incorrect because
  snapshots, failed preparations and auxiliary graphs can multiply payloads.

## Consequences

Other delivered records may be revoked when one subject requests deletion. Serving
metadata/cache is cleared and rollback is invalidated, while model parameters and
their learned influence remain. Caller copies, physical zeroization, opaque graphs,
durable consent recovery and total retained-byte/time guarantees are outside this
implemented integration. Keep unsupported categories denied. Next implement
aggregate owned-array accounting and pre-copy capacity/expiry checks with original
authority, preserving every failed and consumed allowance.
