# ADR-0209: Enroll payload holders before coordinated deletion

## Context

Checkpoint views/prepared models, retired candidates, pending promotions and
rollback bundles retain payload copies. Current-candidate erasure alone cannot
prove removal or prevent restoration. Each owner already has an operation lock.

## Decision

Give each actor one weak ownership registry and automatically enroll supported
holders, retaining the registry across handoff. Configure live/lifetime holder
limits once; never refund consumed lifetime enrollments on collection. An internal
quiescence lease tries all current holder locks before reading retained references.
Metadata snapshots contain no raw owner or payload references.

## Alternatives

Caller-maintained lists can omit hidden copies. Strong holder enrollment would
itself extend data retention. Waiting in a new global lock order risks deadlock
with existing controller/candidate/serving operation orders.

## Consequences

Contention and incomplete initialization cause explicit nonblocking refusal.
Failed checkpoint prepared models remain discoverable; original preparation work
is still charged. Holder counts do not bound nested native bytes or retention time.
Original-authority deletion, quotas/lifetimes, all-owner cleanup and non-resurrection
remain R3.6b2b. No arbitrary graph/callback/caller ownership certificate is claimed.
