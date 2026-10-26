# ADR-0376: Share a physical quota across fresh reward episodes

## Context

The action-mapping qualification spent 94 physical steps under eight episode
limits of 12: one early terminal episode left two steps unused. Equal episode
limits do not supply equal realized exposure for future baseline comparisons.
The existing actor/reward runner already owns delayed receipts, source isolation,
private RNG, native state, clock, training budget and deferred final scoring.

## Decision

Add a small core PhysicalInteractionQuota with a fixed maximum and attempted,
completed and failed counts. Charge before physics; retain failed attempts and
stop further work. Permit one pending attempt and expose detached frozen numeric
snapshots. No history, IO, model, clock or reset operation belongs to this class.

The existing collector optionally uses the remaining quota as its episode
horizon, records before/after snapshots and checks exact expected progress across
callbacks. Disabled behavior stays unchanged. The existing reward loop fills
eight 12-step phases using fresh episodes and the original reward objective.
Each segment has its own update, receipt, behavior state and fresh recurrent
session. All segments share one original 16-update budget, clock and deadline.

The explicit scenario captures physical boundaries 0/24/48/72/96 together with
actual variable native update indices. Signed outcomes use the physical mapping;
native indices are never renamed. At most two segments per phase fit this fixed
game (terminal paths need at least seven steps); unexpected noncompletion fails
closed. Final data remains unavailable until every phase and original preceding
success validates. Keep one sealed active collection and bounded prior records.

## Alternatives

Resetting budgets at each episode would hide paid work. Padding with teacher data
would change each policy's experience. A new orchestration pipeline would duplicate
the existing runtime and guards. Combining episodes into one recurrent objective
would silently alter resets and gradient reduction. Reuse the existing loop and
single-episode update contract instead.

## Consequences

Successful collection has exactly 96 physical TRAIN steps but 8–16 optimizer
updates and variable resets; equal interaction is not equal computation. Preserve
all costs, negative outcomes and failed attempts. Unknown counter changes refuse
work; they are never ignored or refunded. Old defaults and protocols remain.
This prerequisite does not establish learned mapping or a circadian advantage.
Full R4.3 still needs repeated environment instances/seeds and matched policies
collecting their own trajectories under a separately frozen protocol.
