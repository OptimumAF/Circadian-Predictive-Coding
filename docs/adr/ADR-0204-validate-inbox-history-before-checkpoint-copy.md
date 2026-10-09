# ADR-0204: Validate complete inbox history before checkpoint payload copy

## Context

Native snapshots alone omit future labels, duplicate IDs, applied receipts and
arrival/stopped cursors. Reopening an incomplete checkpoint could replay work or
reset budgets. The inbox has no validated snapshot contract, while arbitrary copy
callbacks can access payloads before held-out/corrupt metadata is refused.

## Decision

Introduce a pure exact-version cursor record containing all source/label/applied
histories and observed work/time/stopped metadata. Revalidate roles, permissions,
identity/pair/reference/number/time/diagnostic/capacity/count fields before payload
copy. Capture from a quiescent owner, check dictionary/event indexes, then return a
detached record. Unknown input passes through an exact type/format validator.

## Alternatives

Weights-only capture loses inbox ownership/history. Pending-only capture loses
consumed IDs. Copy-first validation can access forbidden payloads. A standalone
public resume constructor would require native/budget/clock/resource ownership and
retirement rules that this prerequisite has not implemented.

## Consequences

Complete owned inbox information is available for the next atomic checkpoint
handoff. Current model/budget/clock ownership remains unchanged. Stopped captures
describe poison/consumed work and grant no retry capability. Serialization and
physical source/security authority remain outside core. R3.5b remains unchecked
until complete native/candidate/history/budget/sharing restoration and old-owner
retirement are proven; R3.5c's live serving measurements remain subsequent work.
