# ADR-0234: Reserve under the original source lease

## Context

Full composite capture needs retained payload admission while original owner and
source locks remain held. Public payload reservation reacquires a nonreentrant
gate. Current metadata observation already owns that gate but has no payload
reservation need.

## Decision

Add an internal expiring, thread-local reservation capability from the original
budget lease. Separate lifecycle source leasing from record observation, keeping
all original gates, holder pinning and relationship checks. Existing public
metadata capture still observes without charging.

## Alternatives

Releasing gates to reserve permits a mixed epoch. Reentrant locks would change
the existing refusal semantics. A new budget or a refund would renew original
authority. Direct writes to `_charged` would duplicate validation and hide the
admission boundary.

## Consequences

Reservation failures allocate no payload; admitted attempts remain charged even
on later failure. Capabilities expire and refuse cross-thread or changed source
use. No new instance fields, dependencies, native work or public interfaces.
This is a prerequisite only: original consent/elapsed/quiescence checks, complete
native state/alias inventory, parameter copy limits and full recovery remain
required in subsequent increments.
