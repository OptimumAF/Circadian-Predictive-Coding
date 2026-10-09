# ADR-0217: Persist monotone coordinator metadata before worker actions

## Context

R3.5b2a validates metadata relationships and R3.5b2b observes original Windows
process identities, time and RSS. Neither establishes a durable authority record.
A worker checkpoint cannot independently attest its own original caps, ownership
or spent resources. Recovery must preserve charges after ambiguous work.

## Decision

Add a core authority record, transition validation and read/advance port. Implement
the port using existing stdlib SQLite in infrastructure with one private bounded
canonical metadata record. Use exclusive creation, existing-file opening,
DELETE/FULL policy, explicit immediate transactions and no automatic retry.
Compare the entire expected record and commit before returning success.

Require a surviving trusted coordinator's independent high-water witness and
registered-worker observations. Preserve original manifest, clock, start, caps
and cumulative charges. Handoff advances the owner generation only after an
ended-worker observation. An admitted native update remains uncertain; completion
and checkpoint publication require a future protocol.

Why this: SQLite supplies a simple existing transaction boundary without another
dependency. Conservative uncertainty prevents quota renewal while the native
completion and lifecycle codecs are incomplete.

## Alternatives

- A JSON file plus replacement lacks the chosen transaction/locking boundary.
- WAL adds another persistent state surface unnecessary for this single record.
- Reconstructing authority from worker checkpoint metadata loses independence.
- A new distributed lease service exceeds the local surviving-coordinator scope.

## Consequences

The metadata layer is independently testable using fake observations and real
local SQLite faults/contention. It does not implement a live lease, application
dispatch, native restore or coordinator restart. One adapter instance belongs
to one coordinator lane. OS process/crash and power-loss durability still need
separate budgeted validation. Hostile disk changes require authenticated external
anti-rollback; typed values and the private-file policy are not cryptographic
authentication. Full R3.5b2/R3.7/R3.8/G3 acceptance stays unchanged and unfinished.
