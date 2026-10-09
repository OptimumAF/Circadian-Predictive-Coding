# ADR-0232: Capture paired records under one original interval

## Context

Standalone lifecycle and consolidation captures use different intervals. Calling
the public consolidation capture during lifecycle capture would reenter an
already held nonreentrant candidate gate. PromotableActor.version similarly
reacquires its actor read gate. Separate observations cannot establish a coupled
original record state.

## Decision

Expose minimal internal leased read helpers; preserve standalone public capture
signatures and behavior. Capture complete paired metadata plus runtime/inbox
observations and original live references under the full original lifecycle lease
order. Use exact source schemas, joint record/string capacity and relationship
validation before one metadata-only deepcopy. Read the promotable slot directly
while its original actor lease is held.

Why this: reuse of the existing lease order avoids competing ownership rules;
one metadata graph preserves aliases without copying live authority. Capture and
explicit byte encoding are separate acceptance increments c1/c2; their parent
criteria remain intact and unfinished.

## Alternatives

- Compose independent snapshots: does not establish a shared interval.
- Use reentrant candidate/actor locks: changes existing contention behavior.
- Run a consolidation transform to obtain receipts: the installed managed
  lifecycle explicitly forbids arbitrary transforms. Preserve that negative
  result and use synthetic full-history records for schema controls.

## Consequences

Supported capture performs no clock read, native/measurement callback, budget
refund, cleanup, worker operation or authority renewal. Busy, unknown and mixed
records are refused. Original ticking domains, copy charges, quotas and references
survive observation. Metadata validation does not attest live capture provenance;
paired bytes, complete native/inbox composition and restore remain future work.
