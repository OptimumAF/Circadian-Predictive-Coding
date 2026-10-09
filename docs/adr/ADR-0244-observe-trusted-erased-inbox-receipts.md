# ADR-0244: Observe trusted erased inbox receipts before payload removal

## Context

Original inbox cleanup retains applied receipts and payload-free tombstones.
Optional birth-observed history previously pruned the weak live pair witness,
so a later complete capture could not independently certify erased receipts.
Current applied fields or matching tombstone keys cannot establish provenance.

## Decision

Add optional `ManagedReplayOrigins.delete` observation around the existing
original lifecycle deletion. Preflight requires original committed live pairs;
unsupported pending history refuses before lifecycle mutation. An operation-local
scope binds the exact original deletion caller, thread and Context token.

The actual original inbox commit prepares an exact metadata transition before
removing payload references. It compares the originally sealed source/label/
receipt history to actual prepared tombstones and charges new witness metadata
through the same original replay admission before allocation. Publication after
the existing payload removal installs only prepared plain maps and flags.

Erased witnesses are fixed scalar metadata plus weak exact receipt/tombstone
references and content seals. Complete replacement maps/storage are allocated
before removal; the original history/admission objects remain authoritative.
Verification checks complete erased applied coverage, actual current
receipt/tombstone identities and original spent work/revocation history.
Unobserved erasure refuses before pruning its missing lineage; no retrospective
enrollment or refund is issued. Inactive observation changes no persistent inbox
or lifecycle schema and leaves their existing cleanup behavior intact.

## Alternatives

- Relaxing the applied count to live count would silently omit erased history.
- Keeping erased raw source or payload objects would extend their lifetime.
- Reading current receipts at capture would create retrospective authority.
- A generic post-removal callback can fail after deletion and grants an unrelated
  extension point. Use an exact prepared transition instead.

## Consequences

Quota refusal during preparation follows existing failed/stopped cleanup behavior;
revocation and any prior native erasure remain real, and charges are not refunded.
Retrying original cleanup does not grant missing lineage. These failure families
remain broader unfinished criteria.

This g1 prerequisite leaves complete mixed checkpoint capture conservative.
The next g2 must copy and rebind both disjoint live and erased receipt partitions
through the actual memo chain and final proof before handoff. Full g and all
ancestors remain unchecked until their original acceptance criteria pass.
