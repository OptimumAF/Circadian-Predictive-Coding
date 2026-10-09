# ADR-0245: Preserve erased receipt lineage through the actual copy memo

## Context

Original observed deletion releases inbox/native payloads and keeps applied
receipts and tombstones. ADR0244 qualifies a payload-free original witness, but
complete checkpoint capture still requires applied count equal live pair count.
Simply relaxing the count would admit history without proving its origin.

## Decision

Require a complete disjoint union of original live pair and original erased
witness inventories. Keep the existing live source/label/consent checks. Bind
each erased receipt and tombstone through the actual admitted copy memo, with
detached identities, the same original admitted scalar data and complete scalar
diagnostic/time/order equivalence. Preserve the complete cursor sequences.

Charge copied erased metadata through the same original admission and permanent
capacity before every inbox-copy stage. No erased raw payload is copied. Reprove
originals after memo callbacks and captured/materialized identities and content
after final opaque resource/fingerprint ports. Prebuild both history map/storage
sets and publish them with plain assignments at the original trusted handoff.

Why this: separate pure schema/memo helpers keep application orchestration small
and preserve original authority. Prepared maps keep fallible allocation before
publication, consistent with the existing checkpoint and erasure transitions.

## Alternatives

Dropping erased receipts loses cumulative work. Treating tombstone keys or scalar
digest equality as origin grants retrospective lineage. Keeping erased raw arrays
defeats deletion. Creating fresh limits or admissions renews already spent work.
All are incompatible with the original acceptance criteria.

## Consequences

Mixed capture needs birth-enrolled observed erased history and actual copy-chain
evidence. Unobserved/default-off/mutated/late/exhausted cases refuse. Failed attempts
retain charges and permanent copied reservations. Unknown untrained tombstones and
broader native/cleanup families stay unqualified until separately implemented.
Native model and inbox schemas, lifecycle cleanup authority and scientific
protocols remain unchanged. This ADR describes implementation; acceptance needs
the recorded g2 gates and actual evidence.
