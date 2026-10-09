# ADR-0240: Track original candidate replay consent with bounded weak records

## Context

Native replay snapshots contain arrays and scalar descriptors but no original
subject or source identity. Producer/write callbacks expire and completed update
callbacks precede final resource checks. Adding fields to native models would
change existing snapshot and codec contracts.

## Decision

Use a separate opt-in ledger enrolled on a fresh original managed candidate with
empty replay. Observe original producer and actual native copy identities, charge
bounded metadata before copy, and store only weak references to raw graph objects.
Wrap the existing shared operation under the original managed owner gate; verify
original receipts, retained inventory, consent and final budgets under the runtime
gate before publishing row metadata. Failures after an admitted start leave the
ledger uncertain while preserving original exceptions, work, receipts and charges.

Why this: identity establishes origin at the actual producer/copy boundary.
Integrity hashes check later changes to those objects. Deterministic metadata
accounting is documented separately from physical heap/RSS measurements.

## Alternatives

- Content matching: cannot distinguish identical inputs with different consent.
- Strong row payload retention: requires additional raw ownership and accounting.
- Native schema fields: would alter otherwise working snapshot/codec contracts.
- Certifying the completed callback: can precede a committed post-update failure.

## Consequences

Native defaults and schemas remain unchanged; metadata charges never renew or
refund. The new ledger is for a current candidate in this process only. Injected
ports are trusted observational functions. Fork/actor/checkpoint/promotion/restore/
erase lineage and all recovery qualification require additional explicit contracts.
This decision provides no durable provenance, unlearning or scientific admission.
