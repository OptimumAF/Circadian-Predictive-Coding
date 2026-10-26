# ADR-0304: Read original issued checkpoint witnesses during complete capture

## Context

Constructor-copy replay qualification does not prove snapshot lineage. The
existing checkpoint manager records actual native and inbox graph copies and
weak original token/view/state/cursor witnesses. Its public methods reacquire
leases already held during complete capture.

## Decision

Add optional replay_checkpoints to the existing app/NumPy capture boundary.
Require its original ledger and original copies; derive copies only when the
argument is omitted. A small app reader holds the existing nonblocking manager
gate and reuses transient operation validators and original witness proofs.
Every inventory round checks all advertised roles, original issued membership,
complete state/inbox content, rows/receipts/consent and paid history. Only the
original checkpoint snapshots role with its actual issued view qualifies.
Borrowed roots clear on exit. Existing single admission and raw copy charges
remain authoritative, including after late failure.

Why this: reuse the actual producer's evidence without public-method lock
reentry, additional admissions, native methods or content-inferred provenance.

## Alternatives

Calling public origins/capture/restore would reenter leases or perform native
work. Matching keys or bytes would not prove original ownership. Constructor
witnesses alone have no authority over snapshot lineage.

## Consequences

Defaults stay unchanged. This branch captures data from unchanged original
runtime/history only. Later-current history and successful restoration,
publication, other roles, erasure/compound and durable recovery remain open.
Focused controls, scoped static gates, independent review and one prospectively
bounded actual nonempty snapshot integration are required before acceptance.
