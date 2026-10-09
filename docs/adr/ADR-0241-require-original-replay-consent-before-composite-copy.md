# ADR-0241: Require original replay consent before composite copy

## Context

Composite capture validated inbox consent but allowed synthetic replay without
observed producer identity. Capture already holds nonreentrant owner/runtime and
sampler leases, so calling the public ledger API would reacquire those locks.

## Decision

Require a bounded replay inventory before projection/copy. Nonempty original
candidate rows require their exact ledger; other-holder rows refuse pending original
copy-lineage qualification. Use a ledger-only nonblocking lease and original
capture resource checks. Repeat identity/integrity/consent checks around callbacks.
Spend capture attempts from the existing ledger lifetime invocation/metadata
allowance. Preserve failed charges and all original native fields and policies.

Why this: content equality cannot distinguish subjects or establish copy lineage.
The synthetic replay positive test becomes an explicit missing-origin refusal.
Supported empty-replay aliases remain positive; actual trained-row capture receives
separate bounded native integration evidence.

## Alternatives

- Optional bypass: leaves the same missing-origin capture gap.
- Public ledger reads: reenter original owner/runtime/resource leases.
- Bless copied rows by matching hashes: supplies no original producer/holder witness.
- Add native origin fields: changes existing snapshot/schema contracts.

## Consequences

Replay capture is stricter. This increment grants no retained-holder lineage,
serialized provenance, restore permission or scientific admission. Further actual
holder creation contracts and complete variant/history/recovery gates remain open.
