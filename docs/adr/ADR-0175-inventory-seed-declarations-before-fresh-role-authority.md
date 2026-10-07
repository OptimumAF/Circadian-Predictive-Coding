# ADR-0175: inventory seed declarations before fresh-role authority

## Context

P6.7d2a cannot certify five-point precision within ten-seed fixed work limits.
The remaining d2b must audit all prior usage and bind untouched roles before
new source access. Historical execution/scoring contracts deliberately bind
already scored seeds. Reusing them as fresh contracts would misstate chronology.
The source, split, initialization and selector RNGs also use distinct derived
offsets; a base-seed list alone does not establish independent untouched streams.

## Decision

Build a complete retained JSON declaration inventory first, preserving every
physical file identity, declaration pointer and unresolved/unparsed case. Handle
regular/canonical fields, embedded configuration JSON and CLI arguments without
IO in core. Infrastructure owns complete membership and whole-byte checks.
Reject changed/subset catalogs, late corruption, unowned exclusions and external
paths. No source, model, RNG, score, winner or seed selection occurs.

Why this: archived snapshots and escaped manifests can contain the only preserved
seed declarations. Ignoring them or treating failed metadata as an absence would
weaken the usage audit. A declaration does not by itself prove actual execution.
Keep fresh-role and complete prior-usage acceptance false until later evidence.

Split d2b into d2b1 full retained JSON metadata, d2b2 complete remaining usage and
untouched-role contract, and d2b3 full execution fixtures. Preserve every original
d2b/d2/P6.7 criterion. Bind explicit local metadata caps before operations.

## Alternatives

- Select fresh base seeds from six current family manifests alone: misses old
  profiles, failed requests, other reservations and derived stream overlap.
- Scan a few representative requests: cannot prove complete retained coverage.
- Interpret null/symbolic/malformed declarations as unused seeds: hides uncertainty.
- Rewrite the historical confirmation contracts: invalidates accepted provenance.

## Consequences

The JSON inventory can complete with visible unresolved cases, but does not grant
complete prior-usage or fresh-role authority. D2b2 must finish text/source/default/
derived-stream/role-release evidence and handle every ambiguity before declaring
new independent seeds. The full future gates and original scientific acceptance
remain unchanged and unfinished. See [scope and evidence](../p67-seed-usage-inventory.md).
