# ADR-0225 - Preserve supported circadian native array storage order

## Context

The retained r35b2e2 probe showed logical bytes preserved but F native weights
became C arrays. Native pruning creates F storage. Equal logical state alone was
insufficient for complete native round-trip qualification.

## Decision

Version the CPC wire to v2 with an exact C/F tag per array. Native snapshots stay
v2. Traverse/reshape/copy using that order;reject noncontiguous native arrays and
unknown/ambiguous order before payload allocation. Canonical C represents arrays
with both contiguous flags. Include tag bytes in original exact wire bounds.

Why this:explicit layout retains native storage without generic stride graphs,
new dependencies or changed learning behavior. Require fixed source-native
structural continuation and full current variant/corruption/resource gates.

## Alternatives

Keep C normalization:does not meet full native layout acceptance. Generic stride
serialization:wider unsupported graph/alias/byte-budget surface. Reject all F:
would exclude storage produced by the current native pruning path.

## Consequences

Old CPC wire-v1 is explicitly refused;preserved earlier artifacts are unchanged.
Backprop wire-v1 keeps its originally scoped normalization and needs separate
qualification. No implicit migration/live restore/admission;future support needs
an explicit schema version and ownership/resource controls. Limits bound payload
bytes,not total RSS/lifetime copies. No model,seed,metric or baseline tuning.
