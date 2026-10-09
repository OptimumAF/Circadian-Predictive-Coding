# ADR-0233: Bind paired bytes to the complete original observation

## Context

Common-interval capture now retains complete consolidation/lifecycle metadata,
runtime observations and original references. Independent scalar counters do not
exclude a lifecycle or driver epoch from another capture. Separate component
encoding would also lose shared tuple identities across their metadata graphs.

## Decision

Encode one complete explicitly described metadata graph and its immutable alias
graph through the existing inner checkpoint codec port. Require independently
supplied original complete capture, policies, source/content/authority bindings
and all original reference identities. Compare the whole original metadata and
alias graph before wire materialization, after complete raw bounded preflight.
Reuse the lifecycle walkers through trusted internal descriptors with unchanged
defaults. No wire value selects a callable, schema or record constructor.

Why this: a complete original observation prevents partial counter matching from
admitting spliced epochs or refunded charges. One graph preserves cross-component
aliases. Original live references remain separately supplied and unencoded.

## Alternatives

- Bind only revision and update counters: permits different elapsed observations.
- Encode independent lifecycle/consolidation frames: loses shared metadata aliases.
- Serialize live authority objects or addresses: neither portable nor renewed
  original ownership; matching tags cannot authorize model restore.

## Consequences

Every field/type/bit/alias/history and original policy/ref slot is retained.
Strict complete original observation matching requires a new independent codec
configuration when a legitimate original capture changes. Encoding/decoding has
no model, clock, callback, IO or worker operation. Identical portable observations
cannot attest live provenance; original authority remains required. Full native/
inbox/actor/sharing/ownership/live/disk/coordinator loss and scientific/human
criteria remain unfinished and are not waived by the component format.
