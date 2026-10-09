# ADR-0223: Encode explicit complete native components

## Context

CandidateCheckpoint is a trusted in-process transfer with retained owners,
callbacks and original resource identities. Its pickle integrity digest never
loads caller bytes and cannot supply a durable complete restoration contract.
Lifecycle and actor metadata views omit native payload/reference edges.

## Decision

Introduce a typed inner byte-codec port with independent source/policy bindings,
an independently expected byte digest and explicit limits. Implement backprop
first using an exact complete native dictionary schema, float64 frames and its
two required aliases. Refuse missing/future fields and undeclared shared storage.
Keep full R3.5b2e criteria unchanged while adding component task R3.5b2e1.

## Alternatives

Generic pickle can construct arbitrary objects and hides schema/ownership
omissions. Scalar metadata and absent markers omit complete native state.
A universal tagged object graph creates unnecessary extensibility and authority
risk before supported native/domain variants have explicit contracts.

## Consequences

The backprop component preserves every native field and value, including traffic
and aliases. It adds no learning equations or dependency. Unknown future fields
require an explicit new schema. Decoded arrays own detached contiguous storage;
arbitrary original strides are normalized while logical bytes are preserved.
Learning policy and source ownership remain independently supplied. Codec size
limits do not reserve original lifetime copy allowances or grant live restore.
Complete circadian/domain/lifecycle/actor/sharing codecs and original single-owner
admission are still prerequisites for full recovery acceptance.
