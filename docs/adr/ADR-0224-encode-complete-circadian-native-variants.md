# ADR-0224: Encode complete current circadian native variants

## Context

The native v2 snapshot contains 36 base dictionary fields, 73 configuration
fields, a generator, replay deque, structural lineage and optional retention,
exposure and wake-only policy state. A partial scalar checkpoint cannot restore
its native behavior. Generic object deserialization would grant unsupported
types and hide missing state.

## Decision

Implement the inner CheckpointCodec port with three focused adapter modules:
bounded explicit NumPy frames, frozen circadian schema validation, and byte
orchestration. Cover all eight variants constructed through current native
configuration methods. Serialize every parameter, chemical/traffic/age vector,
lineage/mask/cooldown array, counter/history, replay row and policy field.

Use exact PCG64 state including its 128-bit state/increment and cached uint32
fields: this is the generator constructed by the current native model. Refuse
unknown generator implementations, fields and additional shared array storage.
No native interface or learning equations are changed. Additional supported
native variants require explicit schemas and controls before broader recovery.

Require independently expected source/policy/content digests. Validate complete
wire/array sizes and every declared frame before decoding native arrays. Preserve
original snapshots on failure, reconstruct independently owned arrays/deques/
sets/generator, and validate existing native topology/configuration/replay
contracts. Finite float64, int64, int32 and canonical boolean frames have explicit
shapes; the wire uses little endian values and C traversal order.

## Alternatives

Pickle is unsuitable for caller-controlled durable bytes. Generic graph tags
hide schema omissions and add extensibility before ownership rules are proven.
Metadata, missing-native markers and summaries omit native continuity state.

## Consequences

Current complete native variants can round-trip as bytes and continue identically
in fixed fixture-local native controls. Unsupported objects fail explicitly.
Schema changes require a version change; frozen config/native field sets prevent
automatic acceptance of future fields. Strides are normalized and logical values
are preserved. Immutable config records are represented by value; mutable native
storage is independently owned. Matching digests are integrity comparisons,
not source closure, consent, ownership or restoration authority.

Full R3.5b2e still requires complete inbox/consolidation/lifecycle/actor/sharing
codecs and original cumulative copy/resource/owner admission. No disk/live runtime
restore or completed native work is granted by this component.
