# ADR-0231: Bind complete lifecycle bytes to original capture

## Context

Coherent capture now includes every current lifecycle/owner/driver/copy/enrollment
field and 48 original live reference slots. Field values alone omit shared native
consent, provenance, policy and immutable key/anchor relationships. Live references
cannot become newly authorized objects through serialization.

## Decision

Use `LifecycleCheckpointCodec` through `CheckpointCodec[ManagedLifecycleCapture]`.
Require independently supplied complete owner/retention/driver policies, capture
bounds, source/policy/content digests, original capture and authority digest.
The explicit `lifecycle_full_v1` JSON envelope contains complete metadata, immutable
record/tuple alias indices and a presence/alias manifest for all original slots.
It contains no callable, lock, token, event, thread or native model object.

Validate wire length and independent content digest before parsing; reject duplicate
keys. Check the complete envelope, original policies/reference manifest, canonical
bytes, every field/scalar/sequence/string/count/clock/consent/quota relationship and
all alias targets before constructing typed metadata. Revalidate native records
after construction and return the original caller-supplied authority tuple.

Why this: explicit fields and alias records preserve the complete supported
observation without inventing renewed authority. Raw preflight prevents forbidden
or corrupt input from reaching record construction. JSON preserves finite Python
numeric types and signed-zero spelling in the canonical representation.

## Alternatives

Pickle or generic object serialization would traverse live authority. A scalar
summary or independently reconstructed policy objects would omit fields or aliases.
Allowing wire policy values to configure the decoder could enlarge original limits.
Constructing records before checking relationships would weaken resource refusal.

## Consequences

The original capture must remain available. This is a metadata byte component, not
portable authority, durable recovery, a model restore, a budget refund or evidence
that matching tags authenticate the source/owner. Original callback/gate identities
remain local. Fixed schema enum spellings do not consume caller identifier capacity.
Scalar identities are not authority; scalar types/bits and all immutable record and
tuple aliases are preserved. Composition with complete native/inbox/consolidation
components still needs its original exclusive owner and independent closure gates.
