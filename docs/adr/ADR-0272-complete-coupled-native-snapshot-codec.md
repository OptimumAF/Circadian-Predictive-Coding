# ADR-0272: Complete coupled native snapshot bytes through existing codec ports

## Context

New binary PC/CPC learners own complete in-memory snapshots, including precision
and CPC wake policy/chemistry/gates/count. Existing legacy byte codecs do not
cover this format. Actual recovery requires complete state before disk/crash
transactions can be tested; prior incomplete timing comparisons need no rerun.

## Decision

Add an explicit `CoupledCheckpointCodec` through existing `CheckpointCodec`,
`CodecBinding`, `CodecLimits` and canonical NumPy frame helpers. Encode every
supported native snapshot field with canonical versioned JSON and byte-preserving
float64 frames. Validate aggregate resource limits before array scans/copies/
decoding, and all exact types/shapes/fields, native policy/gates and caller-known
binding/digest before returning detached state. No model construction or IO.

Provide a bounded native PC/neutral/active round-trip and exact continuation CLI.
Keep native formats/updates and adapter rates/runtime authority unchanged.

## Alternatives

Parameter-only state would lose CPC wake history and alter continuation. Pickle
would introduce generic object loading. Float lists would obscure dtype/storage
and byte preservation. A new checkpoint framework would duplicate existing
bounded ports. None is necessary for this concrete native-state gap.

## Consequences

The new models can transport complete native snapshots with explicit schema,
size limits, corruption refusal and ownership. Unsupported strided/native/wire
formats fail rather than being silently coerced or partially saved. Digest and
binding consistency remain caller-provided evidence, not provenance or consent.
Disk durability, full runtime/lifecycle reconstruction, cumulative budgets,
single-owner restart and live serving recovery remain separate requirements.
This component alone cannot close R3.7 or support a scientific benefit claim.
