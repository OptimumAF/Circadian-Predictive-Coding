# ADR-0037: Copy the complete NumPy circadian model state in memory

## Context

NumPy circadian sleep had no snapshot/restore API. Its state includes
coupled adaptive-width tensors, earlier hidden-layer tensors, dual
chemistry, traffic, importance, ages, cooldowns, pending gradual prune,
reward baseline, adaptive history, replay batches/priorities, a local
random generator, and P3.2 work clocks. Copying only weights would not
restore future split draws or replay behavior after a rejected event.

## Decision

`CircadianPredictiveCodingNetwork.snapshot_state()` returns a versioned
`CircadianNetworkSnapshot` with a deep copy of all model-owned attributes.
The snapshot carries static input, initial hidden dimensions, width
bounds, and immutable configuration for compatibility checks. The
in-memory state copy includes the NumPy generator and replay deque with
copied arrays. `restore_state()` deep-copies again, verifies format and
static compatibility, checks field coverage and candidate tensor
alignment, then replaces the model state in one commit. Snapshot arrays
can be changed after restoration without mutating the model. A snapshot
created before split/prune can restore the earlier width and its
per-neuron arrays.

Why copy the complete attribute dictionary: model-owned state is
distributed across many coordinated arrays and counters, and the
complete copy prevents a newly added field from being silently omitted.
The format is an in-memory Python object, not a durable checkpoint
schema. Versioned file serialization, strict replay-payload validation,
and incompatible checkpoint diagnostics remain P3.9.

## Alternatives and consequences

An explicit hand-maintained list of arrays would make additions easy to
miss. `deepcopy(model)` alone produces a second model but does not offer
an explicit restore/compatibility boundary. The chosen copy costs memory
proportional to model tensors and replay batches. It is suitable for the
local correctness gate; larger-data rollback memory remains to be
measured under P1.8/P3.7. It does not itself make sleep atomic or log
rollback attempts.

## Evidence

The first snapshot test failed because the NumPy model lacked the API.
Focused tests restore topology, pre-hidden and adaptive tensors, dual
chemistry, active gradual-prune metadata, replay payloads/priorities,
reward and clock state, and the next model-owned random split. Mutating
the saved artifact after restore does not affect the model. Wrong
version/config or corrupt tensor width is rejected before mutation. The
development log records the full quality gate.
