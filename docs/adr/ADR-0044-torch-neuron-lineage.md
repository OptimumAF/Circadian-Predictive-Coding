# ADR-0044: Keep Torch head neuron identity through staged sleep

## Context

Torch sleep selects prune candidates after a potentially noisy split. Its
P3.4b2 preflight simulates that split on detached head tensors and RNG
before touching the live head. Stable IDs therefore need to follow both
the detached candidate and the live structural operations. The head's
in-memory snapshot lists fields explicitly, unlike NumPy's full-state copy.

## Decision

The circadian head owns aligned int64 tensors of active neuron IDs and
birth-parent IDs, plus a monotonic next ID. Initial neurons are IDs
`0..width-1` with no parent. Splitting keeps each source ID and appends a
fresh child ID whose parent is the source ID. Pruning applies the same
mask to IDs as to weights and chemistry. Parent references survive
parent removal; removed IDs are never reused. Sleep results continue
reporting positional indices for compatibility.

`get_neuron_lineage()` returns the same immutable `NeuronLineageSnapshot`
as NumPy. Training/sleep topology checks require width and device
alignment, int64 type, increasing unique IDs, valid birth-parent order,
and a monotonic next ID. Both lineage tensors are in the detached
preflight copy. The integer next ID is copied by value, so rejected
proposals do not allocate a live ID. Head snapshots are now format 2 and
include all three fields; format 1 is rejected rather than inferred.
The existing trained-state hash iterates every head snapshot field, so
lineage now contributes to that hash.

Why keep IDs separate from weights: identity should persist across index
shifts without entering prediction, gradient updates, split/prune scores,
or the existing learning metrics. A child's parent can be absent from
the active ID tensor and still remain its birth reference.

## Alternatives and consequences

Recomputing IDs from positions would lose ancestry when pruning shifts
positions. Keeping an external map would complicate detached planning
and snapshot restore. Two width-sized int64 tensors and one scalar keep
the metadata next to the adaptive state. Full lineage validation adds
one device-to-host condition read during the existing topology check;
this should be profiled if head training latency becomes material.
Historical records of removed children remain a P3.6 telemetry task.
Atomic sleep rollback and durable resume remain P3.7/P3.9.

## Evidence

`tests/test_torch_neuron_lineage.py` first failed on the missing API.
It then passed repeated split/prune and restore continuation, detached
parent/child post-split removal, rejected proposal without ID allocation,
invalid snapshot rejection, and trained-state hash coverage. The
development log records the final full gate.
