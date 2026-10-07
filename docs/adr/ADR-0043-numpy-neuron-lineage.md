# ADR-0043: Keep NumPy adaptive-neuron identity across topology changes

## Context

NumPy split/prune results reported current tensor indices. Those indices
shift after pruning, so repeated growth could not identify a surviving
unit or its split parent. P3.4 requires stable identity and lineage
without changing the learning or evaluation metrics.

## Decision

The adaptive layer owns aligned integer arrays of active neuron IDs and
birth-parent IDs, plus a monotonic next ID. Initial units are IDs
`0..width-1` with no parent. A split keeps the original unit's ID and
appends a fresh child ID whose parent is that original ID. Immediate and
finalized gradual pruning apply the same mask to lineage as to weights
and chemistry. An active child's parent reference remains even after
the parent is pruned. Removed IDs are never reused.

`get_neuron_lineage()` returns a read-only `NeuronLineageSnapshot` with
active IDs, optional parent IDs, and the next assignable ID. Both
external proposals and built-in sleep use the existing mutation methods,
so they share the identity rule. Topology validation checks alignment,
integer type, uniqueness, parent order, and monotonic allocation.
The NumPy in-memory full-state format is now version 2 and includes all
lineage fields; a version-1 snapshot is rejected rather than guessed.
Active gradual-prune training rollback also copies lineage arrays.

Why retain parent IDs after parent removal: a surviving child's origin
must not change when active tensor indices shift. No historical record
of removed child units is exposed yet; P3.6 defines proposed, scheduled,
and actually removed telemetry separately.

## Alternatives and consequences

Using tensor indices as identities would make lineage unstable after
prune. Reassigning compact IDs would lose ancestry. Monotonic integer
IDs add only two width-sized arrays and one scalar. They do not enter
prediction, training gradients, structural selection scores, or metrics.
Torch adopts the same read-only contract under P3.4c2.

## Evidence

`tests/test_numpy_neuron_lineage.py` first failed on the missing lineage
API. It now checks repeated child-of-child splits, a surviving child
whose parent is pruned, never-reused IDs after index shifts, gradual
prune finalization, exact snapshot continuation, and rejection of a
duplicate-ID snapshot. The development log records the full gate.
