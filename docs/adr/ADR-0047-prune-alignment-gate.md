# ADR-0047: Verify pruning against every adaptive array

## Context

Pruning changes hidden width across weights, chemistry, traffic,
importance, age, cooldowns, and stable neuron IDs. NumPy can defer
removal while a marked neuron decays over wake updates; Torch removes
selected neurons during sleep. A position-only test could miss a
metadata shift that later changes learning or selection.

## Decision

P3.6a tests distinguishable values in every per-neuron array, then
verify that immediate pruning keeps exactly the survivors in NumPy and
Torch. It also checks incoming/outgoing/bias dimensions, prediction
shape, and successful wake/sleep clocks. For gradual NumPy pruning, a
pending mark retains its ID and width, consumes minimum-width capacity,
and advances a time-to-live only on successful wake work. Finalization
removes the same ID from all arrays. Torch has no gradual path and is
checked at its minimum width separately.

Why check clocks: a sleep request that only schedules pruning is still
an executed sleep, while final removal during wake should count as wake
work. Neither event should be inferred solely from a changed width.

## Alternatives and consequences

Checking just weights or output dimensions would not detect shifted
chemistry, cooldown, or lineage. Adding new mutation code before finding
an alignment failure would increase risk without evidence. These tests
validate the existing masks and selection capacity; they do not change
metrics or selection. P3.6b will add typed proposed, scheduled, and
actually removed outcomes because the legacy `pruned_indices` field
alone cannot express delayed removal.

## Evidence

`tests/test_prune_metadata_alignment.py` passes NumPy immediate and
gradual and Torch immediate/minimum-width cases. The development log
records the final quality gate.
