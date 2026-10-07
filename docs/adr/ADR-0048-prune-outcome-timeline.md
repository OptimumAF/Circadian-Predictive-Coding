# ADR-0048: Report prune requests, schedules, and removals separately

## Context

The legacy sleep result reports `pruned_indices`. NumPy may only mark
those positions for gradual removal; the unit can disappear during a
later wake update. Torch removes selected positions during sleep, and
its post-split selector may choose a child born in the same event.
Position counts alone therefore cannot say how many units were removed
or identify them after a shape change.

## Decision

`PruneOutcome` is an immutable set of three ordered stable-ID tuples:
validated requests (`proposed_neuron_ids`), marks applied for delayed
removal (`scheduled_neuron_ids`), and units actually removed during the
operation (`removed_neuron_ids`). Rejected selectors do not produce an
outcome. Executed sleep events on both backends expose one outcome;
skipped events leave the optional field empty. NumPy external proposals
return an outcome, and its `CircadianTrainResult` extends the existing
training diagnostic with removals finalized during that wake update.
`get_pending_prune_ids()` derives active marked IDs from NumPy's mask;
Torch always returns an empty tuple because its removal is immediate.

NumPy sleep captures proposed IDs before structural mutation, scheduled
IDs immediately after marking, and actual removals by comparing active
IDs before and after consolidation. This also catches an older pending
unit finalized during sleep replay. Torch maps its post-split prune
positions to IDs before masking, so a newborn child can be recorded as
both proposed and removed even though it appears in neither pre- nor
post-event active lineage. A unit can be both scheduled and removed in
one sleep if later consolidation finalizes its mark.

Legacy `pruned_indices` and application `total_prunes` counters remain
positional request counts for reproducibility. They are not redefined as
actual removal counts. New typed outcomes are available at the core
boundary; P3.10 owns richer application trigger/guard/duration telemetry.

Why derive outcomes from result boundaries: these are observation values,
not learning state. They add no model-owned counters or snapshot schema
change. Full-state restore still determines the next outcome from its
restored pending mask and stable IDs. NumPy topology validation now
rejects a marked mask with missing TTL, wrong dtype, or impossible
minimum-width reservation before restore or training.

## Alternatives and consequences

Renaming the old count would change report contracts and historical
comparisons. Treating a gradual mark as removal would overstate the
structural change. Keeping an unbounded removal journal in model state
would enlarge snapshots and complicate later rollback semantics.
Typed per-operation results retain exact IDs with no new algorithmic
updates. Application aggregates still require P3.10 before presenting
separate status totals in reports.

## Evidence

`tests/test_prune_outcomes.py` first failed on the missing event field.
It covers NumPy immediate, gradual, external, and wake-finalized
removals; Torch immediate and post-split child removal; skipped/no-op
events, rejected proposals, immutable results, snapshot continuation,
and malformed pending masks. The development log records the final
quality gate.
