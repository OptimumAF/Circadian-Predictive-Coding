# ADR-0016: Version a fixed-width capacity control with guarded sleep

## Context

The three-head fixed-feature route starts from equal tensors and widths, but
the circadian head can later split or prune units. Equal starting parameters
alone do not establish a capacity-matched comparison. The memory-enabled
route observes process RSS sequentially and cannot attribute that memory to
one head. A small control is needed before comparative claims.

## Decision

Add `vision_three_head_fixed_width_capacity_memory_v1` as a separate
fixed-data/epoch route. Require the predictive width, circadian starting
width, and circadian minimum and maximum widths to be equal. Require
`target_accuracy=None`, scheduled forced sleep within the epoch cap, an
executable sleep budget, and guard-based rollback. The same frozen backbone
and cached role features feed all three heads. Fixed bounds block splits
and prunes while a forced sleep can still apply chemical reset and
homeostatic work; the inner guard can still accept or roll back that work.

Before opening final test, verify equal initial and final head parameter
counts, unchanged circadian width, zero splits/prunes, and at least one
guarded sleep attempt. Return explicit `capacity_control` metadata with
initial and final per-head counts. Always enable the existing observed
RSS/CUDA telemetry and retain `feature_bytes` as a separate cache measure.
The adaptive-width and other matched routes keep their existing IDs and
default timing behavior.

## Alternatives considered

- Disabling sleep entirely would remove a central circadian mechanism and
  would not test whether the sleep path works under fixed capacity.
- Inferring capacity matching only from equal initial hashes would miss
  structural changes after training begins.
- Reusing the adaptive-width protocol ID would make fixed and expandable
  capacities look equivalent in exported results.
- Treating in-process RSS as head-attributable memory would ignore shared
  features, other resident heads, and allocator history.

## Consequences and open work

The local CPU fixture observes a real chemical-state change during a forced
sleep call with unchanged parameter count, no splits/prunes, and extra guard
evaluations. A paired adaptive-width invocation uses the same backbone and
feature hashes under its original protocol ID. The control is one small
random-feature run, not a statistical comparison or evidence that one head
wins. Observed RSS remains sequential and can miss transient allocations.
ADR-0017 subsequently added process-isolated memory observations. Repeated
capacity-matched runs and the full P1.8 fairness gate remain open.
