# ADR-0596: reuse the live worker for public-context conventional policies

## Context

79demonstrates useful guarded adaptation synchronously. Existing live service
already has a finite worker, original serving-priority admission and drained
publication, but generates v2 queries without public context and reads the legacy
native_updates result field. Replacing that runtime would duplicate accepted work.

## Decision

Add optional strictly immutable per-opportunity public context tasks to the
existing plan and alias the contextual complete update count to the existing
readout port. An explicit existing CLI mode composes the original reference
schedule, worker, gate and promotion controller. Preserve default requests and
native routes. Record applied receipts in the existing schedule's failure finally.

## Alternatives and consequences

A new scheduler is unnecessary. A forced sleep or larger kernel solely to get
overlap changes the experiment without improving the learner. Measure actual
wall-clock request/model-call intersections and preserve zero as valid. The one
case has zero intersections despite two overlapping admission leases. It verifies
integration and permissions, not inference during model execution or speedup.

All postpublication worker tasks succeed and rollback restores source behavior.
This reuses established79 learning, not another learning-benefit claim. Fixed
controller comparisons on a demonstrably learnable policy are a more substantial
next outcome than chasing timing or another provenance increment.
