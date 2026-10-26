# ADR-0527: Compare explicit native projection with the reset-only control

## Context

The owner can project live LoRA operators and reset AdamW once. Literal
compatibility and algebra checks establish state behavior; they do not show
whether projection helps learning. Existing native A/B results include an
accepted reset-only control and original warm history. Comparing projection
only with warm history would conflate two interventions.

## Decision

Extend the existing optimizer-control stream with default-off projection rank
and a fixed accepted reset-only reference. Preserve the original A6/B6 dose,
delayed labels, source/after-A bindings, serving interface and metrics. Project
once after A6, consuming the existing reset authority inside that operation.
Initially exclude rehearsal and update-health combinations.

Why this: one frozen rank-four case answers a concrete integration and task
question without another runner or rank search. The primary signed endpoint
comparison is projection minus reset-only; warm and original baseline are
secondary historical references. All negatives remain evidence.

## Alternatives

- Refit controls: unnecessary while accepted source and assumptions remain valid.
- Compare only with warm history: confounds projection with resetting moments.
- Search ranks or doses: outside this frozen local protocol.
- Shrink stored Parameters: unnecessary for this behavior and would invalidate
  ownership bindings and the declared native training recipe.

## Consequences

Allocated rank stays eight. Projection caps the immediate operator at four;
dormant coordinates can regrow under B training. Additional factor algebra,
copies and reset work are charged, so equal TRAIN counts do not mean equal
compute. Historical controls are nonconcurrent. Neither software correctness
nor one development endpoint establishes generalization, return recovery,
calibration, persistent compression or CPC benefit.

[Protocol and observed evidence](../../artifacts/runs/native-projection-stream-20261022/PROTOCOL.md).
