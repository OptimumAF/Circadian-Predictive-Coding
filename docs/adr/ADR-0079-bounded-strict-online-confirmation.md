# ADR-0079: Freeze a bounded strict-online A→B confirmation before final access

## Context

The v7 ordinary and format-7 checkpointed selection paths have source and
label arrival, inner/outer/final role separation, bounded observed replay,
and a global selection freeze. Their tests prove the individual boundaries,
but P1.3c4 requires one persisted end-to-end A→B comparison. The earlier
fixed smoke used two candidates and seeds 17/19 and showed a negative
circadian final outcome on seed 17. Changing seeds or the objective after
that observation would invalidate the confirmation.

## Decision

Use the same two candidate rates, seeds 17/19, one epoch per phase, 40
source examples per phase, width 4, two inference steps, component sleep,
and replay caps of four examples/96 array bytes. Save the complete request
under `data/` before any confirmation final-test source read. The request
fixes the outer-only mean balanced objective, first-candidate tie rule,
forward/reverse model orders, A and B wake interruption points, required
outcome fields, 96 total model updates, and a 180-second local limit. The
runner rejects a missing or changed request and existing result/checkpoint
paths. It records every outer trial, role/exposure/task/replay ledger,
selection, final seed metric, and signed circadian-minus-baseline difference.

In each order, run one ordinary and one checkpointed comparison. Interrupt
the checkpointed comparison after the first A wake save and the first B
wake save, then resume from the same format-7 file. Assert the global final
source remains sealed, the checkpointed result and fieldwise model-state
digests equal ordinary execution, and method outcomes and model states do
not depend on training order. Report the observed result even when the
circadian method loses. Keep the existing v1/v2 offline results separate.

## Alternatives

- Pick new seeds or rates after looking at the earlier smoke. This would
  turn confirmation into test-informed tuning.
- Run a larger search before the end-to-end gate. The protocol is still
  being validated, and a larger study would spend work without improving
  the correctness claim.
- Reuse the smoke's printed output alone. It has no durable predeclared
  request or end-to-end A/B interruption proof.

## Consequences

The local request and result are ignored scientific artifacts, while this
ADR and the development log record their hashes and outcomes. The tiny
synthetic study is a protocol confirmation, not a representative performance
ranking. It retains both seeds and all 12 outer trials per order. The
observed selected final balanced scores were 0.95/0.70/0.00 for
backprop/PC/circadian on seed 17 and 0.15/0.50/0.35 on seed 19. The
circadian method lost to PC on both seeds, and no candidate, seed, or
metric was revised. Format 7 remains a trusted local checkpoint, not a
format for untrusted files.
