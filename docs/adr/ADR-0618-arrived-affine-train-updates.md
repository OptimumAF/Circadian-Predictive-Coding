# ADR-0618 — Learn affine heads from arrived TRAIN batches

## Context

The offline joint model fits54 TRAIN actions and executes both known-A games,
but already knows future B/C. Chronological learning requires a new untrained
owner. The existing stream already provides delayed labels, receipts and budgets.

## Decision

Add a separate affine TRAIN port over the unchanged fixed least-squares core.
Each homogeneous public-cue batch replaces one head; retain the other without a
row/history bank. Unfitted heads predict uniform K7. Full snapshots have optional
fits, real fit/row counters and bounded latched failure evidence. Retain readonly
qualified27-row validation and commands. Use original TRAIN-only AttributionStream;
score only TRAIN after all fixed updates. Budgets never come from snapshots.
Preserve failed prefixes and refuse completion if receipts become unavailable.

## Alternatives

Starting from the joint model invalidates new-task acquisition. Padding an
episode fit to the readonly27-row schema invents history. New optimizers,
replay and circadian mechanisms are unnecessary for this immediate question.

## Consequences

The fixed A/B/A observation demonstrates TRAIN acquisition, forgetting and exact
A-model recovery at measured cost. Head replacement does not jointly retain A/B.
Native gameplay from causal states remains a separate next outcome. No CPC,
generalization, reliability or release claim follows from these checks.
