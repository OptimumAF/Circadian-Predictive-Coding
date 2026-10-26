# ADR-0538: Separate rectangular choice readout from paired CLM learning

## Context

The existing temporal-rule corpus has four development queries and three active
candidates. CLM's paired frozen-feature batch requires equal row counts because
native symmetric contrastive learning assigns a diagonal positive pair. Relaxing
that training record would conflate candidate ranking with positive-pair labels.

## Decision

Add a separate FrozenChoiceBatch and predict_choices on the existing owned head
learner. Validate independent query/candidate bounds and fixed feature identity,
then use the same projections and state/RNG guards. Permit TRAIN diagnostics and
outer_selection development; reject final/probe roles and all choice records at
paired training admission. Existing paired prediction and objective stay strict.
Aggregate prediction counters cover both APIs; choice-specific candidate/attempt
counters preserve physical costs through restore. Keep the implementation in the
existing core record and adapter modules; add no runner, cache or checkpoint.

## Alternatives

- Relax FrozenEmbeddingBatch: would weaken the native paired training invariant.
- Build a separate mutable predictor owner: would duplicate continuation and
  ownership logic and make changed-head projection reuse harder to reason about.
- Duplicate/pad candidates to force square readout: would change candidate mass
  and potentially ranking rather than preserve the actual choice question.

## Consequences

The learner now directly ranks N queries against M supplied candidates and can
consume the existing corpus. Consumers bind feature rows/columns to exact input
text and candidate IDs; this record does not attest data provenance. Scoped tiny
checks establish API/math/state behavior, not released temporal task usefulness.
The next native experiment must separately freeze data/dose/controls/costs and
retain all negative results. Full R5.3 consolidation and scientific goals remain.
