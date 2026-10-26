# ADR-0410: Catalogue binding through the existing weighted stream

## Context

The accepted weighted catalogue update consumes complete TRAIN sources with
original spans, while the existing sequential array port previously bound one
episode to each wake. Public joint weighting needs catalogue ownership, exact
array/label binding and complete native continuation without another runner.

## Decision

Add `TorchTrainWeightedLearner.from_catalogue(policy, catalogue, updates)` for
bound BP/PC/neutral policies and1..128 updates. Retain one owned catalogue and
the existing immutable episode witness, exact original span identity/value
seals and integer update/row limits. Do not retain128 copies or invent episodes.
Every wake binds original float64 raw296 arrays/labels and fixed aggregate
catalogue weights. Prefix examples equal completed_updates*catalogue_rows.

Reuse the existing independent candidate and full native publication path,
source/input/policy/gradient/RNG/mode guards and snapshot/fork/restore contracts.
Catalogue forks recheck original sources after copying and prove copied witness
values; foreign changes are refused without rollback. Existing episode and
recurrent construction/default math remain unchanged. Caller/stream retain
release authority and permanent execution budgets; the native counter is not
a new lifetime budget or refund mechanism.

## Alternatives

- Repeated source copies would waste storage and source-validation work.
- A fictitious54-step episode would misrepresent complete source boundaries.
- A separate catalogue runner would duplicate release and publication behavior.

## Consequences

The existing delayed TRAIN stream can execute catalogue updates with distinct
objective receipts and unchanged source/actor state. Full native direct parity,
source seals, bounded prefix/fork/restore and original episode/recurrent
regressions pass scoped composite checks. Validation/copy/projection costs
remain additional, with native/RSS/time separately recorded.
Public joint API/CLI composition, ordinary isolated probe templates and scientific
comparison are unfinished. No useful acquisition/retention/recovery or circadian
advantage follows from this engineering result.
