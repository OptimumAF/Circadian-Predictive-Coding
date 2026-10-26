# ADR-0435: Bound a separate polynomial-kernel TRAIN learner

## Context

The fixed original linear control stops short of a full54-row TRAIN fit. This
does not prove inseparability, but motivates a small separate nonlinear control.
Existing NativeLearner, original public-history projection and TRAIN fit stream
already provide isolation, clocks, budgets, receipts and scoring.

## Decision

Use fixed `(1+dot(x,y)/D)^2`, unit multiclass mistake updates and a dictionary
capped at128 exact numerical prototype keys. No RNG, optimizer or hyperparameter
selection. Keep canceled entries; refuse a complete pass atomically when a new
mistaken key exceeds capacity. Return observed uncommitted work in the existing
typed training failure. Complete immutable owned snapshots include dictionary,
coefficients and native work. Bind a separate invariant296/K7 wrapper; expose an
explicit CLI family through the existing stream.

Why this: the rule is simple enough for an independent scalar oracle, can learn
XOR, has a hard storage cap and preserves the original recipes and data roles.
Committed native kernel pairs and derived prediction pairs remain distinct.

## Alternatives

More linear passes would change the frozen dose without testing new capacity.
An unbounded kernel dictionary would violate bounded storage. Eviction or a new
optimizer would add unsupported policy choices. A new runner duplicates existing
isolation and accounting boundaries.

## Consequences

Different capacity/compute means this is not a matched neural baseline or CPC
benefit claim. Synthetic correctness precedes one separately reviewed real fit.
Full acquisition, acquired retention/recovery and release requirements remain
open. Dictionary array-equivalent bytes exclude Python/copy/process overhead.
See [usage and checks](../kernel-perceptron-train-fit.md) and stage evidence.
