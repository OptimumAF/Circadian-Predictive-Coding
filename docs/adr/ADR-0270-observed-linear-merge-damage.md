# ADR-0270: Observe damage before accepting tolerant projected linear merge

## Context

The linear MLP/residual merge is a tolerance-compatible projection. The retained
amplification counterexample proves its coordinate tolerance does not bound
predictions. Original R1.6 requires observed damage and recovery for non-preserving
operations; the existing application gate covers pruning only.

## Decision

Add a merge application gate using the existing copied train/validation admission,
native full-equilibrium scoring and damage record schema. Compare every probe's
prediction/half-MSE/energy with independent inclusive limits, checking resources
before publication. Accepted-only candidate/lineage; rejection/failure returns
owned full source with completed diagnostics. Keep pure merge behavior unchanged.

## Alternatives

Tightening core tolerance would hide the existing non-preserving case instead of
meeting its observed-damage requirement. Prediction-only or half-MSE-only checks
omit part of the required native objective. Duplicated score/record frameworks
would add unnecessary drift; exact public aliases document the reused schema.

## Consequences

Both supported linear families have an explicit selection/recovery boundary.
Negative amplification remains valid and can reject despite task/energy improvement.
No global finite-precision bound, future-SGD identity, new algorithm, final-test
selection, scientific advantage or serving authority is claimed. Original parent
acceptance must be reassessed against actual evidence, not source presence.
