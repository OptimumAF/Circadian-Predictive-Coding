# ADR-0001: Build a Predictive Coding Baseline Beside Backprop

## Context

We need a practical starting point to compare predictive coding with standard backpropagation, while preparing for future dynamic neuron topology experiments.

## Decision

Implement two minimal, directly comparable models:
- `BackpropMLP` using standard gradient descent
- `PredictiveCodingNetwork` using iterative hidden-state inference plus local error-driven updates

Track hidden-layer traffic in both models and define a policy interface for future neuron growth/pruning without enabling structural change yet.

## Alternatives Considered

1. Implement a larger deep architecture immediately.
- Rejected: increases complexity before baseline behavior is clear.

2. Implement dynamic add/remove neurons in first iteration.
- Rejected: mixes two sources of change and makes baseline interpretation harder.

3. Use only conceptual docs without code.
- Rejected: blocks empirical comparison and slows learning.

## Consequences

- We get a reproducible, testable baseline now.
- Structural adaptation remains an explicit next step with clear extension points.
- Predictive coding implementation is intentionally lightweight and educational, not a full research-grade framework.

