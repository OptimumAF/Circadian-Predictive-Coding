# ADR-0030: Keep saturation diagnostics distinct and validate post-sleep widths

## Context

Extreme finite logits expose a deliberate difference between NumPy's
clipped binary diagnostic and Torch's logit-based multiclass
cross-entropy. A wrong-class NumPy logit of magnitude `1e300` yields a
finite diagnostic near `18.42` because probabilities are clipped to
`[1e-8, 1-1e-8]`; it is not the true logit BCE of that example. The
positive and negative tails differ slightly from floating-point rounding
at `1-1e-8`. Torch held-out cross-entropy on logits
`[1e300, -1e300, 0]` remains finite and reports `2e300`, while the PC
training squared-error diagnostic is `1/3`. These quantities must retain
their existing IDs and distinct interpretations.

## Decision

Keep the executed binary clipping and Torch CE/PC diagnostic formulas.
Do not alter baseline metrics to make the circadian method look better.
Test finite saturation explicitly, and validate the internal parameter
and per-neuron widths at the training entry to report an informative
`ValueError` before cooldown or prune state changes. The width check
covers the active post-split and post-prune topology, not only initial
construction. It reads shapes without a Torch device synchronization;
its small host-side cost belongs in training time under P1.8.

## Evidence and limits

The first extreme-positive binary fixture failed an over-strict symmetric
rounding expectation; the assertion now matches the existing clipped
formula exactly. Both backends train after accepted one-neuron split and
immediate prune events with aligned parameters and adaptive arrays.
Corrupting an output-weight row after a split initially produced a late
NumPy matrix mismatch or Torch runtime error; both paths now raise
`ValueError` before cooldown mutation. See
`tests/test_saturation_topology_boundaries.py` and the development log.

These deterministic cases verify finite diagnostics and local training
steps only. They do not make an arbitrary-rate convergence claim or imply
that clipped binary BCE and multiclass CE can be ranked together.
