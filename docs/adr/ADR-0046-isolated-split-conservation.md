# ADR-0046: Validate split conservation separately from sleep effects

## Context

Both adaptive heads duplicate a split neuron's incoming path and divide
its outgoing row between parent and child. Sleep can also replay stored
examples, prune neurons, reset chemistry, or apply homeostasis. A whole
sleep event can therefore change predictions even when the split
operation itself preserves the represented function.

## Decision

P3.5 tests the split in a sleep configuration with replay, pruning,
homeostasis, chemical reset, and split noise disabled. For NumPy float64
predictions the absolute tolerance is `1e-12`; for Torch float32 logits
it is `2e-6`, with zero relative tolerance. The tests compare the same
fixed inputs before and after one split and again after splitting the
new child. They also check that the child's incoming column and hidden
bias duplicate the source and that parent-plus-child outgoing rows equal
the original row within dtype tolerance.

Seeded noisy splits are tested separately. Their outgoing rows receive
opposite nonzero perturbations, so the sum and isolated predictions
still agree within tolerance for the fixed fixtures. This observation
does not impose function preservation on a full sleep event when other
consolidation components run. Fixture seeds cover the code path and
were not selected using model-performance outcomes.

## Alternatives and consequences

Testing only whole sleep would confound split conservation with replay,
pruning, and homeostasis. Checking only outgoing rows would miss an
incoming-column or bias mismatch. The isolated tests add no algorithmic
behavior, tuning, metric change, or runtime dependency. P3.6 retains
pruning alignment; later experiments may evaluate whether noisy splits
help or hurt predictive performance without changing this correctness
criterion.

## Evidence

`tests/test_split_function_preservation.py` passes four NumPy/Torch
single-and-repeated zero-noise and separate seeded-noise cases. The
development log records the final quality gate.
