# ADR-0026: Validate NumPy binary training batches before state changes

## Context

The three NumPy trainers share a binary sigmoid output but previously had
different entry checks. Backprop accepted a zero-row batch, emitted NumPy
warnings and a nonfinite loss, and did not raise. It also accepted a NaN
learning rate. Ordinary PC and circadian PC checked finite inputs/rates but
could reach broadcasting or division with malformed target shapes and
empty batches. Circadian training can change cooldown and chemical state
before a later tensor error, making a rejected call non-transactional.

## Decision

Use `src/core/training_validation.py` at the start of every NumPy binary
training call, before traffic, cooldown, chemistry, replay, or parameter
changes. Require a two-dimensional, nonempty real numeric feature matrix
with the model's input width; a two-dimensional one-column target matrix
with the same row count; finite values; and targets in `[0,1]`. Interior
soft labels remain valid because binary cross-entropy and the executed
`q-target` update both support them. Require a positive finite weight
learning rate. Existing PC/circadian latent-step/rate checks remain.

The validator raises `ValueError` with the failed dimension or value
contract. Deterministic tests cover invalid ranks, widths, row counts,
types, target ranges, nonfinite data/rates, and zero rate across all three
trainers. They verify unchanged weights, traffic, and circadian adaptive
state on rejection, and verify that valid soft labels still train.

## Alternatives considered

- Relying on NumPy's broadcasting and arithmetic exceptions permits
  silent shape expansion, nonfinite metrics, and partial state changes.
- Converting or clipping invalid targets would hide data errors and change
  the optimization problem. Invalid targets are rejected instead.
- Restricting labels to exactly zero or one would reject valid soft-label
  BCE training without a requirement to do so.

## Consequences and open work

Valid binary training numerics and protocol IDs are unchanged. The added
finite/range scan costs one pass over each small NumPy batch; larger-data
timing remains part of P1.8. P2.8b still needs Torch head input contracts.
P2.8c retains post-update finite/topology and saturated-loss boundaries.
