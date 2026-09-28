# ADR-0027: Validate Torch PC head batches before adaptive changes

## Context

The ordinary and circadian Torch PC heads previously checked positive
rates, iteration count, and finite features, then relied on matrix
multiplication and `one_hot` to reject malformed shapes and labels. An
empty batch could pass through ordinary training without an exception.
Circadian training decays cooldowns immediately after its entry check, so
late shape or target failures can change adaptive state before rejection.

## Decision

The shared head entry check now runs before any parameter, traffic,
cooldown, chemistry, age, reward, or energy-history change. It requires a
nonempty two-dimensional Torch feature tensor with the configured width,
a floating dtype matching the head weights, and the same device as the
head. Class-index targets must be a one-dimensional `torch.int64` tensor
on the feature device, with the same row count and values in
`[0, num_classes)`. Features and step controls must be finite, weight and
latent rates positive, and inference steps a positive integer. Rejected
inputs raise an informative `ValueError`.

For a valid batch, finite-feature and label-range reductions are combined
before one `.item()` host synchronization, the same synchronization count
as the previous finite-feature check. A rejected batch may inspect the
failed component with another synchronization to report the right error.
The extra label-range kernel still belongs in future target-hardware
training-time measurements under P1.8.

## Alternatives considered

- Waiting for Torch's matrix or `one_hot` exception allows partially
  changed circadian state and gives inconsistent errors.
- Casting target dtypes or clipping invalid class IDs would hide an input
  contract violation and could change the supervised objective.
- Synchronizing after each latent iteration would add avoidable GPU
  overhead; the existing finite relaxed-state check remains at the
  pre-update boundary.

## Consequences and open work

Valid multiclass learning arithmetic and protocol IDs are unchanged.
Deterministic CPU tests cover both production head paths, malformed
feature/target shapes and dtypes, zero/mismatched batches, out-of-range
classes, nonfinite features/rates, and unchanged parameters plus adaptive
state on rejection. P2.8c still covers nonfinite parameters or updates,
topology compatibility, and saturated-loss behavior. Actual CUDA timing
and CIFAR evidence remain P1.7/P1.8 work.
