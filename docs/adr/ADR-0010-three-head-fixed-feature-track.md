# ADR-0010: Extend the fixed-feature track to the circadian head

## Context

ADR-0009 established a budgeted two-head gate with one frozen ResNet and a
single cached feature bank. The original vision benchmark still has separate
backbones and a linear backprop head. It cannot attribute a difference to the
head learning rule or circadian mechanisms.

## Decision

`run_three_head_fixed_feature_benchmark` adds
`CircadianPredictiveCodingHead` to that same train, guard, validation, and
final-test feature bank. It requires its initial hidden width to match the PC
head and constructs all three heads from the same seed. Initial parameter
hashes must match. The circadian configuration is built by the same helper
used by the image-level reference benchmark. Its sleep trigger and rollback
see only the inner guard batches. Final-test features are materialized after
all three trainers return. The protocol is
`vision_three_head_fixed_feature_v1`; final-test scores are descriptive.

The existing `BackpropResNet50` result retains its historical model name for
compatibility and is explicitly labeled a legacy linear-head reference in
formatted output. It stays in the versioned unmatched protocol.

## Alternatives considered

- Rebuild three separate ResNets with the same random seed: this would rely on
  incidental RNG call order and would repeat feature extraction.
- Reuse the original image-level circadian wrapper: it would build another
  backbone and could receive a different augmented image stream.
- Disable sleep for the matched track: that would omit the mechanism under
  investigation. A zero interval remains an explicit test configuration.

## Consequences and open work

The three heads receive identical fixed input tensors and start from equal
head parameters, but their update rules, learning rates, and work per epoch
differ. One cached augmentation view and a frozen backbone define this
protocol; head training time excludes feature extraction. Capacity, wall-time,
replay/relaxation cost, independent random streams, and model-order invariance
remain open under P1.6–P1.8. The tiny CPU gate is a correctness check and
supports no ranking or claim that circadian control improves accuracy.
