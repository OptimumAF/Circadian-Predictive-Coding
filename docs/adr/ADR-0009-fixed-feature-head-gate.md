# ADR-0009: Stage a two-head fixed-feature comparison

## Context

The legacy vision benchmark compares a linear backprop head with hidden-layer
PC heads on separately initialized backbones. `BackpropMLPHead` already has
the PC head's width, tanh activation, output shape, and initial tensors, but
it was not part of a benchmark. A claim about learning rules needs identical
features as well as identical head initialization.

## Decision

`run_two_head_fixed_feature_benchmark` builds one frozen ResNet-50 and
materializes its train, inner-guard, and outer-validation features once.
Backprop MLP and PC heads receive those same ordered feature batches. A
shared head seed gives equal initial tensors, checked by a content hash.
The backbone state and each role's feature/label batches are also hashed.
Final-test features and labels are opened only after both heads train. The
result uses protocol `vision_two_head_fixed_feature_v1` and records cached
bytes. It reports both outcomes descriptively without choosing a winner.

The route requires the guard-separated vision source protocol and an explicit
frozen-backbone setting. `backbone_weights=none` is a random-feature control;
`imagenet` selects pretrained weights when available. Head training time
excludes one-time backbone feature extraction, so it is not end-to-end
training throughput.

## Alternatives considered

- Instantiate independent frozen backbones under one global seed: their
  weights and batch-normalization buffers need not match.
- Copy one backbone state into three full wrappers: valid, but repeats the
  backbone forward pass and stores multiple large models during this gate.
- Recompute augmented features every epoch: makes exact feature matching and
  model-order checks more complex. This fixed-feature track deliberately
  caches one augmentation view per training sample.

## Consequences and open work

The cached features consume memory proportional to sample count and ResNet
feature width. They create a distinct protocol from the legacy image-level
benchmark and cannot be pooled with it. The circadian head, full
three-model matched report, capacity/compute accounting, and separate
end-to-end practical track remain open under P1.4–P1.8. No comparative
scientific result is inferred from the tiny CPU gate.

The circadian extension was subsequently implemented under ADR-0010. The
remaining fairness and practical-track work stays under P1.6–P1.8.
