# ADR-0012: Version the order-controlled image-level reference

## Context

The v2 guarded image benchmark passes one mutable shuffled training loader
through backprop, PC, and circadian trainers. Its generator advances as each
trainer iterates, and image augmentation can draw from the process RNG.
Changing model execution order can therefore change training inputs or
initialization. The fixed-feature track already isolates those streams, but
the image-level reference remains useful for practical context.

## Decision

`vision_guard_separated_seeded_unmatched_v3` keeps v2's train, guard,
validation, and final-test roles while giving each model a separate
initialization RNG scope. Each model receives a replayable training loader:
the shuffle/worker generator and CPU augmentation generator are reset to
declared seeds for every epoch. All models receive the same ordered image
stream for a shared epoch, independent of their execution order. The runner
accepts a validated model order and records the order and trained-model state
hashes before opening final test. Validation-candidate training uses the
same seeded path. A tiny CPU run, including a forced circadian sleep, matched
trained-state hashes exactly and metrics within `1e-7` absolute tolerance
when the order was reversed. A stochastic-loader test verifies replay across
unrelated process RNG draws. The hash includes the circadian snapshot and
structural RNG state and the predictive head's traffic state, as well as
backbone and head tensors. It does not include optimizer state. A picklable
two-worker fixture also replayed Torch, NumPy, and Python image-view draws
across two epochs on the local Windows CPU.

The v1/v2 implementations and the v2 default remain available for
reproduction. CLI and multi-seed exports accept v3 explicitly. Figure
generation recognizes its distinct protocol ID and keeps outputs separate.

## Alternatives considered

- Resetting one process seed before all three models leaves the shared
  training-loader generator advanced by the first trainer.
- Caching every transformed image for every epoch could require excessive
  memory for CIFAR-scale runs and would change augmentation timing.
- Changing v2 in place would make new v2 outputs differ from prior v2 runs
  without a protocol boundary.

## Consequences and open work

The v3 route still compares unmatched heads and independently initialized
backbones. Its results are descriptive, not learning-rule attribution.
GPU kernels, CIFAR transforms with workers, replay streams outside the vision
runner, and broader random-stream independence remain unverified under P1.7.
No CIFAR download or large sweep was performed for this decision.
