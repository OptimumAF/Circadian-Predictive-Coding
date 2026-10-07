# ADR-0057: Replay a seeded vision loader from a logical batch cursor

## Context

The seeded unmatched vision protocol resets the training sampler and Torch
augmentation stream for each model and epoch. A checkpoint after a wake batch
still needs the unconsumed order and views. A multiworker DataLoader may have
already prefetched later batches in child processes, so its current sampler
generator state alone does not identify the next logical batch.

## Decision

`app.seeded_vision_loader` owns a versioned epoch/batch cursor. It records the
sampler generator state observed at that batch, the entry and current Torch
CPU streams, and current Python and NumPy streams. On a fresh loader, it
re-seeds the epoch, discards already completed logical batches, verifies the
sampler state, and restores the captured process streams before yielding the
next training batch. The entry Torch state is restored when the epoch ends,
matching the seeded protocol's existing `fork_rng` behavior. A failed sampler
verification restores caller and loader RNG state before raising.

The checkpoint methods require a map-style shuffled loader whose sampler
uses the loader generator, ordered delivery, ordinary worker startup, and no
custom worker initialization. They reject persistent workers and changed
prefetch settings, since those can change which child process produced a
stochastic view. The ordinary seeded training route keeps its original
epoch seeds and learning schedule. This module does not identify datasets,
store files, own model state, or read guard, validation, or final-test data.

## Alternatives and consequences

Serializing prefetched worker queues would bind checkpoints to internal
DataLoader implementation details and active subprocesses. Restricting all
resume to zero workers would omit the configured multiworker image route.
Replaying consumed batches costs CPU and transform work, but it reconstructs
worker streams and allows verification at the logical training boundary.

This is P3.9c2a only. P3.9c2b must bind exact development-role content and
protocol identity, save the full classifier and all runner outcomes/counters,
and test accepted/rejected guarded sleep and final-test isolation before the
whole-image checkpoint criterion can be checked. CUDA continuation needs
separate evidence on a CUDA host.

## Evidence

`tests/test_resnet50_benchmark.py` first failed because no cursor API existed.
The bounded CPU fixtures compare a fresh loader with uninterrupted remaining
batches, the next epoch, and next Torch/NumPy/Python draws using zero-worker
Torchvision views, zero-worker mixed stochastic images, and two-worker mixed
stochastic images. The original run interleaves process random draws with
training batches. Invalid cursor shapes, changed batch/prefetch settings,
persistent workers, and a valid but wrong sampler state reject; the latter
leaves caller and loader random streams unchanged. The development log records
the full quality gate.
