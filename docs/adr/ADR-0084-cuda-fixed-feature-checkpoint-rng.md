# ADR-0084: Restore device-bound CUDA random state for fixed-feature checkpoints

## Context

The fixed-feature runner and combined checkpoint previously rejected CUDA.
The circadian Torch head already snapshots its own CUDA split generator,
but the combined checkpoint saved only CPU process random state. Resuming
learning on an actual CUDA device without the process CUDA stream could
change future draws even when head tensors and the local generator match.
The CPU checkpoint-memory route reports process RSS segments but has no
defined CUDA allocator segment scope.

## Decision

Permit a trusted fixed-feature checkpoint on CUDA when memory telemetry is
disabled. The combined payload keeps its version-one CPU fields and adds
optional CUDA device and process RNG fields. A CUDA head uses a distinct
`torch_head_cuda` backend, with its canonical device taken from the live
head tensor. Capture that device's process RNG state alongside the existing
head snapshot, which already contains the local split generator.

Restore checks the saved device, byte-tensor type, and generator state in a
temporary CUDA generator before changing the live head, retry state, or
process streams. It then restores CPU and CUDA process streams after the
model state. Missing CUDA fields in older CPU checkpoints retain their
default `None` values. CUDA checkpoint-memory, wall-time, and fixed-width
capacity modes continue to raise before dataset loading until their device
semantics are verified under P3.9b2b2b2.

## Alternatives

- Re-seed CUDA on resume. That loses draws made before interruption.
- Use the CPU process RNG as a proxy. It does not encode the CUDA stream.
- Enable the CPU RSS checkpoint-memory protocol on CUDA. Its current
  aggregation cannot describe allocator peaks across processes.

## Consequences

On the local RTX 3080 with Torch 2.14.0+cu130, actual CUDA wake,
accepted-sleep, and rejected-sleep file interruptions matched uninterrupted
head state, non-timing report fields, and next Python/NumPy/Torch CPU/CUDA
draws. Head snapshot equality includes the local CUDA split generator.
Missing, malformed, and wrong-device CUDA process state rejected before
live mutation. A public three-head fixture kept final test sealed until
resumed training finished. The bounded test command had a 120-second hard
limit; it completed in seconds. Existing CPU checkpoint tests and an
older-field compatibility fixture passed.

This proves CUDA learning/RNG continuation for the tested fixed-epoch
fixed-feature route. It makes no resumed CUDA allocator or RSS claim.
P3.9b2b2b2 retains
that separate acceptance gate, and the unmatched whole-image runner remains
P3.9c2b2b.

## Subsequent decision

ADR-0085 defines and verifies checkpointed CUDA allocator segments across
process restarts, including memory-enabled deadline and fixed-width capacity
routes. The unsupported-mode statement above describes the state at the time
of this decision.
