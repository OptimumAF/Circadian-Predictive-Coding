# ADR-0017: Measure fixed-width heads in separate processes

## Context

The existing memory-enabled three-head route trains heads sequentially in
one process. Its RSS windows include the shared backbone, cached features,
all resident heads, allocator history, and work from earlier trainers. It
cannot isolate each head's trainer footprint. P1.8 needs a separate local
observation without opening final-test labels or changing the matched inputs.

## Decision

Add `vision_three_head_fixed_width_process_memory_v1`. Spawn one fresh
process per head under one prevalidated fixed-width, guarded synthetic
configuration. Each child seeds and rebuilds the same frozen backbone and
train/guard/validation feature bank, instantiates only its assigned head
from the same head seed, then trains it with the existing matched-head
trainer. No child reads the test loader. The parent checks that split,
feature, backbone, and initial-head hashes, cache bytes, and starting/final
head parameter counts agree before returning reports.

Measure two distinct host windows. Setup sampling starts after Torch import
and device resolution and covers loader/backbone/feature/head setup; report
the pretraining RSS boundary and cached feature tensor bytes separately.
Trainer sampling covers optimizer/wake work, guard checks, circadian sleep
and rollback, and outer validation. On CUDA, reuse the existing synchronized
PyTorch allocator baseline and peak hooks for that trainer window. Return
only descriptive memory/work evidence, not a final-test score or winner.
Require an explicit CPU or CUDA device, synthetic data, zero loader workers,
and a positive finite per-child timeout for this local gate. A tiny script
under `scripts/` makes the spawn-safe CPU check repeatable.

## Alternatives considered

- Sequential measurements in one process preserve shared features but keep
  other heads and allocator history resident, so they cannot confirm the
  isolated trainer footprint.
- Clearing the allocator between sequential heads cannot reset every host
  allocation or model object and would not give equivalent fresh baselines.
- Moving final-test evaluation into each child would open labels during a
  memory-only protocol and add a fourth feature role to setup cost.
- Forking the parent would inherit its resident Torch state; fresh spawned
  children make the boundary explicit on Windows and Linux.

## Consequences and open work

Each child pays its own Python/Torch import, backbone, and feature setup
cost. Those imports precede the setup RSS window, while setup and trainer
observations remain separate. Process RSS includes runtime and shared
backbone/cache costs, so it is not the exact incremental memory attributable
to a head. Highest observed RSS can miss short peaks. The parent verifies
input parity, and the CPU sentinel test proves the worker does not access
the final-test loader. The real tiny synthetic smoke records equal hashes
and 32,835 head parameters in all children; it is not a model ranking.
CUDA telemetry remains unverified locally. Repeated matched confirmation,
larger-data evidence, and full P1.8 fairness conclusions remain open.
