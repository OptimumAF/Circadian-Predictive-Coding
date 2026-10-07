# ADR-0014: Report observed memory separately from cached feature bytes

## Context

The matched-head result reports `feature_bytes`, a sum of cached feature and
label tensor storage. It does not measure model state, optimizer state,
temporary allocations, or the process footprint. A capacity-aware comparison
needs memory evidence, but the three heads run sequentially in one process
and local CUDA is unavailable.

## Decision

Add opt-in memory telemetry to the three-head epoch and wall-time routes,
with distinct `vision_three_head_fixed_feature_memory_v1` and
`vision_three_head_fixed_feature_wall_time_memory_v2` IDs. The prior routes
retain their timing behavior without a polling thread. When enabled, sample
current-process resident memory before, during, and after each head's
trainer, including its outer validation. On Windows, read working-set bytes
through the system process API; on Linux, read resident pages from procfs.
Use a 5 ms polling interval plus boundary samples. Report the starting RSS,
highest observed RSS, and sample count per head. Unsupported hosts report
`None`, never zero. A held 32 MiB allocation fixture checks that the
measurement responds to live memory beyond `feature_bytes`.

For CUDA runs with memory telemetry enabled, synchronize the device, record
allocated bytes, reset PyTorch
allocator peak statistics before each trainer, and report peak allocated and
reserved bytes afterward. CPU results report `None` for these CUDA fields.
No additional package dependency is introduced. Keep `feature_bytes` as a
separate static cache-size measure.

## Alternatives considered

- `tracemalloc` omits native tensor allocations and cannot stand in for
  process RSS.
- The operating system's lifetime high-water mark cannot be reset for each
  head in one sequential process.
- Adding a memory-monitoring dependency would not solve model attribution or
  short-lived allocation gaps; the supported hosts expose RSS directly.
- Running every head in a fresh process would improve isolation but requires
  a larger protocol change for shared feature and initialization identity.

## Consequences and open work

The reported host value is the highest **observed** process RSS in the
trainer window, not a guaranteed instantaneous peak or memory attributable
only to that head. The shared backbone, cached roles, all initialized heads,
allocator reuse, and earlier head work affect its baseline; head execution
order can affect RSS. Samples can miss short-lived allocations, and the
sampler itself adds small runtime overhead. Shared setup and final-test
materialization fall outside each per-head window. CUDA fields reflect only
PyTorch allocator statistics, not all device users, and have not been run on
local hardware. ADR-0015 subsequently added an equal-trial tuning ledger,
ADR-0016 added a fixed-width capacity control, and ADR-0017 added
process-isolated observations. Repeated confirmation and real CUDA evidence
remain open before a memory-based ranking claim.
