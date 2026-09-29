# ADR-0062: Restore a frozen real-CIFAR manifest and version isolated memory

## Context

P1.7e verified the local CIFAR-10 CPU loader. The existing repeated
matched-head confirmation had only been exercised on tiny synthetic data.
Its validation selection could run on real CIFAR, but the process-isolated
memory route explicitly rejected nonsynthetic datasets. A first confirmation
from a saved three-seed manifest reached that guard after fixed-data and
wall-time work; it saved a failure record and no complete result.

## Decision

Keep the saved selection and manifest unchanged. Restore the manifest from
JSON into typed configuration objects and verify its digest before any final
test access. Require the saved selection digest and archive checksum to match
the predeclared request. Extend process-isolated fixed-width memory only to
local CIFAR-10 with zero workers and `download=False`, rejecting other
requests before spawning. Synthetic runs retain protocol v1; local CIFAR
memory uses `vision_three_head_fixed_width_cifar10_process_memory_v2`.

Run one seed across three child processes first, then retry the exact
confirmation manifest. Preserve both the failed attempt and completed
result. Report fixed-data, wall-time, and memory observations separately;
the random frozen backbone and 32 training/16 final-test examples per seed
make accuracy results descriptive.

## Alternatives

- Drop memory from the real-CIFAR confirmation. Rejected because the
  predeclared manifest requires the capacity/memory scope.
- Treat CIFAR memory as synthetic protocol v1. Rejected because it would
  silently broaden an existing protocol's accepted data source.
- Change seeds, trial counts, or metrics after the failure. Rejected because
  fixed-data and wall-time final-test computations had already occurred.

## Consequences

The three declared seeds now yield complete fixed-data, wall-time, and
isolated-memory reports under one unchanged manifest. The local memory
observations include runtime, backbone, feature bank, and sampler overhead;
they are not isolated head allocation or exact transient peaks. Dataset
loading constructs a final-test object during setup, but measured children
never iterate that role. Real CUDA and larger-data fairness evidence remain
open under P1.7/P1.8.
