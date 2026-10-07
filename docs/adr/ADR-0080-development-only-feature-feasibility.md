# ADR-0080: Measure larger CIFAR features without constructing final test

## Context

P1.8n's matched CUDA study used 32-pixel inputs and 1,024 training
examples. Its negative circadian result is retained, but that scale does
not support a general fairness ranking. A larger study needs a local
feature cost and memory estimate before final-test access. The existing
torchvision loader built the CIFAR test dataset even when a caller only
materialized development features; a raising test iterator alone could
not prove that final source construction was absent.

## Decision

Add an opt-in `include_final_test=False` path to the torchvision loader,
the benchmark loader boundary, and the matched tuning feature-bank helper.
It never constructs a `train=False` CIFAR dataset, omits final role IDs
and hashes, and supplies a loader that raises if used. Defaults retain the
reviewed behavior. A fake CIFAR source that raises on final construction
and an app-level bank test verify this boundary.

Freeze one CUDA development-only probe before loading data: ImageNet V2
ResNet-50 frozen features, 224-pixel CIFAR-10 images, seed 173,
4,096/512/512 train/guard/outer-validation examples, batch 32, no head
training, a three-reading quiet-device gate, and a hard 120-second worker
timeout. Bind the verified archive and weight hashes. Record per-role
examples, batches, feature time/bytes/hashes, source IDs, worker RSS, and
CUDA allocator peak; keep final source construction and iteration at zero.

Use the measured cost to save, but not execute, one larger matched-head
request. It fixes 224-pixel 16,384/2,048/2,048 development roles and
4,096 final examples; equal two-candidate optimization grids; selection
seed 179; confirmation seeds 181/191/193; the outer-only validation
objective; and separate fixed-data, wall-time, and process-isolated memory
budgets. The candidate rates are the same base and 0.8× grid as P1.8n.
The request keeps every outcome, including failures and negative scores.

## Alternatives

- Keep a test iterator sentinel after constructing the CIFAR test dataset.
  That cannot establish the requested development-only source boundary.
- Run a larger selection or confirmation before measuring feature cost.
  It would commit to time and memory budgets without host evidence.
- Change seeds, rates, or metrics in response to P1.8n's negative result.
  That would make the new comparison test-informed.

## Consequences

The saved 224-pixel development probe completed 4,096/512/512 examples
in 128/16/16 batches in 9.522 seconds. Its observed worker RSS peak was
1,555,025,920 bytes, and its PyTorch CUDA allocated/reserved peaks were
447,518,208/660,602,880 bytes. The quiet readings were 3%/1%/1% GPU
utilization with at least 8,139 MiB free; the post reading was 2%. No final CIFAR
source was constructed or iterated. The 16,384/2,048/2,048 feature cost
projection is 33.84 seconds per seed at the same resolution, an estimate
rather than a guarantee. The larger request is frozen before new final
access and remains unexecuted. It is still a CIFAR subset with resized
inputs and a frozen ImageNet backbone, so any eventual ranking must carry
that scope.
