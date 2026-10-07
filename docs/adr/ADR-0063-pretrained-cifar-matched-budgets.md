# ADR-0063: Freeze a bounded pretrained-CIFAR matched-head comparison

## Context

The first real-CIFAR matched-head confirmation used 32 training examples,
16 final-test examples, and a random frozen backbone. It verified the
pipeline, but its scores could not support a useful accuracy comparison.
A sealed CPU probe then measured 1.875 seconds to build ImageNet ResNet-50
V2 features for 128/64/64 development-role examples. The cached archive
and pretrained checkpoint had recorded checksums.

## Decision

Predeclare one larger, local CPU study before opening any new final test:
1024/256/256/512 train/guard/outer-validation/final-test CIFAR-10 examples,
32-pixel inputs, batch size 32, one fixed-data epoch, a frozen shared
pretrained ResNet-50 V2, equal width-16 heads, two inference steps, and
forced scheduled sleep. Keep the candidate grid equal across heads: base
and 0.8× base learning rate. Use selection seed 113, then separate
confirmation seeds 127/131/137. Seal final test during selection; persist
its request, all six trial records, and a digest-checked confirmation
manifest before confirmation. Run fixed-data/work, 0.5-second per-head
wall-time, and process-isolated fixed-width memory as distinct scopes.
Limit local selection to 120 seconds and confirmation to 480 seconds.

Why this: the measured setup cost made a single 1,024-example CPU study
practical, while freezing roles, seeds, and budgets before scores prevents
test-informed tuning. The 32-pixel input preserves the existing local
CIFAR transform path and keeps this one step budgeted; it limits external
validity because ImageNet pretraining ordinarily uses a different input
scale.

## Alternatives

- Reuse the 32-example random-feature score as a head ranking. Rejected
  because too few examples and a random representation limit inference.
- Increase the candidate count or alter seeds after seeing validation or
  test results. Rejected because equal predeclared tuning is required.
- Pool one-epoch, wall-time, and observed RSS values into one ranking.
  Rejected because these measure different resources.

## Consequences

All six selection trials completed with common split, feature, backbone,
and initial-head hashes; final test was never iterated during selection.
The unchanged manifest then completed nine fixed-data test rows, nine
deadline heads, and nine isolated memory children in 100.11 seconds.
Fixed-data mean test accuracy was 0.406 backprop, 0.178 predictive coding,
and 0.152 circadian predictive coding. Equal-wall-time means were 0.579,
0.507, and 0.389, respectively. This is a negative result for circadian
under the declared conditions. Work counts and observed process RSS stay
separate in the saved artifact. The small training role, 32-pixel input,
CPU-only timing, and process-wide RSS prevent a general fairness or memory
claim. P1.7 and P1.8 remain open for actual CUDA evidence and a more
representative scale.
