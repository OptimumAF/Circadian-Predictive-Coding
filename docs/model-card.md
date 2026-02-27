# Model Card: Circadian Predictive Coding

## Summary

Circadian Predictive Coding is a predictive-coding-based learner with sleep-phase structural plasticity.
It tracks per-neuron chemical usage, modulates plasticity during wake, and applies split/prune consolidation during sleep.

## Intended Use

- Research and educational experimentation with biologically inspired learning dynamics.
- Controlled benchmarks against backpropagation and traditional predictive coding.

## Not Intended For

- Safety-critical production decisions.
- Unreviewed deployment in medical, legal, or financial decision pipelines.

## Model Family

- Base: predictive coding with iterative hidden-state inference.
- Extension: circadian dynamics:
  - chemical accumulation and decay
  - plasticity gating
  - adaptive sleep triggers
  - structural split/prune
  - optional rollback and homeostatic controls

## Training Data

- Toy two-cluster synthetic dataset (NumPy experiments).
- Synthetic and torchvision-backed vision datasets (ResNet benchmark workflow), including CIFAR-10/CIFAR-100.

## Evaluation

Primary comparison metrics:

- Test accuracy
- Cross-entropy / energy
- Training throughput (`samples/s`)
- Inference latency (`mean`, `p95`) and throughput
- Circadian adaptation telemetry (splits, prunes, hidden dimension trajectory, rollbacks)

## Known Limitations

- Benchmark conclusions are sensitive to sleep hyperparameters.
- Circadian adaptation can underperform if split/prune schedules are too aggressive.
- Current implementation focuses on head-level circadian adaptation in ResNet benchmarks.

## Ethical Considerations

- No personal data is required by default benchmark workflows.
- Public benchmark claims should include dataset, seeds, and configuration details for reproducibility.

## Maintenance Status

Active research repository; APIs and defaults may evolve.
Use release tags for stable references in external projects.
