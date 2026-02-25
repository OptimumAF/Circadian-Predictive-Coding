# ADR-0002: Add a ResNet-50 Speed and Accuracy Benchmark Workflow

## Context

We need a practical way to measure training speed, inference speed, and target-accuracy convergence for backprop, predictive coding, and circadian predictive coding under a shared deep architecture baseline.

## Decision

Add a torch-based benchmark workflow that uses a ResNet-50 backbone with three model variants:
- `BackpropResNet50Classifier`
- `PredictiveCodingResNet50Classifier`
- `CircadianPredictiveCodingResNet50Classifier`

Use a deterministic synthetic image dataset to avoid external download variability and to make repeated speed tests reproducible.

Circadian benchmarking includes sleep events that can split high-usage neurons and prune low-usage neurons in the predictive-coding head.

## Alternatives Considered

1. Benchmark only on real datasets (e.g., CIFAR/ImageNet).
- Rejected for baseline workflow due external download/setup overhead and nondeterministic IO effects.

2. Force all models to train full ResNet weights.
- Rejected for predictive/circadian head baselines because manual local-update logic is currently implemented at the head level.

3. Keep benchmark logic as ad-hoc scripts only.
- Rejected due maintainability and weak testability.

## Consequences

- We get comparable speed and accuracy metrics quickly on most hardware.
- Benchmark remains deterministic and CI-friendly.
- Predictive/circadian ResNet comparison is currently head-centric; full-network local-learning variants remain future work.

