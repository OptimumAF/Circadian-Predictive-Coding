# Changelog

## Unreleased

### Added
- Baseline project structure with layered modules.
- Backprop MLP and predictive-coding network implementations.
- Circadian predictive-coding network with:
  - per-neuron chemical buildup layer
  - plasticity suppression from chemical accumulation
  - sleep events that split high-use neurons and prune low-use neurons
- Three-model experiment runner and CLI entrypoint.
- In-depth aggregate comparison runner (`seeds x noise levels`) with:
  - mean/std test accuracy
  - mean/std final metric
  - mean/std epoch-to-80%-progress
  - circadian split/prune and hidden-dimension end-state stats
- ResNet-50 benchmark workflow with:
  - backprop, predictive-coding-head, and circadian predictive-coding-head variants
  - training speed metrics (samples/s, ms/step)
  - inference latency/throughput metrics (mean, p95, samples/s)
  - target-accuracy early-stop support
  - circadian sleep split/prune and hidden-size tracking
- Neuron traffic tracking scaffold and adaptation policy interface.
- Optional `requirements-resnet.txt` dependency set for torch-based benchmarks.
- Unit and integration tests for baseline behavior.
- Core documentation (`README`, `ARCHITECTURE`, `CONTRIBUTING`, ADRs, module docs).
