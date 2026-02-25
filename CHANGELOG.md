# Changelog

## Unreleased

### Added
- Baseline project structure with layered modules.
- Backprop MLP and predictive-coding network implementations.
- Circadian predictive-coding network with:
  - per-neuron chemical buildup layer
  - plasticity suppression from chemical accumulation
  - sleep events that split high-use neurons and prune low-use neurons
- Side-by-side experiment runner and CLI entrypoint.
- Neuron traffic tracking scaffold and adaptation policy interface.
- Unit and integration tests for baseline behavior.
- Core documentation (`README`, `ARCHITECTURE`, `CONTRIBUTING`, ADRs, module docs).
