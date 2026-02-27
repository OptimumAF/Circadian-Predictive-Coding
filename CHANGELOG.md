# Changelog

All notable changes to this project are documented in this file.

The format is inspired by Keep a Changelog and this project follows semantic intent
for versioning even while in research-stage development.

## [Unreleased]

### Added

- Open-source community baseline files:
  - `LICENSE` (MIT)
  - `CODE_OF_CONDUCT.md`
  - `SECURITY.md`
  - `SUPPORT.md`
  - `GOVERNANCE.md`
  - `CITATION.cff`
- GitHub collaboration scaffolding:
  - issue templates
  - pull request template
  - CI workflow
  - dependabot config
- `pyproject.toml` with centralized tool configuration for `pytest`, `ruff`, and `mypy`.
- Multi-seed benchmark runner:
  - `scripts/run_multiseed_resnet_benchmark.py`
  - JSON and CSV export support for reproducible cross-seed comparison.

### Changed

- Repositioned repository messaging to Circadian Predictive Coding as the primary focus.
- Updated `README.md` with:
  - circadian-first project framing
  - reproducible benchmark commands
  - governance and citation references
- Updated `ARCHITECTURE.md` with clearer module boundaries and circadian-centric design intent.
- Updated `CONTRIBUTING.md` with concrete contribution workflow and quality gates.

### Existing Core Work (Carried Forward)

- Backpropagation, predictive coding, and circadian predictive coding implementations.
- Circadian mechanisms including chemical gating, adaptive sleep policies, split/prune logic, and rollback support.
- ResNet-50 benchmark path comparing all three model families.
- Test suite and deterministic data generation for reproducible experiments.
