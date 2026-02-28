# Changelog

All notable changes to this project are documented in this file.

The format is inspired by Keep a Changelog and this project follows semantic intent
for versioning even while in research-stage development.

## [Unreleased]

### Added

- Review-driven circadian updates in NumPy and ResNet circadian cores:
  - optional reward-modulated wake learning (`use_reward_modulated_learning`)
  - optional adaptive sleep budget scaling (`use_adaptive_sleep_budget`)
  - `get_last_reward_scale()` telemetry helper
- Baseline and ResNet benchmark CLI flags for reward modulation and adaptive sleep budget controls.
- Review follow-up docs:
  - `docs/circadian-model-review-notes.md`
  - `docs/adr/ADR-0004-reward-modulated-wake-and-adaptive-sleep-budget.md`
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
- README visual generation pipeline:
  - `scripts/generate_readme_figures.py`
  - generated PNG benchmark charts under `docs/figures/`
  - illustrative circadian adaptation GIF (`docs/figures/circadian_sleep_dynamics.gif`)
  - interactive Plotly HTML chart outputs under `docs/figures/`
- Docs dashboard and hosting:
  - `docs/index.html` interactive chart dashboard
  - `.github/workflows/pages.yml` for GitHub Pages deployment
  - `docs/.nojekyll` for static pages compatibility
- Ownership and governance metadata:
  - `.github/CODEOWNERS`
  - `docs/model-card.md`
  - `docs/figures/README.md`
- Continual-shift comparison benchmark for retention vs adaptation:
  - `src/app/continual_shift_benchmark.py`
  - `scripts/run_continual_shift_benchmark.py`
  - `tests/test_continual_shift_benchmark.py`
  - shifted/rotated dataset support in `src/infra/datasets.py`
  - new `hardest-case` profile in continual-shift CLI for a stronger stress scenario

### Changed

- ResNet benchmark defaults now enable adaptive sleep budget scaling by default while keeping reward-modulated learning disabled by default.
- Updated circadian unit tests (NumPy + Torch) with coverage for reward scaling and adaptive budget behavior.
- Updated README, model card, and core module docs to document new circadian controls.
- Enhanced benchmark visuals with a compact combined overview figure (static + interactive) and linked it in README/dashboard for faster comparison.
- Added hardest-case dynamics GIF (training progression + inference decision-map evolution) and surfaced it near the top of README and docs dashboard.
- Added an interactive Plotly hardest-case dynamics page with playback controls and circadian internals visualization (node/edge weights, chemical/plasticity state) on the docs dashboard.
- Increased hardest-case difficulty substantially (higher drift/noise, lower phase-B train fraction, longer training horizon) and raised hidden-layer width in hardest-case runs for all three models.
- Added multi-hidden-layer support across NumPy baseline models (backprop, predictive coding, and circadian with an adaptive top hidden layer plus trainable pre-hidden stack).
- Refreshed README benchmark section with a latest master verification run on 2026-02-28 and added raw output artifact under `docs/benchmarks/`.
- Repositioned repository messaging to Circadian Predictive Coding as the primary focus.
- Updated `README.md` with:
  - circadian-first project framing
  - reproducible benchmark commands
  - badges, mermaid circadian loop diagram, benchmark visuals, and results snapshot tables
  - dashboard and interactive chart links
  - governance and citation references
- Updated `ARCHITECTURE.md` with clearer module boundaries and circadian-centric design intent.
- Updated `CONTRIBUTING.md` with concrete contribution workflow and quality gates.

### Existing Core Work (Carried Forward)

- Backpropagation, predictive coding, and circadian predictive coding implementations.
- Circadian mechanisms including chemical gating, adaptive sleep policies, split/prune logic, and rollback support.
- ResNet-50 benchmark path comparing all three model families.
- Test suite and deterministic data generation for reproducible experiments.
