# Changelog

All notable changes to this project are documented in this file.

The format is inspired by Keep a Changelog and this project follows semantic intent
for versioning even while in research-stage development.

## [Unreleased]

### Added

- Verified artifact-only v14 static dashboard and four final-metric PNG
  plots from the descriptive table. Exact regeneration rejects stale or
  hand-edited content; the historical dashboard gains an in-page
  provenance warning without changing its charts
  (ADR-0138).
- Artifact-only v14 descriptive summary table with all configured
  arms/methods, seed count, final-metric observed range, source commit,
  protocol, track, and explicit within-bundle failure scope. It verifies
  the completed source and re-derives its exclusive JSON/CSV output
  without editing fixed v14 results (ADR-0137).
- Opt-in measured absolute process-RSS ceiling for budgeted toy runs.
  Checked wake/sleep/final stops and a final publication check use a
  per-invocation sampler; CLI state records start/observed peak/sample
  facts or a typed unavailable-host error. The sampled ceiling is soft
  and checked resume starts a new memory segment (ADR-0136).
- Opt-in run-level circadian hidden-width ceiling for toy runs. It checks
  selected split growth before sleep or final proposal mutation, including
  transient width that later pruning hides; checked resume and CLI state
  carry current and historical peak width separately (ADR-0135).
- Opt-in total replay-example limit for budgeted toy runs. The core checks
  selected batch lengths before sleep mutation; checked resume restores
  applied exposure, and the CLI records observed/durable replay work with
  a distinct incomplete reason (ADR-0134).
- Opt-in toy CLI update/time flags with an exclusive versioned local run
  state. Completed, incomplete, and error attempts record their budget,
  observed/durable work, reason, and checked checkpoint identity; unsafe
  resume and overwrite are refused (ADR-0133).
- Opt-in total wake-update and per-invocation wall-time limits for the
  toy comparison API. A typed incomplete stop carries exact work and
  checked resume position without final scoring; default runs retain
  their old config and result identities (ADR-0132).
- A prelaunch estimate and explicit planned-update ceiling for the legacy
  three-family Pareto sweep. Its unchanged 34 candidate lists across three
  seeds plan at most 81,600 training updates; default launch refuses the
  work before Torch/data setup, while `--estimate-only` reports it without
  training (ADR-0131).
- A prelaunch work estimate and explicit planned-update ceiling for the
  legacy circadian-policy sweep. `--estimate-only` opens no Torch or data;
  the 18-candidate default launch is refused at the 1,000-update automatic
  ceiling until a higher limit is supplied (ADR-0130).
- A named typed preset and full exclusive configuration/result artifacts
  for the root single-run ResNet CLI. All 110 old defaults and legacy
  flag behavior remain; typed overrides and malformed settings reject
  before the Torch runner (ADR-0129).
- A named historical toy CLI preset, typed existing-field overrides, and
  complete baseline/ordered indepth configuration artifacts. New baseline
  JSON results retain their report fields and include the exact resolved
  settings; invalid inputs reject before training (ADR-0128).
- A typed historical unmatched preset and complete base/per-seed
  configuration record for the multi-seed ResNet export, with strict
  existing-field overrides and pretraining validation; historical
  baseline rates and validation winner logic remain unchanged
  (ADR-0127).
- Declared a single typed `fixed-v14` preset for the versioned bundle,
  rejecting unknown settings before training while retaining its exact
  raw result hashes; audited active experiment entrypoints (ADR-0126).
- Typed, explicit overrides for existing continual-shift presets and
  an exclusive fully resolved configuration artifact, with unknown,
  duplicate, nonfinite, and baseline/protocol field rejection before
  training. The fixed v14 comparison remains unchanged (ADR-0125).
- Opt-in fixed v14 trial-prefix resume with a typed format-10 trusted
  checkpoint, hash-bound hidden run-state cursor, sealed final sources,
  distinct incomplete/failed/canceled states, and exact fresh/resumed
  raw and measured result parity (ADR-0124).
- Atomic same-volume publication for completed v14 run bundles and
  observation sidecars/projections, with hidden incomplete/failed/
  canceled staging records and no partial public result directory
  after caught failures (ADR-0123).
- Opt-in genuine v14 wake metrics from existing NumPy training return
  values, stored in a separate completed-run sidecar and additive
  measured JSONL/CSV projection. Two bounded local runs repeat all
  metric and final data bytes without changing fixed v14 protocols,
  baselines, or scores (ADR-0122).
- Opt-in P5.2a versioned projection of verified v14 train-only and scored
  records into deterministic JSONL and final-row CSV, with exact
  re-derivation verification and explicit unavailable wake metrics;
  original v14 results remain byte identical (ADR-0121).
- Opt-in versioned v14 local run bundle with strict P5.1 provenance schema,
  execution Git/workspace and environment capture, exact source-role and
  payload hashes, complete-grid disk verification, and unchanged fixed
  v14 JSON bytes (ADR-0120).
- Current NumPy/Torch backend capability matrix and hash-bound v14
  NumPy-only result metadata sidecar; fixed v14 artifacts remain byte
  identical and no Torch replay mechanism was added (ADR-0119).
- Fixed v14 globally sealed full-stack trigger outcomes for all six
  seed/arm trials and three NumPy methods, with exact matched replay,
  structural lineage preflight, common final roles, all paired metrics,
  and deterministic local JSON (ADR-0118).
- Fixed v14 train-only guarded NumPy runner for periodic, unchanged
  adaptive, and no-sleep arms with matched replay only after accepted
  circadian sleep; six-trial preflight and deterministic unscored JSON
  preserve the final-role seal (ADR-0117).
- Fixed v14 train-only all-opportunity shared replay schedule for future
  full-stack periodic/adaptive/no-sleep controls; exact v9 periodic
  parity, sealed roles, and repeatable local JSON without model scores
  (ADR-0116).
- Fixed v13 NumPy sleep-trigger timing comparison across stationary noisy
  and axis-shift streams: globally sealed periodic, unchanged adaptive,
  and no-sleep arms with equal wake work/capacity, deterministic local
  JSON, and a zero-event adaptive result at its defaults (ADR-0115).
- Fixed v12 NumPy/CPU-Torch 32-cell structural comparison with independent
  wake modulation, importance-history reward weighting, and importance
  score factors; globally sealed final roles, equal work/change caps,
  deterministic local JSON, and no observed history-weighting benefit
  in this bounded run (ADR-0114).
- Read-only NumPy/CPU-Torch structural rank audit separating the existing
  reward-weighted importance history from ordinary importance scoring
  and wake learning-rate changes; no new ranking heuristic (ADR-0113).
- Fixed v11 NumPy/CPU-Torch matched difficulty comparison with 24 globally
  sealed clean, label-flip, and feature-outlier trials, equal work, local
  deterministic JSON, and a null modulation result (ADR-0112).
- Fixed train-only NumPy/CPU-Torch difficulty-signal audit for clean,
  label-flipped, and feature-outlier rows; no training heuristic selected
  (`docs/difficulty-modulation-audit.md`).
- Phase-budget regression coverage for NumPy typed structural proposals;
  separated request parsing, eligibility, and ranking while preserving
  Torch's distinct post-split planner (ADR-0110).
- Opt-in NumPy `wake_only_adaptive_v1` replay side-effect policy and a
  globally sealed eight-trial matched-control ablation; the fixed result
  is null for side-effect choice and retains the baseline advantage
  (ADR-0109).
- Versioned NumPy matched replay controls with one train-only FIFO or
  seeded bottom-k schedule for circadian, predictive coding, and backprop;
  guarded applied-work accounting and a globally sealed fixed two-seed
  outcome artifact (ADRs 0106–0108).
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
