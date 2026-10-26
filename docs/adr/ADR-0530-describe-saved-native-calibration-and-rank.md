# ADR-0530: Describe saved native calibration and rank evidence

## Context

The terminal consolidation run is accepted as a negative: retained A2/2, B0/2,
anchors pass; A/B NLL worsen versus warm. R5.4 asks for rank/calibration/
task-retention results versus unchanged and native-adapter baselines. Existing
reports retain probabilities, fixed readout identities, spectra and costs.

## Decision

Add one saved-report CLI using four fixed accepted report pins, bounded input
and fresh output. Reuse the existing probability decoder, Brier/rawNLL/five-bin
ECE functions and original task criteria. Require raw response, typed result
and saved metric agreement and exact source-group/role/request/target identity.
Describe retained spectra without another SVD or rank experiment. Preserve
historical total costs separately from derivation's zero model work.

Why this: required descriptive evidence can be computed from already-paid
predictions. Another intervention or model run would not be needed for these
metrics and could encourage selecting a favorable result.

## Alternatives

- Fit a new temperature or select a new rank: changes the scientific question
  after observing outcomes and requires a separate protocol.
- Duplicate metric formulas: risks inconsistent ECE confidence/bin/NLL rules.
- General archive loader: unnecessary for four specific accepted inputs.

## Consequences

No new model/runtime/core API or dependency. Fixed source paths/pins intentionally
limit this CLI. Six cases are descriptive, not strong calibration evidence.
Different intervention timing/history and extra computation remain explicit.
Reported pretruncation rank is not installed float32 rank, memory saving or
performance at untested ranks. Negative results and stronger open scientific
claims remain preserved independently of scoped software acceptance.
