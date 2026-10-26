# ADR-0373: Fixed independent-data replay removal in the existing recipe

## Context

R2.8 had accepted component controls and an exploratory historical join, but no
prospectively frozen family after explicit replay removal. Existing final data
had already been inspected and cannot become fresh confirmation evidence.

## Decision

Expose one fixed removal-confirmation scenario through the existing rehearsal
CLI. Reuse the ordinary stream, native factory, wake/replay sessions and clocks.
All four methods share original initialization23 and prospectively fixed new
TRAIN7302/FINAL9102 data. The shared builder adds validated keyword-only RNG
partitions while preserving old defaults, IDs and bytes. A small adapter composes
eight arms and descriptive pairs; private _run_one remains an internal adapter
boundary, not a new public learner interface.

Why: independent data answer the bounded component-removal question without
rerunning accepted baselines or selecting settings from their final scores.
Keep all endpoints, unequal work and negative tradeoffs. No new optimizer,
selection, margin, runner, evaluation authority or tuning surface in the CLI.

## Alternatives

Relabeling old inspected data as confirmation would be invalid. Another full
runner would duplicate working arrival, source ownership, replay and final-score
isolation. A larger seed/hyperparameter sweep is unnecessary for the bounded
original task and would not make equivalence valid without a prespecified margin.

## Consequences

The single192update/1200row family passes current focused and native gates with
independent review. Record the minimal bundle as unresolved; no circadian benefit,
equivalence, generalization or equal-work effect follows. Original R2.8 permits
this finding. Broader scientific and release requirements remain open. Final
scores cannot guide subsequent learner settings. See stage RESULTS/REVIEW for
exact protocol, failures, measured work and source assumptions.
