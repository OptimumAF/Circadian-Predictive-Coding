# ADR-0114: Factorize structural ranking without changing historical cores

## Context

P4.7a showed that the current reward-weighted importance EMA can change
split/prune rank at fixed state. That causal possibility does not show
a learning or retention benefit. Turning the existing reward switch on
or off alone would change wake parameter updates and importance history
together, so it cannot attribute a structural outcome.

## Decision

Use a separate v12 app-level factorial with wake parameter scaling,
reward weighting of the importance EMA increment, and importance in
split/prune scores as independent two-level factors. Every cell uses
the existing core reward computation. For the no-wake-scale control,
the app divides only the just-applied parameter delta by the observed
scale. For plain importance history, it divides only the just-added EMA
increment by that scale. A parity test compares both corrections
together to the unmodulated core after a nonunit-scale update. The
controls are confined to this versioned experiment; core configuration,
checkpoints, and earlier protocols are unchanged.

Use new seeds 23/29 and the existing sealed four-role A/B source. Freeze
the 8+8 full-batch update schedule, one A-boundary component sleep,
one split and one prune slot, and all metrics before training. All
thirty-two trials must preflight role hashes, equal initialization,
pre-sleep weight identity within each wake factor, actual A-phase
reward traces, work, stable structural IDs, width, and parameter count
before any final source field opens. Release final roles once per seed,
score every saved A/B state, and repeat local JSON byte for byte.

## Alternatives

- Add a direct reward term to structural scores now. That duplicates an
  existing path without outcome evidence.
- Compare only the historical reward switch. It confounds wake learning
  with structural importance history.
- Change seeds, thresholds, or stopping after inspecting final scores.
  That would invalidate the prospective comparison.

## Consequences

All thirty-two fixed trials trained and applied one split and one prune
under matched caps. Reward weighting of importance history changed no
selected stable ID or held-out accuracy/forgetting pair in this run.
The existing importance score mix changed two prune choices, without a
consistent accuracy benefit. The synthetic study has weak A/B learning
in some cells, so it cannot establish general irrelevance of the
existing signal. It provides no measured reason to add another reward
ranking heuristic. Exact outcomes and limitations are in
`docs/structural-ranking-comparison.md`.
