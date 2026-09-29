# ADR-0117: Commit matched replay only after a v14 guarded sleep

## Context

The v14 all-epoch schedule offers prediction-independent rows, including
epochs where adaptive or no-sleep arms perform no consolidation. The v9
runner applies replay only at forced periodic boundaries. Applying the
v14 selection to PC and backprop before the circadian inner guard commits
would leave unmatched updates after a rollback. Treating every offered
row as applied work would also misstate the treatment.

## Decision

Keep a separate v14 train-only runner above the existing arrived source,
shared buffer, and guarded sleep function. Arm-specific configurations
change only the declared sleep schedule and current adaptive switch:
periodic forces phase-local interval four, adaptive uses interval zero
with unchanged readiness thresholds, and no-sleep uses interval zero
without adaptive triggering. Source, wake rates, inference counts,
retention, replay learning rate, guard tolerance, and capacity caps stay
fixed. The runner checks circadian retained IDs, order, selected IDs, and
detached content before every decision. It then records one typed event
per wake epoch and gives each baseline private copies of the selected
rows only when circadian guarded sleep accepts and commits the exact two
updates. Rollback, skipped, and core error paths apply no baseline replay.

A separate train-only study rederives the schedule and development roles,
checks all six arms/seeds, initial trainable state within each seed,
per-method wake work, actual replay/structural/guard ledgers, model clocks,
and final-role seals. The local adapter serializes only deterministic
train and decision facts; wall-clock durations and model scores are
excluded. Held-out release and outcomes remain a separate P4.8b2b gate.

## Alternatives

- Reuse v9's periodic runner for all arms. Its schedule omits nonperiodic
  opportunities and rejects the adaptive switch.
- Replay baselines before the inner guard and roll them back later. This
  adds a cross-model transaction for no scientific benefit.
- Give baselines their own event decisions. That would break matched
  replay exposure within an accepted sleep.

## Consequences

Applied replay work can differ across arms when trigger rates differ.
The difference is recorded per method and must not be labeled equal
compute. Guard rollback preserves the circadian snapshot and leaves
both baselines without replay. The train-only result can establish
conditional application and feasibility, but cannot establish an
accuracy or retention benefit. P4.8b2b must audit the full unscored
Cartesian before opening any final role and retain every null or
negative outcome.
