# ADR-0146: Compare every schedule training fact before outer scoring

## Context

P6.3c5 completed two byte-identical train-only schedule results on all 33
matched cells. Periodic replay committed; the current adaptive policy stayed
inactive under its fixed variance threshold. The next task requires all
policy outcomes and paired costs without changing that protocol or using
outer values to choose its completion path.

## Decision

Add a separate c6 scored orchestration that retains after-A model copies,
reuses frozen c5 helpers, and exactly compares the complete all-seed train
object before reading any outer input/label. The public boundary verifies
the saved c5 bundle and binds its own selected source/adapter identities.
Reuse c4 scoring types/arithmetic and the core two-task metric contract.
Publish all 33 cells and all nine within-method policy pairs per seed,
including inactive/null/negative rows. Preserve c5 source bytes and all
confirmation/final seals.

## Alternatives

- Score each completed seed immediately: a late mismatch could occur after
  earlier scores were known, violating the all-seed gate.
- Edit the frozen c5 trainer to return models: invalidates its source witness.
- Train c5 and then retrain for scoring: doubles work and weakens checkpoint
  identity. The explicit small scored orchestrator reuses all train helpers.
- Force an adaptive event or exclude its identical no-sleep rows: changes the
  protocol and suppresses a valid inactive finding.

## Consequences

The app retains small additional model copies and explicitly depends on c5
private helpers and c4 scoring helpers. Hash binding and complete fact equality
make that dependency reviewable. Scores are descriptive development evidence;
different replay/guard costs remain visible. Full-minus-one, growth controls,
independent confirmation and final release still require separate gates.
