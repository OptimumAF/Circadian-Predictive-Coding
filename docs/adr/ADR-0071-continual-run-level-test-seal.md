# ADR-0071: Seal all configured seeds before final-test access

## Context

V4 with two seeds scores and commits the first seed before training the
second. A source-level sentinel raised on seed 17's final-test input while
only seed 17 had finished. Per-seed isolation therefore cannot support a
run-level setting freeze or a later seed's independent training decisions.

## Decision

Add opt-in `continual_global_test_seal_v5` with a distinct config type.
The ordinary runner first trains both phases for every configured seed,
holding model states and deferred final-test roles in memory. Only after
the last seed finishes does it bind held-out hashes and score seeds in
their original order. V5 inherits v4's phase-local sleep and bounded
observed-example replay; v1–v4 execution paths and outputs remain
available. The first v5 increment rejected checkpoint requests before
data loading because the per-seed completed-result format could not
represent unscored trained seeds safely. ADR-0072 subsequently added a
separate format-5 unscored-state checkpoint.

## Alternatives

- Change v4's seed loop in place. Its checkpoint format commits test
  digests and reports after each seed, so a timing change would silently
  change existing resume semantics.
- Score only the last seed. That discards declared evidence and cannot
  support the existing aggregate report.
- Regenerate and retrain earlier seeds at reporting time. That adds
  unreported compute and creates another state-matching risk.

## Consequences

Two-seed source sentinels now pass in both model orders. V5's per-seed
reports equal v4's exactly for the same data and settings, while final
labels are first read after both seeds train. Perturbing seed 17's final
labels changes its test hash but leaves both trained states and seed 19's
report unchanged. The route retains every pending model and role until
scoring, increasing memory roughly with seed count. V5 is one frozen
configuration with no inner guard/outer-selection split, no setting
selection, and no arrival ledger. Its tiny scores are descriptive and
cannot be promoted as strict-online confirmation. Format-5 checkpoint
continuation is covered separately by ADR-0072.
