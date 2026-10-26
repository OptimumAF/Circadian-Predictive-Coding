# ADR-0412: Exact weighted PC proposal readout

## Context

The frozen joint study's weighted PC/neutral learners fit16/54 TRAIN decisions
versus ordinary30/54. This is a negative result, not evidence for another seed,
horizon or baseline change. The existing realized-update diagnostic described
only the unweighted path. Ordinary PC already uses CE update directions; its
squared-error statistic is a legacy diagnostic. All-one native parity is accepted.

## Decision

Extend existing diagnose_pc_update with explicit class_weights=None. None keeps
the original report and numerical path. Explicit validated seven-class weights
use class_weighted_pc_update itself to obtain the exact detached rounded native
candidate. This adds no learner update or new update law.

Report ordinary and weighted prior CE separately. Geometry compares weighted
prior-CE gradient with rounded parameter delta. Fixed-state energy after the
proposal keeps the original final latent z, recomputes the candidate prior, and
uses the candidate output weights; it is not another latent relaxation. Preserve
source/snapshot/gradient/RNG/mode guards and partial failed-work receipts.

## Alternatives

- Reusing unweighted formulas would misdescribe the actual weighted update.
- Copying the weighted relaxation would risk a second divergent update law.
- Changing training recipes before measuring this result would mix hypotheses.

## Consequences

93 scoped cases pass, including independent NumPy arithmetic, complete original
diagnostic regressions and four exact native PC/neutral S2/S64 oracle cases.
Numerical proposals consume work even though native_train_calls/optimizer_steps/
published_updates are zero in the diagnostic. Primitive-port counters separately
charge attempts/returns/rows/nominal work; internal intermediate traces are not
collected. Weighted public joint readouts still refuse the option until their
existing source/actor/checkpoint/report/work guards support this new schema.
No comparative experiment or scientific benefit is established by these tests.
