# ADR-0582: Predeclare fresh TRAIN windows under one lifetime owner

## Context

The live ticket runner consumes a single bounded TRAIN window before publication.
Its inbox already supports paired observation/label versions. After publication,
relabeling old feedback or constructing a fresh budget would lose causal or
resource history. The private learner and deployed actor are separate owned models.

## Decision

Add an explicit immutable two-window plan to the existing live runner. Declare
all capacity, update/decision limits and later source ticks before work. Keep the
same runtime, full private native state, clock, budget, sharing gate and action
envelope. Use the existing first-window drain/guard/typed-commit boundary.

Only a verified original publication admits the later window. Its predictions
and labels retain their actually observed actor version. Preserve invalidated
old feedback and all cumulative receipts. Recheck current generation after
prediction and before admission; revalidate exact plan types. Retain incomplete
prefixes and permanent charges. There is no forced drain or implicit retry.

The published actor remains its first complete snapshot while the private owner
continues. Original full rollback is preserved. Original single-window defaults
and final/controller-study separation remain unchanged.

Why this: version pairing and the owned runtime already exist. One finite opt-in
schedule improves continued learning without a new framework or authority reset.

## Alternatives and consequences

- New runtime/budget after publication: rejected; lifetime accounting would reset.
- Rebind previous labels: rejected; they were observed under another generation.
- Second publication/candidate rebasing now: deferred; it needs its own explicit
  ancestry contract and is not required to demonstrate fresh private learning.
- Bounded storage: retain the same finite16 source/label/update history and fixed
  publication list; no unlimited full-model success-record chain.

The engineering case passed16updates96rows with deployed104/private112 and
rollback96. This is self-reviewed scoped correctness, not independent acceptance
or a controller/CPC/generalization/efficiency claim. Full parents remain open.
Evidence: issue66 and `artifacts/runs/post-publication-learning-20261023/RESULTS.md`.
