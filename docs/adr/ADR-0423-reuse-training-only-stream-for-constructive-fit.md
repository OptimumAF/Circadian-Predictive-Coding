# ADR-0423: Reuse the training-only stream for a constructive fit diagnostic

## Context

R2.1 still lacks acquisition of all six original TRAIN traces. Exact input
aliasing is absent, but original BP128 fits45/54 and a linear least-squares
control fits52/54; neither failure establishes finite H8 incapacity. Activation,
gradient, margin, reserved-task and input-bound diagnostics already exist.

## Decision

Add a separately named disposable diagnostic with the unchanged H8 BP recipe
and a fixed maximum512-update work cap. Reuse the existing invariant learner,
one-wake catalogue template, delayed training-only stream, original budget and
saved-source CLI. Score all fixed checkpoints only after training; persist raw
probabilities and unknown/partial work. Keep all benchmark defaults frozen.

The orchestration function owns invocation-local guards and completion state;
these closures deliberately stay together. Model math, source validation,
delayed update authority and budgets remain in their existing modules.

## Alternatives

A new optimizer or runner would confound the question and duplicate working
ports. Repeating128 updates would duplicate an accepted result. Tuning seeds,
stopping early on a successful checkpoint or replacing the baseline would alter
the scientific comparison. Another alias inventory would add no new evidence.

## Consequences

One prospective local case can provide a constructive fit witness or preserve a
negative result. It is not matched-compute evidence or proof of incapacity.
The stream owns up to512 input-array copies and at most3 checkpoint states;
wall/RSS, projections, snapshots and score work remain explicit. No deployed,
benchmark, reserved or final evaluation state enters this diagnostic.
