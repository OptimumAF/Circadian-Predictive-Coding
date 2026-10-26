# ADR-0603 — Explicit closed-loop controller-development metric

## Context

The existing contextual controller fit uses TRAIN-derived classifier accuracy,
while its later validation judges complete workflow outcomes. Shared templates
and source state do not make these metrics equivalent. The repository already
has disposable version-bound trials, typed task execution and the outer objective.

## Decision

Add an explicit task option to the existing trial/study. Run private frozen
before/after actors through existing six-task `_episodes`, with controller_train
role admitted before predictor access. Require complete exact phases, tasks,
native/version/work evidence before task measures. Reuse the original objective,
controller and wait/wake arms. New CLI runs four development trials and freezes
only; it performs no live application. Default classifier study remains intact.

Why this: actual local task outcomes address the demonstrated metric mismatch
without a new learning algorithm, orchestration framework or guard relaxation.

## Alternatives

Relabeling classifier correctness as task success would misstate evidence.
Changing frozen82 or validating/refitting against the same application would
invalidate chronology. Running another live comparison before the trial metric
is explicit would leave the underlying uncertainty unresolved.

## Consequences

Task readouts add charged typed calls/copies and sampled costs. Known development
templates are not heldout generalization. Shadow success still differs from
guarded deployed utility and a one-step arm misses delayed repair. Later validation
must preserve frozen metrics, fit, final/probe isolation and honest negative results.
No scientific advantage or repository-wide readiness follows from correctness.
