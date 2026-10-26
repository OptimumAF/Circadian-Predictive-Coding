# ADR-0336: Keep task-fit diagnostics explicitly training-only

## Context

The delayed stream previously required final-test packets even when the intended
question was whether a learner can fit its training episodes. Relabeling copied
training data as final would obscure the metric's resubstitution meaning.
Existing isolated fresh-task probes answer a separate reserved-data question.

## Decision

Add an explicit exact-bool training-only stream mode requiring empty heldout data.
Reuse the existing runtime, release loop, receipts, budget and checkpoint observer.
Finish seals training and performs no heldout scoring. Bind the admitted mode
across continuation and leave the default evaluation contract/report unchanged.

Compose one fixed train-fit recipe on owned learners/checkpoint forks. Derive a
non-learning prior/majority reference solely from training labels, retain every
fixed checkpoint and count repeated exposure honestly. Training-fit predictions
cannot drive promotion, benchmark updates or an adaptive budget.

## Alternatives

Passing training packets under final-test aliases misstates their role. A new
training loop duplicates established budget/release behavior. Extending reserved
fresh-task probe budgets changes accepted experiments and their question.

## Consequences

The API can support direct train-fit diagnostics while final isolation remains
visible. Correct execution does not prove fitting; even complete fitting does
not establish generalization, retention, recovery or circadian benefit. Reports
must preserve negative curves and diagnostic computation/memory costs.
