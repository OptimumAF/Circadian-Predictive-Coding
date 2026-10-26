# ADR-0453: Explicit binary-gripper objective in continuous BC

## Context

More fixed optimization, frozen normalization and a public episode clock have
not yielded task success for the current MLP baseline. Gripper labels in the
qualified TRAIN demonstration have exactly two levels, and some predictions
retain the wrong sign. This motivates testing classification on that actuator;
it does not prove the cause of failure or qualify a strong BC baseline.

## Decision

Add an optional last policy field declaring a normalized gripper axis with
endpoint labels. Use logit2z cross entropy on that axis and unchanged normalized
half-MSE on the remaining active axes, averaging by the original rows/active
axes. Preserve default all-axis MSE, predictions, initialization and parameter
counts. Full state owns the declaration; training objective names differ, while
common evaluation diagnostics and task metrics remain unchanged.

Why this: test a task-label interpretation through the existing learner, without
an extra classifier, runner, objective weight or action threshold.

## Alternatives and consequences

Implicitly thresholding continuous labels changes data semantics. Extra output
parameters would change initialization and comparison width. A tunable axis
weight adds another selection opportunity. Explicit strict labels and the
sigmoid(2z) interpretation avoid those changes. Legacy default metadata needs a
narrow missing-field-to-null compatibility rule in its existing readers; exact
prechange source copies and all old results are preserved. This is not a new
persistence codec or a claim that the new objective improves robot behavior.

604 focused cases/scoped statics and one matched TRAIN study pass independent review. Aggregate common error improves modestly while grasp/align/translation errors worsen. No native binary-objective case was run; full robot and broader scientific/release requirements remain open.
