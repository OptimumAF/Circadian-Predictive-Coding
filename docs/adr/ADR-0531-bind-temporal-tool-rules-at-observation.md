# ADR-0531: Bind temporal tool rules at observation

## Context

The original R5.4 result delivery is accepted, with negative consolidation
outcomes preserved. R5.5 needs changing tools/rules and old-domain returns under
declared current truth. Existing fixed-catalog examples keep targets constant.
The linear predictor already masks inactive tools, but the TRAIN compiler only
admits the complete union catalog.

## Decision

Add a fixed pure temporal corpus and a small script composing the existing
AttributionStream, perceptron and evaluator. Keep the known union schema fixed;
explicit default-off subset compilation permits only known current candidates
and an allowed target, preserving the native full-union mistake objective.
Each case binds its observed policy version/target and separate role/group.
Old-domain return keeps v2; a still-allowed obsolete route choice is an error.
Train the full fixed dose first, then evaluate every checkpoint plus unchanged
and frozen-A controls under identical current-policy requests.

Why this: domain recurrence does not imply policy reversal. A mask-only success
must be distinguished from learning a revised rule. Observation-bound labels
also prevent late feedback from being silently reinterpreted by a newer policy.

## Alternatives

- Relabel old packets with current rules: changes the observed training truth.
- Restore v1 on return: tests a different temporal task and rewards obsolescence.
- Another orchestration framework or dynamic output growth: unnecessary; current
  ports already support the required bounded known-union example.

## Consequences

No new dependencies, learner mathematics or default behavior. Future tool slots
and concept vocabulary are public upfront, limiting the claim. Concept overlap
and one deterministic dose do not establish generalization or modern-model/CPC
benefit. Frozen controls need readouts and remain distinct histories. External
native integration, stronger empirical utility and parent R5.5 remain open.
