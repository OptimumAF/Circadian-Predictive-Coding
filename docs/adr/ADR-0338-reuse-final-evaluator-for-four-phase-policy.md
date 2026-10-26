# ADR-0338: Reuse final evaluator for four-phase policy

## Context

The accepted A/B/C_train/A recipe fits C then forgets it on return. Train trace
accuracy does not show what happens when the policy encounters its own actions.
The existing post-training evaluator already owns cached scoring, actual episode
activation, fixed source identities and complete learned-state/cost guards.

## Decision

Add an explicit five-checkpoint four-phase route to that evaluator, leaving
three-phase defaults unchanged. Collect final packets only after256updates and
stream.finish. Score known A/B/C_train and current-run-heldout D at all five
points and both cues. D has appeared in older evaluations; do not call it a new
unseen domain. Centralize the existing C_train descriptor so training/evaluation
use identical geometry. Allow validated owned layout tuples on the existing
live runner, without changing its global defaults.

## Alternatives

Reporting only train accuracy omits actual policy failures. Reusing historical
final packets for learning violates evaluation isolation. Another runner would
duplicate working orchestration and guards. Changing seeds/updates/metrics from
final failures would invalidate the frozen evidence.

## Consequences

All40 episodes, including failure/unreached slots, remain visible. Final work
cannot train, promote or select a learner. Costs include the necessary training
prefix because historical opaque optimizer states were not saved. Exact old
train readouts/receipts can be compared; opaque state identity cannot. Scoped
correctness alone establishes no retention, generalization, recovery savings or
circadian advantage. Future comparisons need separate fixed protocols and
matched computation/exposure controls.
