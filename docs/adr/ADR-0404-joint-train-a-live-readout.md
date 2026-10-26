# ADR-0404: Reuse the target-free A readout for joint checkpoints

## Context

Fixed64 PC/neutral controls improve some teacher-forced fit but acquire0/6exact
traces. The existing conflict reducer already reports54unique canonical inputs
and0conflicting labels. Rebuilding that readout would not answer another learner
question. The current A-game helper instead rejects genuine joint128/6912 states,
assuming nine rows per update and an A-only training selector.

## Decision

Allow the existing train-a-rollout option with exact joint prerequisites and no
extra diagnostics. Reuse its evaluator, frozen runner, paid-prefix handling,
separate zero-update budget and original A-blue/red game plan. An explicit
catalogue_method binds only the six existing exact recipes, A-only evaluation and
checkpoint128. Reuse the shared full-policy/54-row counter; require original
untrained source0 before forking and throughout subsequent snapshot callbacks.
Native algorithm identity remains separate from fixed64 outer recipe aliases.

Keep old omitted/A-only evaluator admission, game policy and learning doses.
Report live predictions/steps/projection/time separately. No E/F/final or teacher
authority is granted to decisions. Why this: actual frozen policy games provide
a runnable connection between teacher-forced fit and agent behaviour without
changing a learner or creating another orchestration framework.

## Alternatives

- Duplicate label-conflict reporting: existing measured evidence already answers it.
- Ignore row counters or accept arbitrary expected policies: breaks source/state
  admission and permits relabeled training history.
- Add another evaluator or train a new policy: unnecessary for the existing
  genuine frozen checkpoint and target-free game runner.

## Consequences

Known-task live behaviour becomes reachable from the joint runner. Correct
synthetic/native integration does not prove navigation, useful adaptation or a
circadian advantage. Existing failed/SPENT experiments and scientific requirements
remain unchanged; any real-game measurement needs a separate frozen budget.
