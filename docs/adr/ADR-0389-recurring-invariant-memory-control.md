# ADR-0389: Compose invariant memory into the existing recurring-task runner

## Context

The raw count runner already executes a bounded delayed A/B/C_train/A stream with
five checkpoints and TRAIN-only native readouts. The invariant count learner and
typed snapshots already exist, but the raw evaluator cannot admit them. The two
deferred A/F native attempts remain incomplete and spent. Further endpoint-chain
infrastructure would not answer a different learner question.

## Decision

Add an exact boolean option and scenario-specific CLI flag to the existing runner.
Compose the existing bound learner and a typed TRAIN-only evaluator entry point
into its existing stream and checkpoint evaluator. Keep raw defaults, count law,
capacity, inputs, exposures, seeds, metrics and final exclusion unchanged.

Why this: this small vertical slice tests successive acquisition and a returning
task using working components. A new runner or an expanded checkpoint framework
would duplicate orchestration without improving this outcome.

Protect private paid training, returned readouts and final publication across
fallible external ports. Missing admitted receipts retain failed/unknown work
plus genuine earlier lower bounds. Keep five bounded private reports, original
budgets/clocks/source proofs and full typed state admission. Account canonical
projection and count work separately from neural SGD and process overhead.

## Alternatives

- Relabel a raw native state as invariant: rejected because representation and
  policy are part of the complete typed learner state.
- Create another experiment runner: rejected because the existing stream,
  source collection, evaluator and outcome helpers already satisfy the protocol.
- Retry the prior A/F cases: rejected; their failed budgets and outcomes remain.

## Consequences

The optional route can answer a narrow known-TRAIN acquisition/return question.
Final and disposable-probe data remain outside this experiment. Scoped tests and
independent source review precede the separately frozen single native case.
Negative or incomplete runs do not authorize a retry. Known-task count success
would not establish held-out skill, equal-compute superiority, neural recovery,
reward learning or circadian benefit; the full roadmap stays open.
