# ADR-0007: Version toy and continual NumPy evaluation splits

## Context

The toy and continual runners used only training and test arrays. The
continual runner already deferred final-test scoring until both training
phases finished, but neither runner saved split identities or had a
validation role. Historical benchmark text and commands used all original
training examples, so reserving a holdout changes the task.

## Decision

Default to `toy_validation_v1` and `continual_validation_v1`. Reserve a
deterministic, class-stratified 20% holdout from each original training
split, record hashes for every role, and pass only training arrays into wake
and replay updates. The phase-B scarcity fraction is applied to the
remaining training examples. The continual runner generates phase-B data
only after phase-A training and scores final test only after all models
finish phase B. The toy runner reports validation accuracy after training,
followed by final test accuracy. Reports and CLI output name the protocol.

Keep `toy_legacy_train_test_v0` and
`continual_legacy_train_test_v0` as explicit routes with the old sample
allocation. Refuse to overwrite an existing continual output file.

## Alternatives considered

- Silently reserving validation in the old route would make historical
  settings appear comparable despite fewer fitted examples.
- Reusing test examples as a validation guard would leak final labels.
- Removing the old route would make historical behavior harder to reproduce.

## Consequences

Corrected and legacy scores must not be pooled. The NumPy validation sets
are currently recorded and reported; they do not yet control sleep or outer
model selection. Continual sleep scheduling still knows the planned A+B
duration, so this does not claim strict-online operation. Direct role-access
tests cover wake, replay, adaptive triggering, and threshold resolution;
remaining vision and aggregate-report gates stay in the development plan.
