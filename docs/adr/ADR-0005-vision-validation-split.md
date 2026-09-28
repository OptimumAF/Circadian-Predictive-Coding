# ADR-0005: Reserve validation data for vision benchmark decisions

This records the first corrected vision protocol. ADR-0008 adds a separate
guard role and supersedes the shared-guard selection design below for new
default runs.

## Context

At the reviewed commit, the ResNet benchmark used `test_loader` for early
stopping and pre/post sleep rollback. Its tuning scripts also ranked trials by
test accuracy. Those results cannot be read as untouched final-test estimates.

## Decision

The vision data builder now returns explicit train, validation, and test
loaders with immutable ordered sample IDs and SHA256 hashes. The synthetic
task creates independent, fixed-seed datasets for each role while preserving
the old test seed. CIFAR validation is a seeded, disjoint holdout from the
official training source. It has deterministic evaluation transforms; training
alone may use augmentation. A requested training subset keeps its sample count
when the source has room for the validation holdout. With a full training
source, the validation count is reserved and the training count shrinks.

ResNet epoch stopping and sleep acceptance use validation labels. The
validation set is the current **guard**: it is available before final testing
and its labels are permitted for these inner decisions. Its examples are
excluded from wake training and replay, so its access and storage cost must be
counted in later budgets. `test_loader` remains the final reporting set.

## Alternatives considered

- Reuse official CIFAR test examples as the guard: rejected because it leaks
  the final test labels into model decisions.
- Apply training augmentation to validation: rejected because the guard
  decision would depend on stochastic views and shared transform RNG state.
- Remove the sleep rollback guard: rejected because it would change the
  algorithm at the same time as correcting evaluation access.

## Consequences and open work

The corrected split changes the protocol and may change rankings. Historical
outputs remain labeled legacy/test-informed; no historical artifact is
rewritten. The main vision runner now trains all three models using a view
without the test loader before final evaluation. Compatibility single-model
helpers used by the older tuning scripts still evaluate the final test for
every candidate trial. Those scripts now rank by validation accuracy and write
new files, but per-trial test access remains a protocol limitation. P1.1/P1.2
remain open until final testing is sealed from all selection paths, with
label-invariance regression tests across all models. P1.3
will define a separate outer model-selection set where practical and specify
strict-online continual-learning access before comparative claims. P1.9 will
assign a distinct protocol identifier to corrected artifacts.
