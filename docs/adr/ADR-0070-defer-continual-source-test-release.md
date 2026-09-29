# ADR-0070: Defer v4 source test-field release

## Context

V4 disabled final-test validation and hashing during training, but the
shared train/validation splitter still read `DatasetSplit.test_input` and
`test_target` immediately to construct a `LabeledData` role. A source-level
sentinel failed before Phase A training in ordinary and checkpointed runs,
in both model orders. The runner did not use those values for decisions,
but the source boundary could not withhold them.

## Decision

Add an opt-in deferred final-test role to the synthetic split helper.
With `hash_test=False` and `defer_test_access=True`, it carries the source
reference and reads its test fields only when final hashing and scoring
open the role. V4 alone requests this option for A and B in both execution
paths. Reject a request to defer access while hashing the test. Keep the
v1/v2/v3 split behavior, configuration identity, and checkpoint formats.

## Alternatives

- Make every `hash_test=False` call defer source access. That would change
  the role objects carried by the reviewed v2/v3 checkpoint routes.
- Generate a new test set only after training. That changes the data
  protocol and needs a separately declared comparison and checkpoint
  identity; it is not required to close this source-release seam.
- Leave the early field read because the role is not passed to training.
  That prevents a source-level sentinel from enforcing label release.

## Consequences

V4 source input and label fields are not requested until the per-seed A/B
models finish training. Existing test content, hashes, scores, and replay
decisions remain deterministic. The synthetic generator still allocates
held-out arrays when constructing the phase; this decision delays release
through the source interface, not array generation. The deferred role
retains its source object until scoring. Across multiple seeds, a completed
seed is scored before later seeds train. Disjoint inner guard and outer
selection roles, a global setting-freeze boundary, and a role/label
arrival ledger remain open under P1.3c3b. This is not full strict-online
evidence.
