# ADR-0069: Delay ordinary continual final-test access until training ends

## Context

Checkpointed v4 runs request `hash_test=False` when building phase roles
and bind final-test hashes after all Phase B training. The ordinary v4
path still used the default splitter behavior, which hashed Phase A's
final-test inputs and labels before Phase A training and Phase B's
before Phase B training. A raising hash sentinel exposed the difference
in both model orders. Final-test label perturbation did not alter trained
state, but early hashing violated the intended label-access boundary.

## Decision

For v4 ordinary runs only, construct both phase roles with final-test
hashing disabled. Once all three models have finished both phases, bind
the two held-out hashes, then score. Keep the v1/v2/v3 ordinary timing
and all checkpointed behavior unchanged. Do not pass held-out roles
into either phase training helper.

## Alternatives

- Leave early hashes because they do not currently affect training.
  This would allow a later decision path to use a premature label read.
- Change the common splitter default. That would change reviewed v1/v2/v3
  behavior and unrelated callers.
- Treat the existing early materialization as proof of strict-online
  label arrival. The synthetic generator does create held-out labels
  before training; the runner's access seal is a narrower claim.

## Consequences

Raising role and hash sentinels now pass in both model orders and in
ordinary/checkpointed v4 execution. Perturbing only final-test labels
changes test hashes and scores, while development-role hashes, all three
trained states, sleep decisions, and replay choices remain identical.
V1/v2/v3 results are unchanged. The source generator still materializes
held-out labels early. Physical label-release records and disjoint
inner guard/outer selection roles remain P1.3c3b; full strict-online
acceptance and bounded confirmation remain open.
