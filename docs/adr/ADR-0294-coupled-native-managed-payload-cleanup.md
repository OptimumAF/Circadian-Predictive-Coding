# ADR-0294: Compose coupled native ports with existing managed cleanup

Date: 2026-10-15. Status: accepted; R3.7x validated in the retained stage RESULTS/REVIEW.

## Context

Bare managed consent prevents future use but retains raw inbox/checkpoint data.
The existing lifecycle coordinator can clear delivered registry-owned copies and
invalidate pending checkpoints. It requires explicit native footprint/erase and
payload measurement ports; coupled native learners currently retain no raw replay
batches in learned state.

## Decision

Provide a small adapter with typed zero-replay footprint/erase callbacks for exact
CoupledLearner instances and byte measurement for exact real numeric NumPy arrays.
Compose these callbacks into the existing lifecycle coordinator before admission.
Refuse optional owned-copy or elapsed-retention policies explicitly until their
additional measurement ports are supported. Keep all basic ingress, tick, holder
and preparation limits intact.

Add a separate fixed six-case `--managed-cleanup` integration. Transfer one
candidate, retain both runtime owners and one caller inspection copy, then opt out
a subject while a checkpoint is pending. Require the existing conservative cleanup
to revoke all delivered keys, clear both inboxes and invalidate the pending token.
Compare complete native state/policy and original accounting before and after.
Demonstrate exhausted lifetime ingress still refuses new data after raw cleanup.

Why this: explicit outer callbacks reuse proven app orchestration without adding
another cleanup implementation or changing native learning rules.

## Alternatives

Changing application code to import coupled adapters would reverse dependencies.
Returning unsupported policy settings as though enforced would weaken resource
contracts. Deleting learned parameters would implement a different operation.

## Consequences

Cleanup conservatively includes other delivered subjects and must report those
revocations. It drops registry-owned raw references and records payload-free
erasure metadata. Caller snapshots, callback-owned data and RAM overwrite remain
outside the guarantee; learned parameters remain and this is not unlearning.
No process recovery, provenance attestation or comparative learning benefit claim.
