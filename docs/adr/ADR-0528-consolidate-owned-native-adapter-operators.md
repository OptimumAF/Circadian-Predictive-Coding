# ADR-0528: Consolidate two owned native adapter operators

## Context

The fixed rank-four projection experiment learned B and lost A. That valid
negative result does not establish retention from rank reduction. R5.4 still
requires actual adapter consolidation; the pure multi-term algebra exists, but
the native owner could previously project only its current factors.

Ordinary native snapshots expose caller-mutable tensors and contain head,
gradients and optimizer state that this operation does not need. Accepting such
a snapshot as unchanged historical evidence would require additional provenance
or a weaker contract.

## Decision

Append default-off consolidation authority requiring the existing one-use
projection and reset allowances. Capture one bounded private float32 factor
reference at an explicit caller boundary. Keep object/version and module
metadata guards; return detached, tensor-free metadata. Continue ordinary TRAIN,
then project the operator `.5 * reference + .5 * current` to a declared rank via
the accepted core and existing native staging/reset/publication path.

Why this: a single factor-only reference supplies the missing two-state learner
behavior without another loader, runner, authority registry or hashing system.
It avoids factorwise averaging cross terms. Current factors already contain
the earlier history: this is interpolation of complete operators, not addition
of independent task deltas. The current scalar head is preserved explicitly.

Capture feasibility precedes copies. Standalone projection/reset/fork refuse
while the reference is pending. Reference integrity is rechecked after reset
and before publication. Success releases the private tensors; permanent capture,
consolidation, projection and reset costs/allowances survive restore and fork.
Paid failures poison and preserve prefixes; they do not refund or roll back.

## Alternatives

- Borrow a mutable full snapshot: does not prove unchanged historical factors.
- Retain unlimited references: unnecessary and unbounded for this outcome.
- Average A and B separately: introduces unwanted cross terms in their product.
- Add another runner or tune ranks: neither is needed for owner correctness.
- Copy a pending reference during fork: hidden storage/copy costs and unnecessary
  lifecycle complexity for the first stream integration.

## Consequences

One private factor slot and bounded historical receipts; no new data roles,
labels, selection rule, random initializer, dependency or app/core interface.
Current head/base/Parameter identities/allocated rank remain fixed. Inactive
coordinates may regrow; truncation tails describe the factor operator, not
rounded native error or task performance. Additional memory/compute are charged.
No task benefit, return recovery, calibration improvement or CPC advantage is
inferred from literal owner qualification. Existing stream integration and a
separate frozen native study remain required.

[Protocol and scoped evidence](../../artifacts/runs/native-adapter-consolidation-20261022/PROTOCOL.md).
