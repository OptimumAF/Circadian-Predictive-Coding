# ADR-0395: Check owned immutable schedule leaves without serialization

## Context

The fixed weighted BP comparison reached its original180s cap. A model-free
profile measured repeated complete schedule serialization at about29ms for a
128-episode/6.2MB payload. That is a concrete cost, not proof of total runtime
causality. The schedule must still refuse changed evidence before native work.

## Decision

Use a pure core `DemonstrationScheduleWitness` for bound BP. At admission it
requires exact complete TRAIN records/native primitives/tuples and validates all
episode semantics. It retains tuple and record/dictionary ownership and every
mutable field. Ordinary checks reread all fields, dictionary keys/order/types,
scalar values, signed-zero rewards and scalar string identity. Exact tuples of
exact native floats cannot change in place; changed tuple identities refuse.

On construction/fork, full copied values/float bits/string references are checked
against the admitted owner before binding fresh child witnesses. Existing cached
payload bytes remain identity-bound; ordinary BP operations need no serializer.
The recurrent adapter keeps its original validation and pickle path.

## Alternatives

- Repeated pickle proof is complete but has measured recurring cost.
- Skipping checks, TTL sampling, broad counter exemptions or hashing only selected
  fields would fail the unchanged-evidence requirement.
- A new persistent format or success chain adds unnecessary state and storage.

## Consequences

Witness storage is bounded by512 episodes×16 transitions per owner, with no
history. It adds setup work and memory while reducing repeated validation cost.
Private frozen witnesses are trusted authority like prior private byte seals;
this is not cryptographic defense against replacing all private authority.
Equivalent private tuple/string replacements can be refused more strictly than
before. Callers should use public fork/restore instead of mutating owned data.
Core depends only on core; adapters compose it with native state publication.
No objective, learner update, release/clock/budget, probe or final access changes.

[Focused behavior, timing, costs, failures and review](../../artifacts/runs/weighted-schedule-witness-20261018/RESULTS.md).
