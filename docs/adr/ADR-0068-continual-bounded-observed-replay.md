# ADR-0068: Bound observed-example replay in a versioned continual route

## Context

The existing NumPy circadian replay buffer stores one entire training
batch per wake epoch. `replay_memory_size` limits batch snapshots, not
retained examples or bytes. In the continual runner, a batch can hold
many examples, and there was no retained-ID or memory ledger. The v3
route isolated Phase A sleep scheduling but did not solve this limit.

## Decision

Add opt-in `continual_bounded_replay_v4`, a separate configuration and
seed-result type, and checkpoint format 4. Its configuration declares
positive maximum retained examples and copied-array bytes. Component
sleep and at least one replay step are required so the replay path can
operate independently of structural changes. The NumPy core receives
the budget before training and stores individual labeled training rows.
It deduplicates by SHA-256 of canonical input and training-label bytes,
then keeps the smallest content hashes subject to both caps. This rule
does not inspect error, validation, final-test score, or future-phase
examples to choose retention. Sleep replay may still prioritize among
the retained rows using the existing configured replay selector.

The result records the declared caps and retained IDs/count/bytes after
Phase A and after Phase B. Bytes mean the copied NumPy input and target
arrays only; Python object, deque, hash, and checkpoint overhead are
excluded. At resume, the app recomputes allowed IDs from arrived
training roles, rejects unobserved rows in the active buffer, and checks
the frozen A buffer against Phase A only before any model update.
The core restore also checks buffer shape, uniqueness, and both caps.

## Alternatives

- Reinterpret `replay_memory_size` as an example count. This would
  silently change historical v1/v2/v3 replay and their checkpoints.
- Keep only the newest rows. A small Phase B batch could overwrite all
  retained Phase A examples, making cross-phase replay vacuous.
- Rank retained rows by model error. That adds a model-dependent memory
  selection choice before the strict-online comparison is specified.

## Consequences

V4 preserves the older config/checkpoint shapes and replay semantics.
The hash rule is deterministic and can retain both A and B rows, but its
phase balance is not guaranteed and hash order is not a learned memory
policy. The raw-array byte cap is not process memory. Tests exercise
independent count and byte caps, selected-row provenance in both model
orders, checkpoint A/B continuation, and rejection of a future B row,
unobserved B validation row, or B row injected into the frozen A buffer
before a resumed update. V4 remains a partial protocol: inner guard,
outer selection, complete label-arrival/task-info ledger, and bounded
confirmation remain P1.3c3–c4 and parent P1.3c.
