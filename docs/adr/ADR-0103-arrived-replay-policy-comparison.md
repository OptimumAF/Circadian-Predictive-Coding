# ADR-0103: Compare replay retention on one arrived-role manifest

## Context

P4.2a added bounded FIFO and seeded bottom-k retention in the NumPy core.
The arrived v6 runner requires an exact config type and its format-6 resume
reconstructs the historical content-hash model. V7 selection also binds v6
candidates. Adding policy fields to either route would change saved meaning.
Sleep telemetry counts replay work but did not identify distinct replayed
examples or duplicate wake observations.

## Decision

Use a separate `continual_replay_policy_comparison_v8` ordinary manifest with
one unchanged v6 arrived-role config, ordered unique seeds, and exactly the
declared FIFO and seeded reservoir policies. The manifest and its SHA-256
digest are public result fields. Both policies use the same A/B source and
role splits, training configuration, model order, example/array-byte caps,
sleep schedule, and replay-step limit. Train every policy/seed before
releasing any A or B final-test source field. Retain all scores and per-seed
reports; this route performs no winner selection.

Only explicitly selected nondefault retention policies add an audit ledger
to the model snapshot. It records all distinct wake content IDs, IDs with
repeated wake arrivals, duplicate occurrences, distinct IDs successfully
used in sleep replay, and replay updates. The A and B reports are cumulative;
the B report includes A exposure. A failed or rejected sleep restored by the
existing model transaction does not become applied exposure. Snapshot restore
checks ledger shape, counts, and internal ID consistency before replacing
live state. V8 checkpoint preflight still needs to bind IDs to arrived roles.
The old content-hash and whole-batch snapshots retain their field sets.

The audit ledger can grow with the number of distinct observed or replayed
IDs. Its memory is reporting overhead outside the declared **retained replay
array** budget; the public `max_bytes` remains copied input/target arrays,
not Python object, model, or audit memory. This differs from the retention
algorithm, which needs no global seen-ID ledger. Checkpointed v8 continuation
needs its own format and preflight before P4.2b is complete.

## Alternatives

- Add policy to v6/v7 config or reuse format 6/8. That would silently change
  their identity and checkpoint interpretation.
- Infer distinct exposure from sleep update counts. One row can replay more
  than once, so the count cannot identify distinct content.
- Choose a policy from the observed final scores. That would consume final
  labels for model selection and bias this comparison.

## Consequences

The fixed two-seed, two-policy local run uses 40 source rows per phase,
two wake epochs per phase, 4-example/96-array-byte caps, and one replay
step per scheduled sleep. Both policies applied four replay updates per seed
and kept four rows at both phase boundaries. They retained different B
content IDs and had different distinct replay-exposed counts, while their
balanced scores were identical in both seeds. That null accuracy result is
kept in `data/continual_replay_policy_v8_smoke.json`, with exact manifest,
all seed scores, role/guard/sleep histories, and the audit ledger. This tiny
synthetic result is a correctness and comparison artifact, not a policy
ranking or generalization claim. P4.4 still owns replay-capable PC/backprop
controls, so these are retention-policy comparisons within the circadian
method, with unchanged non-replay baselines.
