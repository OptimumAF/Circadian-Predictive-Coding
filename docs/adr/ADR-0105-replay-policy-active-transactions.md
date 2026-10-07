# ADR-0105: Continue active v8 replay-policy trials within format 9

## Context

ADR-0104 made completed policy/seed trials durable, but a stop during Phase A
or B still restarted that trial. V6 already has a transaction boundary for
model-order wake work, before and after sleep, and A/B arrival. Its format-6
identity describes the historical v4 content-hash policy. Reusing that file
identity for FIFO or seeded reservoir would give an old checkpoint a new
meaning.

## Decision

Keep the v8 manifest and format-9 file identity. The outer checkpoint holds
a completed policy-major/seed-minor prefix and at most one active transaction
for the next trial. The active value is a separately typed format-9 cursor
containing its policy index, seed index, A/B phase, wake model index or sleep
boundary, model snapshots, development-role audit, and typed sleep history.
It contains no final-test state.

Use the existing v6 transaction engine with an explicit retention policy at
the model-factory boundary. Its save callback is adapted into the format-9
envelope; v6's default policy, digest, format, and file interpretation stay
unchanged. A completed active trial becomes one unscored prefix record, then
the active slot clears. Final-test fields open only after every policy and
seed has completed training.

On resume, check manifest, trial order, policy index, phase, model cursor,
and frozen Phase A presence before opening a source. Rebuild only arrived
development roles, validate role IDs/hashes and event order, reconstruct the
policy-bearing NumPy model, and reuse model, retained-ID, guard, and typed
sleep-history preflight. Recompute exact observed and duplicate IDs from the
arrived train rows at the active circadian wake cursor. Match applied replay
updates to the sleep history, including a pending before-sleep boundary.
Phase B data cannot enter a Phase A cursor. All checks finish before another
training update or final release. This is a trusted local pickle contract.

## Alternatives

- Interpret the v6 active record as a policy checkpoint. That would alter
  historical format-6 semantics and permit a mismatched model factory.
- Resume only after a full policy/seed trial. That repeats potentially long
  A/B work and weakens exact interruption coverage.
- Save scored rows as trials finish. That would expose final labels before
  the predeclared global training set is complete.

## Consequences

Small fixed tests interrupt A and B at wake, before-sleep, and after-sleep
cursors in both model orders, including a reservoir policy boundary. They
require ordinary/resumed result and model-state equality and no repeated
wake updates. Tampered manifest, phase, future-role replay, duplicate ledger,
or guard history rejects before another update; a forged Phase B cursor
rejects before its source arrives. The reporting ledger remains outside the
retained-array memory cap. The two-seed comparison keeps its null policy
result. Replay-capable PC/backprop controls remain a separate P4.4 task.
