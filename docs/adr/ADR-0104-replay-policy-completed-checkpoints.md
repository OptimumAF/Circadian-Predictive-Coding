# ADR-0104: Save completed v8 policy trials in a separate format-9 file

## Context

The ordinary v8 comparison holds all policy/seed models until every trial
finishes, then releases final tests. A process interrupted after one full
trial had no durable unscored state. V6 format-6 checkpoints and v7 format-8
selection checkpoints reconstruct the historical content-hash model; using
either file identity for FIFO or seeded reservoir would change its meaning.

## Decision

Add a format-9 envelope for a prefix of completed trials in the v8
manifest's policy-major, seed-minor order. The envelope binds the complete
manifest digest, next trial cursor, and detached v6-shaped unscored arrived
record for each finished trial. Its file magic differs from v6/v7. It
contains no final-test values, hashes, or access events. A save occurs only
after A and B training for one policy/seed are complete; final sources open
only after every trial is complete and durable.

On resume, reject a changed manifest or invalid prefix before building any
source. Rebuild each saved trial's A and B development roles, then reuse
the arrived completed-state, baseline, guard-event, sleep-history, replay
budget, and wake-progress validator with an explicit policy model factory.
Also derive exact observed content-ID counts from the arrived train rows and
declared wake epochs. Compare those to the saved A/B observed and duplicate
ledgers, require exposed IDs to be observed, and match replay-update counts
to successfully applied sleep events. Bind both cumulative ledgers with a
separate digest. A forged duplicate count still fails when that digest is
recomputed. Restore only the validated unscored state; the remaining trials
train once in the original manifest order. Trusted local pickle remains the
storage contract, so the file must not be loaded from an untrusted source.

This is a completed-trial checkpoint. It does not yet save a mid-epoch,
before-sleep, or A/B arrival transaction. P4.2b2b will add those active
cursors and finish the original P4.2b2 acceptance criteria.

## Alternatives

- Reuse v6 format 6 or v7 format 8. Their fixed config and model identities
  would no longer describe the saved policy state.
- Save final scores after each trial. That would open final labels before
  the full predeclared policy/seed run is frozen.
- Retrain completed trials on restart. That would waste work and could hide
  a changed source or random stream behind a nominal resume.

## Consequences

Tests interrupt after one durable trial, then resume the remaining three
without retraining the first; the result equals ordinary execution. A
terminal checkpoint also rescored without training. Changed policy seed,
changed A training data, forged exposure ID, and a duplicate count whose
digest was recomputed all reject before another training update. A source
sentinel checks that both final-test sources stay sealed until all four
records are durable. The v6 loader rejects format-9 file magic. Existing
v6/v7 regression tests remain green. Active A/B interruption and exact
continuation are still open, so P4.2b2 and P4.2 remain unchecked.
