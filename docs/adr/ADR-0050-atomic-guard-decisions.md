# ADR-0050: Restore guarded sleep after rejected or failed scoring

## Context

The fixed-feature matched-head and unmatched vision runners snapshot the
circadian head before a guarded sleep. They restored it when a finite
guard delta exceeded tolerance, but an exception during scoring or a
nonfinite guard value could leave the executed sleep in place. A NaN
delta compares false against the tolerance and could be accepted.

## Decision

Both guard routes validate pre/post accuracy and cross-entropy as finite
and require a finite rollback delta. With rollback enabled, pre-scoring,
sleep, and post-scoring run inside one restore-on-error boundary. A
rejected finite delta restores the same snapshot and returns the legacy
empty event with `rolled_back=True`. An error or nonfinite score restores
the snapshot and re-raises; it does not produce a successful report.

The vision guard remains head-only because its training route freezes
the backbone. Local runner attempt and rollback counters are outside
the head snapshot. They still count a rejected completed attempt, while
the head's sleep-event counter reverts. The rollback tolerance must be
finite and nonnegative before training begins.

## Consequences

Accepted finite events keep their old result and report meaning.
Rejected events have the same no-change result as before. Guard scoring
exceptions now fail the run with the learning state restored, so later
resume or caller handling begins from the last accepted state. Durable
run failure status and external checkpointing remain P3.9/P5 work.

## Evidence

`tests/test_guarded_sleep_atomicity.py` first failed because the vision
guard was embedded in training and had no separately testable boundary.
It now covers accepted/rejected events, full head state and next seeded
sleep/wake continuation, pre/post scorer errors (including scorer-side
mutation), nonfinite pre/post scores, and local runner counters on both
routes. The development log records the quality gate.
