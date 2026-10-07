# ADR-0092: Keep arrived guard facts and checkpointed sleep history beside model state

## Context

The v6 arrived runner checks each scheduled sleep attempt with the phase's
disjoint inner guard, but its prior `GuardDecision` ledger retained only
accuracy and acceptance. NumPy core telemetry already contains the complete
structure, replay, and chemistry proposal. A rejected guard restores the
model, so recording only its final state would lose the proposal. The guard
uses two accuracy evaluations; it does not measure cross-entropy. Active and
completed v6 checkpoints previously had no typed sleep history.

## Decision

Attach the measured inner-guard accuracy scores, phase role hash, tolerance,
delta, and total scored-example exposures to the core event. Leave both
cross-entropy fields absent. On a rejected attempt, retain the core proposal
and its duration while clearing applied structural and replay effects and
setting final width and chemistry to the restored entry state. Record one
event for each completed phase-local sleep decision, including schedule
skips, beside the trained state in ordinary pending seeds, active checkpoint
cursors, completed unscored seeds, and final v6 per-seed reports. The v6
smoke script exports these records as finite JSON.

Keep the existing role-event digest and v7 trial digest unchanged. A distinct
version-one sleep-history extension and digest bind measured events to that
role ledger. Before model restoration or final scoring, validate epoch order,
phase-local trigger, role hash, two accuracy exposures per guard example,
scores, tolerance, acceptance, and the typed event invariants. Reject old or
incomplete checkpoint histories instead of inferring missing facts.

## Alternatives

- Recompute cross-entropy on the guard: adds two evaluations and changes the
  established work ledger without affecting the accuracy decision.
- Put durations in trained state: makes trained-state hashes depend on wall
  time and breaks the existing final-label invariance check.
- Extend the existing role-event or v7 trial digest: changes selection
  identity for facts that are observational rather than selection criteria.

## Consequences and evidence

The new ordinary fixture first failed because v6 returned no typed events;
the checkpoint fixture first failed because resumed reports lost them. The
focused tests now cover accepted/skipped phase-local facts, a real rejected
NumPy split proposal, v6 rollback and resumed parity in both orders, and
active/completed history rejection before training or scoring. Existing v6
role/final-test and v7 selection suites pass. ADR-0093 subsequently extends
this version-one history with failed attempts and a retryable epoch cursor.
