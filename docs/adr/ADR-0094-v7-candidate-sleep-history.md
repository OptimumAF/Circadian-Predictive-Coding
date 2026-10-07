# ADR-0094: Expose v7 candidate sleep history beside the trial ledger

## Context

The v7 selector trains every declared candidate through the v6 guarded
runner. Its ordinary pending seeds and checkpointed unscored seeds already
retain typed sleep attempts, but the public v7 result omitted them. Final
scoring also called the common scorer without the selected circadian
candidate's history. The outer trial digest predates telemetry and binds
the score, work, and role ledger used for choice. Sleep durations vary
between equivalent runs and must not become a selection input.

## Decision

Expose one `ArrivedCandidateSleepHistory` per candidate and seed in the
public result, in declared candidate/seed order. Carry the history from the
chosen circadian candidate into the final seed metric. Mark the new history
field out of dataclass result equality; compare its typed content directly
when testing continuation, omitting only measured durations. Include both
candidate and selected histories in the bounded v7 smoke JSON with strict
finite-number serialization. Leave the original outer trials, trial digest,
choice objective, freeze digest, and checkpoint format unchanged.

## Alternatives

- Put timing-rich events into `ArrivedOuterTrial` now. Its existing `asdict`
  trial digest would change, and the checkpoint validator would need a
  separate provenance digest and format migration. That remains P3.10c2c2.
- Report only the chosen candidate. That would hide the guard attempts made
  while evaluating the other predeclared settings.

## Consequences

The two-candidate/two-seed fixture first failed on the absent public field.
It now checks ordinary, fresh checkpointed, and interrupted/resumed runs in
both model orders, both phases, selected final history, JSON safety, and
unchanged trial digest. The original v7 score and final-release tests still
pass. Completed candidate records already carry v6's versioned, validated
sleep history; explicit v7 trial/freeze binding, guard error/skip cases, and
count reconciliation remain open in P3.10c2c2. This decision does not
change training, candidate choice, baseline work, or final-test timing.

ADR-0095 completes the separate trial/freeze provenance gate and advances
the trusted selection checkpoint format to 8.
