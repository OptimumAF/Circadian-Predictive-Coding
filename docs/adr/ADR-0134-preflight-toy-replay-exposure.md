# ADR-0134: Preflight toy sleep replay exposure

## Context

P5.5c bounds toy wake updates and wall time at checked cursors. A NumPy
sleep can also replay whole retained wake batches. Their lengths can differ,
and selection may follow cached priorities rather than buffer order. Counting
after sleep would allow a total replay-example ceiling to be exceeded after
structural and weight mutation. Replay retention's example cap bounds stored
rows, not repeated example presentations across sleeps.

## Decision

Add an opt-in `max_replay_examples` total to the toy execution budget, outside
the scientific `ExperimentConfig`. At a due sleep, the core selects the same
snapshots it will replay and sums their actual batch lengths before any sleep
mutation. If that plan exceeds the remaining allowance, a typed core limit
exception rolls back the sleep transaction. The app converts it to an
`incomplete` stop at the checked `before_sleep` cursor, counts only applied
telemetry, and restores the cumulative count from a validated trusted
checkpoint. A resume may raise the total cap without changing its scientific
request. A cap lower than already checked work stops before more training.

The CLI adds `--max-replay-examples` to its existing budgeted baseline route.
Its version-1 run state adds this budget and observed versus checkpointed
replay-example counts, while its checkpoint identity includes the durable
count. It publishes no result for an incomplete attempt. Existing unbudgeted
sleep invocations retain their original call shape and selection path. Previous
version-1 state files lacking the new replay count remain resumable when
their original checkpoint hash, cursor, and update count still match.

## Alternatives

- Count replay updates or retained rows: rejected because selected batches
  can contain different numbers of examples and can be presented repeatedly.
- Check after sleep: rejected because that would exceed the cap before stop.
- Put the limit in `CircadianConfig`: rejected because an execution ceiling
  should not change model/config/checkpoint scientific identity.
- Forbid sleep at an exact reached cap: rejected because zero-replay and
  skipped sleeps still have valid work to finish.

## Consequences

The ceiling limits actual replay-example presentations at completed toy sleep
events. The check does not bound wake work, replay memory retention, transient
structural width, or process RSS; P5.5d2/d3 retain the latter two run-level
bounds. The local checkpoint is trusted pickle input, and the CLI still binds
resume to its exact byte hash. No fixed-v14 artifact, seed, baseline, metric,
or previous result is reinterpreted.
