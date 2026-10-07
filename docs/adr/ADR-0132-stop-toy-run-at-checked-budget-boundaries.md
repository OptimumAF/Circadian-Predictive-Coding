# ADR-0132: Stop a toy run at checked budget boundaries

## Context

P5.5c requires actual runtime limits and explicit stop reasons. The toy
comparison has one ordered wake update per model and epoch. Its optional
trusted checkpoint records a validated cursor after intermediate model
updates and before/after sleep. Final-test scoring occurs only after the
training helper returns. The existing `ExperimentResult` represents a
complete final-scored run and cannot truthfully represent a partial one.

## Decision

Add an opt-in `ToyExecutionBudget` outside `ExperimentConfig`, with a
non-negative total wake-update ceiling and a non-negative finite wall-time
ceiling. A per-invocation monotonic clock starts before dataset setup and
is checked before each model update, before sleep, and before final-role
release. At a limit, raise `ToyExecutionStopped` with status `incomplete`,
reason `max_training_updates` or `max_wall_seconds`, completed wake-call
count, elapsed time, and the last successfully saved checked cursor.
The update reason wins when both limits are reached at an update boundary.
No checkpoint means the stopped invocation is explicitly non-resumable.

On resume, the runner first validates the existing checkpoint and uses its
three loss-history lengths as the total committed update count. It may
resume with a higher total-update ceiling; the wall clock starts fresh.
An exact full-run update ceiling permits completion. A stopped run never
returns a completed result or reads final-test arrays. The existing
checkpoint format, config digest, protocol, order, metrics, and default
call path remain unchanged. The separate CLI lifecycle and error record
is P5.5c2.

## Alternatives

- Return a partial `ExperimentResult`: rejected because its final-test
  accuracy fields would be missing or fabricated.
- Add execution limits to `ExperimentConfig`: rejected because a local
  resource policy would change scientific config/checkpoint identity.
- Interrupt inside a model update or sleep: rejected because the current
  checkpoint records only complete boundaries and an internal interruption
  would not have a restorable state.

## Consequences

The wall-time ceiling is observed at boundaries; a single update or sleep
may run past it before the next check, and final scoring may run past it
after the last check. Total wake updates
exclude replay and structural work, which remain P5.5d. A caller can
inspect a precise incomplete reason and cursor; CLI completed/incomplete/
error artifact publication remains open until P5.5c2. No scientific
candidate, seed, metric, or historical output was changed.
