# ADR-0504: provisional Push labels on frozen student states

## Context

The fixed112 learner improves Push TRAIN fit yet fails the native task and loses
Pick20. Its executed trajectory differs from the expert demonstration. Existing
Push collection executes expert commands and requires success for demonstration
export; existing Pick correction code assumes that task's raw42 state.

## Decision

Add a pure core correction batch and one outer collector using the existing Push
port, teacher and native BP. Query teacher targets on observed physical states with
actual student history, execute only student commands, and keep targets distinct
from the native trajectory. A complete negative student episode may release a
provisional correction packet after all labels arrive. Incomplete/unknown work
cannot release. Keep the original successful-demonstration gate unchanged.

Use existing study composition for reconstruction and one separately budgeted
collection; require the accepted112 student trajectory to reproduce exactly before
accepting the outer packet. No training, promotion or final/probe access in this
collection scope. Preserve all earlier negatives and permanent charges.

## Alternatives

- More optimizer updates on the same expert-only data: this already improves fit
  without acquiring the task and worsens old-task results.
- Relabel saved relative12 inputs as physical states: absolute geometry is not
  observed in those features and must not be invented.
- Relax successful expert export: would confuse learner corrections with a
  successful demonstration and weaken existing acceptance criteria.

## Consequences

New code owns immutable labels, arrival/history validation and finite paid
collection. No optimizer, dependency, native-port or inner-runtime changes. Teacher
queries and repeated source reconstruction are explicit costs. Successful data
release does not prove that subsequent correction fitting will acquire Push or
retain/recover Pick; those require a separately frozen protocol.
