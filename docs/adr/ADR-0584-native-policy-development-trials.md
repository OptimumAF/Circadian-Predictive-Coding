# ADR-0584: Isolate policy-phase native trials from legacy final scoring

## Context

The verified12batch policy corpus differs from the native reference runner's
eight-batch/v2-only recipe and old task grouping. Treating opportunities as
completed updates, silently reusing old final metrics or force-draining a finite
schedule would change its resource/scientific meaning.

## Decision

Add one explicit policy-change mode to the existing native runner. Keep original
qualified full BP state/optimizer, actor/runtime/clock/budget/shared gate and finite
action envelope. Declare12updates/13opportunities before work, and record actual
committed work at four fixed phase opportunities. No budget reset or forced drain.

After TRAIN, existing owned BP decoder/readouts run actual sandbox development
episodes with explicit policy versions. Bound storage to four full phase records;
validate their original counts/final candidate and refuse changed evidence during
readout. Deferred mode captures states without opening any task evaluator. Source,
actor, final candidate and RNG protections remain; fitted source/version binding
is unchanged. No new schema, source-loader, simulator or orchestration framework.

Legacy final and fixed-group utility paths reject policy report markers before
source/clock/native access. Old metrics/final results stay unchanged. A later
comparison must explicitly freeze its new phase-matched procedure and include
costs; it cannot reuse incompatible old grouping by accident.

Why this: the old runtime already provides controlled learning; explicit data and
evaluation semantics are the actual missing integration. Separate opportunity
axes and current policies prevent misleading acquisition/retention conclusions.

## Alternatives and consequences

- Broaden all old recipes to arbitrary batch counts: rejected; only the fixed
  new12batch corpus receives the explicit extension.
- Score phase records through legacy final tasks: rejected; labels/groups differ.
- Drain pending work after the horizon: rejected; opportunity/update limits are
  part of the original predeclared procedure.
- Claim gain from cross-policy success: rejected. The first native case's raw
  +1parcel change also changes truth/masks and uses an already v2-qualified source.
  Same-phase no-learning controls are required before attributing learning benefit.

The sole fixed native case passes engineering assertions with6SGD36rows and
6unapplied batches; v1invoice acquisition remains unshown. Self-review is not
independent/scientific/release acceptance. FullR7.5/#26/#18 stay open. Evidence:
issue68 and `artifacts/runs/native-policy-trial-20261023/RESULTS.md`.
