# ADR-0457: Use the actual stable actor at the robot boundary

## Context

The continuous delayed-label recipe now learns on a private candidate through
the existing runtime, but the frozen robot smoke accepts only concrete learners.
Restoring a candidate or actor snapshot into a new learner would bypass the
actual serving owner. Native serving needs version attribution and unchanged
complete state while unrelated candidate labels arrive.

## Decision

Add a mutually exclusive exact StableActor/nearest-backend route to the existing
smoke. Retain one actor/backend identity/version/full-state binding. Call the
actual actor's public locked prediction and state methods; backend inspection is
type/identity admission only. Validate versioned records, complete native state
and actions around prediction, controls and cleanup. Preserve all old routes.

Explicitly label derived actor work because the stable actor returns actions,
not a nearest-work report. Do not fabricate selected indices or substitute a
restored learner. No runtime/model/port rewrite, promotion or new native case.

## Alternatives

A restored learner copy would verify a different serving path. Broad duck typing
or accepting promotion-capable subclasses would require additional concurrency,
version and state semantics. Adding a new runner would duplicate existing
controller and accounting boundaries.

## Consequences

The runtime-owned warm actor can drive existing controls while candidate updates
remain isolated. Equal-valued invalid snapshots and equal-state backend
replacement are refused. Snapshot/copy overhead is retained, not described as
distance work. Fake-native integration is verified separately from real simulator
execution, learning benefit, promotion, visual features and release readiness.
