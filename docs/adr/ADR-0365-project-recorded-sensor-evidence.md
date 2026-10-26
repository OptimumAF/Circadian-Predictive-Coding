# ADR-0365: Project recorded sensor evidence through the shared inspector

## Context

Existing sensor prearrival/heldout/Cprobe evidence cannot use the saved game/
MiniGrid inspector dispatch. It already contains useful negative results and paid
partial work. Resource artifacts lack a shared unique report identity.

## Decision

Add a pure bounded sensor projection and offline template behind the existing
shared dispatcher/file publisher/CLI. Every recorded cell resolves to its exact
embedded source pointer; missing evidence stays unknown, null and signed negatives
remain visible. Retain partial statuses, paid charges and inactive/unobserved
controls. Display separate resource evidence with explicit unverified association.
Do not recompute scientific metrics, authenticate costs or run a learner.

## Alternatives

A new dashboard/server would duplicate working IO/security boundaries. Planned
protocol counts cannot substitute for missing measured work. A new provenance
scheme would expand infrastructure without a producer identity requirement; mark
the present limitation explicitly instead.

## Consequences

One entry point handles existing game/MiniGrid and sensor reports. File-size,
finite/duplicate JSON, safe embedding and exclusive publication remain in infra.
The UI shows one recorded report at a time; full R4.7 structural/sleep/cross-run
coverage remains open. Correct inspection does not establish scientific benefit.
