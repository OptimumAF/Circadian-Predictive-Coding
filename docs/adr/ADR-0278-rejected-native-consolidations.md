# ADR-0278: Bound repeated rejected native consolidations

## Context

The existing actor/shadow runtime charges consolidation IDs before trusted
transforms and preserves actor availability through a separate serving gate.
R3.7 still needs an actual repeated native refusal stream and bounded history
observations beyond single-failure unit controls and checkpoint failure demos.

## Decision

Compose existing coupled learner ports, PromotableActor, ActorShadowRuntime,
consolidation cursor and process-memory sampler in a fixed three-family example.
Eight unique callbacks mutate only detached copies, demonstrate actor cache
availability while the candidate gate is held, then fail. Explicit duplicate and
exhausted-ID probes must refuse before callbacks. Keep full before/after native
state and serving snapshots, and expose attempt/revision changes separately.

Observe actual cursor bytes, array bytes, cache entries and boundary process RSS
under finite budgets. Keep all original runtime and learning rules unchanged.

## Alternatives

- Fake-only tests do not establish complete native CPC or serving-cache behavior.
- An unlimited loop would obscure original attempt quota and consumed IDs.
- Resetting or discarding failed history would renew resources after rejection.

## Consequences

The runnable integration demonstrates bounded failure history with stable actor
availability across PC and both CPC controls. Boundary RSS remains observational,
not a hard or long-run memory proof. No new dependency, algorithmic feature,
scientific result or full recovery authority is introduced. Existing original
long-run and managed recovery criteria remain unfinished.
