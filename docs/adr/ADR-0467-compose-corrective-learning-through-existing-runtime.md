# ADR-0467: Compose corrective learning through existing runtime

## Context

TRAIN-only corrective collection exists, but it has not fed a private learner.
The current frozen smoke exports JSON and keeps a usable paired batch internally;
the runtime already supports full optimizer continuation and permanent updates.
Both earlier native source tasks fail; the mixed objective has regressions.

## Decision

Expose a typed programmatic result from the existing smoke body and keep the
CLI wrapper intact. A small adapter applies eight complete arrived correction
batches through the existing actor/shadow runtime with unchanged MSE policy,
statistics and optimizer. Preserve original label times, source/actor weights,
paid receipts and bounded storage. Compose collection19, private updates and
frozen17/19 rollouts in the same existing script, stopping at any failed stage.

## Alternatives

A new simulator runner, checkpoint/model archive or supervision framework adds
no needed capability. Replacing executed student actions with teacher labels
would falsify trajectories. Refit, objective selection, horizon/seed/epoch search
and held-out tuning would confound the frozen case and exceed its budget.

## Consequences

One bounded native intervention can test the combined recipe. The candidate's
extra eight updates prevent equal-compute/query-value claims; provisional
off-reference labels are not a qualified expert. Negative results and all open
R4.4/R4.5/research/release requirements remain. No new dependency or serving
publication is introduced.

## PC/CPC extension, 2026-10-20

Reuse the existing recorded PC facade and full snapshot boundaries to admit PC,
neutral and active CPC in the same eight-update composition. Why this: frozen
rollout already supports these owners; a separate optimizer or simulator runner
would duplicate working boundaries. Keep native energy/event costs distinct from
BP arithmetic and retain bounded paid reports before outer validation. No new
scientific claim follows from integration correctness.
