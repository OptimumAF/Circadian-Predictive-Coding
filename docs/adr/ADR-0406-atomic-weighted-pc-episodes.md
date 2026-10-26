# ADR-0406: Share the native commit for weighted PC episodes

## Context

The accepted weighted categorical PC primitive proposes four parameters and
detached partial gradients. Complete neutral CPC state also includes chemistry,
importance, traffic, ages, counters, cooldowns and legacy history. Publishing four
parameters alone would make continuation invalid. Existing weighted BP and native
learners already stage owned snapshots before publication.

## Decision

Extract the existing PC/CPC post-gradient transition into private native
`_apply_training_gradients` methods without changing its operation order. Ordinary
native steps retain their original gradient calculation and call this transition.
The new complete-TRAIN-episode weighted PC/neutral adapter stages an independent
head, passes accepted weighted partials and raw final-state errors to the same
transition, validates complete state and donor guards, then publishes one payload.

Keep actual weighted energy separate from original legacy MSE history. Preserve
native policies/defaults, all-one complete-state identity, foreign-change refusal
without rollback, and existing complete snapshot/fork/restore formats. Require
source admission before numerical work; no reserved evaluation/probe input.

## Alternatives

Reimplementing neutral counters/chemistry in an outer adapter risks state drift.
Running an ordinary native update then replacing its parameters would perform
extra relaxation and derive adaptive state from the wrong update. Publishing
only four parameters violates native continuation. These approaches are declined.

## Consequences

The shared transition also serves active CPC's original native update, so focused
active/native finite-update regressions accompany weighted adapter checks.
Bounded candidate arithmetic is repeated by the pure proposal and shared commit;
there is one relaxation and one complete native transition per admitted episode.
Weighted schedule composition and scientific evidence remain separate open work.
