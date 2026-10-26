# ADR-0449: Diagnose continuous action errors on TRAIN data

## Context

The fixed task-space teacher produced a qualified 29-step robot demonstration.
The first frozen BC pilot reduced its aggregate TRAIN loss, while both frozen
probe policies failed. Aggregate loss alone conceals which actions remain
underfit. Probe outcomes must not drive tuning or data selection.

## Decision

Add pure bounded per-axis and per-stage normalized physical-action summaries,
and a TRAIN-only diagnosis method on the existing NumPy BC learner. Use one
existing prediction call, preserve all stages and axes, and report diagnostic
work separately. Refuse non-TRAIN policies/records and invalid metadata before
prediction. Keep the deployed/training state unchanged.

Why this: a complete TRAIN decomposition gives a concrete next learning question
without changing the objective, selecting a favorable evaluation result, or
introducing another orchestration framework.

## Alternatives

More simulator probes cannot diagnose TRAIN fitting and would spend evaluation
exposure. Immediately changing the architecture, loss weights, or data producer
would confound an unresolved optimization question. Retaining every prediction
in an unlimited history is unnecessary for the bounded summaries.

## Consequences

Core depends inward on existing continuous-action math and records. The adapter
adds role authority; the pure function alone grants none. Reports are bounded
and immutable. Constant axes are excluded from the denominator; physical output
rounding at extreme narrow bounds is explicit. Correct diagnostics do not prove
robot skill or explain probe failure. R4.4 strong BC and adaptation remain open.
