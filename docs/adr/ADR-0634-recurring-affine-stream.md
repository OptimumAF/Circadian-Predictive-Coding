# ADR-0634: Ordinary recurrence through the existing affine TRAIN stream

## Context

The explicit ten-arrival composer supports three earlier packets per cue, but
the ordinary affine runner hard-codes six updates and A/B or ABC sources. Private
F rehearsal cannot be promoted to ordinary TRAIN. The next useful learner path
needs a fourth original TRAIN context and return without another runner.

## Decision

Add exact keyword `recurring_catalogue=False` to `run_context_linear_stream`.
True requires `retain_arrived=True` and excludes ABC mode. Admit eight ordered
complete nine-row original TRAIN sources A/B/C_train/G_train blue/red. Execute
indices0..7,0,1 through the existing composer10/retention3 and AttributionStream,
with label delay1, checkpoints0/2/4/6/8/10 and one original ten-update allowance.
The existing90s/512MiB runner limits remain. Preserve default six-update modes.

Reuse original source witnesses, source/actor isolation, counter reconciliation,
receipt capture, failed-work reporting and detached TRAIN scoring after all fits.
The existing final observer receives an actual completed endpoint; it cannot
substitute report reconstruction or incomplete receipt evidence. Existing public
CLI/data-loading/workflow callers are unchanged.

## Alternatives

A new runner/replay service would duplicate delayed admission and cost ownership.
Broadly widening old defaults would alter frozen protocols. Reusing F probe or
D final packets as G TRAIN would break role isolation. None is adopted.

## Consequences

Ten fits account for90arrival+162replay=252rows;48TRAIN monitor calls432rows
follow learning. Extra exposure, memory and computation remain explicit; this
is not matched compute or a circadian comparison. Literal integration fixtures
prove software behavior, not native acquisition/retention/generalization. A real
original G TRAIN layout/source must be qualified separately using existing ports
before a small prospectively frozen outcome observation. There is no new native
allowance or retry of failed120/108. Single-agent self-review is not independent
acceptance; full goal/parents remain open.

Evidence: [results](../../artifacts/runs/recurring-affine-stream-20261024/RESULTS.md),
[usage](../recurring-train-catalogue.md#ordinary-chronological-train-integration).
