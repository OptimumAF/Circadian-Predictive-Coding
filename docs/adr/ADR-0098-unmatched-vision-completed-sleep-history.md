# ADR-0098: Carry completed unmatched-vision sleep decisions beside model state

## Context

The v1/v2/v3 unmatched-vision runners used a repeated guard for sleep
rollback but returned only aggregate attempts and retained changes. A
rejected `SleepEventResult` discarded the core's proposed stable IDs. Trusted
format-1 vision checkpoints saved the classifier and counters without a
decision history, so a resumed report could not explain a guard rejection.

## Decision

Reuse the typed Torch decision adapter for vision. Record an event for every
completed epoch's attempted or skipped decision in the circadian outcome and
public test/development reports; baseline histories are empty. For a guarded
attempt, record scores, selected metric/delta/tolerance, exact examples
completed by both guard passes, and measured attempt time. A rejected event
retains the core proposal and zeroes applied work. The legacy v1 repeated
validation role remains `validation`; the disjoint v2/v3 role is
`inner_guard`. The role hash uses the selected split's sample-ID digest; the
existing checkpoint data digest binds raw train/guard/validation content.

Store active and completed histories in format-2 trusted vision checkpoints.
Before restoring a model or opening final test, validate ordered epoch/wake
clocks, nested event invariants, role/metric/exposure, canonical resolved
outcomes, and runner attempt/rollback/split/prune/cooldown counters. Guard
batch sizes come from the fixed evaluation loader's metadata, without
iterating data or advancing random streams during preflight. Measured event
durations are excluded from uninterrupted/resumed semantic comparisons.

## Alternatives

- Reconstruct events from report counters after resume. Counters cannot
  recover scores, skipped reasons, or rejected proposals.
- Put runner guard facts into the head snapshot. Evaluation roles belong to
  the runner, and volatile durations would change trained-state hashes.
- Reuse format 1 without a version bump. Earlier files have no required
  history and could silently return incomplete reports.

## Consequences

Bounded CPU v1/v2/v3 runs now return JSON-safe accepted, rejected, guarded
core-skip, and unattempted records. Active and completed CPU checkpoint
histories match uninterrupted non-timing facts and reject tampering before
restore/final scoring. Existing format-1 vision files must be regenerated.
Failed-attempt persistence and actual-device CUDA telemetry are separate
P3.10c4b/c gates; their acceptance is not inferred from CPU tests.
