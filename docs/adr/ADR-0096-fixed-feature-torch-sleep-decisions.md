# ADR-0096: Preserve fixed-feature Torch sleep decisions beside model state

## Context

The Torch head already emits model-owned sleep facts, but the fixed-feature
runner discarded guard scores, cooldown suppression, and rejected structural
proposals. Its format-1 trusted checkpoint saved report counters without an
ordered decision history. A resume could therefore reproduce the model while
leaving the public report unable to explain each sleep attempt.

## Decision

Describe runner decisions in `torch_sleep_decisions.py` without mutating the
head. For a completed guard, record the inner-guard feature/label digest,
both accuracy and cross-entropy scores, the selected metric's delta and
tolerance, and two exact completed evaluation-pass counts. A rejected event
keeps the core's proposed stable IDs and chemistry while zeroing applied
changes. An unattempted epoch records disabled, not-due, or cooldown reason
without calling core sleep. Core-skipped attempts retain their core reason.

Attach these typed events to the circadian report; baseline reports carry
empty histories. A format-2 fixed-feature checkpoint saves the ordered
history beside the combined model/RNG snapshot and validates stage length,
epoch/wake clocks, guard role digest, and operational counters before model
restore. Measured event durations are excluded from equivalence comparisons,
while all other event facts must match. `SleepBudgets.time_limit_seconds`
remains `None` because this head has no per-sleep time limit; a wall-time
benchmark's separate per-head deadline remains in its existing result field.

## Alternatives

- Put runner guard facts into the head snapshot. That would make a model
  snapshot depend on evaluation roles and change trained-state hashes.
- Replace legacy attempt/rollback counters with event-derived counts. That
  would change report semantics before all supported modes are verified.
- Reconstruct events from counters on resume. That cannot recover guard
  scores, cooldown reasons, or rejected proposals.

## Consequences

Ordinary and interrupted CPU fixed-feature runs now produce the same
non-timing sleep history and trained model state on covered accepted,
rejected, skipped, and cooldown paths. Older format-1 fixed-feature files
must be regenerated. Failed-attempt persistence and actual-device CUDA
history validation remain P3.10c3b2/c3c; a failed guarded call still raises
after restoring the head, as before.
