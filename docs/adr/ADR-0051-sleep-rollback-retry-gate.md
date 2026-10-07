# ADR-0051: Gate a rejected sleep retry on elapsed epochs and new wake work

## Context

P3.7 restores the circadian head after a guard rejects sleep. With an
empty training phase, a periodic or adaptive attempt could then run from
the identical restored head and split generator on every epoch. Those
attempts repeatedly score the same rejected proposal and consume guard
time without new learning evidence.

## Decision

The runner owns a `SleepRollbackCooldown` separate from model state. A
guard rejection records the completed epoch and the count of successful
wake batches. For a positive configured cooldown, another due attempt
requires both an epoch later than the fixed cooldown and at least one
new successful wake batch. A suppressed due attempt never calls sleep
or scores the rollback guard. Suppressions and rejections are append-only
operational counts; restoring a head does not restore them. Accepted
events do not arm the gate.

`circadian_sleep_rollback_cooldown_epochs=None` resolves to zero in
`legacy` and `disabled` modes and one in corrected `components` mode.
Explicit nonnegative overrides, including zero, are allowed. This keeps
the reviewed legacy schedule reproducible while giving component runs a
predeclared policy. The resolved cooldown, actual sleep attempts, and
suppressed due attempts appear in matched-head and unmatched vision
reports. The CLI exposes an explicit override. The policy has a small
versioned, validated snapshot for deterministic operational resume;
P3.9 still owns a durable combined run checkpoint.

## Alternatives and consequences

Retrying on every epoch can repeat an identical rejected proposal when
no wake update occurs. Permanently disabling sleep after one rejection
would suppress later adaptation even if learning state changes. The
fixed one-epoch default is a correctness rule, not a tuned performance
hyperparameter. A new wake batch changes the model's successful-work
clock; the policy does not claim that every retry will pass the guard.
Repeated failures after genuine wake work remain visible in rollback
counts. No metric, seed selection, or baseline configuration is altered
to favor the circadian result.

## Evidence

`tests/test_sleep_retry_policy.py` covers mode resolution, invalid
configuration, no-wake suppression, epoch and wake gates, and an
incompatible operational snapshot. `tests/test_sleep_retry_runners.py`
first failed because the config field was absent, then covered both
guarded routes, periodic/adaptive due attempts, legacy and explicit-zero
behavior, accepted/disabled paths, public report fields, CLI mapping,
and identical seeded repeated runs. The development log records the
full quality gate.
