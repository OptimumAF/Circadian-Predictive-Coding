# ADR-0035: Separate runner epochs from core sleep clocks

## Context

The original `sleep_event(current_step, total_steps)` arguments were
supplied with runner epoch numbers, while NumPy `_epochs_since_sleep`
advanced once per `train_epoch` call and Torch `_steps_since_sleep`
advanced once per `train_step` batch. NumPy replay invoked the training
path without advancing its wake counter but did advance traffic. The
shared word "step" obscured these different units, especially with
multiple Torch batches per epoch or replay during sleep.

## Decision

Runners pass validated `SleepEpochProgress(completed_epochs,
total_epochs)` to core sleep calls. These are completed outer-loop epochs,
used for warmup and split/prune progress windows. The older
`current_step`/`total_steps` inputs remain accepted for direct callers;
providing both forms at once raises `ValueError`. The numeric meaning of
the legacy inputs is unchanged, and the four runner routes now pass the
same numbers through the typed form.

`get_sleep_clocks()` returns a read-only `SleepClockSnapshot` on NumPy
circadian models and Torch circadian heads/classifiers. It counts
successful wake batches, wake examples presented (including repeats),
wake batches since the last executed sleep, NumPy replay updates, and
executed sleep events. A skipped/disabled attempt does not reset the
wake-since-sleep clock or increment sleep events. NumPy replay increments
its own update count only after a successful replay step; it does not
increment wake batches/examples or the adaptive wake clock. Torch has no
replay and reports zero replay updates. Torch snapshots include the new
counters so guard rollback restores them.

Historical configuration names retain their units: NumPy
`min_epochs_between_sleep` counts successful `train_epoch` calls, which
are wake batches when callers supply one batch at a time. Torch
`min_sleep_steps` counts successful `train_step` batches. Both
`sleep_warmup_steps` fields are compared to completed **runner epochs**
only when progress is supplied; direct calls without progress bypass
that warmup check. `sleep_energy_window` counts wake diagnostics,
excluding replay, and cooldown/age updates follow successful wake
calls. Runner sleep attempts are separate operational decisions from
core executed-event counts.

## Alternatives and consequences

Renaming existing config fields or reinterpreting the adaptive minimum
as runner epochs would change published/legacy behavior. Counting replay
as wake would also change adaptive trigger timing. Those options are not
adopted. New core counters add state that full NumPy snapshots and
checkpoint/resume must preserve under P3.3/P3.9; this increment does not
claim an atomic NumPy sleep transaction. The width-sensitive diagnostic
history policy remains P3.2c.

## Evidence

The first typed-clock test failed on the absent module/API. Tests now
cover invalid epoch progress, warmup skip versus performed sleep,
successful wake batch/example counts, NumPy replay isolation, a
two-batches-in-one-epoch adaptive minimum on each backend, legacy-input
equivalence, ambiguous-argument rejection, and Torch snapshot restore.
The development log records focused and full quality gates.
