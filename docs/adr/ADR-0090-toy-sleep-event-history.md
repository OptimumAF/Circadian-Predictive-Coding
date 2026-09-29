# ADR-0090: Persist one toy sleep decision per epoch

## Context

The toy runner owned periodic/adaptive scheduling and legacy sleep counts,
while the NumPy model returned only facts from actual `sleep_event()` calls.
An unscheduled epoch cannot call that method merely to obtain telemetry:
adaptive readiness could then change training. Existing version-1 toy
checkpoints stored only aggregate counts, so they cannot reconstruct earlier
decision reasons after a resume.

## Decision

The runner records one `SleepEventTelemetry` for every completed epoch. For
an attempted sleep it retains the model-owned structure, replay, chemistry,
and core duration and adds the periodic/adaptive trigger and measured outer
attempt duration. For a schedule miss or disabled mode it reads immutable
chemical summaries and records a zero-duration skipped decision without
calling sleep. The event sequence is compare-excluded from summary equality
because timings vary, while explicit semantic projections can be compared
across resumes. The existing legacy `event_count` and learning path remain
unchanged.

The trusted toy payload is version 2 and validates the complete epoch-indexed
event sequence at wake, before-sleep, and after-sleep cursors. A version-1
payload is rejected before model restoration, since filling its missing
history with guessed events would make the report incomplete. The existing
file envelope remains checksummed; the app payload version controls semantic
compatibility. `--json-result` writes a complete result after final scoring
to a new local path, preserving the ordinary console output.

## Alternatives

- Call core sleep on every epoch: changes adaptive scheduling and the
  historical training trajectory.
- Reconstruct skipped epochs from counts at resume: loses trigger and
  chemistry facts, and cannot recover attempt outcomes from the aggregate.
- Keep version-1 checkpoints with a partial event list: violates the complete
  history contract and makes resumed reports depend on interruption timing.

## Consequences and evidence

`tests/test_toy_sleep_telemetry.py` first failed on the absent event sequence.
It covers periodic, adaptive, combined, disabled and not-due decisions,
typed/JSON fields, wake/before-sleep/after-sleep interruption in both model
orders, and pre-training rejection of old or incomplete histories. The
existing toy resume suite covers baseline trajectory and sealed final-test
timing. A scheduled no-topology legacy sleep remains recorded as an attempt
without increasing the legacy counted-event total.
