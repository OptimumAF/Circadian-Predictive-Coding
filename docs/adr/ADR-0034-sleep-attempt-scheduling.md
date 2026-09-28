# ADR-0034: Treat periodic sleep intervals as attempts

## Context

The toy and continual runners called sleep only at periodic epochs, even
when adaptive criteria were ready. Vision runners checked both paths, but
could spend guard evaluation on a `disabled` sleep call. The word
"trigger" conflated a scheduled attempt, a forced model call, and an
executed event. The existing NumPy legacy protocol needs its original
interval-only behavior for reproducibility.

## Decision

`sleep_schedule.decide_sleep_attempt` takes a sleep mode, completed runner
epochs, an epoch interval, adaptive readiness, and whether periodic calls
are forced. An interval divisible by the completed epoch count schedules
an **attempt**. It does not guarantee an executed event: an unforced call
still needs the core adaptive criterion, and warmup or structural budget
rules can still skip it. A forced periodic call bypasses only the adaptive
criterion. An adaptive-ready call can be attempted between intervals and
is not automatically forced. `disabled` schedules no attempt, including
no vision rollback guard exposure.

Toy and continual `components` runs check adaptive readiness after each
completed epoch, even when the interval is zero. Their `legacy` route
keeps interval-only calls. Vision unmatched and matched-head runners
already checked adaptive readiness independently and now use the same
attempt decision. Vision `sleep_attempts` counts scheduling attempts;
`SleepEventResult.performed` and component-mode NumPy event counts record
execution. These values can differ. The fixed-width capacity control
requires sleep and both structural switches to be enabled, so disabled
or nonstructural-only configurations fail before data loading.

The scheduler uses **completed runner epochs**. It does not reinterpret
the core model's wake-update counter or the older `current_step` and
`total_steps` budget inputs. Those clock names and replay/event counts
remain P3.2b; width-dependent adaptive input handling remains P3.2c.

## Alternatives and consequences

Making every periodic call unconditional would change historical
`force_sleep=False` behavior. Extending adaptive off-interval calls to
NumPy legacy would change its sequence of updates. Both are deferred to
explicitly selected routes. The pure decision object makes attempt and
force semantics testable without a dataset or a guard. It does not itself
record a durable attempt ledger or protect a sleep transaction.

## Evidence

The first component-mode tests with interval zero failed in toy and
continual runners despite ready adaptive criteria. The new schedule
matrix covers periodic, adaptive, forced, disabled, and zero-interval
combinations. A matched-head CPU fixture verifies disabled sleep records
zero attempts; two capacity-route fixtures initially reached data loading
and now reject disabled or structurally switched-off sleep. The
development log records focused and full quality commands.
