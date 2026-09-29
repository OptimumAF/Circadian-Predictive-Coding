# ADR-0067: Isolate Phase A sleep from the future Phase B horizon

## Context

The phase-arrival v2 path defers Phase B data construction, but both
ordinary and checkpointed Phase A trainers pass the configured A+B epoch
horizon to core sleep. Core sleep uses the completed/total fraction to
select split and prune budgets. In a fixed seed-17, two-epoch Phase A
fixture, changing only Phase B from one to seven epochs changed v2's
second Phase A sleep from no split to a split, and changed trained state.

## Decision

Add opt-in `continual_phase_local_schedule_v3`. Its Phase A sleep uses
the Phase A epoch count as the progress horizon in both execution paths.
Keep the Phase B progress horizon at A+B after Phase B arrives. Reuse v2's
phase-specific development-role construction and test seal, but give the
new route checkpoint format 3 so trusted files cannot be confused with
v2 checkpoints. Preserve the v1 and v2 schedules and defaults exactly.

The v3 name states the implemented boundary. It does not claim to be a
complete strict-online protocol.

## Alternatives

- Change the v2 schedule in place. That would silently change reviewed
  offline outputs and the meaning of existing checkpoints.
- Pass no progress horizon during Phase A. That would bypass the current
  split/prune timing policy instead of giving it the phase information
  actually available.
- Rescale Phase A onto a speculative B duration. That still lets an
  unarrived phase influence decisions.

## Consequences

Tiny forced-sleep tests compare two future B durations in both model
orders and both ordinary/checkpointed Phase A paths, including the final
Phase A checkpoint state. They also show the v2 fixture retains its
historical future-horizon sensitivity. A Phase A interruption resumes to
the same v3 final result and state as uninterrupted execution, and both
match the ordinary result.

The runner config and checkpoint identity still contain future Phase B
settings, even though Phase A sleep no longer uses its duration. V3 also
lacks a declared replay memory budget, guard/outer-selection arrival
rules, and label/retention reporting. P1.3c2–c4 and the full P1.3c
acceptance criteria remain open.
