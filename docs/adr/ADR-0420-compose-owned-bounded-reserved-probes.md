# ADR-0420: Compose bounded reserved probes on an owned warm schedule

## Context

R2.1 already supplies isolated fresh-task/readout/return execution. Bounded
TRAIN-only schedules and exact base diagnostic admission prevented composition.
Native counters include warm history and must not be zeroed to fit a new schedule.

## Decision

Add an explicit from_probe_snapshot factory to the existing weighted wrapper.
Own full warm state, a relative schedule origin,1..16original probe_train episodes
and optional exactly16originalTRAIN return calls. Separate reserved backend and
classweight ports preserve source roles and existing TRAIN refusal. Reuse the
existing bounded atomic publication and complete-field/pickle schedule proof.
The existing probe admits the exact validated richer diagnostic and serializes
independent native/scheduling failure scopes only at the training port.

## Alternatives

Relabeling reserved records or broadening TRAIN APIs would obscure release
boundaries. Resetting counters would erase native history. A new orchestration
framework would duplicate the existing isolated runner without a new requirement.

## Consequences

Source/donor/RNG isolation and full-state continuation are verified in scoped
native tests; base wire formats remain intact. Snapshot format stays unchanged
and does not prove historical stopping/role provenance. Comparative acquisition,
retention and recovery remain unestablished. Unfinished parent/release requirements
and all prior negative evidence remain open/preserved.
