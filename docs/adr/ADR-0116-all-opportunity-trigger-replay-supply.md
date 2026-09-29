# ADR-0116: Offer shared replay at every train-only trigger opportunity

## Context

P4.8a isolates trigger timing with chemical reset alone. A full-stack
comparison needs structural change, homeostasis, guarded sleep, and matched
replay controls. The existing v9 shared schedule emits rows only at forced
periodic boundaries and explicitly rejects adaptive triggering. Modifying
that protocol would change the meaning of its saved results. Sampling
replay only after a trigger decision could also allow arm-specific model
state to influence which training rows are offered.

## Decision

Use a separate fixed v14 manifest and all-opportunity train-only session.
After every arrived wake epoch, one shared FIFO buffer observes exactly
the train role and offers the newest two retained labeled rows. Selection
does not depend on predictions, sleep readiness, guard outcome, or final
data. The 8-row/192-byte caps and role-content digest are checked before
advancing; every method receives its own detached copy. Phase B arrives
only after all twelve A wake epochs. The schedule records potential work
only. A later runner must commit baseline replay solely after accepted
guarded circadian sleep, and must record actual work separately.

The v14 manifest is exact and prospective: two new seeds, a rotated and
translated B source, three arms, fixed wake/sleep/replay/guard settings,
and structural capacity caps. A changed manifest is rejected before any
source access. The historical v9 schedule and result identity stay as
they were. Its periodic subset serves as an exact parity oracle for the
new supply, including retained order, selected IDs, and method work.

## Alternatives

- Extend v9 to support adaptive/no-sleep arms. That would invalidate its
  fixed periodic-only semantics and artifacts.
- Give each arm its own prediction-prioritized selection. That would no
  longer expose the baselines to the same labeled examples.
- Treat offered rows as applied replay. Guard rollback and no-sleep would
  then be misreported as training work.

## Consequences

The schedule is a reproducible source and budget gate, not a full-stack
outcome. P4.8b1 can close only with exact repeat artifacts, v9 periodic
parity, source/role sentinels, and quality checks. P4.8b2 must still prove
actual event, replay, structure, guard, capacity, and global final-role
behavior. Adaptive inactivity is a valid result and cannot justify
retuning the frozen threshold.
