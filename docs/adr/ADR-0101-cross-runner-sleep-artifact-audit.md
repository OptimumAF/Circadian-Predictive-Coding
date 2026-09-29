# ADR-0101: Reconcile typed sleep artifacts without changing legacy counters

## Context

Toy, continual, fixed-feature, and unmatched-vision runners expose the same
typed sleep event but retain older operational counters with different scopes.
An attempted sleep may roll back after proposing structural or replay work;
an executed replay-only sleep may retain no topology change. The fixed-feature
guard-example counter also includes regular epoch stopping evaluations, not
only the two passes around a scheduled sleep.

## Decision

Audit each public local JSON sequence against its typed report and trusted
checkpoint history. Require every dataclass field, strict finite JSON, aligned
chemical summary widths, and zero applied work on rollback or error. Compare
ordinary and resumed event semantics while excluding measured durations.

Keep three quantities separate: scheduled attempts, completed core sleep
events, and retained split/prune changes. Reconcile each runner's existing
counter to its documented meaning rather than deriving all counters from one
generic event count. Count fixed-feature guard exposure as regular epoch
stopping evaluation plus completed sleep guard passes. The audit uses bounded
synthetic inputs and controlled rejection scores to check accounting; it does
not compare scientific model performance.

## Alternatives

- Redefine legacy event counts as every scheduled attempt. This would change
  historical report meaning and make rejected proposals look like applied
  work.
- Treat every accepted sleep as a topology change. Replay-only component
  sleep can perform work with zero retained split or prune.
- Compare measured event durations across ordinary and restarted runs.
  Restart and file boundaries alter elapsed measurements.

## Consequences

`tests/test_sleep_artifact_audit.py` now checks all four runner families,
including arrived continual selection, strict JSON, saved and resumed
histories, replay-only execution, and rejected proposals. The Phase 3 exit
gate separately exercises forced split/prune, no-op, rollback, aligned state,
and future seeded behavior. This audit changes no model, baseline, metric,
seed, protocol ID, or final-test release rule.
