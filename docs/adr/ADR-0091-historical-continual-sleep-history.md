# ADR-0091: Keep historical continual sleep history outside trained state

## Context

The continual v0–v5 benchmark schedules sleep per phase but its ordinary and
checkpointed calls do not provide an inner guard. Its low-level sleep helper
has an optional guard for separate callers, while the arrived v6/v7 protocol
uses an actual disjoint inner role. Adding guard scoring to v0–v5 would change
their evaluated behavior. The existing model-state hash tests also serialize
the trained state; measured event durations vary and must not enter that
state. Existing v0–v5 checkpoints have aggregate counters but no history
from which earlier triggers, skips, chemistry, and proposals can be recovered.

## Decision

Record one version-one `SleepEventTelemetry` per completed global epoch on
the historical runner's ordinary and checkpointed paths. Each record keeps
the phase-local periodic/adaptive schedule decision, full NumPy core facts
for attempted sleep, and runner attempt duration. Skipped schedule decisions
read chemistry without invoking core sleep. The guard field is explicitly
absent because those routes do not assess an inner role. The legacy counted
event signal and all model updates remain unchanged.

Keep event history in the pending seed, per-seed report, active checkpoint,
and unscored seed record, beside the trained-state objects. This preserves
trained-state serialization and final-label invariance. The existing v0–v5
checkpoint `format_version` still identifies the protocol route; an explicit
sleep-history extension version requires complete global epoch indexing at
every wake/before-sleep/after-sleep cursor and rejects old eventless payloads
before restoration. The complete report can be written as finite local JSON
after scoring with `--json-result`, while the existing text output remains.

## Alternatives

- Add a guard to v0–v5: changes the original baseline protocol and retention
  trajectory instead of observing it.
- Put measured durations inside trained state: makes state hashes depend on
  wall time and breaks final-label/order invariance checks.
- Reconstruct events from aggregate counters on resume: loses unscheduled,
  skipped, and proposed facts, and cannot produce a complete history.

## Consequences and evidence

The new historical continual fixture first failed on the absent report field.
It covers all six protocol IDs, two seeds, both phases, model-order variation,
actual checkpoint interruption, and JSON output. An existing trained-state
hash test caught the first in-state design; moving history beside the state
restored that invariant without changing the scientific route. Guarded
v6/v7 outcomes, role scores, and rollback proposals remain P3.10c2b–c.
