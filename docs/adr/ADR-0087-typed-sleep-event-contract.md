# ADR-0087: Use one immutable sleep-event fact contract

## Context

The NumPy and Torch sleep methods own structural budgets, stable neuron IDs,
replay work, chemistry, and actual core changes. Runners own the periodic or
adaptive trigger, guard role and scores, rollback decision, cooldown, and
durable report. Existing `SleepEventResult` values summarize core work, but
a rejected runner guard replaces the result with an empty one. That loses
the proposal and cannot explain a rejection. Timing also differs between
core execution and the full guarded attempt.

## Decision

`src/core/sleep_telemetry.py` defines frozen, version-one, JSON-safe value
objects for one attempt. Stable `(parent ID, child ID)` split pairs and
separate proposed, scheduled, removed, and applied prune IDs survive a
rollback without claiming that rejected changes persisted. Width checks
use actual removals; scheduling alone does not shrink a model. Primary,
fast, and slow chemical summaries retain counts and finite min/mean/max.
Replay records exact proposed and applied examples and updates. Guard data
names its development role, metric, pre/post scores, deterioration delta,
tolerance, and scored-example count. An absent guard is explicit. Core and
total-attempt durations are nonnegative seconds. Outcome and reason are
separate so accepted, skipped, rolled-back, and failed attempts remain
distinguishable.

The contract validates each value and cross-field consistency, then uses
ordinary dataclass serialization. It does not inspect a dataset, perform
sleep, write a log, or decide whether a proposed change should be accepted.
The next implementation steps populate model facts and attach runner facts.
Measured durations must be kept out of deterministic learning-state
equality checks; this contract does not yet alter either backend's result.

## Alternatives

- Add optional scalar fields to both existing backend results: duplicates
  validation and still omits runner-owned guard and trigger decisions.
- Let runners log ad hoc dictionaries: permits missing fields and ambiguous
  proposed versus applied changes after rollback.
- Save only the final accepted state: cannot explain a rejected proposal or
  prove replay/structure was restored.

## Consequences and evidence

`tests/test_sleep_telemetry_contract.py` first failed because the module was
absent. It now covers immutable JSON serialization, accepted structural and
no-topology events, rolled-back structural/replay proposals, skipped and
pre-guard error records, and malformed budgets, identity, width, guard,
replay, chemistry, time, trigger, and reason fields. This is the P3.10a
schema gate. At this stage, model capture and runner propagation remained
P3.10b/c. No learning rule, baseline, seed, metric, or evaluation role
changed.

P3.10b later attached model-owned NumPy and Torch records to core sleep
results (ADR-0088/0089). Runner guard decisions and durable event sequences
remain P3.10c.
