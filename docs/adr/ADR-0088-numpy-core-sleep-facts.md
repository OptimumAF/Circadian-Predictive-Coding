# ADR-0088: Attach measured NumPy core facts to sleep results

## Context

The version-one sleep-event contract needs resolved model facts before a
runner can add guard and persistence facts. NumPy sleep may split a parent,
schedule a gradual prune, replay batches of different sizes, and finalize an
older pending prune during replay. Existing `SleepEventResult` equality is
used in seeded rollback and resume checks. Core timing varies between
otherwise identical trajectories, and storing it in the model snapshot
would make deterministic state comparisons misleading.

## Decision

Every returned NumPy `sleep_event()` result carries a typed `telemetry`
value. The field is excluded from dataclass equality and is not stored on
the model. Executed events report resolved split/prune limits and a replay
update limit capped by retained snapshots, stable `(parent ID, child ID)`
pairs, selected/scheduled/removed IDs, exact successful replay example and
update exposures, pre/post primary/fast/slow chemical summaries, and width.
Core duration starts after argument normalization and ends after model
validation and summary calculation, just before constructing the record.
An unguarded executed core event is `applied`; disabled, not-due, warmup,
and legacy zero-budget returns are `skipped` with explicit reasons. Core
events have no guard and set attempt duration equal to core duration so a
runner can replace it with the measured guarded attempt later.

The typed record is built inside the existing atomic sleep boundary. A
telemetry construction error restores the same weights, chemistry, replay,
topology, counters, and RNG as any other post-mutation error. Returning a
record does not advance model clocks or change replay selection. A later
runner rollback can retain the proposal while clearing applied facts.

## Alternatives

- Store last-event telemetry in the model: it would enter NumPy's complete
  `__dict__` snapshot and change learning-state equality for timing alone.
- Infer replay examples from `replay_steps` or buffer size: snapshots hold
  different batch sizes, and configured steps can exceed retained entries.
- Derive removed IDs only from newly selected prunes: replay may finalize a
  neuron scheduled in an earlier event.

## Consequences and evidence

`tests/test_numpy_sleep_telemetry.py` first failed on the missing result
field. It exercises forced split identity and chemistry, immediate prune,
scheduled prune followed by replay finalization, exact one- and two-batch
exposure, replay-only and executed no-op, four skip reasons, JSON encoding,
and a telemetry failure after mutation with exact snapshot and next seeded
sleep continuation. Existing atomic, component, lineage, clock, prune, toy
resume, and continual resume tests pass. The Torch head and runner-owned
guard/report facts remain P3.10b2/c. No baseline, seed, metric, training
rule, or final-test access changed.
