# ADR-0089: Capture Torch sleep identity before post-split pruning

## Context

The Torch head may select a prune candidate after a noisy split. The
candidate can be the split parent or its newly appended child. Final lineage
alone cannot identify a child removed in that same event. The head has no
sleep replay, and existing seeded comparisons depend on its model-local
split generator and deterministic `SleepEventResult` equality.

## Decision

Torch head `sleep_event()` now returns the same version-one typed core fact
record as NumPy. It captures `(parent ID, child ID)` pairs immediately after
the live split and before the post-split prune, then records selected and
actually removed stable IDs. The replay budget, examples, and updates are
zero. Primary, fast, and slow chemical summaries are extracted from the
selected device at entry and after validated work. Scalar extraction
synchronizes device work before the elapsed core duration is read. Executed
events are `applied`; disabled, adaptive-not-due, warmup, and legacy
zero-budget returns are `skipped` with reasons.

The record is attached to `SleepEventResult` with `compare=False` and does
not enter head state or a checkpoint snapshot. Construction occurs inside
the existing atomic snapshot boundary, so a telemetry error restores the
live head and local split generator. Runners may later add guard timing and
acceptance without changing this core boundary.

## Alternatives

- Infer split children from final lineage: fails when the child is pruned
  during the same event.
- Treat selected positional prune indices as stable IDs: indices shift when
  parents or children are added and removed.
- Add a Torch replay field with a nonfunctional switch: falsely implies a
  capability the head does not have.

## Consequences and evidence

`tests/test_torch_sleep_telemetry.py` first failed on the missing result
field. It covers forced split and chemistry, parent/child post-split prune,
prune-only and minimum-width no-op, four skipped routes, JSON-safe zero
replay, and injected telemetry failure with exact state and next seeded
split continuation. A tiny actual-device CUDA test verifies finite scalar
extraction and JSON serialization on the local GPU under a 60-second cap.
No baseline, learning rule, seed, metric, or final-test access changed.
