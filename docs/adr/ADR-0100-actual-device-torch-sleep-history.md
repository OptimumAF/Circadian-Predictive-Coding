# ADR-0100: Gate Torch runner sleep history on actual CUDA continuation

## Context

The default `.venv` contains CPU-only Torch, but this checkout also has a
working `.venv-cuda` environment on an NVIDIA GeForce RTX 3080. CPU tests
proved the typed fixed-feature and unmatched-vision history contracts but
could not establish device process-RNG, head-local split-generator, or CUDA
checkpoint-memory continuation.

## Decision

Exercise the existing checkpoint protocols on the actual CUDA device with
small synthetic inputs and controlled guard scores. For fixed features,
compare accepted/rejected and pre/core/post error histories across explicit
resume, process and head random streams, a written local JSON result, and
two-process allocator-memory routes. For unmatched vision, exercise v1/v2/v3
accepted/rejected/error histories, their distinct guard roles and selected
exposure, error rollback, sealed final test, written local JSON, and exact
non-timing model/report continuation. Keep measured event durations out of
cross-process semantic equality; require every other event fact to match.

The controlled scores are correctness fixtures for guard decisions. They do
not select research seeds, retune a baseline, change an experiment metric, or
claim that circadian learning wins. CUDA tests use the saved protocol IDs and
existing small training configuration.

## Alternatives

- Infer CUDA behavior from CPU tests. This cannot exercise the selected
  device's process stream, head generator, or allocator segments.
- Compare elapsed event durations across processes. Scheduling and file
  boundaries change durations even when the trained state and decisions are
  identical.
- Force a rollback by replacing only the computed delta. It would disagree
  with the recorded guard scores and violate the telemetry contract; the
  fixture instead supplies consistent pre/post scores.

## Consequences

The bounded device fixtures test CUDA continuation without a sweep. The
fixed-feature memory worker now removes only volatile event durations from
state equality. Existing allocator segments, model hashes, baseline metrics,
and final-test release assertions remain in place. The cross-runner artifact
audit remains P3.10c5.
