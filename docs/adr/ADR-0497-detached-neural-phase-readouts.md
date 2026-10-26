# ADR-0497: Detached neural phase readouts

## Context

The existing GN-trained neural policy completes one known TRAIN17 robot return.
Controlled before/after-B comparisons need intermediate policies, while the
existing neural runner returns only its final candidate. The nearest-stream
runner already exposes detached observations at fixed boundaries.

## Decision

Add an optional observer to the existing neural checkpoint boundaries. Pass
owned native forks and small metadata; require unchanged readouts and protected
benchmark authority, retain bounded paid results, and keep callback work separate.
None preserves the existing schema/callshape. Initially admit only plain TRAIN
A/B/A, including BP/PC/CPC and explicit BP GN; do not couple it to probes/replay.
Results cannot influence the scheduled training sequence or promotion.

## Alternatives

Reconstruct each intermediate model independently: repeats optimization and
creates another orchestration path. Monkeypatch private checkpoints in each
experiment: obscures the supported isolation/cost contract. Introduce a new
runner or serialization scheme: unnecessary for in-process frozen serving.

## Consequences

Existing learner and runtime remain the training owners. Callbacks receive no
live runtime authority, no initializer is added, and failures retain paid work.
Fork/snapshot/result storage is bounded; opaque callback computation must be
budgeted externally. Controlled native adaptation remains a separate frozen
scientific question after software checks and independent acceptance.
