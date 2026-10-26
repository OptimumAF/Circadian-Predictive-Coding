# ADR-0502 — Explicit physical cube task state

## Context

PushCube has different structured observations and success semantics from
PickCube. Equal model width cannot establish equal feature meaning. Its standard
observation lacks the grasp flag used by the existing Pick task-space model.

## Decision

Use a new pure immutable physical state (observed grasp, world TCP/object/goal
positions) and explicit D12 displacement/executed-command history projection.
A separate optional TRAIN-only PushCube port obtains the real Panda grasp signal,
checks native controller/world-axis alignment and keeps task-specific sparse
success/end flags. It records the additional sensor work and bounded paid trace.
Existing PickCube representations and defaults stay unchanged.

Native scalar quaternion roundoff is admitted only under the explicitly reviewed
two-float32-epsilon unit-norm bound, with exactly zero vector components. Both
earlier strict-predicate failures remain failed; no angular tolerance is added.

## Alternatives

Reusing flattened Pick offsets would invent Push feature semantics. Filling grasp
with zero would invent sensor evidence. A new orchestration framework is not
needed: complete TRAIN records can feed the accepted existing-source continuation.

## Consequences

Push observations explicitly include an extra contact sensor; future compared
learners must share it and pay its costs. Python position subtraction need not
match previously rounded Pick displacement bits. Port correctness alone proves
no learning, retention or circadian benefit. Distinct Push supervision remains a
separate versioned requirement. [Evidence](../../artifacts/runs/push-cube-train-port-20261021/RESULTS.md).
