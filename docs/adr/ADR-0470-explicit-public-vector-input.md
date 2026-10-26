# ADR-0470: Explicit fixed public-vector model input

## Context

The accepted camera acquisition/return case learns stored trajectories using
RGB789 nearest memory. Its public21 prefix is identical across the two camera
teachers. A matched simpler input control is missing; official42D state would
change the input contract. Existing raw RGB observations, costs and owner guards
must remain inspectable.

## Decision

Add a pure, immutable, owned known789-to21 projection and an explicit opt-in at
the existing frozen rollout boundary. Validate the full raw source before
dropping RGB. Keep raw789 traces, rendering, roles and controller contracts.
Require a matching21D policy before forks/native creation. Refuse unsupported
clock/correction combinations and bind projection accounting to prediction
progress. Preserve default calls and output keys.

Why this: a small model-input view isolates the actual missing comparison
without another learner, runner, camera condition or statistics-fitting rule.

## Alternatives

Changing the encoder globally would hide raw evidence and change defaults.
Official42D state or selected coordinates would create a different comparison.
A new orchestration framework is unnecessary. Ignoring invalid dropped pixels
would admit corrupt source evidence.

## Consequences

Scoped correctness can be accepted independently of a separately frozen native
comparison. The model may reduce prototype storage and distance coordinates;
renderer/acquisition work is unchanged and projection validation adds work.
No scientific benefit follows from the implementation. Reuse the accepted A29
statistics' exact21-prefix through the existing typed constructor in the next
protocol, rather than refitting and introducing reduction-order differences.
Full neural/adaptation/generalization/serving/release parent requirements remain
open. See docs/pick-cube-public-vectors.md and the implementation stage RESULTS.

## Follow-up evidence, 2026-10-19

The separately frozen public-vector-native case succeeds on shifted B and fresh
A using the same A29-only actor without new learning/labels. Both full raw789
traces equal teachers. This supports local simpler-input sufficiency and limits
the earlier RGB-adaptation interpretation, without establishing neural learning,
visual invariance/generalization or CPC advantage. All earlier negatives remain.
See docs/public-vector-camera-control.md and the linked actual independent review.
