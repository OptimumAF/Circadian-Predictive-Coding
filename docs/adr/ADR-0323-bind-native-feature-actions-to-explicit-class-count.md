# ADR-0323: Bind native feature actions to explicit class count

## Context

Actual native heads already support categorical outputs, but the owned feature
adapter fixes two classes in constructors, validation, state geometry and
prediction. Accepted binary demonstrations/live episodes therefore cannot admit
a third action. This blocks key/door imitation through the existing stream.

## Decision

Append `num_classes=2` to the existing feature policy, bounded to native integer
2..16. Bind it through every native state/constructor/output/optimizer geometry.
Accept hard integral class indices in the existing float64 B×1 label envelope.
For count2, preserve the old B×1 class1 prediction exactly; for explicit count≥3,
return B×K full native softmax probabilities. Keep native objectives and atomic
owned train/snapshot/restore/fork semantics.

Why this: the existing native head and learner contracts already express these
operations. A second learner or orchestration framework would duplicate native
training and continuation. The explicit count avoids inferring action geometry
from data, final scores or the largest label observed in a batch.

## Alternatives

- Unconditionally returning all binary columns breaks existing consumers.
- One-hot labels would change the public training contract without a native need.
- Silently mapping extra actions into two classes loses behavior information.
- Extending all environment and RL boundaries at once would obscure this first
  native correctness/continuation gate.

## Consequences

This is a concrete native learner prerequisite; original R4.2 stays open. Existing
binary consumers remain unchanged and categorical callers must deliberately
consume the full matrix. Current typed snapshot format1 owns the expanded policy;
no historical pickle/durable recovery compatibility is asserted. No dependency
or native loss/metric/seed/rate changes. Next extend categorical episode/attribution
and scoring semantics through the existing stream, before key/door experiments.
