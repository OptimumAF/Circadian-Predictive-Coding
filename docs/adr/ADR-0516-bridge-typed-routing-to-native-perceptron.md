# ADR-0516 — bridge typed routing to the native perceptron

## Context

R5.2 has a verified lexical reference but no learned typed-routing control. An
existing deterministic categorical perceptron and TRAIN-only attribution stream
already own learning and scheduling. Evaluation labels must not enter training.

## Decision

Add a fixed-vocabulary pure projection and an adapter that builds only TRAIN
packets and reads detached native snapshots. Reuse the existing stream with no
heldout data. Evaluate every frozen development checkpoint after all updates.
Compute permitted affine logits before softmax, retaining native coefficients
and update semantics. Declare the six-word new/old/return example prospectively.

## Alternatives

Another trainer/stream would duplicate working components. A released modern
model requires separate source/weight/native-input pins. Global probability
renormalization can lose every allowed candidate to underflow. Vocabulary fitting
from development queries would mix partitions. None is needed for this control.

## Consequences

No new dependency or old learner/metric change. Catalogs currently require at
least three entries because existing two-class stream batches have binary
semantics. Unknown words do not generalize automatically. Development overlap,
memory/compute cost and every failed or adverse result remain explicit. The
reference does not establish modern-family equivalence, calibration, native task
performance or circadian advantage.
