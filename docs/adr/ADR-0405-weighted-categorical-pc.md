# ADR-0405: Explicit weighted categorical PC proposal

## Context

The existing complete-episode weighted feature update supports BP only. Native
PC manually descends categorical CE plus a hidden quadratic penalty, while its
legacy diagnostic is a different MSE statistic. Current negative acquisition
results do not establish a cause or justify changing baselines.

## Decision

Add a detached core parameter proposal using explicit validated TRAIN-role
weights. Normalize the entire per-row CE-plus-hidden energy by observed weight
mass. Preserve native `B*dE/dz` relaxation and detached partial parameter
gradients, and exact native parameter operations for uniform target mass.
Keep actual energy and legacy statistic separately named. Independently verify
derivatives, native all-one identity and input/state isolation before composing
a learner adapter. No default, native state format, schedule or experiment changes.

## Alternatives

Weighting CE alone changes the balance against the hidden prior rather than
weighting each row's objective. Dividing latent gradients by B additionally
changes the finite-step native solver even at all-one weights. Differentiating
through relaxation would implement a different algorithm. All are declined for
this matched primitive.

## Consequences

The core has no update authority or adaptive counters. Atomic PC/neutral
integration and a prospectively frozen scientific protocol remain separate,
unfinished work. Correctness is not evidence that weighting improves learning.
