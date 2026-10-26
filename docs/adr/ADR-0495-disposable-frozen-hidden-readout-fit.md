# ADR-0495 — Disposable fixed-hidden TRAIN head-fit diagnostic

## Context

The existing neural learner improves recorded imitation error but its frozen
robot returns remain unsuccessful. Same-input nearest memory works in known
contexts. The explicit gripper rule did not repair the neural control gap; stage
balancing already failed. More doses, seeds or threshold sweeps lack justification.

## Decision

Fit one disposable output head on the existing owned stream readout's frozen
hidden representation. Use fixed clipped inverse-tanh least squares, fixed rank
cutoff and original physical-target scoring. Require complete TRAIN episodes and
source isolation. Keep the benchmark learner, optimizer, actor and defaults intact.

## Alternatives

Another residual wrapper repeats information already available. More native
rollouts cannot answer fixed-feature fitting capacity economically. Changing the
benchmark optimizer or deploying the fitted head would combine distinct questions.

## Consequences

A constructive TRAIN fit can guide optimization work; a poor fit is evidence at
this numerical convention, not a proof of representation impossibility. Endpoint
clipping, ill-conditioning, nonuniqueness and all costs remain visible. No fit
score selects probes/final data or confers deployment authority. No new dependency,
framework, configuration or orchestration layer is introduced.
