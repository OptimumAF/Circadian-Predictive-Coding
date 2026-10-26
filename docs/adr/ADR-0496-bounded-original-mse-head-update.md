# ADR-0496 — Explicit bounded original-MSE BP head update

## Context

Fixed hidden features admitted near-exact37-row TRAIN fitting, but only with a
large-norm ill-conditioned inverse-tanh readout. That diagnostic is undeployed,
not the original objective. Neural control negatives and memory successes stand.

## Decision

Add a pure fixed damped Gauss–Newton head proposal and explicit methods on the
existing BP learner. Keep the original normalized MSE and require strict actual
loss decrease. Freeze hidden parameters, cap the global head step, preserve
hidden momentum and reset only head momentum on accepted commit. Retain paid
rejected work and source mutation evidence. Default SGD and PC/CPC stay intact.

## Alternatives

Installing the diagnostic head would violate its nondeployment contract and
confound objectives. More training-dose/seed/decoder sweeps repeat negative
avenues. All-parameter second-order fitting is a larger change; the head-only
proposal tests the observed fitting gap with fewer parameters and no new state.

## Consequences

This is an optimizer hypothesis, not a promised native control improvement.
Saturated derivatives, conditioning and missing coverage remain possible limits.
Existing learner snapshots/counters suffice; no new orchestration or dependency.
Any stream integration must declare optimizer choice, momentum semantics,
solver/diagnostic costs and rejects before comparative or native experiments.
