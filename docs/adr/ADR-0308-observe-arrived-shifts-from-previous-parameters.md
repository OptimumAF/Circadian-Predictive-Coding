# ADR-0308: Observe arrived shifts using previous copied parameters

## Context

The matched stream runner already emits copied checkpoint states and arrived
training metadata before final scoring. The learnability pilots found no CPC
advantage; nonmonotonic probe curves do not justify a renewal mechanism. Useful
adaptation needs signals available from ordinary data rather than final scores.

## Decision

Add bounded pure entropy/input-mean/residual summaries and frozen-reference
window/persistence decisions. Compose an observer with the existing runner at
every update. Retain one owned previous state and pass a separate copy to a
target-free predictor. Compute after the current update from preceding parameters
and arrived training data; label this timing explicitly. Preserve failures,
unreached rows and costs. Signals change no training or promotion policy.

## Alternatives

Current post-update parameters would absorb some of the label's effect before
the measurement. Label-conditioned latent diagnostics cannot be used before
labels arrive. Reading final evaluation for a trigger would violate isolation.
A new runner would duplicate the existing copy/arrival/scoring boundary.

## Consequences

Reference/window/report storage has explicit finite bounds. Measurements remain
model-independent; the adapter binds existing matched snapshots to feedforward
prediction. Strict thresholds and overlapping-window persistence are descriptive,
not independent statistical evidence. Mean drift can miss concept-only changes.
Any misses and false alerts remain evidence; deployable efficacy and future
policy feedback require a separate frozen protocol. No causal or CPC-benefit
claim follows from this engineering integration.

## Training-prefix calibration follow-up, 2026-10-16

The fixed-rule pilot missed every recurring change. Its uncertainty reference
near0.984 leaves less than0.016 positive headroom, below threshold0.05. Preserve
that negative protocol. Add opt-in calibration on a declared stationary arrived-
training prefix; use the same maximum observed nonnegative score plus fixed
margin rule across all methods, and freeze fitted thresholds before monitoring.
Report strict algebraic reachability without lowering thresholds to force it.

Why this: a prefix fit and domain check directly address an observed unusable
rule without final-evaluation access or outcome-dependent per-method tuning.
Reuse the existing detector by replaying bounded scalar summaries once at fit,
accounting for those extra detector steps; discard the prefix afterward.
Keep original defaults/report schema unchanged unless explicitly opting in.
Report row IDs and caller-declared run/update identity, not a claim of parameter
attestation. Training parameters evolve during calibration.

Two overlapping fit windows cannot establish false-positive calibration or
stationarity. Clipping to the measured headroom could conceal an unusable
channel, so fitted thresholds are capped only at the mathematical domain upper
bounds and unreachable channels remain explicit. Useful discrimination still
requires a separate frozen small protocol and honest misses/false alerts.
