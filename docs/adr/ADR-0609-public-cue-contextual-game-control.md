# ADR-0609: Reuse two public-cue heads for the existing game

## Context

The conventional contextual CPU learner has complete owned state and working
ticket behavior. Neural MiniGrid controls still miss known TRAIN-A cue turns;
standardization improved mean fit but both A games failed. The simpler FIFO36
control already solves known cue tasks. R2.1 and matched native controls exist.

## Decision

Add an opt-in raw296 MiniGrid adapter over the original two-head unit-mistake
learner. Validate/canonicalize both public frames; route from exactly one visible
initial red/blue cue, with all canonical features retained. Bind snapshots to
this encoding. Extend the core context bound to296 head inputs plus2routing
columns, preserving original update/default behavior and finite capacity.

Compose A8/B8/C8/A8 through the existing delayed stream and diagnostic CLI.
Read all five owned checkpoints after completing32updates; reuse the existing
frozen known-A game evaluator. Its facade requires the unchanged original
fully spent32-update budget and clock, so no new training authority is created.
Prior paid results, source/actor, inputs and original charges remain guarded.

Why this: a small conventional reuse candidate addresses acquisition without
adding a new optimizer, orchestration framework or provenance/capture sequel.
Context identity comes from public observations; layout names and labels are
excluded from routing. This is an explicit capacity change, not a CPC feature.

## Alternatives

The existing cue-relative transform changes key-color coordinates and does not
route heads. Altering its semantics would invalidate historical evidence. More
joint BP exposure, width/rate/seed search or repeated probes lacks a new declared
hypothesis and would not repair an observation-to-control gap automatically.

## Consequences

Two heads cost4158float64 coefficients/33264numeric bytes plus copies. These
blue/red heads do not discover layout contexts. Six TRAIN fits measure imitation;
known-A games measure actual behavior. Stronger B/C gameplay, final generality,
neural/CPC advantage and release readiness remain separate requirements. The
one frozen case may be negative; every failure and spent allowance is retained.
Evidence: issue93 and its local protocol/results.
