# ADR-0458: Declare bounded public RGB features in the existing PickCube port

## Context

R4.4 requires state observations followed by frozen visual features. The current
continuous port uses raw state and disables rendering. The existing 2048D image
backbone exceeds the continuous spec's 1024D bound. A public visual route is
needed before robot visual learning or domain recurrence can be claimed.

## Decision

Add opt-in RGB mode to the existing port, teacher and two smoke CLIs. Freeze one
789D analytic recipe before camera outcomes: qpos9/qvel9/public goal3 plus 768
normalized 8x8 spatial block means from fixed 128x128 RGB. Select only declared
public SDK arrays; exclude privileged object geometry. Keep current frame only,
owned exports, explicit work counters and exact recipe checks. Preserve original
state/default behavior and existing controllers, learners and app runtime.

Why this: deterministic pooling supplies a small visual control without learned
weights, downloads or a new orchestration layer. Public goal coordinates are an
explicit task instruction because the SDK hides the goal marker from its sensor.
Teacher privileged geometry is label generation only. No new dependencies.

## Alternatives

A pretrained 2048D backbone needs a separate dimensional/interface and dependency
decision. A learned encoder would add fitting and selection authority. Arbitrary
camera/encoder settings would prevent a fixed reproducible recipe. A new runner
would duplicate existing role, control, ownership and work boundaries.

## Consequences

Existing NN/MLP and stable actor interfaces can consume the declared features.
Current payloads are bounded, but renderer initialization, hidden SDK renders,
Python allocation overhead and whole-lifetime descendant memory are not fully
accounted. One native reconstruction establishes public visual IO and frozen
lookup behavior only. Generalization, useful visual adaptation, domain drift,
promotion, comparative benefit and release remain open. Future recipes require
explicit versioning and independent verification; this recipe is not retuned.
