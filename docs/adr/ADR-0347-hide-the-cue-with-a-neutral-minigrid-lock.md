# ADR-0347: Hide the cue with a neutral MiniGrid lock

## Context

The actual MiniGrid boundary supports native observations and learner ports,
but Memory is not the roadmap's recurring key/door game. A standard colored
MiniGrid door would reveal which key is needed after a transient cue disappears.

## Decision

Construct a bounded native MiniGrid scene with a transient ball cue, two keys,
neutral grey locked door and goal. Remove the ball before the first native step.
The private lock checks the carried key without exposing its required color in
images, mission or info. Correct unlocking delegates to native door opening;
wrong-key attempts terminate explicitly. Preserve full public observations and
native action/step/end semantics.

Use a pure teacher whose only input is the public observation; cue and navigation
phase derive from public history. Declare TRAIN layouts and separate final D.
Reuse existing records, codecs and native stream ports. Extract the existing
private episode lifecycle so both actual environments share failure accounting.

## Alternatives

A colored door would eliminate the memory requirement. Passing a target or
layout to the teacher would weaken its public-observation contract. Duplicating
the complete adapter lifecycle would risk divergent cost/end/role behavior.

## Consequences

The task is a custom environment built from actual pinned MiniGrid primitives,
not a registered upstream benchmark. Its shaped success reward and wrong-key
failure are explicit. Scripted success is separate from learned adaptation.
The full recurring learning, retention, held-out and recurrent RL requirements
remain open; a separate small reviewed protocol must precede comparisons.
