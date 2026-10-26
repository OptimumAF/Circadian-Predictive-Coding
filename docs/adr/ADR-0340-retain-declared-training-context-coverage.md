# ADR-0340: Retain declared training-context coverage

## Context

The accepted four-phase recent and partitioned 16-transition controls both have
zero final B/C live success. Recent selection is blue-only with the frozen
alternating arrivals and period-two replay. More checkpoint infrastructure does
not address this demonstrated exposure limitation.

## Decision

Add one opt-in bounded episode store through existing StreamRehearsal, keeping
latest released train episode per declared task/cue key. Select eligible keys in
fixed round robin without resetting on replacement. Refuse undeclared contexts;
reuse existing codec, content IDs, native consolidation and exposure budget.
Compare only against explicit recent48/partitioned48 controls with the same
48-transition/12,096-byte ceiling and fixed native/replay update counts.

## Alternatives

Changing the existing recent baseline would invalidate preserved controls.
Increasing unbounded history would hide storage cost. A new orchestration layer
is unnecessary because released episode admission and replay already exist.

## Consequences

Coverage has stronger declared task/cue information and different exposure
composition. A difference cannot establish a pure reservoir, age, capacity or
circadian effect. Preserve all old and new outcomes, including negative ones.
Source correctness and a frozen single-seed case do not establish general utility.
