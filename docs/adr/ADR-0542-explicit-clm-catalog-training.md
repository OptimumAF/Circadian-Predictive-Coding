# ADR-0542: Explicit full-catalog CLM training

## Context

The accepted temporal CLM stream and reserved probes show incomplete adaptation.
Inspection establishes that native paired training scores paired positive-action
rows; the full active candidate catalog is available only for readout. Those
results do not establish that missing negatives caused the learning failure.
Existing consolidation binds LoRA factors rather than these dense CLM heads.

## Decision

Add an opt-in supervised catalog learner with explicit TRAIN query features,
ordered unique candidate IDs and indexed targets. Name its directional row-mean
cross-entropy objective separately. Reuse the owned learner's atomic optimizer,
clipping, scheduler, state guards and failure behavior, with permanent candidate
and logit exposure counters. Keep objective-specific public snapshots distinct.
Integrate through the existing temporal runner, not another orchestration layer.

## Alternatives

Changing native paired CLM silently would invalidate existing reproduction and
comparison assumptions. Repeating paired runs cannot fill the missing interface.
New circadian mechanisms or a dense-head consolidation framework would add
confounds before establishing conventional task learning.

## Consequences

Existing defaults remain paired. Catalog training changes both the objective's
directionality and candidate composition; it is not an isolated causal test of
negative coverage. Correctness is tested with independent scalar math, gradients,
owned-state/failure checks and affected stream regressions. One fixed native case
may compare with an accepted historical paired control only after source,
environment, text/token and pre-update probability bindings pass. Such reuse is
historical and does not establish matched total computation or broad benefit.

See [usage and module boundaries](../clm-catalog-learning.md) and the preserved
[protocol](../../artifacts/runs/clm-full-catalog-20261022/PROTOCOL.md).
