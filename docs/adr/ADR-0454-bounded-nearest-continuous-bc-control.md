# ADR-0454: Bounded nearest-observation continuous BC control

## Context

Repeated MLP optimization, normalization, public time and binary-gripper loss
retain grasp error; native MLP cases remain negative. A data-matched retrieval
control can separate executable demonstration cloning from this optimizer.
Existing categorical exact-key action memory does not support continuous units.

## Decision

Add pure bounded squared-L2 retrieval and an owned nearest continuous learner.
Use mandatory frozen statistics, earliest-row computed ties, strict role/spec/
action validation and atomic append with capacity refusal. Keep all retained
rows; full snapshot/fork/restore state includes normalized prototypes, physical
labels, policy and counters. Expose distance work alongside predictions.

Narrowly extend the existing frozen smoke with this exact learner and a separate
typed state route. Preserve MLP guards/defaults and distinguish append counts
from optimizer work. No new runner, disk codec or simulator adapter.

Why this: test the released demonstration through a simple data-matched control
with explicit memory/compute costs before adding further MLP variants.

## Alternatives and consequences

Interpolated actions, adjustable neighbor counts, metric tuning or eviction add
choices not needed for this control. A full query/prototype/feature tensor has a
larger temporary allocation; per-query distances keep it bounded. Ordinary
floating arithmetic may create computed ties, resolved by fixed row order.

683 scoped WSL checks,79 Windows checks/statics and one native case pass review.
All29 teacher transitions are exactly reconstructed with sparse success. This is
one-trajectory memorization/reconstruction with full replay memory, not strong
BC/generalization/plasticity/retention/recovery/CPC qualification. Broader R4.4
requirements and every prior negative result remain open/preserved.
