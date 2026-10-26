# ADR-0379 — Small repeated-instance game comparison

## Context

The existing reward and frozen routes support own-policy96-step quotas and explicit
environment instances. A matched comparison across more than one instance is
missing. The historical frozen0/23 supervisor qualification failed and cannot be
silently admitted or retried as if it passed.

## Decision

Add a fixed four-cell composition of existing single-arm recipes to the existing
CLI:9/14 and18/5, each reward then frozen, unchanged initialization17/sampler29.
Capture protocol/options/runner/clock/wall limit before work; detach raw results,
validate preceding success, and stop on first failure. A pure app projection joins
physical stages and preserves every failed/not_run cell. Aggregate only when both
declared pairs complete. Keep original signed metrics and unequal actual costs.

Why this: existing runtime/learner/final-isolation behavior is sufficient. A small
declared matrix provides useful descriptive evidence without introducing another
runner or changing objectives/baselines to favor an outcome. Structured fresh
catalogue pairs avoid reusing the failed historical qualifier; choices precede
their outcomes, not selected for performance. Old negatives remain preserved.

## Alternatives

Substituting old teacher trajectories violates own-policy exposure. Admitting the
failed supervisor artifact hides a failed gate. A seed sweep or new orchestration
framework adds cost without resolving the current comparison question.

## Consequences

Every arm owns its original budget; no retry/renewal within an arm. Final data
never enters later training or selection. Two finite related pairs do not justify
statistical/generalization/CPC claims. Frozen is a nonlearning control, not the
full trained-baseline requirement. Native execution needs a separately frozen,
reviewed local protocol and resource caps. Experiment driver preserves returned
raw cells before subsequent work; parent failure does not erase known results.
