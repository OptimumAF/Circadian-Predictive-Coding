# ADR-0380 — Continue an owned TRAIN prefix through mapped quotas

## Context

The scratch reward/control study has no positive TRAIN returns. Existing
acquisition/warm-start routes train native recurrent policies, but mapped quotas
reject their combination. New infrastructure cannot resolve this learning gap.

## Decision

Add a fixed A-only192 TRAIN prefix to the original reward execution routine and
compose frozen-prefix then reward-prefix arms in a small adapter/CLI route.
Regenerate both original learners; match full optimizer/gradient/parameter/counter
state at192 before reward collection. Keep one original budget/clock per arm,
absolute native counters and original quotas; freeze learned state for control.
All finals remain unavailable until both arms complete all planned TRAIN work.
Generalize the existing physical scorer with explicit native origin and original
prefix context history; omitted origin preserves native-zero semantics.

Why this: native demonstration/reward ports, original quota runner and final
scorer already implement the required objectives and boundaries. A prescribed
prefix allows acquisition to be assessed without adding a new runner or selecting
successful seeds/doses. A-only leaves B remapping untrained before continuation.

## Alternatives

Loading historical scalar reports cannot recover native tensors or optimizer
history. Resetting counters/budgets after prefix hides work. Increasing scratch
budgets or selecting favorable seeds from final results changes the frozen
comparison. Duplicating the actor/evaluator would create unnecessary drift.

## Consequences

Prefix work is charged and visible, even if acquisition fails. Fixed A-only192
is a new condition and carries no guarantee from the earlier mixed A/B/A prefix.
The known-A initial skill is interpreted after final execution, never used for
selection/promotion. Original scratch negatives stay preserved. Frozen has fewer
updates than reward, and neither supplies trained matched PC/CPC controls.
Scientific benefit and full roadmap completion require further evidence.
