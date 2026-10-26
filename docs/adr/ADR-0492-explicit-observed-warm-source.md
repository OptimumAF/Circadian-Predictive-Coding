# ADR-0492: explicitly admit the qualified observed warm source

## Context

The existing observed teacher qualified TRAIN17 in15 native controls, while
conventional learners trained on the old20-row teacher and private corrections
still failed. The learning artifact reader and stage bindings admitted only v1.

## Decision

Extend the existing reader with explicit observed_grasp=True and the robot CLI
with --observed-grasp-warm-teacher, requiring readiness+A-only+nonvisual inputs.
Validate original serialized bool feedback/raw42 float flags before normalization,
then reuse full selected qualification. Keep old defaults/schema/marker refusals.
Stage bindings retain their two fields and derive each definition from known
TRAIN identity, validating selected current/next flags before any copy. B19 stays
read-only at this CLI boundary. No new learner, loss, runner or dependencies.

Why this: it tests whether the already-qualified source can teach useful control,
using existing stream/rollout infrastructure and preserving every negative result.

## Alternatives

Relabeling the selected source as v1 discards observed-contact semantics. A new
loader or orchestration framework duplicates working paths. More seeds/thresholds
or dose search would change the current fixed scientific question.

## Consequences

Old v1 input remains available; selected data requires an explicit route. The
A15 source gives192updates2880rows rather than old3840 and is not matchedcompute.
A tiny real-learner software gate and independent review precede one separately
frozen native question. Implementation correctness does not establish learned
acquisition, retention, recurrence, generalization, safety or a CPC advantage.
