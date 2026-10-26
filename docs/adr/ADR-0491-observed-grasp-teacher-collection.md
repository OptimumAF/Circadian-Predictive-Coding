# ADR-0491: Collect the existing observed-grasp teacher explicitly

## Context

The selected observed teacher was exercised as an off-policy labeler. Correct
feedback and fixed reference replay still yielded failed learned control. The
existing demonstration collector only ran geometry-only v1, so selected-policy
closed-loop usefulness had no native qualification.

## Decision

Extend the existing collector/CLI with explicit observed-grasp selection, reusing
the existing core teacher and thresholds. Use a separate frozen keyword-only
result subclass carrying current-feedback values and a distinct episode identity.
Bind native current state and actual returned controls; save paid decisions and
failed-step work before validation/cleanup can refuse later release.

## Alternatives

Adding feedback fields to the base result changes default serialized schemas.
Replacing v1 silently would invalidate old source admission and results. Another
runner or teacher algorithm is unnecessary. More learner optimization without
qualifying the supervisory policy leaves the demonstrated dependency unresolved.

## Consequences

Defaults and old artifacts remain; the old learning reader refuses new selected
data. Software correctness and one fixed native teacher qualification are separate
gates. Qualified teacher data still needs explicit learner admission, and success
of this teacher is not success of a learned policy or a circadian benefit.
