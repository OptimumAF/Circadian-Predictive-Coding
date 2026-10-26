# ADR-0493: Version observed-teacher settling against the native goal radius

## Context

The fixed384-update BP TRAIN17 trial grasped but did not achieve task success.
Source investigation and pure counterexamples exposed a separate supervisory
contract gap: max-axis20mm entry/30mm retention admits SETTLE outside PickCube's
Euclidean25mm position ball. This is not proof the gap caused that trajectory.
Pinned Panda success also requires maximum arm velocity<=.2; geometry cannot
certify that condition. Old v1 evidence and default behavior must remain valid.

## Decision

Add GoalConsistentObservedGraspPickCubeTeacher as a narrow subclass overriding
held transport/settle selection only. Euclidean<=20mm enters settle; previous
settle retains it through inclusive25mm. Keep the existing entry margin, approach,
grasp, feedback, scalar-state bounds and lost-grasp behavior. This threshold is a
native task compatibility constraint, not an experimental search.

The existing collector/CLI require explicit goal_consistent=True / --goal-consistent
alongside observed-grasp readiness. A distinct frozen result subclass, definition
and episode ID preserve policy identity. The result retains the existing observed
feedback fields and actual final-success qualification; a hold command never
qualifies success. Existing v1/default schemas are unchanged. Existing learning
readers refuse v2 until explicitly extended and verified.

## Alternatives

Changing the shared max-axis helper or legacy thresholds would silently change
approach rules, v1 supervision and historical evidence. Merely clipping the old
30mm hysteresis still permits box corners outside the placement ball. A new
runner, learnable threshold or hard gripper decoder has no demonstrated need.

## Consequences

This repairs an observable teacher decision contract through existing components;
it does not establish native teacher success, learner acquisition, retention,
reliability, safety or CPC benefit. Software checks and any later native teacher
qualification remain separate gates and budgets. Preserve all old negative,
partial and failed runs. Subsequent learning admission must carry the explicit
definition without relabeling old data or exposing final/probe/guard labels.

## Verification and next dependency

122 focused checks/platform and independent software acceptance passed. The
separate native TRAIN17 teacher qualification passed in15controls/75knownphysics.
Its spec/transitions/stages are exactly identical to the accepted v1 demonstration;
this is no new learning coverage. Defer duplicate warm admission/training and
apply explicit v2 supervision through the existing student-arrival corrections.
[Software](../../artifacts/runs/goal-consistent-observed-teacher-20261020/RESULTS.md) ·
[Native](../../artifacts/runs/goal-consistent-observed-teacher-native-20261020/RESULTS.md).
