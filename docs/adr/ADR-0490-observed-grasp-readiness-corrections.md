# ADR-0490: Bind readiness corrections to observed grasp

## Context

A frozen mixed BP student grasped/lifted in TRAIN17, but the geometry-only query
teacher offered50 ALIGN/open labels because the student never followed its hover
phase. Eight arrived updates removed closing behavior and worsened both original
TRAIN reference errors. This accepted negative result is preserved in
artifacts/runs/bp-binary-gripper-readiness-recovery-20261020/.

## Decision

Offer explicit observed_grasp_corrections=True for nonvisual TRAIN readiness
queries. Read the current official raw42 float0/1 flag at18, validate native port
pairing before geometry/decision and retain exact bool feedback per paid label.
A separate ObservedGraspPickCubeTeacher requires exact bool feedback. True chooses
existing closed cube-to-goal lift/settle commands and tolerances; false cannot
promote six closing intentions into contact. Original v1 teacher stays unchanged.

Distinct ObservedGraspCorrectionBatch and provenance/definition retain flags
through ownership, arrival gates, canonical42/history47/taskspace12 views and
existing private learner updates. V1 schema stays unchanged. RGB/public21 and
teacher roll-in do not supply this selected feedback and are refused explicitly.
Source training, native actual soft commands, objectives and original budgets do
not change. Keep software acceptance and native usefulness separate.

## Alternatives

More updates/replay would repeat invalid held-object supervision. Widening hover
thresholds would change the original teacher without using available observed
state. Closing commands are not contact evidence. A new controller or orchestration
framework is unnecessary; the existing query/runtime already accepts paired labels.

## Consequences

The opt-in path costs bounded flag reads/boolean retention/validation in addition
to existing geometry and learning. It uses privileged TRAIN observation feedback;
no heldout/final/probe access or deployment promotion follows. The official flag
reports the current state, not persistent grasp security or safety. Preserve all
old records/negative results and version any future semantics separately. Native
benefit needs a separate prospective trial with unchanged metrics and honest costs.
