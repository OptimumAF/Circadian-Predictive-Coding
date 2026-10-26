# Observed-grasp readiness corrections

## Student-prefix teacher completion

The existing observed-v2 query pipeline now supports a fixed student prefix then
teacher completion, using actual executed taskspace history and distinct mixed
behavior receipts.100 checks/platform and a fixed native question are independently
accepted: four teacher controls finish the task after12 student controls; the
updated pure student still fails and both original fits worsen.
[API and scopes](readiness-roll-in.md) ·
[Full result and costs](../artifacts/runs/student-prefix-teacher-completion-native-20261020/RESULTS.md).

## Optional stage-balanced A-only acquisition

`run_continuous_learnability(..., a_only_control=True,
stage_balanced_packets=True, diagnostic_stage_bindings=(warm, new))` selects
15 original warm TRAIN rows per update: three per recorded ALIGN, DESCEND, GRASP,
LIFT and SETTLE stage, cycling deterministically within each stage. The full
qualified warm source must have 15 rows and all five stages. Original targets and
already encoded actual histories are gathered directly; no reordered trajectory
is fabricated. B remains read-only. Use official state42 or the declared existing
history12/47 recipe; RGB, replay and probe composition are refused in this scope.

The robot learning CLI requires the explicit combination:

```powershell
python -m scripts.run_continuous_robot_learning --output <fresh-output.json> --warm-teacher <qualified-warm.json> --new-teacher <qualified-new.json> --readiness-teachers --readiness-stage-residuals --a-only-control --stage-balanced-packets
```

For the qualified observed-grasp source, also use
`--observed-grasp-warm-teacher`; the tested BP taskspace case uses
`--binary-gripper --taskspace-history executed`. Defaults retain uniform packets
and their previous report schema. The new option changes empirical sample
weighting. Complete A/B readouts remain on their original unweighted distributions.
Row-index provenance, actual unique/repeated rows, stage exposures, gathering and
selection costs are reported; validation and copy overhead is additional.
The existing stream and robot boundary refuse changed prior packet/receipt/clock
or source evidence before further work and before releasing the frozen return.

Why this: one-row GRASP/SETTLE stages were underrepresented in the qualified
15-row demonstration, and the accepted first corrective step locally worsened
grasp fit. Equal stage exposure is a fixed TRAIN hypothesis; it does not create
new state coverage or establish a learned controller. **44 focused checks per
platform and scoped statics passed with independent acceptance.** Selected PC/CPC
execution remains unqualified. Freeze a separate protocol before model/native
work. [Software evidence and costs](../artifacts/runs/stage-balanced-acquisition-20261020/RESULTS.md).
The separately frozen 384-update native case was independently accepted as
negative: 42 grasp-feedback steps, no task success, closest goal distance 31.880 mm.
Improved rare-stage fit did not deliver useful closed-loop control.
[All outcomes and costs](../artifacts/runs/stage-balanced-acquisition-native-20261020/RESULTS.md).

## Explicit goal-consistent correction supervision

The existing query and collection/private-update/return API accepts
`goal_consistent_corrections=True` alongside `observed_grasp_corrections=True`
and `readiness_corrections=True`. The smoke CLI equivalent is
`--goal-consistent-corrections --observed-grasp-corrections --readiness-corrections
--corrective-queries` with an admitted TRAIN learner. This selects the existing
v2 teacher's Euclidean20mm entry/25mm retention rule for held-object labels.
Raw nonvisual observations and no teacher roll-in remain required.

GoalConsistentObservedGraspCorrectionBatch retains the existing feedback/label
fields but uses distinct provenance provisional_geometry_observed_grasp_goal_consistent_v2
and definition pick_cube_geometry_observed_grasp_goal_consistent_v2. Raw42,
canonical42/history47/taskspace12 views and private update reports preserve that
exact family. Old observed-v1/default schemas and supervision are unchanged.

Only collection receives the query option. The deployed/source learner stays
frozen, updates apply to a private candidate, and the return runs the frozen
candidate without correction queries. Endpoint labels never replace actual soft
student commands or history. Original arrival clocks, budgets, receipts, failure
charges and optional replay exposure remain explicit. Declare a separate budget
before executing any model or simulator work.

Why this: v2 qualified as a teacher but produced exactly the existing demonstration
at the known reset. Repeating that training data would add no coverage. This path
lets the versioned rule label states the student actually visits; useful native
learning remains a separate experimental requirement.

Executed evidence:47 focused checks per platform, scoped lint/format/typing
and independent acceptance passed. The fixed384-source/eight-private-update
native question also received independent acceptance, with a negative result:
both50-control episodes failed, both original TRAIN fits worsened, and student
labels contained no SETTLE examples. The original teacher's one hold example is
also imperfectly fitted. This does not establish a live causal explanation.
[Software](../artifacts/runs/goal-consistent-readiness-corrections-20261020/RESULTS.md) ·
[Native result, costs and next action](../artifacts/runs/goal-consistent-readiness-recovery-20261020/RESULTS.md).
The optional TRAIN step-direction readout below is now implemented and accepted;
retain strict reserved probe roles and freeze/review any later experiment.

The geometry-only v1 teacher remains the default. The opt-in path uses the current
official grasp observation to give closed transport/settle supervision even when
the student did not visit the teacher's hover phase. False feedback still permits
approach/grasp attempts using existing thresholds, but six closing commands cannot
stand in for contact. This is a TRAIN teaching signal, not a qualified expert,
persistent attachment, safety or learned success claim.

Why this: an accepted mixed-BP recovery trial produced50 ALIGN/open labels while
the student actually held the cube; the update lost closing behavior and worsened
both original reference fits. This change fixes the missing supervision input.
[Decision](adr/ADR-0490-observed-grasp-readiness-corrections.md).

## Optional first-step TRAIN diagnostic

`run_continuous_correction_update(..., diagnostic_stage_bindings=(warm, new))`
selects a read-only diagnostic on the first paid corrective step. Each reference
is an existing exact `ReadinessStageBinding` for complete TRAIN17/19 data in the
learner's frozen model space, ordered warm17/new19. For history inputs, construct
the references with the existing encoding and pass the same recipe; the adapter
validates the executed-action history. Known TRAIN data is not a fresh-task probe.
The existing v2 warm-ID refusal stays in place.

The adapter records exact native before/after snapshots around that update,
validates its original receipt/charge, then runs the existing physical half-MSE
direction calculation before any replay. `report["step_diagnostics"]` contains
overall/align/descend/grasp/lift/settle groups for both references. An empty stage
has zero rows and `result=None`; it performs no gradient computation. The default
omits this section and incurs no diagnostic math. No extra optimizer step, fit,
prediction, simulator, promotion or final/probe access is requested.

A negative gradient dot realized step predicts only a first-order decrease in
physical half-MSE at the old parameters. It does not measure actual after-loss,
closed-loop success or PC energy descent. Native mixed BP/PC objectives are named
separately. Overall and stage gradients overlap and both computations are charged;
work distinguishes attempted from returned gradients, with copying/validation/
containers/RSS additional. A diagnostic failure preserves its paid first update,
partial work and error, stops further wakes/replay, and releases no candidate.
Declare a separate budget for using this opt-in in tests or experiments.

[20 focused checks per platform, scoped statics and independent acceptance](../artifacts/runs/readiness-correction-step-diagnostics-20261020/RESULTS.md)
cover the tiny mixed-BP integration, default equality, partial paid failure and
input/evidence refusals. PC/CPC support is source-reviewed, not executed by these
BP-only fixtures; no native or scientific benefit claim follows.

## Existing API

```python
from pathlib import Path
from scripts.maniskill_continuous_smoke import run_corrective_adaptation
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe

report = run_corrective_adaptation(
    mixed_bp_source,
    output_directory=Path("new-exclusive-directory"),
    visual_observation=False,
    readiness_corrections=True,
    observed_grasp_corrections=True,
    action_history_recipe=PickCubeTaskSpaceHistoryRecipe(mode="executed"),
    binary_gripper_axis=3,
    acquisition_only=True,
)
```

This performs simulator and optimizer work; declare a separate local budget before
running it. run_frozen_rollout accepts the same observed flag for query collection.
The smoke CLI flag --observed-grasp-corrections also requires --readiness-corrections
and --corrective-queries with an admitted learner. The new mode requires nonvisual
raw42 observations, no actor/clock/context intervention and no teacher roll-in.
The current official flag at18 must be exact float0/1 and match the current native
port after all metadata callbacks. Actual commands remain student soft actions.

The released ObservedGraspCorrectionBatch carries native bool feedback alongside
unchanged ReadinessCorrectionLabel records and original observation/arrival ticks.
True labels are lift/settle (closed), false labels approach/grasp. Ownership and
raw→canonical42/history47/taskspace12 views preserve family and feedback; model12
uses grasp feature0. Invalid flags, aliases, lost pairing or substituted old family
refuse work/release while retaining paid evidence. RGB/public21 is unsupported.

run_continuous_correction_update derives provenance/definition from the actual
owned batch. Existing MSE/BP binary/PC/CPC/nearest objectives, original budgets and
permanent charges remain; selected binary still requires its explicit argument.
The runner sends the observed option only to collection and returns a pure frozen
student. Optional reference rehearsal keeps its separate original exposure budget.

## Verification and extension

Focused commands and executed scope live in
[the software stage](../artifacts/runs/observed-grasp-corrections-20261020/RESULTS.md).
No new dependency/environment variable or model math. Extend using the existing
query/view/learner path and distinct semantics; preserve exact observed flag and
paid label pairing rather than inferring contact from commands or relabeling old
trajectories. Native usefulness and all broader research/release gates stay open.

## Observed evidence

Software146 checks per platform and independent review passed, including observed
MSE BP/PC/neutral/active private updates, mixed BP fake robot flow and preserved v1
defaults. A separate native question correctly supplied10ALIGN/40LIFT labels with
current flags, but the updated pure student still never closed/grasped or succeeded.
A TRAIN fit worsened; B improved slightly. [Native result, costs and auxiliary failures](../artifacts/runs/observed-grasp-readiness-recovery-20261020/RESULTS.md).
No reliability/CPC benefit follows. The existing reference composition subsequently
passed six focused checks/platform, but its fixed A20 replay treatment also failed
to retain grasp or task success and worsened both TRAIN reference fits. All old
budgets/results remain closed/preserved.

## Reference rehearsal composition

The existing optional reference API also accepts the observed batch family:

```python
from src.app.continuous_stream_rehearsal import ContinuousReplayPlan

updated = run_continuous_correction_update(
    source,
    observed_batch,
    now=arrived_tick,
    action_history_recipe=recipe,
    binary_gripper_axis=3,
    replay_plan=ContinuousReplayPlan(every_updates=2, max_replay_examples=80),
    reference_episodes=(original_train_model_episode,),
)
```

The reference must be a complete TRAIN episode in the same frozen model space;
it is not a raw42 episode when the learner uses taskspace12. The80-row example
permits four replays of an original20-row bank. Eight correction charges and
their original arrival clocks remain separate from replay exposure charges.
The source/actor stay unchanged and the candidate receives both kinds of update.

[Six focused checks/platform and independent acceptance](../artifacts/runs/observed-grasp-reference-retention-20261020/RESULTS.md)
verify this composition with a tiny real learner. They establish correctness;
useful retention and native control require separate evidence. Extend through the
existing bank/plan arguments and retain reported extra memory/computation costs.

[The fixed native replay question](../artifacts/runs/observed-grasp-reference-retention-native-20261020/RESULTS.md)
completed with zero success/no updated grasp and worse A/B fit than correction-only,
despite four extra updates/80rows. Another replay ratio or dose is not indicated by
this case. The historical next action was teacher qualification; the collector
now supports observed v1 and explicit goal-consistent v2, both qualified at the
known reset. Their demonstration contents are identical. The current accepted
v2 correction result above supersedes that handoff; neural control remains open.

## Accepted first-step finding and next repair

The separately frozen no-simulator case matched the original384 source and all
eight native mixed correction receipts. Its first step points against GRASP fit
on both original references and toward SETTLE fit; inherited momentum points
toward better warm grasp fit. Do not infer a momentum cause or eight-step phase
trajectory. [All groups, work and independent review](../artifacts/runs/readiness-step-interference-20261020/RESULTS.md).

Next implement an opt-in fixed15-row A-only packet with3original examples per
recorded stage and deterministic index cycling. Keep complete source episodes,
original targets and already encoded histories intact; report repeat/stage costs.
This is an unproved base-acquisition repair, with correctness and a separately
frozen scientific/native question still required. No new loss, learner port,
optimizer update, RNG or runner is needed; B remains read-only.
