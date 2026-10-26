# Observed-grasp teacher collection

## Explicit goal-consistent policy

`collect_readiness_teacher(observed_grasp=True, goal_consistent=True)` and the
existing CLI's `--geometry-readiness --observed-grasp --goal-consistent` select
the versioned Euclidean transport/settle policy. It enters hold within20mm and
retains it within the native inclusive25mm goal radius. Approach/grasp/lost-grasp
behavior is inherited. Default/v1 behavior, definitions, IDs and data are preserved.

The new result is GoalConsistentObservedGraspTaskTeacherResult, with definition
pick_cube_geometry_observed_grasp_goal_consistent_v2 and episode identity
pick-cube-goal-consistent-observed-grasp-teacher-{seed}. It retains the existing
current-feedback tuple and complete paid-work/final-success qualification. Zero
XYZ only requests a hold; native success also requires arm-static feedback.
Current learning readers refuse this new policy until explicit admission is added.

Use a fresh output and separately frozen native allowance before running:

```powershell
python -m scripts.maniskill_task_teacher_smoke --geometry-readiness --observed-grasp --goal-consistent --train-seed 17 --output fresh-goal-consistent-teacher.json
```

[Rationale and boundaries](adr/ADR-0493-goal-consistent-observed-teacher.md).
The software passed122 checks per platform and independent acceptance. One
separately budgeted native case qualified at TRAIN17 in15controls/75knownphysics,
with final success. Its spec/transitions/stages exactly equal the accepted v1
demonstration, so it adds no new training coverage. Do not retrain duplicate data
because the version changed. Next extend existing student-arrival corrections
with the explicit v2 rule; keep v2 warm-source admission deferred.
[Native result and costs](../artifacts/runs/goal-consistent-observed-teacher-native-20261020/RESULTS.md).
This is one known-reset teacher, not a successful learned policy or reliability.

The existing teacher collector can opt into ObservedGraspPickCubeTeacher for
closed-loop TRAIN collection. This reuses its thresholds and commands; it adds no
learner, optimizer, seed search, retry, intervention or new runner.

```powershell
python -m scripts.maniskill_task_teacher_smoke --geometry-readiness --observed-grasp --train-seed 17 --output fresh-observed-teacher.json
```

Native execution requires the optional pinned CPU simulator stack and a separately
frozen local allowance. The output must be fresh. Default geometry-only v1 and
legacy CLI invocations retain their schemas and behavior. Selected mode requires
raw42 TRAIN17/19; RGB, guard labels and interventions refuse before collection.

```python
from src.infra.maniskill_task_teacher import collect_readiness_teacher

result = collect_readiness_teacher(train_seed=17, observed_grasp=True)
episode = result.training_episode()  # Refuses unless full qualification passes.
```

## Data and responsibilities

Selected collection returns the distinct frozen ObservedGraspTaskTeacherResult
with required keyword-only observed_grasp feedback, the existing selected teacher
definition and pick-cube-observed-grasp-teacher-{seed} episode identity. The base
TaskTeacherResult fields/asdict remain unchanged. Feedback is one bool per actual
returned native control and must equal its CURRENT raw42 official flag18. Current
and next flags must be exact float0/1 before copying can normalize values.

Returned paid teacher decisions are counted even if later validation rejects the
decision or control fails; feedback covers only returned controls. Original port,
teacher state, executed float32 commands, prior records, full episode and work stay
bound around callbacks. Failed-step work is saved before cleanup and cannot be
erased by a later counter regression. Errors and partial evidence remain available.

Qualification retains complete TRAIN binding, bounded commands, valid phase/action
semantics, exact work and final sparse success, with no cleanup/error/trace failure.
Selected family, definition, identity and feedback cannot be silently substituted.
Default geometry-only learning admission refuses this version, including feedback
markers on relabeled v1 data. Explicit selected admission validates original JSON
bool feedback and raw42 float flags before numeric normalization, then reuses the
full selected result qualification.

The infra module owns simulator IO and qualification; core teacher behavior is
unchanged. The existing CLI selects/binds the requested family and saves paid
payload even on mismatch. No final/probe/guard/promotion/deployed learner access.

## Verification

Final90 checks per platform, scoped statics and independent acceptance are in
[the software stage](../artifacts/runs/observed-grasp-teacher-20261020/RESULTS.md).
The separately frozen [native case](../artifacts/runs/observed-grasp-teacher-native-20261020/RESULTS.md)
qualified at TRAIN17:15 controls/75knownphysics, sparse1/finaltrue/terminated,
zero learner work. This supports the supervisory policy at that known reset;
useful neural control, retention, generalization, safety and CPC remain unproven.

## Explicit warm-source learning

The existing robot learning CLI accepts `--observed-grasp-warm-teacher` with
`--readiness-teachers --a-only-control` and nonvisual inputs. Only TRAIN17 opts
into the selected reader; original TRAIN19 remains a read-only reference. Optional
`--readiness-stage-residuals` retains each source's actual teacher definition.
The core stage binding keeps its existing two-field schema and owns the full
episode, validating selected flags before copying in raw42/history47/taskspace12.

This command spends learner and native work; use a fresh output and a frozen
local protocol before executing it:

```powershell
python -m scripts.run_continuous_robot_learning --output fresh-learning.json --warm-teacher qualified-observed-17.json --new-teacher original-readiness-19.json --readiness-teachers --observed-grasp-warm-teacher --a-only-control --learner-family bp --binary-gripper --phase-updates 64 --taskspace-history executed --readiness-stage-residuals
```

The explicit input path passed189 focused checks per platform with scoped
statics and independent actual acceptance. Its one tiny real-learner fixture used
3updates6Arows, B0 exposure/0fits and two fake controls.
[Executed software evidence](../artifacts/runs/observed-grasp-train-input-20261020/RESULTS.md).

The qualified15-row source changes the original192-update acquisition exposure
from3840 to2880 A rows. It is a changed supervisory dataset, not matched compute
or evidence of learned success. No new seed, threshold or optimizer search.

Extend this path through the existing collector/CLI, keep exact current-feedback
and paid-work semantics, and preserve old data/definitions. No new dependencies
or environment variables are introduced.

## Executed learner question and remaining scope

The fixed A15-only native question completed with producer-reported zero success
and truncation at50 controls. Full independent acceptance is incomplete: a stdout
byte-binding assumption failed; the separate semantic checker resolved that binding
but exceeded its200k traversal cap before native-row validation. Both attempts and
all originals remain. No native/model retry or third saved checker was launched.
[Exact observed scope and failures](../artifacts/runs/observed-grasp-train-acquisition-20261020/RESULTS.md).

The separate fixed phase128 (384-update/5760-A-exposure) question is independently
accepted as negative. The unchanged final policy grasped on41current rows but
never achieved sparse task success in50controls. Closest goal error26.787mm,
final68.352mm. Final A half-MSE.01562665 is not native acquisition. The initial
optional-field checker failure remains; a separately budgeted native-only check
completed the missing scope without retraining or rerunning the simulator.
[Protocol, results, costs and failures](../artifacts/runs/observed-warm-fit-depth-20261020/RESULTS.md).

The stopping-rule repair is now explicitly versioned, tested and qualified as
described above. Its known-reset demonstration is exactly unchanged, so repeating
the learning run on those rows would add no new coverage. The next repair is
explicit v2 supervision through existing student-arrival corrections, where it
can label transport/settle states reached by the learner. Preserve source/candidate
isolation, raw feedback, endpoint labels, receipts/clocks/costs and old v1 evidence.
V2 warm-source admission remains unfinished/deferred; the current reader refuses
it. No native learner improvement or trajectory causality is established yet.

B transfer is deferred until useful base control exists. The existing Python
run_learning_rollout API already supports A/B/A, so no extra runner/CLI flag is
needed. Its tiny integration fixture and adaptation question remain unfinished.
