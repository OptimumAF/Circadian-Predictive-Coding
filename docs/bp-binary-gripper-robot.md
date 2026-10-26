# BP binary-gripper likelihood in robot learning

The robot learning API and CLI can explicitly select the existing BP mixed
objective on normalized gripper axis3. Translation axes retain their existing
continuous loss; the gripper uses logit cross-entropy. Predictions and native
serving remain soft bounded tanh actions: this option adds no thresholded action,
controller, phase input, new learner or observation representation.

Why this: conventional taskspace12 acquisition still failed after correction and
reference-replay trials. Saved source TRAIN maximum normalized errors included
vertical.7188 and gripper1.1803. The existing mixed objective retains a gradient
for a wrong gripper sign. Exposing it supports a declared conventional baseline
development question; it does not promise to fix translation drift or favor CPC.
All earlier MSE/negative results remain unchanged.

## Run

```powershell
./.venv/Scripts/python.exe -m scripts.run_continuous_robot_learning `
  --warm-teacher original-qualified-17.json --new-teacher original-qualified-19.json `
  --readiness-teachers --taskspace-history executed --readiness-stage-residuals `
  --learner-family bp --a-only-control --phase-updates 64 --binary-gripper `
  --output new-exclusive-result.json
```

This command spends training and simulator work; use only under a separately
frozen/reviewed local experiment budget. The API equivalent is
`run_learning_rollout(..., binary_gripper_axis=3)`. Omit the argument/flag for the
original MSE mode. The only accepted selected axis is exact native integer3;
bools/floats/other axes and PC/CPC selection fail before models/input-file reads
at the CLI. Both TRAIN input episodes require endpoint−1/+1 gripper targets before
fitting/model construction, including B when retained as read-only diagnostics.
No final/probe data are admitted and no model is promoted.

The selected candidate must carry the requested policy axis and mixed diagnostic
identity in every A-only receipt. Missing/unrequested/substituted head or objective
refuses native serving while retaining paid training reports. Existing phase
updates, source identity, clocks, charges, fixed physical/history preprocessing
and frozen full-state guards remain in use. Selected private correction is
explicit separately, as below.

## Costs and interpretation

Use the existing work report, including binary_gripper_loss_rows. Dense matrix/
parameter-gradient counts are unchanged for equal shape/exposure; logaddexp and
other mixed-likelihood elementwise operations are additional work. Do not compare
mixed training losses numerically to MSE losses. Shared normalized residual metrics
can describe fit, but neither fit nor correctness establishes closed-loop success,
retention, generalization, CPC advantage or safety. Native qualification is separate.

The robot entry point, existing correction adapter and smoke runner compose the
existing core/stream math. New tests cover robot admission and corrections in
tests/test_bp_binary_gripper_{robot_learning,corrections}.py.
[Protocol and focused commands](../artifacts/runs/bp-binary-gripper-rollout-20261020/PROTOCOL.md).
No new dependencies, environment variables or project layout. To extend, preserve
explicit requested objective/policy/receipt binding and the family-specific native
likelihoods; do not turn mixed losses into cross-family matched metrics.

## Executed evidence

[Software79 checks per platform, statics and independent acceptance](../artifacts/runs/bp-binary-gripper-rollout-20261020/RESULTS.md).
One frozen native mixed-objective run attained grasp after control9 and lifted
the cube, but overshot the goal:50 controls, zero task success. Native behavior and
model work were independently accepted; full protocol conformance remains open
because an instrumented scalar-readback counter did not account for recursive
comparison work. No retry or retrospective waiver. [Qualified result and costs](../artifacts/runs/bp-binary-gripper-acquisition-20261020/RESULTS.md).
No native mixed-correction recovery trial is implied or proven useful by this
evidence. Implementation correctness and observed recovery remain separate.

## Private arrived corrections

```python
report = run_corrective_adaptation(
    mixed_bp_source,
    output_directory=new_exclusive_directory,
    visual_observation=False,
    action_history_recipe=executed_taskspace_recipe,
    readiness_corrections=True,
    acquisition_only=True,
    binary_gripper_axis=3,
)
```

The lower-level run_continuous_correction_update accepts the same explicit axis.
Only exact BP/axis3 is admitted, bound to its actual owned policy. Omitting the
argument preserves the original MSE contract and refuses a mixed policy. Actual
arrived correction labels require endpoint gripper supervision before private
forks. Student executed commands remain soft and remain the history authority.
Optional replay_plan/reference_episodes uses the existing complete TRAIN model-
space bank, endpoint supervision and separate original exposure budget. The bank
does not reach the native collection/return callbacks. Eight correction updates
and extra replay retain mixed diagnostic receipts, clocks, permanent charges,
full source/momentum guards, and binary_gripper_loss_rows including replay.
The deployed source is unchanged; frozen return serves the detached candidate.

Why this: the mixed source reached grasp/lift but missed the goal. This composition
makes existing readiness supervision usable at those observed failure states,
without changing the learner, controller, representation or default baselines.
Software evidence is recorded in the
[correction stage](../artifacts/runs/bp-binary-gripper-corrections-20261020/RESULTS.md).
No new environment settings/dependencies. Extend using existing runner options and
preserve explicit policy/receipt binding; separately freeze any native trial.

## Arrived correction result

[Selected correction software169 checks/platform accepted](../artifacts/runs/bp-binary-gripper-corrections-20261020/RESULTS.md).
A separate one-use native recovery question completed negatively: original mixed
student grasped, but all50 provisional targets were ALIGN; eight private updates
400rows yielded a pure return with0success/no grasp. Original TRAIN A and B common
half-MSE both worsened. [Accepted result and full costs](../artifacts/runs/bp-binary-gripper-readiness-recovery-20261020/RESULTS.md).
The next dependency is explicit observed-grasp feedback in readiness supervision,
with versioned semantics, before more training. Existing v1 records, thresholds,
all negative results and unresolved earlier readback qualification remain intact.
