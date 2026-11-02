# TRAIN command fidelity

`run_training_readout(..., command_fidelity=True)` optionally describes the
controller commands from the same predictions used by its existing TRAIN
residual readout. It adds no prediction, forward, gradient, fit, optimizer update
or simulator call. Existing conditioning still pays its original two forwards
and one backward per group; residuals pay their original single prediction.
The default call shape, report fields and primary objectives remain unchanged.

```python
from scripts.push_cube_training_readouts import run_training_readout

# learner and groups must already have their original TRAIN authority.
report = run_training_readout(
    learner, groups, label="declared-training-diagnosis",
    guard=original_owner_guard, command_fidelity=True,
)
if not report["complete"]:
    print(report["error"], report["work"])  # retain paid work; do not retry implicitly
else:
    print(report["groups"]["push"]["command_fidelity"])
```

Keep the existing `corrective`, `aggregated` and `complete_b22` flags when those
groups are present. The four/five/six/seven-group row limits and64KiB phase-result
bound still apply. No eval/probe/final role is accepted by the TRAIN readout.
This API does not collect, qualify files, release labels or grant training work.

The pure `src.core.continuous_command_fidelity.continuous_command_fidelity`
helper accepts exact finite float64 action/target matrices, normalized root
translation/gripper units and fixed(-1,+1) bounds, with1..256rows and bounded stage
labels. It converts each value to IEEE float32 and back, matching the existing
native action port. Numerical signed zeros compare equal. It reports:

- Original row indices whose four native command values differ; per-axis mean
  and maximum absolute error, with the first maximum's original row index.
- Original translation rows with different negative/zero/positive directions
  after rounding. Zero differs from both positive and negative commands.
- For explicit `binary_gripper_axis=3`, endpoint target sign times the analog
  native gripper command, retaining margins and positive/zero/negative rows.
  Targets must be exact±1. With axisNone this endpoint description is omitted.
- Stage labels and original row indices, plus per-stage mismatch counts.

The gripper remains a continuous command. Its binary-axis option changes the
training objective; neither the learner nor this diagnostic thresholds its
native predictions. Margin sign is descriptive endpoint alignment, not policy
correctness or evidence that a grasp/task succeeds. Exact command agreement on
stored inputs also does not prove closed-loop task success.

`CommandFidelityWork` counters accumulate native scalar conversion attempts/
returns, equality/error/direction work, optional margin products/classifications
and known owned numeric action-array payload. A failed conversion retains its
attempt and preceding returns. Counters are cumulative across group calls:
do not sum each group's cumulative `work` snapshot. Final parent
`work.command_fidelity` is the aggregate. The prediction/target seal copies have
their separate `prediction_capture_numeric_bytes`; other validation, snapshots,
temporary arrays, Python/JSON and process memory are additional.

The adapter retains returned residual payload and original prediction charges
before command diagnosis. Failed diagnostics preserve the paid prefix and refuse
more model work. Source/detached model state, identities, TRAIN inputs, prior
payloads, captured arrays and work views are guarded. Borrowed predictions do
not escape the readout and no learner is returned or promoted.

Existing normalized half-MSE and native sparse-success outcomes remain primary.
These TRAIN descriptions do not redefine old metrics, rank methods, explain
causality or justify tuning a reserved reset. The all-negative neural/affine
native results and negative reserved24 transfer remain unchanged. No actual
saved-model/data diagnosis was run by that software increment. The later
[fixed-checkpoint diagnosis](reports/train-command-diagnosis-20261101.md) computed
all seven TRAIN groups, finding remaining direction disagreements and9negative
analog gripper margins, but FAILED its outer phase serialization/postcheck gates.
Its captured observations are accepted with those qualifications; the full case
remains partial and inference allowance SPENT. A metadata-only JSON bridge was
tested without rerunning the model. Future actual work needs a separately frozen
input/model/read/computation protocol, preserving all historical allowances.

[Scoped checks, commands, costs and review](reports/train-command-fidelity-20261101.md).
To extend safely, keep new descriptions pure and default-off; reuse the same
returned TRAIN predictions rather than running another forward or importing
evaluation labels into training diagnostics.
