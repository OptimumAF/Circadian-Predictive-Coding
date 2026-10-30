# Calibrated before-wake observations with a fixed diagnostic reference

`scripts.arrived_candidate_shift.ArrivedCandidateShift` has an optional
calibration binding. Omit its new arguments for the existing fixed-threshold
current-candidate observer and original report schema.

```python
from scripts.arrived_candidate_shift import ArrivedCandidateShift
from src.core.shift_calibration import ShiftCalibration

reader = ArrivedCandidateShift(
    predict_state,
    calibration=ShiftCalibration(),  # existing six-batch prefix/margin defaults
    calibration_model_id="declared-baseline-run",
    calibration_prefix_rows=tuple((f"stationary-{i}",) for i in range(6)),
)
```

All three optional declarations are required together. Declare ordered TRAIN
row-ID tuples before the first prediction; they are checked only when each
original label arrives. This is a declaration of stationary-prefix membership,
not a statistical stationarity test. It grants no new data/budget/source authority.
The existing8batch/48row ceiling remains, with room required for monitoring.
IDs and the caller model label have explicit UTF8 bounds. No probe, final or
task-readout data may become calibration input or select its prefix.

The first admitted candidate state is copied as one frozen diagnostic reference
plus a detached value pin. These two constant retained reference copies do not
grow into a state history. There are also live/transient snapshot and working
copies; no absolute model-byte bound follows from the copy count. Native callers
remain responsible for their original memory/copy/forward budgets.

Calibration and all later monitoring predictions use that fixed reference.
Current learner state can update separately; each row names its current native
counter and the fixed `prediction_native_updates` counter. The caller model label
is metadata, accompanied by the observed actor/candidate/wake/native binding;
it is not independent source authentication. Callback, owned configuration,
detector identity and reference replacements/corruption refuse without rebasing.
Changing the initial baseline or prefix requires a new explicit protocol; this
reader does not reset itself or renew prior budgets.

The existing core calibrator suppresses prefix alarms and fits maximum prefix
score plus fixed margin, preserving channel ceilings and unreachable thresholds.
It replays scalar prefix metrics, not model predictions. The report exposes
thresholds, prefix/replay rows, reachability and incident onset/continued/clear.
Source and working inputs remain protected; lost returns retain incurred costs.
A completed scalar fit is reported even if its return or later incident handling
fails. Such a stopped/incomplete observation is cost evidence, not accepted
feedback or a resumable calibration grant.

Use the existing delayed schedule's `read`/`require_current`/`require_paid` ports;
this increment does not expose calibration through the live CLI automatically.
The fixed-reference alarm describes change from a baseline, not current learner
error after learning. Combining it with current-candidate ACK or wake selection
needs explicit schedule semantics and a prospectively bounded protocol.

[Software checks, failures and limits](reports/arrived-calibrated-shift-20261029.md):
literal probability fixtures verify the contracts without a learned model. They
do not establish native usefulness, calibrated false-positive control, task
learning, CPC advantage or deployment readiness. Existing negative shift results,
the supervised-error baseline and all spent studies remain unchanged.
