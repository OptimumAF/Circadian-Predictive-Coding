# #407 / R2.1-CTRL.ALIGN — TRAIN command-fidelity software accepted

The existing TRAIN readout now optionally describes native32 action errors,
translation directions and analog gripper endpoint alignment using its original
residual predictions. It adds **zero model passes**. Default reports, primary
normalized half-MSE, native success definitions and policy behavior stay intact.
This is verified software functionality, not an actual diagnosis of the saved
neural checkpoint or evidence of improved task learning.

The full goal remains ACTIVE. #399's neural220 lower-TRAIN-error/four-native-failure
result and inner-budget deviation remain; #398 affine negatives, #403 negative24,
#405 known-training native23 and #406 CPU workflow evidence are reused without
reopening their spent raw/model/data/native allowances. No new baseline training,
physics, final/probe labels, seed selection or threshold tuning occurred.

## Source and behavior

New `src/core/continuous_command_fidelity.py` is a pure bounded action description.
It requires exact finite float64 matrices with1..256rows/four normalized root/
gripper axes, original stage labels and declared(-1,+1) bounds. Native conversion
matches IEEE float32 commands. Signed zeros compare equal numerically; translation
directions are negative/zero/positive after rounding. It retains original row/
stage indices, mismatches and per-axis native error maxima/means.

For explicit binary_gripper_axis3 it requires endpoint targets±1 and reports
target-sign×ANALOG gripper command margins. Neither policy nor simulator thresholds
those predictions. Margin alignment is not discrete policy correctness or task
success. With axisNone, endpoint gripper summary is absent. No existing metric is
redefined and no inference about causes, retention, generalization or CPC follows.

`src/adapters/continuous_learnability_modes.py` adds an explicit prediction observer
for a residual readout. The default BP call still delegates unchanged to the
existing learner method; default PC logic uses the same existing kernels. The
default-off `command_fidelity=True` branch in `scripts/push_cube_training_readouts.py`
retains already-returned TRAIN actions briefly, seals original residual payload/
charges, then invokes the pure helper. It returns no arrays or learner and grants
no role/label/promotion/clock/budget authority. Existing four/five/six/seven-group
admission and64KiB phase bound remain. Explicit episodeID/context/TRAIN provenance
is attached to command records.

Existing conditioning still pays2forwards1backward/group and residuals1prediction;
those are counted. The new description adds conversions/comparisons and copies,
not neural arithmetic. Main parent command counters are cumulative across groups;
do not sum each group's cumulative work snapshot. Prediction seal copies are
separate; state snapshots, validation, Python/JSON/temporary/RSS costs additional.

## Checks and observed costs

Physical D:\Dev\Repositories\Circadian, HEADd9119e8/inherited dirty work preserved.
All TEMP/TMP/TMPDIR/cache output D:, CPU threads1. No dependencies installed.
Frozen definitions/caps: `artifacts/runs/train-command-fidelity-20261101/PROTOCOL.md`.
Two before-source copies preserve exact prior dirty modes/readout code.

```powershell
.venv\Scripts\python.exe -I -B artifacts/runs/train-command-fidelity-20261101/check.py initial --tb=short
.venv\Scripts\python.exe -I -B artifacts/runs/train-command-fidelity-20261101/check.py corrective --tb=short -k 'mutated_accumulator or late_command_failure or default_and_optin'
.venv\Scripts\python.exe -I -B artifacts/runs/train-command-fidelity-20261101/check.py final --tb=short
```

Initial27PASS/1.4172114s/72450048B; corrective9PASS22deselected/1.0838062s/70836224B;
final combined32PASS/1.35737s/72400896B. **All68executionsPASS**, total3.8583876s/
max72450048B sampled RSS. All3new command slots SPENT. No numerical test failures.
Initial three typing errors (optional witness and malformed axis test inference)
fixed with explicit witness refusal/Any fixture annotation; no ignores. One patch
anchor failed verification before mutation, corrected using exact current lines.

Cumulative actual software work:32literal BP source objects (explicit parameters,
zero initializer/RNG/fit/SGD),192forwards1554rows,64gradients518rows,64predictions/
518rows,79command attempts1047attempted rows4094native conversion attempts,
216public snapshots131328known numeric snapshot payload bytes; other copies
additional. SVD0, recorded-data/SDK/teacher/native/final work0. Failed/refused
commands and partial conversions count. Source-object ceiling32 is SPENT; no
rerun/fixture allowance remains. Original draft row ceilings were refined BEFORE
any tests specifically to cover the necessary generated220-row H2 payload case:
three executions alone need1980forward-row/660gradient-row/660prediction-row
exposures. Final prospective4096/1024/1024 row bounds held; no scientific/user
limit or actual study quota changed.

Meaningful checks include exact/native32-rounded equality/signed zeros, tiny
residual with opposite translation, zero/negative analog endpoint margins,
noncontiguous stage provenance, dtype/finite/bounds/axis/endpoint refusal, partial
conversion charges, same default/opt-in residuals/conditioning/model-work counts,
TRAIN-only roles, late failures and corrupted/replaced captured actions/work/
payload/root groups/operations. A genuine generated220-row H2 seven-group readout
completed at **56806bytes < unchanged65536** phase bound. This is not an H64
diagnosis of real demonstrations or a qualification of all possible payloads.

Final four-file checks PASS:

```powershell
.venv\Scripts\python.exe -m ruff check --no-cache src/core/continuous_command_fidelity.py src/adapters/continuous_learnability_modes.py scripts/push_cube_training_readouts.py tests/test_train_command_fidelity.py
.venv\Scripts\python.exe -m ruff format --check src/core/continuous_command_fidelity.py src/adapters/continuous_learnability_modes.py scripts/push_cube_training_readouts.py tests/test_train_command_fidelity.py
.venv\Scripts\python.exe -m mypy --no-incremental --follow-imports=silent --cache-dir D:\Dev\_Temp\circadian-407-mypy src/core/continuous_command_fidelity.py src/adapters/continuous_learnability_modes.py scripts/push_cube_training_readouts.py tests/test_train_command_fidelity.py
```

Scoped git diff/new-file whitespace checks PASS. No full suite, new Linux
numerical fixtures, clean clone/current wheel/install, SDK/browser/device/GPU,
remoteCI/native controls, saved-model/data/scientific comparison or release test
was executed: this is a bounded software diagnostic increment. Unchanged baseline
math and historical controls were reused, not rerun. No commit/push/reset/delete/
global configuration/unrelated dirty edit.

## Review findings and acceptance

Lead self-reviewed the combined actual source, procedure and results. One finite
independent peer reviewed math/default paths and state/data/paid-work boundaries
in four read-only batches, with no imports/tests/data/models/native/replication.
The final batch0.4s inspected actual corrective source/final32PASS/usage and
accepted scoped SOFTWARE. Independent review is not independent test execution.
Exception ended; see stage/REVIEW.md for precise scope.

Review found that a mutable command accumulator could forget earlier groups.
The caller now validates exact successful native-int row deltas and returned
cumulative work, retains birth command/ACK before validation, and requires the
current accumulator equal private paid totals at every boundary AFTER callbacks.
Both immediate reset and late reset/bool-zero replacement refuse before more
model work. Failure restores trusted paid payloads/costs. Selfreview fixed missing
command-operation returned status and root group/operation replacement fallback;
all focused cases pass. No acceptance criterion was weakened.

## Critical path and one next action

New diagnostic code is ready; no saved-model or dataset command diagnosis was
performed. The next useful outcome is #408/R2.1-CTRL.ALIGN.E: one inference-only
TRAIN command diagnosis of the already-fixed neural220 checkpoint, not another
fit/dose/native baseline. Source inspection located the existing typed `restore`
function and54461B peer-qualified FITTED-STATE export; neither export nor dataset
was opened here. Any new input use must be explicitly prospectively bounded and
reviewed, preserve old spent allowances/failures, and justify the changed
diagnostic question. No model/read authority arises from issue creation.

Exact next: inspect only that existing restore function/export-envelope source,
then freeze the one input-use/inference protocol and finite procedure fixtures
before opening inputs. Reuse the original220TRAIN bundle and at most7prediction
forwards/220rows, zero fitting/conditioning/backward/RNG/native/teacher/evaluation,
matching the cached producer where needed. Keep original MSE/native success
primary. A negative/descriptive finding is valid; choose any subsequent learner
repair from TRAIN evidence, never tune reset24 or assert a causal/CPC advantage.
Full acquisition/retention/recovery/reliability/generalization/science/release
parents and repository goal remain OPEN/ACTIVE.
