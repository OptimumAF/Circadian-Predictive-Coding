# Generic owned minimum-change updates, 2026-10-29

R2.1-CTX.LS.M, [#307](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/307).
Implemented and software-qualified. This is not native task-retention, CPC,
generalization, deployment or release acceptance. Full goal/#21/#18 remain open;
all earlier minimum-norm results, including #305/#306 return losses, stay frozen.

## Delivered behavior and boundaries

```text
src/core/categorical_linear_fit.py       shared residual kernel + typed generic update
src/core/wide_categorical_linear_fit.py  existing warm API delegates to that kernel
src/adapters/linear_fit_learner.py        explicit atomic minimum-change operation
tests/test_linear_minimum_change_owner.py analytic/ownership/stream behavior checks
```

`LinearFitLearner.train_batch_minimum_change` explicitly uses prior coefficients
in the existing wide residual formula `W + pinv(Z) @ (Y - ZW)`. Its first unfitted
call uses the same cold solver. Default `train_batch` is ALWAYS the original cold
minimum-norm fit; the new operation is nonsticky. No constructor/policy/snapshot/
serialization mode or stored row/history chain was added. Caller chooses the
operation every time and retains role/label/budget/lease authority.
[Usage, objective and limitations](../linear-minimum-change-owner.md).

Shared arithmetic preserves prior components invisible to the current design,
and an already-solved batch can leave coefficients stable. It does not otherwise
protect old predictions or solve conflicting old/current labels. Tests expose
that overlap limitation. Rank/singular values/residual/fit_rows describe the latest
batch only; successful lifetime update/example counts remain cumulative.

`update_categorical_linear` requires an exact valid fitted generic prior and
current1..128rows/1..296features. A generic prior with128last-fit rows is accepted
without converting it into the wide record. Existing wide1..16rows/1..4096features,
fixed intercept/RCOND1e-12/one-hot targets/equal row weights and CLM role/encoding
interfaces remain unchanged. Cold `_solve_design` and cold-fit arithmetic unchanged.
This reuses existing #162 minimum-change arithmetic, not another algorithm/runner.

The atomic owner stages a FULL detached prior separately from the unchanged
comparison pin, plus detached feature/target arrays. Source/policy/state owner
replacement, source value changes, staged-prior/input changes and RNG changes
refuse publication. A simultaneous staged-prior/source mutation cannot alter the
before pin. Foreign mutations are preserved rather than silently restored.
Only after all checks does one model/counter replacement publish.

New diagnostic definition:
`minimum_change_one_hot_least_squares_sum_squared_residual_v1`. Value is current-
packet residual squared sum, not CE/lifetime loss/retention. Lost or malformed
returns use the existing TrainingUpdateFailed scalar attempted/returned work,
unknown flag and zero candidate commits under this new intent. Default diagnostic
and failure behavior unchanged. Caller budget successful-update counts are not
rewritten as attempts; failed existing streams stop and prohibit more work.

## Executed evidence and limits

Both platforms passed71focused cases:32new cases,27existing generic-owner cases,
10existing wide-update cases and2cold math oracles. Independent rank-one expected
coefficients verify invisible-direction preservation; solved-batch coefficients
stay unchanged; conflicting labels change old predictions as explicitly allowed.
Actual generic128-row packet/prior and296feature update pass; wide17-row input
still refuses before solving. New malformed prior/backend/geometry/overflow and
12mutation/lost-return modes refuse safely with preserved accounting.

Unfitted minimum-change and ordinary fits produce identical complete snapshots;
fork/source/readout ownership holds. Calling ordinary train_batch after warm
updates produces the same model as the cold control: no hidden update mode.
The cold old-point scores are near a floating tie; tests check the cold model/
scores instead of asserting a platform-dependent rounding sign as task success.

Test-only explicit port/fork composition exercises EXISTING AttributionStream:
three delayed TRAIN updates oldâ†’newâ†’old return on orthogonal augmented constraints,
both synthetic constraints remain predicted correctly, source/actor unchanged,
label causality/counters intact. Non-TRAIN roles refuse before solver. A lost warm
return preserves attempted work, leaves candidate/source/actor unchanged, stops
the stream and refuses another attempt. This is a small synthetic behavior fixture,
not a native six-task result. Contextual routing/probe/CLI integration is UNFINISHED.

Physical repository `D:\Dev\Repositories\Circadian`, HEAD0aae1d6 plus inherited
dirty/untracked work. Windows existing Python3.14.7; Linux existing3.12.3/NumPy1.26.4.
Every project shell used TEMP/TMP `D:\Dev\_Temp`; Linux also TMPDIR `/mnt/d/Dev/_Temp`.

```powershell
$tests = @('tests/test_linear_minimum_change_owner.py','tests/test_linear_fit_learner.py','tests/test_wide_categorical_linear_update.py','tests/test_categorical_linear_fit.py::test_should_match_closed_form_intercept_slopes_and_unconditional_residual','tests/test_categorical_linear_fit.py::test_should_use_one_fixed_rcond_solve_and_recompute_even_when_solver_residual_is_empty')
.venv/Scripts/python.exe -B artifacts/runs/linear-minimum-change-owner-20261029/check.py artifacts/runs/linear-minimum-change-owner-20261029/windows-final-work.json @tests
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/linear-minimum-change-owner-20261029/check.py artifacts/runs/linear-minimum-change-owner-20261029/linux-final-work.json @tests --capture=sys
```

Each71PASS0fail/skip/terminal0. Windows2.215556200s81,215,488B sampled peak;
Linux17.872556889s76,292,096B. Each48core solve attempts215rows,46actual NumPy
lstsq invocations213rows. Two mocked nonfinite backend returns per cohort cause
the difference; mocked counters are not physical matrix work. Each8probability
calls19rows/4affine calls7rows. Named wrappers bind canonical/imported aliases
and charge before delegation with shared locks; not exhaustive scalar-operation
or process-tree accounting. Analytic oracle/copy/diagnostic work included in
invocation time/RSS. Fault-only RNG seed mutation is restored by fixtures; no
model initialization randomness, neural/optimizer/latent/GPU/Torch or native case.

Preflight command used the same check.py/windows-pure-work.json with new test file
and `-k 'invalid_admission or nontrain_stream'`:8PASS,0solver/forward work,
1.139397600s66,240,512B. All tests across both platforms:96core attempts430rows,
92NumPy invocations426rows,21.227510689s summed/max81,215,488B. Within original
shared192attempts1536rows, <=3pytest calls/platform120s512MiB; actually2Windows/
1Linux calls. No numerical test failure, budget amendment, native/cohort allowance
or #287FAILED/SPENT retry. Prior source fixtures use literal metadata where stated,
not unaccounted source training or a native success claim.

```powershell
$checked = @('src/core/categorical_linear_fit.py','src/core/wide_categorical_linear_fit.py','src/adapters/linear_fit_learner.py','tests/test_linear_minimum_change_owner.py')
.venv/Scripts/python.exe -B -m ruff check @checked
.venv/Scripts/python.exe -B -m ruff format --check @checked
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent @checked
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent --platform linux @checked
```

Final statics2.json all four terminal0/6.882359300s, staticRSS uninstrumented.
statics.json retains both-target mypy failures for intentionally invalid test
prior argument typed object; marked that test input Any, not weakened production
types/admission. Preliminary checks before adding that test had passed. No source
change after final combined checks. One documentation patch missed an inline-link
context; a guarded direct append repaired the docs, with no production/test edit
or numerical repeat.

Lead self-reviewed source/math/default compatibility/prior pin/failure/stream
procedure and actual receipts. Finite independent source/math/state/procedure
and final result/static review accepted:four readonly calls/about0.671s command
wall, no project imports/tests/models/network/writes; exception ended. Not
independent experimental replication or native usefulness acceptance.

Artifacts `artifacts/runs/linear-minimum-change-owner-20261029/`:6files10,996B at
closeout, below4MiB. Exact arguments/counts/outcomes and failed static receipt
retained. Local/unpublished. Skipped broad unrelated math/model-family/CLM-Torch/
raw-backed/native/simulator/GPU/remoteCI/clean-clone/current-wheel/release checks;
unchanged #162 role and earlier native assumptions reused where applicable.
#279 wheel predates current code. No installs/downloads/deletions/global changes/
commit/push, unrelated dirty/ignored/frozen/failed data preserved. #285 negatives,
#305/#306 losses and #287 FAILED/SPENT remain unchanged.

## Precise next implementation action

Add the corresponding explicitly named minimum-change port to the existing
ContextLinearFitLearner, reusing its current atomic routing/whole-state checks and
this generic owner operation. Preserve inactive-head/source/actor ownership,
cumulative native counters and all default cold paths; test cross-head mutation,
failure and explicit old/new/return behavior under a prospective software budget.
Do not invent a second numerical implementation or snapshot mode. Contextual
probe/CLI composition and any native retention comparison follow only after those
gates, separately frozen; no experiment allowance or promised remedy here.
