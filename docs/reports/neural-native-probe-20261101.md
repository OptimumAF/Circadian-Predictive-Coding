# #411 — isolated neural adaptation: fresh native failure, retention incomplete

The fixed private 16-update SGD probe completed its optimization. Both reserved
PickCube resets failed before and after adaptation, despite lower imitation
error. All three old-task imitation errors increased. Native retention remains
unknown: the first old rollout stopped at a logical copy-accounting cap and the
other two did not run. This is FAILED/SPENT/PARTIAL, not full #411 acceptance.

| Readout | Before | After |
|---|---:|---:|
| Reserved probe_train23 normalized half-MSE | .04443458050903264 | .01134174458195636 |
| Reserved probe_test24 normalized half-MSE | .06436728616819498 | .019411707498635647 |
| Old17 normalized half-MSE | .0015257813261950286 | .032833591187138286 |
| Old19 normalized half-MSE | .002662961974564868 | .006914129266285016 |
| Old20 normalized half-MSE | .0009462260756697378 | .008692289342499016 |
| Native23 | Failure,50controls,return0 | Failure,50controls,return0 |
| Native24 | Failure,50controls,return0 | Failure,50controls,return0 |
| Native old17 | Accepted #410 success,22controls | Incomplete after4controls |
| Native old19 | Accepted #410 success,20controls | UNRUN |
| Native old20 | Accepted #410 success,15controls | UNRUN |

Before old-native results reuse the unchanged qualified #410 endpoint; they were
not rerun. MSE measures imitation, not native acquisition or retention. The
generic `fresh_acquisition_gain` field .044955578669559335 is imitation gain
only. There is no actual old-loss/return phase, recovery evidence, reliable
generalization, matched-compute optimizer superiority, CPC benefit or release
claim. Existing negative affine, neural, Push and reserved24 results remain.

## Delivered behavior and isolation

The existing `ContinuousProbePlan`/`run_continuous_probe` and
`ActorShadowRuntime(training_role="probe_train")` already supported private
updates and detached exact before/after readouts. No new runner or optimizer was
needed. The missing narrow native boundary admitted explicit fixed23 scoring
only for nearest-memory learners. `scripts/maniskill_continuous_smoke.py` now
also admits exact BP/PC/neutral-CPC/active-CPC with the same role, reset, TS12,
policy and exclusive-mode checks. Default/24/nearest branches retain their
contracts; the public full-network fitting API remains TRAIN-only.

The new neural23 route reuses the reserved independent control witness. A lost
native return or throwing/forged-zero work getter preserves paid attempted and
returned work instead of erasing it. Frozen state/history, original source
owners/counters and no native optimization remain checked. The literal software
cohort covers all four families; only BP was executed in this real experiment.

One RNG-free restoration of the accepted #410 full D12/H64/seed17 record retained
80updates/17600exposures, four parameter and velocity arrays, zero momentum and
the original mixed objective. The private child reached96/17888 via16SGD×18rows.
Only original23 labels trained it. Original24 remained inference-only; final
evaluation was unavailable. Old220TRAIN examples supplied old readouts only,
not replay. No deployed/source promotion, initializer, model RNG, new teacher
collection, seed/dose/metric search, baseline refit or role conversion occurred.

The qualified checkpoint seal was captured before restoration/public getters;
the first public AND direct native record had to equal it. Full original and
borrowed-owner guards ran throughout paid boundaries. Four completed fresh
native cases retained frozen state/history proofs and zero optimizer work.
Terminal source/input/pin qualification was not reached; do not replace that
missing end proof with a current source hash check. The private adapted model
was not exported and cannot resume after process exit.

## Commands, failures and observed costs

From physical `D:\Dev\Repositories\Circadian`, process-local TEMP/TMP/TMPDIR
were `D:\Dev\_Temp`. Software commands, each exactly once:

```powershell
.venv\Scripts\python.exe -I -B artifacts/runs/neural-native-probe-20261101/check.py initial
.venv\Scripts\python.exe -I -B artifacts/runs/neural-native-probe-20261101/check.py final
.\artifacts\runs\neural-native-probe-20261101\run-case.ps1
```

Initial14tests:12PASS/2FAIL,1.644487100s/85,528,576B. Two CPC literal fixtures
incorrectly paired zero wake history with80updates; corrected to0/0 without any
wake-update work. The initial hook missed the distinct PC feedforward path's
two returned calls/rows. Those known two are permanently charged in addition
to unchanged initial JSON; remaining caps were reduced, not renewed.

Final16tests:16PASS,1.438393900s/85,553,152B. Cumulative30executions,28PASS/2
fixtureFAIL,3.082881000s;30literal sources,16fake scenes,28controls,28predictions/
rows and28true forwards/rows,546public snapshots/331,968selected payload bytes.
Both software slots and the fake-scene cap are SPENT. Constructors, model RNG,
SGD, gradients and SDK work were zero in these literal tests.

Final scoped commands passed:

```powershell
.venv\Scripts\python.exe -m ruff check scripts/maniskill_continuous_smoke.py tests/test_neural_probe_training_native.py artifacts/runs/neural-native-probe-20261101/study.py artifacts/runs/neural-native-probe-20261101/check.py
.venv\Scripts\python.exe -m ruff format --check scripts/maniskill_continuous_smoke.py tests/test_neural_probe_training_native.py artifacts/runs/neural-native-probe-20261101/study.py artifacts/runs/neural-native-probe-20261101/check.py
.venv\Scripts\python.exe -m mypy --follow-imports=silent scripts/maniskill_continuous_smoke.py tests/test_neural_probe_training_native.py artifacts/runs/neural-native-probe-20261101/study.py
```

PowerShell host syntax passed. Initial type annotation errors and stale patch
anchors were corrected before final checks. The first static preservation
assertion excluded the wrong enclosing function and failed; the corrected check
shows only `_execute_frozen_rollout` changed and52other top-level definitions
AST-identical. An initially misreported65 was corrected to52 in the GitHub
receipt. Later handoff discovery guessed `initial.json`/`final.json` instead of
`initial-tests.json`/`final-tests.json`, and a PowerShell wildcard rg failed;
ordinary file enumeration resolved both, with no experiment rerun.

The real case ran cached LinuxPython3.12.3/NumPy1.26.4/ManiSkill3.0.1/Sapien3.0.3,
software Mesa, CPU numerical threads1/GPU0. Producer16.560353472s, sampled peak
534,278,144B; GNU16.94s/523624KiB. HostPID15704 exit1/19.6337128s/no host stop.
No install/download/network/global SDK change occurred.

The selected cumulative copy counter reached2,147,500,992B:2GiB+17,472B, one
returned full-state payload above the frozen cap. This is logical copy traffic,
not retained memory/RSS. A prospective pre-input increase from1GiB/65536validator
returns to2GiB/131072 was justified by literal payloads/nested guards, but that
forecast was still insufficient. Both forecast and actual failure remain.

Paid work:16SGD+16gradient returns/288rows;231forwards/725rows;215predictions/
437rows;10residuals/232rows;8forks/2readout restores;13,298public snapshots;
109,613validator returns. Nine input reads645,081B:checkpoint1,bundle2,23three,
24three. The final checkpoint/bundle reads did not run. Five scenes closed;
204returned native controls/1020known control-physics steps. The205th outer
attempt was charged before a pre-step guard refused; it does not establish
another physical dispatch. The partial old17 port reports4/4controls/20known
physics and zero possible failed native controls. Constructor/reset physics
and unmeasured copies/work remain additional unknown costs.

Inner update budget wall/RSS fields were deliberately unset; outer gates were
separate. Do not claim an inner limit existed. No timeout or physical RSS cap
was hit. Full result complete/budget_ok false; permanent numerical/IO stop.

## Review, artifacts and unfinished work

Lead reviewed the final combined source/procedure/results. The finite peer
review found and verified fixes for work-getter erasure and an unqualified
first-snapshot baseline. PRE GO preceded the actual case. POST confirmed the
new443351B result SHA256
`1ce05cb1f108da87cb88ea2fe851ac6927d0cc43336e94006bac35d33ab2f821`, all20current
source pins, paid counters and host costs; accepted partial observations only.
Its displayed projection truncated middle native-case fields, so those case
summaries rely partly on the lead's cached receipts, not complete independent
verification. All four peer batches SPENT/exception ENDED; no replication.

[Frozen protocol and all artifacts](../../artifacts/runs/neural-native-probe-20261101/PROTOCOL.md),
[software protocol](../../artifacts/runs/neural-native-probe-20261101/SOFTWARE-PROTOCOL.md),
[POST review](../../artifacts/runs/neural-native-probe-20261101/POST-REVIEW.md).
Original raw/protocols/pins/markers remain immutable. Lead and peer raw-analysis
read slots are spent; reuse `LEAD-SCALARS.json`. No automatic retry or private
state export. Retention/full conformance criterion remains unchecked on
[#411](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/411).

Skipped: old unchanged probe/math/native suites, extra models, diagnostics,
native repeat, new seeds, Linux literal rerun, broad suite, clean clone/package/
release, GPU, remote CI and all final data. No commit/push/delete/global changes;
unrelated dirty master d9119e8 work preserved. Broader R2.1/R5/science/release
parents and full goal remain OPEN/ACTIVE. Accepted repeated-update three-update
runtime is reused, not reopened.

Roadmap decision: the useful next outcome is a role-bound disposable fitting
control, not another capture/copy audit or another SGD dose. Existing pure GGN
and TRAIN-only owned publication suffice mathematically; the existing probe
wrapper only delegates SGD. Why this: distinguish weak fixed-dose fitting from
closed-loop adaptation limitations without selecting seeds or claiming a win.
[#412](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/412) scopes
the explicit private optimizer path using the existing runtime and unchanged
objective. First implement that isolated path with prospectively bounded
literal checks, preserving default SGD and public TRAIN-only authority. No new
input/model/native experiment grant follows from this handoff. Actual old-native
retention and context-return recovery remain required and unproved.
