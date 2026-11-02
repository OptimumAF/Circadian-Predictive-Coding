# #413 — stronger private fitting still fails fresh control and loses old tasks

The fixed16-update private full-network experiment completed all measurements.
Fresh23training imitation error fell to .0000296891, but both fresh native resets
still failed. All three previously successful old native tasks also failed after
adaptation. Independent review accepted the completed negative measurement.
There is no native acquisition, simultaneous retention, recovery or CPC advantage.

| Native reset | Reused qualified80 endpoint | After16private GGN updates |
|---|---|---|
| Reserved probe_train23 | Failure,50controls,return0 | Failure,50controls,return0 |
| Reserved probe_test24 | Failure,50controls,return0 | Failure,50controls,return0 |
| Old TRAIN17 | Success,22controls,return1 | Failure,50controls,return0 |
| Old TRAIN19 | Success,20controls,return1 | Failure,50controls,return0 |
| Old TRAIN20 | Success,15controls,return1 | Failure,50controls,return0 |

Only the five after-cases ran anew. Before fresh evidence reuses completed
#411before observations with original frozen proofs; before old evidence reuses
#410. Source record and inference/reset/observation/action assumptions match.
The #411overall failure, missing old-native retention and terminal gates remain
unresolved. This completed experiment does not retroactively repair them.

| Normalized half-MSE | Before | After |
|---|---:|---:|
| Training23 | .04443458050903264 | .00002968911380734017 |
| Initial training23 row | .004263776804704623 | .000000011779137497928179 |
| Readout24 | .06436728616819498 | .04614745444803071 |
| Old17 | .0015257813261950286 | .029124389236546408 |
| Old19 | .002662961974564868 | .07832418965873827 |
| Old20 | .0009462260756697378 | .06918157489832814 |

The original mixed training objective fell .047391164874249186→
.000051607548215780025, with16strict-descent proposals. Low stored-trajectory
error was insufficient for successful closed-loop behavior in this case.
This does not identify a causal failure mechanism or prove that coverage is
the cause. The generic `fresh_acquisition_gain` .018219831720164274 is imitation
gain only. Existing16SGD fresh native negatives remain valid; that run's24MSE
after .019411707498635647 is lower than this run's despite weaker training fit.
No method wins across all readouts. Equal16updates are not equal computation.

## Executed isolation and work

Same qualified #410D12/H64/seed17/TRAIN/binaryaxis3/no-normalization source,
exact80updates/17600exposures, four parameter/momentum arrays and zero initial
velocity. A new checkpoint-input purpose restored the complete qualified state
without an initializer, model RNG, warmup fit or old raw-analysis renewal.
The private candidate reached96/17888 via16×18original23rows. No old TRAIN
replay, final/test training, new teacher, role conversion, source/deployed
publication, private-weight export, budget renewal or seed/dose selection.

Existing #412plan/runtime and unchanged owned optimizer/kernel supply the case.
The producer forbids public full/head TRAIN APIs and ordinarySGD; only the
explicit reserved binding reaches the shared inner update. Original objective,
damping.001/rcond1e-12/normcap1 and one trial per update stay fixed; all16accepted
updates reset momentum. Both synchronous before/after observer phases completed.
All five native cases had frozen state/original-owner/history proofs, zero
native optimizer updates,50returned predictions and no possible failed controls.
The actor/source remained unchanged and no private learner escaped the callback.

Paid work:16full updates/288rows;32gradient calls/576rows;16proposals and dense
1092square solves/17472aggregate solver rows;308forwards/1346rows,260predictions/
482rows,10residuals/232rows. Selected fitting arithmetic1,375,340,544dense MACs
and163,286,016numeric payload bytes. Solver arithmetic/workspace, elementwise
work, additional copies and Python/SDK/reset work remain unknown extras.

One RNG-free restoration,8forks,2readout restores,16,109public snapshots,
132,481validator returns and2,596,164,480selected logical copy bytes. The
prospectively frozen4GiB/262144validator ledger used #411's measured2.1475GB
at204controls to budget250controls (~2.63GB extrapolation plus fit). It held;
the original failed2GiBcase/forecast/costs remain immutable. This is copy
traffic, not retained RAM. No counter changes were ignored or budgets renewed.

Eleven bounded input reads1,062,784B:checkpoint2/bundle3/23three/24three. Original
triple reserved reader/receipts and frozen role/episode owners stayed intact;
final checkpoint/bundle byte equality, full source/data/paid-prefix proofs and
all24pins beforeinputs/beforeprobe/after passed. Five scenes closed;250controls/
1250known control-physics steps, constructor/reset physics additionally unknown.

## Commands and qualification

From physical `D:\Dev\Repositories\Circadian`, TEMP/TMP/TMPDIR point to
`D:\Dev\_Temp`. One new stdlib-only source qualification, then one actual case:

```powershell
.venv\Scripts\python.exe -I -B artifacts/runs/private-full-fit-native-20261101/qualify.py
.\artifacts\runs\private-full-fit-native-20261101\run-case.ps1
```

Qualification10checks PASS/.013626600s/23,355,392B, zero project imports/model/
input/numerical/native work; original seven helpers+Meter AST-identical and
fixed caps/plan/after-order verified. Before-native reuse proof matched the
#412before-BP file to the #411source pin; predict/snapshot/restore/fork AST and
forward/native port/smoke code remain unchanged. Accepted #41221literal checks,
#409math and native boundary evidence were reused, not rerun.

Scoped statics and PowerShell host syntax passed:

```powershell
.venv\Scripts\python.exe -m ruff check artifacts/runs/private-full-fit-native-20261101/study.py artifacts/runs/private-full-fit-native-20261101/qualify.py
.venv\Scripts\python.exe -m ruff format --check artifacts/runs/private-full-fit-native-20261101/study.py artifacts/runs/private-full-fit-native-20261101/qualify.py
.venv\Scripts\python.exe -m mypy --follow-imports=silent artifacts/runs/private-full-fit-native-20261101/study.py
```

Actual exit0/hostPID67680,28.2110061s/no host stop. Producer25.142452076s,
sampled peak536,576,000B; GNU25.61s/530708KiB (user20.08s/sys.97s). These held
110s/1792MiB workstop, GNU125+5/130saccept2GiB, host150s, stage8MiB/log512KiB/
report2MiB/per-native1MiB limits. RSS is observed, not OS-enforced. The original
inner16-update budget deliberately has no wall/RSS fields; outer gates are
separate. CPU threads1/GPU0, cached Linux3.12.3/NumPy1.26.4/ManiSkill3.0.1/
Sapien3.0.3/softwareMesa. A GLVND warning was logged; all native contracts still
completed. No installs/downloads/network/global setting changes occurred.

## Review and next learner outcome

Lead reviewed source, procedure and actual results. Finite independent review:
two source/PRE batches then one POST raw-byte pass/compact result and host/current
pin check, ~.6s tool time within180s. Directly verified all five complete negative
cases without a truncated middle projection. ACCEPT completed measurement,
not numerical replication or general task/release readiness. Exception ENDED.
No new production edits or numerical fixture failures in this increment. An
initial whole-file diff due line endings was replaced by normalized source diff;
the first lead display was verbose/truncated, so compact output came from the
already-written scalar cache. Neither caused another raw read or model run.

[Frozen protocol](../../artifacts/runs/private-full-fit-native-20261101/PROTOCOL.md),
[PRE](../../artifacts/runs/private-full-fit-native-20261101/PRE-REVIEW.md),
[POST](../../artifacts/runs/private-full-fit-native-20261101/POST-REVIEW.md).
Raw545604B SHA
`839f222c40bee0cfd8f2bf1ca053a9890aa2cd91a918f4f4200983593b1cdc88`;
lead1+peer1new raw-analysis slots SPENT, use `LEAD-SCALARS.json`. All protocols,
pins, read/host/case markers, original native traces/costs and negative results
remain. No repeat, next dose, private export or dead-process resume allowance.

Skipped old unchanged numeric/math/default/native suites, extra seeds/models,
new teacher collection, Linux fixture repeat, broad suite/cleanclone/package/
release/GPU/remote CI and all final data. No commit/push/delete/global change;
unrelated dirty masterd9119e8 preserved. #412software and #413scoped investigation
complete; broad R2.1/R5/acquisition-retention-return/science/release/full goal open.

Next [#414](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/414):
an explicit fixed23/probe_train correction path on learner-visited states.
Existing `PickCubeCorrectionQuery`/correction DTO/rollout path only admits
TRAIN17/19/20; `collect_probe_teacher` knows reserved roles but executes teacher
trajectories. That is the concrete missing boundary, not another optimizer dose.
Reuse query/teacher/rollout/runtime logic with separate role-bound provenance;
keep original TRAIN and24/final refusals. Original23native evidence motivates
this testable coverage repair; it does not establish coverage causality or use24
for tuning. New corrective teacher labels need explicit new consent/budgets,
finite software checks and a separate reviewed native protocol before collection.
