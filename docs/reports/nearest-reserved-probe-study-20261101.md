# #401: one reserved memory append improves imitation, old TRAIN scores unchanged

One private18-row append from the original reserved23 packet reduced score-only
reset24 normalized half-MSE from **.036546455682723106 to .008950245041976411**.
The signed acquisition gain is+.027596210640746695. All three original old TRAIN
readouts stayed unchanged. The source remained220rows/7updates; only a disposable
private candidate reached238rows/8updates. No model was released or promoted.

This is a scoped reserved-reset imitation result within PickCube. The old inputs
are already stored TRAIN prototypes; unchanged scores do not establish retention
on new rollout states. Physical24success, Push retention, relearning after
forgetting, a different physical task, population reliability, generalization,
safety and CPC advantage remain unproved. No native episode ran.

## Fixed outcomes

| Readout | Role/rows | Before normalized half-MSE | After | Gain before−after |
| --- | --- | ---: | ---: | ---: |
| reserved24 |probe_test/18|.036546455682723106|.008950245041976411|+.027596210640746695|
| Pick17 A |TRAIN/15|0|0|0|
| Pick19 original B22 |TRAIN/22|6.472884083870503e−18|same|0|
| Pick20 |TRAIN/15|0|0|0|

All three reported old retention losses (after−before) are exactly0. Preserve the
tiny B22 rounding residual, rather than rewriting it as0. Full overall/per-axis/
stage diagnostics and signed values remain in the local result/cache.

| Reserved24 action axis | Before half-MSE | After |
| --- | ---: | ---: |
| root dx |.1094459932745077|.034830949931091086|
| root dy |.012481663299032492|.00048753809517846074|
| root dz |.02425816615735222|.0004824921416360928|
| gripper |0|0|

Data/seed/row order/metrics/directions and one-update dose were frozen before
reads; no early best, favorable endpoint, model choice, tuning, retry or sweep.
Only original probe_train23 labels updated the private copy;24labels and old TRAIN
readouts scored only. Final29was unavailable. These are the already reserved
teacher trajectories, not new independent collection or a matched-compute neural
comparison. Existing memory successes and BP/PC/neutral/active probes were reused.

## Preparation was necessary and paid

The accepted complete-B22 writer/stream reports durable_checkpoint=False, not a
restorable learner. This new question therefore explicitly prepared the unchanged
source through existing run_sequence(fixed_input=True,maximum_releases=6):
A15,B50,TRAIN20_15,Push18,oldcorrections50,currentcorrections50,originalB22_22.
The original B22 stays distinct from derivedB50, and all198previous rows/order
remain. One empty nearest constructor,0RNG,7appends220rows (warm15+six delayed205).
Source preparation is new paid work, not a new baseline outcome or confirmation
of accepted old native success. No historical neural parameters were rebuilt.

Only final tick17/7updates/220rows enters the existing probe; other12detached
observer entries perform0predictions. Original preparation source/actor remainA15;
the probe source/actor remain220. Existing qualified readers preserve original
history/role/identity/receipts/consent, byte hashes and exact row targets. The
reserved loader's diagnostics=True plan is copied toFalse only, because nearest
has no gradient diagnostics. Original18+18episodes/update1/roles are unchanged.

Both inner runtime budgets intentionally configure update ceilings only, not
wall/RSS. The explicitly declared outer job gates below cover the whole process.
This does not repeat #399's incorrect assertion of configured inner time/RSS caps.
Original receipts/clocks and all earlier spent budgets remain unchanged.

## Executed work and resources

Exact ONCE from physical D:\Dev\Repositories\Circadian:

```powershell
.\artifacts\runs\nearest-reserved-probe-study-20261101\run-study.ps1
```

HostPID74104 exit0/no timeout,3.03971s. Producer complete/errornull/budgettrue,
2.859871357s/sampled54681600B. GNU2.94s/max54672KiB/exit0. Existing cached Linux
Python3.12.3/NumPy1.26.4, CPU/OMP/BLAS/MKL1, physicalD and D:temp,0GPU/SDKscenes.
No installs/downloads/network/global tool changes. All declared40sTERM+5sKILL/
45saccept/35sworkstop/768MiBwork/1GiBobserved/60shost/8MiBstage limits held.

| Work | Observed |
| --- | ---: |
| Source preparation |1emptyconstructor/7appends/220rows|
| Private reserved probe |1append/18rows,0optimizer-gradient-denseNNwork|
| Total append attempts/returns/rows |8/8/238|
| Readout predictions/returned rows |8/140|
| Distance candidates/feature values |32060/384720|
| Score-only24 rows, before+after |36|
| Old TRAIN score rows, before+after |104|
| Prototype payload per source/private owner |28160→30464B (+2304)|
| Fork attempts/returns |18/18|
| Public snapshot attempts/returns |141/141|
| Known validator attempts/returns |1075/1075|
| Known returned numeric validator-copy payload |24425344B|
| File opens/read attempts/returns |16/16/16|
| Actual source bytes read |2087127|
| Forbidden operations |0|

Known validator payload covers the learner's original _snapshot and two admitted
aliases; public snapshots are included in that count, not an additional copy
total. Generic deepcopy, Python containers, validation temporaries/other copies
are additional. Logical prototype/readout/input byte fields are not total
resident memory or operation counts. The sampled RSS and GNUhigh-water readings
are separate observations. No per-method efficiency claim follows.

Stream elapsed.343757095s INCLUDES probe elapsed.168708601s; do not sum them or
call their subtraction an isolated preparation benchmark. Whole process time
also includes imports/input qualification/guards/capture. The three old readouts
remain the original at-most3contract; Push retention was not scored.

Input opens are exactly3memory+3B22teacher+5probe_train+5probe_test. Four literal
hash/length bindings were checked by the existing readers and again before
preparation/after probe. Actual bytes2,087,127<3MiB.18direct source pins agree
before inputs/before preparation/before probe/after and independent POST.
Original data/plan/receipt/row identity+value guards and paid result seals passed.
No unreturned append; no cap or failure-policy amendment/retry occurred.

## Verification, review and limitations

Reused accepted #40036case software proof; its numerical fixtures were not rerun.
Five NEWAST-only observer tests pass/.5845289s/47579136B,0models/datasets/native:
lost read return, exact bytes retained after a late ACK, per-file permanent cap,
validator ACK/copy bytes retained after deadline, unrelated artifact refusal.

```powershell
.venv\Scripts\python.exe -I -B artifacts/runs/nearest-reserved-probe-study-20261101/check.py
.venv\Scripts\python.exe -m ruff check --no-cache artifacts/runs/nearest-reserved-probe-study-20261101/study.py tests/test_nearest_probe_study_observer.py
.venv\Scripts\python.exe -m ruff format --check artifacts/runs/nearest-reserved-probe-study-20261101/study.py tests/test_nearest_probe_study_observer.py
.venv\Scripts\python.exe -m mypy --no-incremental --follow-imports=skip artifacts/runs/nearest-reserved-probe-study-20261101/study.py tests/test_nearest_probe_study_observer.py
```

Final scoped statics2files PASS. Initial three fixture annotation errors
(ASTlist variance and BytesIOoverride signature) were corrected without ignores;
passed tests were not repeated for those annotation-only edits. Host syntax and
actual storage functions passed without child creation. No production algorithm
or library code changed. Fullsuite/newLinuxfixtures/cleanclone/wheel/browser/
device/GPU/remoteCI/release skipped; this actual study itself ran cached Linux.

Independent finite PRE reviewed full actual procedure/protocol/host/five test
definitions before input reads and accepted GO. POST single NEWactual byte pass
plus current pins/host/resources accepted outcomes/accounting. Three read-only
batches1.0stools, no imports/tests/input/oldraw/model reads or numerical
replication. Review exception ENDED. Lead actual source/procedure/results
self-review complete, not described as independent execution.

Stage `artifacts/runs/nearest-reserved-probe-study-20261101/` holds frozen protocol/
producer/host/pins, SPENT markers, observer checks, actual report/log/resources and
LEAD-SCALARS. NEWactual39074B SHA256
fc640a4ae7d85e9191ab1edde7fbfe8246eb347c6665b8eb23c458152903b45e.
Lead1+peer1raw reads SPENT; use cached scalars, no reopen/model/native retry grant.
All original positive/negative/partial evidence and unrelated dirtywork preserved;
no commit/push/delete or raw artifact publication.

## Remaining useful outcome and next action

#401/R2.1-MEM.PROBE.E completes this scoped imitation question. #399four neural
native failures/inner-budget protocol deviation remain unchanged/open. Broader
learning/retention/recovery/science/release parents and full goal stay OPEN/ACTIVE.

Native24was not attempted: existing check_scoring_request restricts reserved native
scoring to exactBC/PC, and _check_guard_native_frozen has only those validators.
Recorded imitation gains cannot stand in for a robot outcome. This is a concrete
next dependency, not a reason to repeat the probe/capture/diagnostic increment.

Next #402/R2.1-MEM.NATIVE: add an explicit fixed-input nearest probe_test24 path
through EXISTING run_frozen_rollout, with literal admission/frozen-owner/role/
reset/history/lost-control/cleanup tests first. Preserve ordinaryBC/PC/TRAIN,
guard31/37and final29refusal, no corrections/interventions/labels/promotion. Full
prototype owner/array/counter/identity proof must survive callbacks. No new runner
or simulator allocation. A later paired native observation requires a separately
frozen budget, including any necessary live preparation/append again; expired
owners/clocks are not free or restorable authority. No automatic rerun or sweep.
