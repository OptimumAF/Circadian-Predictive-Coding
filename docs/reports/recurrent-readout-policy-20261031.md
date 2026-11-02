# Fitted recurrent readout through stream and native loop — #377

Implemented an explicit owner with two categorical readouts on original frozen
public-cue recurrent features, plus the existing stream/evaluator handoff.
The new firstparty example fits both original A TRAIN episodes before any
target-free game. Ordinary learner/weighted/probe/default behavior and existing
core mathematics remain unchanged. The separately frozen firstparty native case
also succeeds on BOTH known A tasks in9steps each; broader claims remain open.

Changed: new src/adapters/minigrid_recurrent_readout.py and
scripts/recurrent_readout_example.py; narrow new facade/local session
classification in minigrid_policy_evaluation.py. Existing public generic
admission does NOT accept the new method as an old Torch checkpoint.
Usage docs/recurrent-readout-policy.md; full output remains JSON, no new runner,
loader, solver, serialization authority or dependency.

## Ownership and inference

The owner keeps frozen original raw148/H8/K7/seed17 native states at0updates/
0rows and separate immutable float64 categorical fits/cue counters. TRAIN role
is caller-owned at the array port, enforced at complete episode/firstparty ports.
Actual first public pixels select the head; context metadata/labels never route.
All native parameters/gradients/momentum and inactive readouts remain guarded.
Snapshot/fork/restore preserve exact policy and frozen representation, rejecting
changed/replaced prior containers/tensors/readouts/counters. A failed operation
retains an independent known-before prefix plus attempted/returned feature/fit
work and terminal latch; it cannot restore/continue or report an unavailable
failed capture as healthy. No claim that corrupted live state was repaired.

Sessions keep at most16actual frames, zero reset and one first-public cue.
Existing activation reader recomputes the whole prefix, then existing core
probabilities use only its last hidden row. All seven float64 actions are
validated and output detached. n-step game costs n(n+1)/2 native forward rows
and discarded projections; no teacher, lookup table or stored action sequence.
Each returned feature call is charged before subsequent corruption refusal;
lost returns remain uncertain, terminal and cannot reset the prefix.

The firstparty command uses the existing delayed TRAIN-only AttributionStream,
two fits18rows, independent source/realactor/candidate, original source/role/
protocol witnesses, sealed preceding training report and zero-update/replay
readout budget. A narrow exact2fits/18rows facade with custom copier uses the
existing native loop; existing generic admission still refuses the new label.

## Focused checks and preserved failures

Prospective SOFTWARE freeze: comment6548530909/stage PROTOCOL-SOFTWARE.md,
before fixtures/models. At most40distinct/platform,2cohorts/platform,120s512MiB
each/hostaggregate360s/statics120s; stage16MiB/cache128MiB. One tiny native
stream/session fixture/platform, no games/historical TRAIN reads. Scripted
delegate ports and fake environment work separately bounded/observed.

A bounded test-author exception wrote30scripted+one native test. Lead reviewed
tests, added output/generic-loop/state-copy/admission regressions and executed
them. The same worker independently reviews production it did not author.
This is not independent review of its own test authorship or numerical
replication. Max2agents total, finite stage review ownership.

Independent production review caught and lead fixed, before models:
- Returned feature costs were being counted after a corruption guard.
- A mutated detached native head could enter failed capture; independent
  known-before baseline now supplies failure evidence.
- Session snapshot copy needed final original-owner/value/saved-equality checks.
- Global recurrent-method registration could broaden old generic admission;
  classification is local to the already-admitted native loop instead.
- Fault monkeypatches could bypass port measurement. Tests use stable outer
  ports/override hooks; all intended calls/lost returns now remain measurable.

First Windows cohort39PASS/1FAIL. Native fixture returned one2row feature read,
then NumPy inputs hit the old Tensor-only comparator fallback before any LSfit.
Only this owner now aliases the existing NumPy/Torch _same_value comparator;
shared helper/default semantics stay untouched. Original work/failure retained.
The other39script/default cases already used that comparator via fixtures, so
their evidence is reusable. Initial fake-loop counters missed late-installed
fake class; moved installation into fixture setup. Earlier2constructs/4steps
are source/test-derived, not retroactively instrumented as observed counters.

Windows2 reran only affected native+fake-loop cases:2PASS. Linux1 executed
final complete40cases:40PASS. Final union40passing distinct cases/platform,
81passing call outcomes+one preserved failure (82calls total). No skips/errors.
No numerical budget amendment or unchanged math/control cohort rerun.

| Cohort | Body seconds | Host seconds | Peak bytes |
|---|---:|---:|---:|
| Windows1 |11.606494000006933|12.0205946|342478848|
| Windows2 |8.409314099990297|8.773846|332460032|
| Linux1 |13.797270482999998|55.9050993|344309760|

Host aggregate76.6995399s<360; WSLstartup stays included. The initial Windows
session92484 and Linux70990 were polled to terminal, never restarted on yield.

Physical software work across all three cohorts:6native constructions/
18private randn/7824drawn parameter scalars,6optimizer constructions,0native
TRAIN/SGD/backward/autograd/predict;4LSfits8rows/4LAPACK;17actual activation
calls30rows,30single-row native forwards30rows;12actual probability readouts/
16rows;3003snapshot-port calls including scripted validation, plus65Tensor.clone
calls33364cloned scalars. Snapshot-port totals are not exclusively physical
native Tensor validations.
The counter initial_values measures randn values;90zero bias entries are
source-derived additional parameter initialization, not random draws.

Scripted delegate work:156feature attempts754rows/152returns748rows;
84fit attempts378rows/82returns374rows;62predictions114rows, all returned.
Differences are intentional lost-return cases, not silently refunded work.
Fake-loop2construct4steps are measured in Windows2 and Linux1 each; initial
Windows1same2/4 is known from executed assertions/source. No actual physics/
teacher/environment or archive read. These named costs are not allcopies/
allocations/FLOPs/I/O or independently replicated math.

Final five-file Ruff/format and three production-file mypy default+Linux-target
PASS, statics-final.json4.9729564s. Earlier preparation unused-import/type
annotation failures and patch/JSsyntax misses retained in issue history; after
repeated matching failures, annotation edits were simplified to literal native
PowerShell replacements. No model work in preparation. Actual Linux statics,
fullsuite/wheel/cleanclone/install/GPU/browser/remoteCI/final/controller/
comparative/baseline/source cohorts skipped: unchanged behavior/math beyond
these boundaries. No install/download/push/delete/global settings or reset.

Exact commands, physical D: cwd and TEMP/TMP D:\Dev\_Temp:

    .venv\Scripts\python.exe -B artifacts/runs/recurrent-readout-policy-20261031/check.py windows-1 tests/test_recurrent_readout.py tests/test_recurrent_readout_native.py tests/test_public_cue_recurrent_learner.py::test_should_refuse_nontrain_record_before_native_work
    .venv\Scripts\python.exe -B artifacts/runs/recurrent-readout-policy-20261031/check.py windows-2 tests/test_recurrent_readout_native.py tests/test_recurrent_readout.py::test_actual_generic_loop_uses_frozen_readout_sessions_and_closes_fake_environments
    wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp PYTHONDONTWRITEBYTECODE=1 /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/recurrent-readout-policy-20261031/check.py linux-1 tests/test_recurrent_readout.py tests/test_recurrent_readout_native.py tests/test_public_cue_recurrent_learner.py::test_should_refuse_nontrain_record_before_native_work

Stage artifacts/runs/recurrent-readout-policy-20261031/ retains preimage,
frozen protocol, all cohorts/failed costs/host/static/source/review receipts.
PhysicalD:HEADd9119e8, inherited dirty work preserved. Independent SOFTWARE
production/procedure/execution review ACCEPT7reads/1.0895268s; initial test
authorship is not independent test review or numerical replication. All four
original review findings and the actual failed cohort remain recorded.

After software acceptance, comment6548943222 froze ONE firstparty from-birth
2fit/18row→two target-free A game case, including all recomputed prefix/native/
fit/physics/copy/RNG/budget/capture/time/RSS/storage costs. Its prepared protocol
and observer are in artifacts/runs/recurrent-readout-policy-native-20261031/.
Complete negative is valid. No16step/seed/metric/control selection, retry,
recovery of disposed376objects or old132/140/143weights, or further
rank/gradient/capture/audit sequel.

## Native firstparty qualification

The sole frozen native invocation completed BOTH original A-blue/red tasks in
9steps each, return.49375, terminated/closed. TRAIN exactness9/9each; blueCE
1.165422180485596/red1.1654221804855949. All7float64 firstargmax decisions use
actual public frames. This is useful known-task behavior from two TRAIN fits,
not reliable acquisition, retention/recovery, unseen transfer or CPC advantage.

Exact physicalD: command (TEMP/TMP D:\Dev\_Temp):

    & ./artifacts/runs/recurrent-readout-policy-native-20261031/run-native.ps1

Hidden ownedPID74112/samehandle/terminal0/no expiry. Observer6.983781199989608s/
340889600B, host7.4133175s; example2.4925828000123147s/340709376B. All attempts
returned/capsheld; SPENT, no retry.2coldnativeheads6private draws2608scalars+
30derived zero biasentries/2optimizers;2LS18rows/2designsolves/2LAPACK18rows;
22activations126rows/126single-row forwards including18fit+18score+90liveprefix
rows, discarded original output projections included.20probabilitycalls36rows,
3604snapshotcalls/21Tensor.clone10332scalars;2env/reset/localGymseeds,
18physicssteps/2returnedcloses,2distinctclosedowners. Source-derived2postfit
matrixproducts18rows/20predictionproducts36rows; named counters are not all
copies/FLOPs/allocations/IO.0nativeTRAIN/SGD/backward/autograd/nativepredict/
teacher/probe/final/replay/promotion/historicalmodel work. One original bundled
94705B TRAIN source read; no previous disposed fit reused.

Training originalbudgetcharges2/0; readonly0/0 across both games, original
births/lastclocks retained in actual.json. Source/actor/threeRNG/stateunchanged
true. JSON55481B/stdout1687B/stderr0B; native stage<8MiB/no cache. Full costs,
scope/skips/protocol/terminal receipts in native-stage RESULTS.md. PRE GO read8
was before models; POST ACCEPT reads9–10, total10formalreads/1.5318864s.
Finite exception ended, no numerical replication. No production change after
software acceptance. All seven scoped #377 criteria met; #378 PLANNED with no
fixture/model/native allowance, full parents remain open.

Critical path reassessed: R2.1 diagnostic/isolated probe and matched controls
already exist and are accepted. Existing affine joint rehearsal already solves
freshF2/2/oldABC6/6; original incremental rehearsal feature remains deferred.
Do not rebuild those or add another rank/gradient/capture/audit increment.
Existing public-cue recurring pilot binds native SGD/128 clocks/factory, so
silently labeling readout fits as those updates would be incorrect. Next expose
the new fitted-readout learner to an explicit bounded A→B→C→A TRAIN recipe in the
existing stream/example, preserving original meanCE pilot/defaults and separate
fit clocks. Report current-task acquisition, old-task retention and reacquisition
on return separately. FIRST reconcile the existing recipe/readout helpers and
add the smallest opt-in composition with focused scripted checks; new native
case only after its own frozen protocol. No new replay algorithm is justified
before this plain sequential control is observed.

Full goal and acquisition/retention/recovery/reliability/heldout/scientific/
release remain open; no CPC or matched-compute claim.
