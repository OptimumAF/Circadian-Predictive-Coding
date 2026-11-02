# #431 / R2.1-CTRL.REPLAY — private neural joint rehearsal

## Delivered behavior and remaining claims

The existing run_continuous_probe API now accepts explicit replay_plan,
replay_episodes and replay_budget for exact BP/full-gauss-newton only. It uses
one joint native update per original delayed reserved-packet release. Original
probe_train rows precede separately admitted old TRAIN rows, with original roles
and episode spans retained; probe rows never enter the fixed TRAIN bank. Old
training_readouts alone remain score-only. Source, serving actor and benchmark
learner remain unchanged; private child is not exported/adopted/promoted.

Reused components: prepare_replay_bank/_BudgetBinding, original ToyBudgetSession,
_FullProbeBinding/native full-network kernel and existing ActorShadowRuntime.
The ordinary TRAIN rehearsal helper cannot bind the private probe_train runtime;
existing affine/context joint controls justify this smaller composition. No
runtime role widening, new runner, optimizer, normalization fit or RNG introduced.
Default no-replay reports/behavior preserved. Unsupported SGD/PC/CPC/nearest,
corrective, velocity and diagnostic combinations refused explicitly. Existing
three-old-readout BP limit retained, not broadened to new contexts/future labels.
Optional context-return recovery remains unfinished and separate; no return phase
or inference of recovery from a charge was added here.

Files:

- src/adapters/continuous_probe_rehearsal.py — fixed TRAIN/new reserved binding,
  original separate exposure budget, joint full-fit application, admitted metadata.
- src/adapters/continuous_probe.py — optional API composition through existing
  loop/ports; actual joint row accounting, additional newTRAIN readout, outer guards.
- tests/test_private_continuous_rehearsal.py — new pure/literal/tiny-kernel cases.
- docs/private-continuous-rehearsal.md — API/role/cost/unsupported-mode usage;
  docs/continuous-fresh-task-probes.md links the explicit new option.

Architectural decision: adapter composes existing app authority and native fit;
no app/core dependency points outward. Full-network math and generic runtime
unchanged. Separate replay arguments make consent explicit rather than training
on score-only monitors or pretending reserved data is TRAIN. No new dependency.

## Execution and reproducibility evidence

Physical D:/Dev/Repositories/Circadian, HEADd9119e80926a139a2f38460564d7eab1b6d2a2b0,
inherited unrelated dirty work preserved. Prospective PROTOCOL.md frozen before
edits/checks. Process TEMP/TMP/TMPDIR D:/Dev/_Temp, no bytecode, threads1/caches D:.
Actual tests Windows Python3.14; Linux-target types, not Linux execution. No
original episodes/checkpoint/raw scientific artifacts/SDK/teacher/physics/final
inputs read. Existing #410/#411/#413/#415/#429 concise results reused, not rerun.

Three newly warranted cohorts24/27/31allPASS; final31distinct,82case executions.
Tiny manually specified D2/H2/A4 model states, zero RNG/source __init__/SGD/PC.
All actual fixture native work is charged:28fullGGN/106jointrows,56grad/212gradrows,
456forwardrows,28lstsq(maxmatrixdim64),1973selectedbp._snapshot/568224numericB.
Counters cover selected snapshot path, not all deepcopy/elementwise/workspace
traffic. No total all-in copy/computation claim; actual RSS also recorded. Failed
injections include one real kernel with lost return; its work remains counted.
No test/numeric retry after a cap violation; all frozen observed limits held.

Actual body times2.4827771/1.6271669/1.9372346s, total6.0471786s; peaks69689344/
69513216/69779456B, maximum<384MiB. Launcher peaks~5.9MB are separately labeled,
not body memory. No torch/maniskill/sapien modules imported. 8static checks total
20.8074877s, Ruff/format/Win32+Linux-target mypy3filesallEXIT0. Three formatter
commands make11commands<=16, under180s scoped statics. Final6source/helperpins
match executed tree; new-source whitespace and scoped git diff --checkPASS.

Exact executed test command shape, for N=1,2,3, each stored unabridged in
artifacts/runs/private-continuous-rehearsal-20261102/tests-N.json:

```powershell
.venv/Scripts/python.exe -B artifacts/runs/private-continuous-rehearsal-20261102/check.py tests-N .venv/Scripts/python.exe -B artifacts/runs/private-continuous-rehearsal-20261102/pytest_body.py tests-N -q tests/test_private_continuous_rehearsal.py -p no:cacheprovider --basetemp=D:/Dev/_Temp/circadian431-testsN
```

Final commands (statics-2.json records exact arrays, outputs and time):

```powershell
.venv/Scripts/python.exe -B -m ruff check src/adapters/continuous_probe.py src/adapters/continuous_probe_rehearsal.py tests/test_private_continuous_rehearsal.py
.venv/Scripts/python.exe -B -m ruff format --check src/adapters/continuous_probe.py src/adapters/continuous_probe_rehearsal.py tests/test_private_continuous_rehearsal.py
.venv/Scripts/python.exe -B -m mypy --follow-imports=silent --platform win32 src/adapters/continuous_probe.py src/adapters/continuous_probe_rehearsal.py tests/test_private_continuous_rehearsal.py
.venv/Scripts/python.exe -B -m mypy --follow-imports=silent --platform linux src/adapters/continuous_probe.py src/adapters/continuous_probe_rehearsal.py tests/test_private_continuous_rehearsal.py
```

## Proved boundaries, finding and independent scope

Checks execute the real joint fit and compare its full result/state with explicitly
joined arrays through unchanged native kernel. Two delayed releases preserve source/
actor, receipts/clock, physical rows8/newrows4 and separately charged replay4.
All three score roles remain separate. Default no-replay fits2newrows and never
implicitly trains old readouts. Pure guards refuse unobserved/duplicate/reserved/
wrong-role/spec/capacity/spent/budget/replaced inputs and unsupported modes.
Exhausted replay stops before overshoot; failed native attempts keep charges.
Lost native return is unknown work, not zero. Corrupted preceding paid result is
restored for capture and refused before further work. Equal replacement of an
original replay budget is refused, including after first successful fit.

Independent review found one genuine initial gap despite27passing checks: a
post-native helper guard could throw before the recorded port saved a successful
paid result. Remedy returns that result immediately for existing capture, caches
physical row count BEFORE native call, retains admitted metadata independently
of later bank corruption, then checks original replay time/RSS/authority at outer
boundary. Three post-native budget/rows/bank mutations and one post-return wall
stop now prove a failed root retaining original rich result+charge/no secondfit.
Initial finding and pre-fix limitation remain in REVIEW-FINDING.md/probe.diff;
probe-final.diff is actual combined delta from preserved before source. No actual
scientific measurement was relabeled or retried. All original negatives retained.

Finite private_rehearsal_review:3readonly batches .8076782s tool wall, outputs
4658/4411/3574tokens within5000each, no toolingerrors. Initial2found blocker;
unusedthird original batch inspected remedy and cached final EXIT0/statics.
Scoped role isolation/joint rows/budget ownership/paid capture accepted, no remaining
blocker.31test count lead-supplied, not independently counted/executed; some surrounding
final metadata initialization/outer hook lines not reread. No peer imports/tests/
models/data/SDK/raw scientific records/writes/replication. Exception ENDED.
Lead separately self-reviewed actual combined source/procedure/31checks/results;
self-review is not independent or whole-repository acceptance. REVIEW.md exactscope.

Other tooling limitations: initial broad orientation/source excerpts truncated,
then focused ranges used. One guessed nonexistent context_joint_probe.py lookup
failed; actual private-joint source located via rg --files/docs. No model/native
study/tool retries or missing-output claim. No cap increase required. New helper
and targeted delta introduced no unrelated mutation. No new dependencies/install/
download/globalconfig/commit/push/reset/delete. Existing original source/counters/
spends/failures and negative scientific results are preserved.

Stage~0.1MiB before report, <=8MiB; test temporaryfiles0B, scoped mypycache16748773B
on D:. Full suite/unchanged old math or native cohorts/Linux fixture execution/
cleanclone/wheel/GPU/remoteCI/browser/release skipped: new role/receipt/composition
boundary only. Kernel math reused, explicit actual native qualification next.

## Critical path and one next action

Scoped #431software criteria complete. This is usable isolated API behavior plus
tiny correctness evidence, not successful robotics acquisition/retention. #429
nearest256already5/5knownfinalTRAIN with more memory/compute; does not satisfy128
resource constraints, neural/generalization/CPC/reliable-return/release goals.
The #413no-replay neural fit still failed all five after-cases; #415corrective
coverage remains negative; #411FAILED-SPENT/other partials unchanged.

Next #432/R2.1-CTRL.REPLAY.E is one genuinely new fixed intervention, not another
viewer/capture/audit or dose/seed/mixture search: old-TRAIN rehearsal during original
16fullGGNprivate updates. Reuse accepted qualified80source/before/no-replay evidence,
original23/24/old17/19/20 roles/seeds/metrics/horizon. Proposed originalold3bank52
rows +18newrows =>70jointrows perwake,832extraold/1120physicalrows, subject to source
metadata confirmation BEFORE data/model observation. Extra exposure/compute explicit,
not matched all-in compute. No new teacher/corrections, final data or adoption.

ONE exact next action: inspect accepted #413report and existing
artifacts/runs/private-full-fit-native-20261101/study.py plus original checkpoint/
frozen-port caller SOURCE, reconcile #431explicit replay args, then freeze/review
one local protocol/producer/input-use and resource/analysis budget BEFORE original
inputs/models/native work. No actual experiment grant from software closure or
issue creation; any engineering remedy prospective, failed charges remain spent.
A negative rehearsal result or simpler control preferable is valid. Broad18/21/23/
R2/R4/R5/128/heldout/acquisition-retention-return/CPC/reliability/release/fullgoal
OPEN-ACTIVE. GitHub livecanonical; plans/log archival evidence, not new current
handoffs. No fourth/fifth-update, capture, or provenance sequel prerequisite.


Canonical scoped completion: https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/431#issuecomment-6571715473 . #431CLOSED/all5scopedcriteria; #432PLANNED singleoriginalneuralrehearsal outcome. Broadparents/fullgoalOPEN-ACTIVE.
