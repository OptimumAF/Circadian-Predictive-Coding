# R2.7-CTX.S — scripted calibrated incident scheduling, 2026-10-29

[Issue291](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/291)
owns live status. Tracker18/parents21/26/33/full goal remain OPEN. Previous turn
PROGRESS290; its observer/core evidence and289native supervised serving reused,
not rerun. HEAD0aae1d6 unchanged. Inherited dirty270–290/user changes and every
negative/SPENT scientific capture preserved. No old raw scientific evidence read.

## Working behavior and limits

The existing `_reference_schedule` optionally composes declared stationary TRAIN
calibration, fixed-reference arrived observations, CURRENT-candidate correct ACK
and onset-bounded FIFO error fitting. No new runner/controller law/threshold/CLI.
Each onset fixes the already-arrived boundary; continued alarms never renew it.
Fits require fresh current errors and original credit. Original horizon remains;
short/continued-only cases truthfully leave work incomplete. Causal labels,
source/native origin, receipt identities, candidate pins and paid evidence remain
enforced. An optional expected_candidate is checked inside the original ACK lease
before diagnosis/commit, including after the first input hook.

Positive scripted case: six prefix packets ACK, two changed packets fit, candidate
counter12 to14, source/actor unchanged, all8TRAIN receipts consumed. Delays1/2
pass. One fit can make the next packet correct: it ACKs on current learner13
while diagnostic reference stays12. A deferred wake retries with fresh current
feedback and two actual updates, without credit/horizon renewal. Stationary
no-incident case leaves wrong packets unfitted. Continued-alarm case retains
boundary7 and leaves packet8 pending. Real PromotableActor declined-drain callback
runs once at tick9 after2fits+6ACK; completed=True/accepted=False. This qualifies
the drain callback, NOT accepted publication. Native289guard/promotion evidence
applies only to its original source/recipe; it was not rerun or extended here.

Implemented/executed scripted correctness is distinct from native usefulness.
Stationarity/false-positive behavior, useful wake selection, comparative adaptive
learning, CPC advantage/generalization/release/full repository readiness remain
OPEN. Model/stationarity labels remain caller declarations. Old misses/false
alerts,279wheel limitations and287UNKNOWN/SPENT remain unchanged.

## Validation, repairs and preserved failures

Windows final combined tree111PASS/0FAIL/0SKIP:25new schedule cases plus86existing
pure observer/calibration cases. Linux109PASS before final pre-ACK repair, then
12affectedPASS on final tree including2new cases:111distinct overall, NOT one
final111case cohort. The12 include the no-update mutation check. Unchanged paths
reuse Linux109 evidence; no gratuitous full rerun beyond the original callback
cap. Default supervised ACK has an explicit new compatibility case.

Three failed numerical invocations remain in their original receipts, no refund:

1. Windows1:0PASS/1FAIL. Feature2 fixture correctly rejected by unchanged[-1,1]
   domain after23literal entries/6scalar replays,0training. Fixture changed to
   distinct normalized values; no boundary weakening.
2. Windows2:0PASS/1FAIL after successful2scripted updates. Test compared label
   event_id to batch sample_id. Corrected assertion after inspecting receipt type;
   production unchanged.
3. Linux1:0PASS/1FAIL at tick0. New receipt check addressed AdaptationPoll.updates
   rather than .training.updates. Scoped mypy found the same mistake, but the
   shell continued to pure tests. Code/test field corrected, original receipt
   preserved. Subsequent numerical gates followed fresh passing statics.

Scoped mypy initially15typing errors, repaired with explicit dict annotations,
None narrowing, an ordinary input function and actual restore_state port. Later
one poll-field error as above. Final3file Ruff lint/format/default+Linux-target
mypy PASS. Final git diff --check PASS with inherited CRLF advisories. One report
writer failed Windows default encoding on an arrow; its empty owned file was
replaced through the UTF8 patch tool. No tests/model work in that writer, no lost
experiment receipt. No ignore/config weakening/new dependency/install/download/
global change/commit/push/delete.

Independent readonly source review found observation-side re-pinning and callback
replacement of exported costs. Repaired previous-state pins and private cost
ownership with detached exports and checks before ACK. Final source read found
the first-input-hook gap; optional expected_candidate now verifies the prior pin
inside the existing lease. Exact refusal test asserts0ACK/0fit/0diagnosticattempt.
Other faults exercise failed/current/paid/episode mutation, no-update poll and
cost preservation. Lead self-review covers combined source/procedure/results.
Independent review is scoped software review, not replication/native acceptance.
Final visible-evidence disposition is recorded after the review below.

## All-attempt costs and bounded procedure

Frozen per platform:<=4pytest invocations,180s cumulative body,512MiB sampled RSS;
<=768named scripted probability callback entries/3072rows,256scripted training
updates/1024rows,256scalar prefix replay observations/1024rows. Stage8MiB;
statics120s/512MiB. No transfers/renewal. Named callback entries include wrappers
and nested literal callbacks, not all scaffolding/NumPy/FLOPs. Scripted updates
are fixture state transitions, not learned model optimization. Rows are1/packet.

| Receipt | Exit | Pass/fail/skip | Body seconds | Peak RSS B | Literal entries/rows | Scripted updates/rows | Scalar replay observations/rows |
|---|---:|---|---:|---:|---:|---:|---:|
| win-1-work.json | 1 | 0/1/0 | 2.786779600 | 82,014,208 | 23/23 | 0/0 | 6/6 |
| win-2-work.json | 1 | 0/1/0 | 2.139450000 | 78,299,136 | 25/25 | 2/2 | 6/6 |
| win-3-work.json | 0 | 111/0/0 | 5.194565000 | 78,274,560 | 472/472 | 17/17 | 114/114 |
| linux-1-work.json | 1 | 0/1/0 | 17.187287834 | 80,572,416 | 1/1 | 0/0 | 0/0 |
| linux-2-work.json | 0 | 109/0/0 | 18.179987092 | 80,461,824 | 453/453 | 15/15 | 114/114 |
| linux-3-work.json | 0 | 12/0/0 | 9.397012806 | 75,423,744 | 233/233 | 11/11 | 48/48 |

Windows:3actual invocations10.120794600s, maxsampledRSS82,014,208B;
520literal entries/rows,19scripted train rows,126scalar replay rows.
Linux:3actual invocations44.764287732s, maxsampledRSS80,572,416B;
687literal entries/rows,26scripted train rows,162scalar replay rows.

All captured native/model/LS/affine/routing/teacher/task episode/typed dispatch
counters0. Torch never imported in recorded tests. No random model draws/GPU/
final/probe/old scientific raw/comparative cases. All pytest count/time/sampledRSS/
callback/training/replay bounds and stage bound held; sampled RSS does not prove
every sub-sample peak. Statics tool wall totals<120s; static subprocess RSS was
not instrumented, so all-attempt static-memory compliance is NOT attested.
Finite independent review<=4readonly calls120s, max2agents incllead; no execution,
imports, writes or model work. No more numerical work planned; unused allowance
stays unused and provides no scientific grant.

Stage `artifacts/runs/calibrated-shift-schedule-20261029/`:11,987B,
check.py plus six work receipts. Recorder checks cumulative caps against all
prior same-platform receipts and saves attempts before final assertions. No
receipt-date/missing-peak recovery occurred here. Full command argv/cost/outcome
is embedded in every receipt.

## Exact executed commands

Physical D: cwd, process TEMP/TMP D:/Dev/_Temp. Recorder supplies
`-q -ra --tb=short --maxfail=1`. Failed invocations included below.

```text
.venv/Scripts/python.exe -B artifacts/runs/calibrated-shift-schedule-20261029/check.py artifacts/runs/calibrated-shift-schedule-20261029/win-1-work.json tests/test_calibrated_shift_schedule.py tests/test_arrived_candidate_shift.py tests/test_arrived_calibrated_shift.py --basetemp=D:/Dev/_Temp/circadian-calshift-sched-win1
.venv/Scripts/python.exe -B artifacts/runs/calibrated-shift-schedule-20261029/check.py artifacts/runs/calibrated-shift-schedule-20261029/win-2-work.json tests/test_calibrated_shift_schedule.py tests/test_arrived_candidate_shift.py tests/test_arrived_calibrated_shift.py --basetemp=D:/Dev/_Temp/circadian-calshift-sched-win2
.venv/Scripts/python.exe -B artifacts/runs/calibrated-shift-schedule-20261029/check.py artifacts/runs/calibrated-shift-schedule-20261029/win-3-work.json tests/test_calibrated_shift_schedule.py tests/test_arrived_candidate_shift.py tests/test_arrived_calibrated_shift.py --basetemp=D:/Dev/_Temp/circadian-calshift-sched-win3
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/calibrated-shift-schedule-20261029/check.py artifacts/runs/calibrated-shift-schedule-20261029/linux-1-work.json --capture=sys tests/test_calibrated_shift_schedule.py tests/test_arrived_candidate_shift.py tests/test_arrived_calibrated_shift.py --basetemp=/mnt/d/Dev/_Temp/circadian-calshift-sched-linux1
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/calibrated-shift-schedule-20261029/check.py artifacts/runs/calibrated-shift-schedule-20261029/linux-2-work.json --capture=sys tests/test_calibrated_shift_schedule.py tests/test_arrived_candidate_shift.py tests/test_arrived_calibrated_shift.py --basetemp=/mnt/d/Dev/_Temp/circadian-calshift-sched-linux2
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/calibrated-shift-schedule-20261029/check.py artifacts/runs/calibrated-shift-schedule-20261029/linux-3-work.json --capture=sys tests/test_calibrated_shift_schedule.py -k "ack_prefix_then_fit or ack_after_a_fit or resume_deferred or complete_fit_and_ack_drain or changed_evidence_inside or preserve_failed_current or pre_ack_input_hook or existing_supervised_ack or unobserved_state_change_during_wait" --basetemp=/mnt/d/Dev/_Temp/circadian-calshift-sched-linux3
.venv/Scripts/python.exe -B -m ruff format --check scripts/run_bp_ticket_serving.py scripts/candidate_training_feedback.py tests/test_calibrated_shift_schedule.py
.venv/Scripts/python.exe -B -m ruff check scripts/run_bp_ticket_serving.py scripts/candidate_training_feedback.py tests/test_calibrated_shift_schedule.py
.venv/Scripts/python.exe -B -m mypy --follow-imports=silent scripts/run_bp_ticket_serving.py scripts/candidate_training_feedback.py tests/test_calibrated_shift_schedule.py
.venv/Scripts/python.exe -B -m mypy --platform linux --follow-imports=silent scripts/run_bp_ticket_serving.py scripts/candidate_training_feedback.py tests/test_calibrated_shift_schedule.py
git diff --check
```

Skipped native/teacher/rollout/289model rerun/baselines/old raw/full repository/
clean clone/current wheel/remote CI/release checks. No native accepted promotion
or scientific benefit inferred from script correctness. Core detector/controller
formulas and unrelated dirty changes remain intact.

## One precise next implementation action

Add an explicit calibrated opt-in to existing context_head_live.run_owned_case/
run_case, using the owned LS source and a declared six-prefix/two-joint TRAIN
recipe within8batches/48rows. Keep original gates/horizon/receipt ownership and
qualify options with pure tests first. Before ANY native work freeze a small
stationary/shift protocol with SAME-stream supervised error/ACK control, explicit
model/prefix/diagnostic/rehearsal/storage/fit costs and failure policy. Do not tune
thresholds, open final/probe data, repeat old baselines or revive287. Native results
may fail or favor the simpler control. No native allowance granted here; full
goal and parents remain ACTIVE/open.


Final independent source/visible-evidence review accepted the repaired software and honest evidence scope. Four readonly calls, approximately1.6s observed tool execution; no writes/imports/tests/model work. Exception ended. This is not replication, native usefulness, accepted promotion or scientific acceptance.
