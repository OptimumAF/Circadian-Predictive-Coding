# Matched public history for the local key/door stream

Preview the fixed recipe without collecting episodes or constructing models:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-public-history-train --describe
```

The same command without `--describe` runs the fixed TRAIN-only qualification.
Its common deadline is 90 seconds, CPU threads1/GPU0. The existing four native
methods BP/PC/neutral CPC/active CPC receive identical public initial8+current8
inputs, hidden width8, five classes, initialization17 and eight A/B/C/A teacher
episodes with label delay1. Each method retains its original objective/config.
The existing AttributionStream supplies all updates. All four streams finish
before six original TRAIN traces are scored at checkpoints0/2/4/6/8.

Why this: a current local A frame can require different actions under the two
earlier public cues. One initial frame separates that demonstrated pair. The
encoder resets for every episode, retains one frame, and uses no target, context
ID, hidden movement law or future frame. It does not solve hidden B mapping or
prove that all histories are identifiable. Existing8-input routes and the
official MiniGrid148→296 public-history contract retain their behavior.

## Boundaries and costs

```text
src/core/key_door_public_history.py             pure8→16 episode transform/reset
src/adapters/key_door_pilot.py                  existing TRAIN collector/default route
src/adapters/key_door_public_history_pilot.py   existing learners/stream/TRAIN scores
scripts/run_coupled_attribution.py             fixed opt-in scenario
tests/test_key_door_public_history.py          causal transformation behavior
tests/test_key_door_public_history_pilot.py    matched execution/isolation/failures
```

The core depends only inward on episode records; collection, native learners and
CLI remain outer adapters. Targets/actions/rewards/terminal flags/roles/identities
are preserved. Complete source/serving state, original receipts/budgets/clocks,
completed and unreached rows are guarded before further work. A failed native
call preserves original receipts and the last observed successful state; a
stopped candidate remains unavailable rather than being presented as atomic or
resumable. Native prediction returns and verified score completions are distinct.

The TRAIN-only scenario has no final evaluation, probe, replay, live policy,
promotion or selection.
Each model has181parameters. Shared input arrays grow4608→8704bytes; one logical
initial frame is64bytes. Copies, metadata, adaptive/native state and diagnostics
also consume memory/time. Equal exposure and initialization do not imply equal
computation: BP uses8autograd/SGD updates; the three PC methods use24manual energy
updates with48batch latent iterations in total. Total256TRAIN rows,
120readout calls/960rows,8teacher episodes/64physical steps.

## Executed evidence and extension

The one reviewed native case completes in10.3686s (11.4369s supervisor), external
peak317607936bytes. Initial parameters match across all four methods; PC and
neutral CPC parameters/readouts match exactly.159focused checks and scoped Ruff,
format and Windows/Linux mypy targets pass. Actual Linux execution and whole
repository/release gates were not run.

All20checkpoint full-sequence accuracies are zero. End mean TRAIN decision
accuracy is22.9167% BP and18.75% each PC/neutral/active CPC. These are small fixed
teacher-forced resubstitution diagnostics, not live/generalization/adaptation or
circadian advantage evidence. Original failed/inconclusive studies stay intact.
[Protocol, commands, failures and raw curves](../artifacts/runs/matched-public-history-20261017/RESULTS.md).

The original TRAIN-only route retains its accepted descriptor and behavior.
The optional reserved probe scenario below composes the existing helpers.

Focused tests:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_key_door_public_history.py tests/test_key_door_public_history_pilot.py
```

Use the seven paths and exact Ruff/format/mypy commands recorded in
`artifacts/runs/matched-public-history-20261017/closing-statics.json`.
No dependencies, environment variables or configuration files were added.

## Matched fresh-task capacity

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-public-history-probe --describe
```

Without `--describe`, this fixed scenario performs the same four benchmark
streams and original TRAIN scores, then collects the existing reserved E16
packets once and encodes each independently. Existing `run_checkpoint_probe`
and `run_fresh_task_probe` run on disposable full-state initial0/end8 forks.
Each of eight probes has an independent16-update budget and points0/4/8/16.
Only probe_train labels enter those copies; probe_test supplies metrics. No
probe result or label updates the benchmark, serving actor, promotion or final
evaluation. The original common90-second deadline covers all work.

Pure boundaries check original source/actor/candidate/checkpoints, receipts,
budgets/clocks, TRAIN/reserved data and completed/unreached probe evidence.
The helper's active result is private; only its selected slot is published.
Previously published slot/report replacement and reserved-array replacement are
refused. A failed probe retains charged updates and partial metrics; subsequent
probes remain not_run. Later arms that already finished TRAIN are reported as
benchmark_complete until both own probes complete. Native work and failure
receipts remain additional to all historical studies.

Maximum additional invocation work:160parameter updates/1280TRAIN rows, including
128probe updates/1024probe rows. BP40autograd/SGD and PC-family120manual energy
updates have unequal objectives and compute; PC240batch latent iterations.
184score calls/1472rows,26teacher episodes/208physical steps; shared encoded
arrays28288bytes, including19584reserved bytes. Input/copy/report/native state
and all guard cost count in whole time/RSS. Genuine initial0/end8 is distinct
from relative probe0/16 (native ends16/24).

E is deterministic teacher-forced fit with class1absent; new record IDs do not
make it independent generalization. Initial→end reflects the whole benchmark
history. Full curves, negative results, signed metrics and original seed/dose
are retained; there is no tuning for CPC benefit. This comparison does not
reopen accepted original R2.1, establish useful initial acquisition, or prove
old-task retention, recovery savings, live skill, CPC advantage or release
readiness. [Frozen protocol and executed checks](../artifacts/runs/matched-history-probes-20261017/RESULTS.md).

Focused probe integration checks:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_public_history_probes.py
```

Use `accepted-statics.json` in that artifact directory for exact scoped Ruff,
format and Windows/Linux mypy commands. No new runner or learner is introduced.


### Executed matched capacity case

The one reviewed case completes in24.1387s, external peak319873024bytes.
All252focused checks and scoped statics pass; original benchmark row fields and
TRAIN demonstrations match the previous accepted case exactly. All eight probes
reduce cross-entropy and finish at50% accuracy. Every endpoint predicts only
class3, the50% majority action; all exact sequences remainzero. This is incomplete
supervised fit, with no live skill or CPC benefit established. PC-neutral reports
match exactly apart from elapsed time. All curves, costs and historical failures
are retained in the linked evidence.

Next add owned per-episode public-history prediction sessions to the existing
TRAIN-layout live diagnostic path, preserving raw8/RNN defaults. Reset before
any action using only the first public observation, keep checkpoint weights
frozen, and use the existing episode runner. Report legal/illegal actions,
returns, terminal/censored failures and all inference/encoding/session costs;
do not mask or repair actions. Freeze/review a separate protocol before native
execution. No further E dose, probe horizon or seed selection is proposed.


## Closed-loop TRAIN-layout history

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-public-history-live --describe
```

Without --describe, unchangedfourbenchmarkstreams/120TRAINscores finish before
120fixed knownTRAIN A/B/C± liveepisodes atpoints0/2/4/6/8. Rawargmaxfirsttie,
max12steps; illegalactions remainunapplied, no mask/repair/retry. No final/probe/
replay/promotion. KnownTRAIN resubstitution doesnot establish generalization or
useful forgetting/recovery when initialskill is absent.

Why this: majority-action teacher fit does not establish actual navigation.
Report pooled andper-layout success/return/steps/illegal/timeout atallfivepoints,
with costs separate from skill. Each episode owns fullfrozennativefork and one
initialpublicframe, resetbeforeanyaction. Inputonly exactfloat64(1,8), output5
probabilities; no context/cuemetadata/targets/legal-actionlist inprediction.
Nativeparameters/gradients/optimizer/counters frozen; sessioncounters separate,
max12calls, failstops. Actualnative returns countedbeforepostguardrefusal.

```text
src/adapters/key_door_public_history_session.py  owned public8→16 predictor
src/adapters/key_door_pilot.py                  existing liveowner/episode runner
src/adapters/key_door_public_history_pilot.py   fixed TRAIN live composition
tests/test_public_history_session.py          ownership/reset/failure counters
tests/test_public_history_live.py             real game/fake-native integration
```

Existing _live/run_key_door_episode own execution; activeoutputprivate, selected
slotpublicationonly, previous/unreached results+identities sealed. Originalsource/
actor/candidate/checkpoints/receipts/data/budgets staychecked; onlyexplicitly
observed original last_clock advances canupdate seal. Defaults/raw8/RNN/E16
routes unchanged. Fullownership construction/copy/readback/encoding/report costs
count in invocationtime/RSS, no zero-cost historyclaim.

Maximum32parameterupdates256TRAINrows:BP8SGD/PC24manualenergy48batchlatent.
120TRAINscores960rows+<=1440liveprediction/physicssteps; teacher64physics,
<=1504total/120liveepisodes andsessions/up to240nativeforks.181params/model,
8704sharedTRAINarrays/64logicalframeB/session exclude nativeauxiliaryallocations.
Distinct attempted/returned/encoded/completed/appliedcounts retain failed and
censored work. Shortillegalepisodes are not efficient skill.

Focusedtests: python -m pytest -q tests/test_public_history_session.py tests/test_public_history_live.py.
Exact scopedRuff/format/mypyWindows+Linux targets in accepted-statics.json under
[executedprotocol/evidence](../artifacts/runs/matched-history-live-20261017/RESULTS.md).
No dependency/environmentvariable changes. Freeze/review separately before new
native experiments; existingcases confer no futureallowance.


### Executed native live case

The one frozen case completes in68.7098seconds; externalpeak317452288bytes.
All120readouts stop on an unappliedillegal action, success/returnzero; no room
reaches terminalgoal.156native predictions lead to36appliedsteps. Additional
32TRAINparameterupdates/256rows,120TRAINscores960rows and64teacherphysicssteps
bringknownphysicalworkto100steps/128episodes. Earlierbenchmarkrowfields and
TRAINdemonstrations match acceptedsourceexactly; PC-neutral livepayloads match
apartfrom their declaredmethod/versionnames. Costs include120freshsessions/
240liveforkinterfacecalls, allencoding/copy/guard overhead.

122focusedchecks and78closingchecks pass with scopedstatics/independentreview.
The broad80s engineeringgate remainsincomplete, andthe reproducedconstruction
regression failure isretained. No usefulacquisition/retention/recovery/CPCbenefit
or releaseclaim follows from this scopedintegration. Allpriornegatives remain.
Next implement a capped supervised exact-public-observation memorycontrol with
explicit conflict/unseen rules and ownstate. Reuse existing stream/liveports
underitsownmodelidentity; compareagainst retainedfourcontrolnegativeswithout
tuningthem, and accountfor itsdifferentmemory/compute. Freeze/review before
comparativeexecution. Full requirements andcommands stay inlinkedplans/evidence.
