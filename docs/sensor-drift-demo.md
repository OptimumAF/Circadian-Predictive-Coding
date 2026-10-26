# Recurring sensor demo

## Isolated C probes from saved inversion states

The [saved-state qualification and assay](../artifacts/runs/r21-saved-inversion-probes-20261017/RESULTS.md)
reuse existing SensorLearnability and native restore on recorded4/8/12snapshots,
with no benchmark retraining. Only initial parameters were captured at0; the
canonical zero-update method control is reconstructed and labeled explicitly.
The prepared input excludes final scores and forecasts before native probing.

All four methods have small positive C BCE gains after four isolated updates;
accuracy drops at wake0/8 and improves at4/12. PC-neutral matches exactly;
active CPC gains are slightly lower than PC at all wakepoints. This is a fixed
C-task observation, not joint A/B/C retention, recovery or a generalization claim.
All64native states/scorepoints,16health/SVD records, permanent charges and both
RSS measurements are retained. Stage scripts use exclusive single-run outputs;
read existing results rather than rerunning a spent experiment. See the linked
exact commands, fixed protocol, tests, independent review and complete costs.

## Fixed contrast-inversion condition

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_sensor_drift_demo --describe --contrast-inversion --monitoring --heldout
```

`--contrast-inversion` keeps the original A data and exact A return. Only B
pixels become `1 - A pixels`; object-identity labels stay the same. The camera
negative reverses raw left/right contrast, so the class rule seen in pixels
changes. No context ID or added training noise is provided. The final B set is
the negative of the original fixed noisy final A set, with the same labels and
seed27031. Stationary and contrast inversion are mutually exclusive.

All existing ordinary BP/PC/neutral/active modes and BP scheduled/replay/incident
controls compose with this condition. `--heldout` retains finish-only scoring;
ordinary `--learnability` probes remain disposable and separate from benchmark
updates. Initial seed17, rates, four batches per phase, delay2, monitoring settings,
metrics and budgets are unchanged. Default brightness/stationary calls and
protocols remain unchanged. The current inspector accepts brightness/stationary
reports; contrast-inversion inspector presentation is not yet implemented.

Why: the accepted easy fixture has perfect cold accuracy and cannot demonstrate
recovery. This fixed alternative is specified before any model execution; it is
not selected from a search over transformations. Four B updates may be inadequate.
A negative or censored result remains valid. See ADR-0371 and the
[frozen qualification protocol](../artifacts/runs/r46-contrast-inversion-20261017/PROTOCOL.md).

The frozen qualification completed89native updates/350training rows across all
seven cases. All B prearrival accuracies remained0; recovery is censored, and A
return has no deficit. Incident/periodic heldout B accuracy at checkpoint8 is
.40625/.15625, returning to0 by12. Both have paid replay costs and positive old-A
BCE losses. The original bounded R4.6 demo criteria are met; successful recovery,
statistical false-positive control and circadian benefit remain unproved.
[Exact commands, costs and all outcomes](../artifacts/runs/r46-contrast-inversion-20261017/RESULTS.md).


## Stationary monitoring control

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_sensor_drift_demo --describe --stationary --monitoring
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_sensor_drift_demo --stationary --monitoring --output new-stationary-sensor.json
```

`--stationary` uses twelve exact baseline-A batches with independent arrays;
it adds no pixel noise or RNG. It requires monitoring and keeps the original
seed17, models, rates, delay2, four-batch calibration and detector policy.
The existing ordinary BP/PC/neutral/active methods and scheduled/replay/incident
BP controls remain selectable. Default recurring behavior is unchanged.

`sensor_monitoring.false_alarms.stationary_A` counts actual observed post-prefix
ticks4..11: eight opportunities when complete, fewer after a failure. It retains
all three channel alarm/onset counts and rates. Zero trials give null rates;
0/8 is descriptive, not statistical false-positive control. Frames/windows are
dependent. Absent B and return phases have zero trial denominators and recovery
`not_applicable`, null ticks/labels-to-recovery and no fictional `no_deficit`.

The unchanged periodic replay deduplicates the all-A data to two content rows:
two one-row updates at each boundary4/8/12, six replay updates/rows and18total
native updates/54rows, with the original12row allowance. Incident replay retains
its conditional upper bound and pays only admitted rows; no threshold is tuned
to create an incident. Scheduling still has12wake receipts/3optimizer steps.
Monitoring adds no model forwards beyond the original12prearrival predictions.

`--heldout` and `--learnability` retain separately frozen recurring B/return
protocols and are refused with this new control before model work. Stationary
extensions of those assays remain unspecified. The [existing saved-report inspector](key-door-inspector.md#stationary-sensor-controls)
now opens all seven stationary controls with explicit absent phases, partial
unknown costs and source pointers. No native rerun is needed.

Focused checks use `tests/test_stationary_sensor_monitoring.py` and
`tests/test_stationary_sensor_demo.py` plus affected sensor suites. See ADR-0370
for the decision and the [frozen small qualification protocol](../artifacts/runs/r46-stationary-sensor-20261017/PROTOCOL.md).

The combined274checks and scoped lint/format/types pass. One independently
reviewed seven-case native qualification completed81updates/342rows and84original
forecasts: each case recorded0/8alarms in all channels; scheduled3optimizersteps,
periodic6replay charges and incident0/three skips. Allcold/prearrival accuracies
are1.0; no deficit/recovery or statisticalFPR/circadian benefit is established.
[Exact commands, full costs, results and limitations](../artifacts/runs/r46-stationary-sensor-20261017/RESULTS.md).

## Original recurring recipe

This small NumPy qualification exercises the existing delayed stream with matched
BP or existing PC/CPC methods. It is not a comparative study or a deployment/
promotion demonstration.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_sensor_drift_demo --describe
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_sensor_drift_demo --output new-sensor-result.json
```

Output must be a new file in an existing directory. No downloads or new
dependencies are needed. The CLI bounds BLAS threads to one in its own process.
The fixed body limits are 12 wake receipts, 30 seconds, RSS256MiB and JSON128KiB.
Default BP performs12optimizer updates; scheduled and replay modes have the
separate optimizer/exposure limits documented below.
Qualification's external parent wrapper also enforces45seconds including setup.
There are no new environment variables; existing `.env.example` is unchanged.

## Structure and behavior

```text
src/core/sensor_stream.py          deterministic pixels and separate labels
src/core/sensor_retention.py       fixed final data and pure signed summary
src/app/coupled_attribution.py     existing stream, optional prearrival hook
src/adapters/sensor_drift_demo.py  fixed native composition, delayed score joining
src/adapters/sensor_learners.py    existing native methods, fixed sensor geometry
src/adapters/sensor_learnability.py isolated checkpoint probes and feature health
scripts/run_sensor_drift_demo.py  CLI, new-file output
```

The fixture has two unique 4×4 patterns, repeated as four balanced rows in each
of12batches. A4→brightness+0.25 B4→exact A4 return preserves the left/right label
rule. Context IDs are absent from learner inputs.48frames are row exposures,
not48unique examples. The default path admits no heldout or final data; the
opt-in finish-only context readout is documented below.

`AttributionStream(..., observe_features=callback)` calls
`callback(PrearrivalObservation(tick, row_ids, features), CandidateState)` before
same-tick label arrival, including zero delay. Features/state are detached. The
candidate carries current shadow version/wake/consolidation indices. The runtime
lease refuses nested training; source/actor/candidate/budget authority changes
are refused without repairing state. Existing retrospective checkpoint callbacks
keep their original meaning. Callback counters remain available after failure
as `prearrival_report`; actual native forwards require separate accounting.

In this demo labels arrive two ticks later. Each stored prediction is joined to
its arrived TRAIN labels by row IDs, after the corresponding wake. Scores never
recompute predictions using updated weights. These stream-row scores are
prequential training observations. The opt-in heldout scores are separate
finish-only measurements. The stable
actor remains frozen throughout; shadow adaptation is not deployed adaptation.

## Observed qualification

[Artifacts, commands and independent review](../artifacts/runs/r46-sensor-prearrival-20261017/RESULTS.md).
142focused tests and scoped statics pass. One seed17 BP16→8→1/LR.1 case completed
12SGD/48rows. Existing BP candidate-gradient validation means24training
forward/backward evaluations96rows, plus12diagnostic forwards48rows: total36F/
144rows and24B/96rows.145parameters, fixture arrays6528B; total measured peakRSS
42,614,784B, parent.6850371s. Model/runtime/report copies are included in RSS,
not in fixture-array bytes.

All12observed batch accuracies were1.0; first A was1.0 before any update.
Cold-B accuracy was not measured: its first observation uses two updates.
BCE first.5742944534,
last.4361568362. This demonstrates temporal execution and repeated-pattern fitting;
there was no accuracy deficit/recovery event. No circadian benefit follows.

## Checks and extension

### Opt-in calibrated monitoring

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_sensor_drift_demo --describe --monitoring
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_sensor_drift_demo --monitoring --output new-monitored-sensor-result.json
```

This condition retains prearrival features/probabilities until labels arrive,
then applies the existing calibrated detector. Reference2/window2/persistence2,
prefix4/defaultmargin1e-6 fit only the first four stationary-sensor A batches.
The model may learn during that prefix. B/return data never refit thresholds and
diagnostics never feed learning or promotion. Calibration excludes all pre-shift
A batches from monitoring, so its false-alarm rate is null, not zero.

[Monitoring evidence](../artifacts/runs/r46-sensor-monitoring-20261017/RESULTS.md):
105focused/scoped statics and one reviewed12SGD48row qualification. Predictions,
scores and receipts exactly match the previous case. Feature alarms occur at
featureticks5..8 (known only when labels arrive at7..10). Returning A has one
continued alarm in four monitor batches/rate.25/no new onset, because its first
rolling window still includes B. This is not statistical false-positive control.

Recovery criterion is accuracy≥.75 for two consecutive batches after the first
deficit in B or returning A. Outputs distinguish no_deficit/recovered/censored;
labels_to_recovery counts only phase labels arrived strictly before the confirming
prediction. Labels used later to confirm its accuracy are excluded. Partial
horizons are explicit. In this actual case both phases have no_deficit/null
labels_to_recovery; synthetic tests exercise recovery/censoring without claiming
the learner recovered. Source/actor unchanged, no final/probe data.

Monitoring adds no native forwards. This new case costs36F144/24B96 including
12SGD48rows; peakRSS43,352,064B/parent.4994921s/output35,893B, pending features
1536B maximum. Detector fitting replays four metric summaries, without training.
Sensor total24SGD96rows across two engineering cases; all earlier results preserved.

New modules: `core/sensor_recovery.py` owns pure phase timing/recovery summaries;
`app/sensor_monitoring.py` composes existing calibration/incidents and reports
bounded scalar rows. They receive no learner or runtime authority. Failed
monitoring retains the already paid forecast/score and cumulative training charges.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_sensor_recovery.py tests/test_sensor_monitoring.py tests/test_sensor_monitoring_integration.py tests/test_sensor_stream.py tests/test_sensor_drift_demo.py tests/test_prearrival_observation.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check src/core/sensor_recovery.py src/app/sensor_monitoring.py src/adapters/sensor_drift_demo.py scripts/run_sensor_drift_demo.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check src/core/sensor_recovery.py src/app/sensor_monitoring.py src/adapters/sensor_drift_demo.py scripts/run_sensor_drift_demo.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL src/core/sensor_recovery.py src/app/sensor_monitoring.py src/adapters/sensor_drift_demo.py scripts/run_sensor_drift_demo.py
```

### Earlier prearrival checks

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_prearrival_observation.py tests/test_sensor_stream.py tests/test_sensor_drift_demo.py tests/test_attribution_checkpoint_observer.py tests/test_training_only_stream.py tests/test_coupled_attribution.py -k "not test_should_run_eight_native_updates_with_matched_pc_and_neutral_scores"
.venv/Scripts/python.exe -X utf8 -B -m ruff check src/core/sensor_stream.py src/app/coupled_attribution.py src/adapters/sensor_drift_demo.py scripts/run_sensor_drift_demo.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check src/core/sensor_stream.py src/app/coupled_attribution.py src/adapters/sensor_drift_demo.py scripts/run_sensor_drift_demo.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL src/core/sensor_stream.py src/app/coupled_attribution.py src/adapters/sensor_drift_demo.py scripts/run_sensor_drift_demo.py
```

### Scheduled finetuning control

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_sensor_drift_demo --describe --scheduled --monitoring
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_sensor_drift_demo --scheduled --monitoring --output new-scheduled-sensor-result.json
```

`adapters/scheduled_backprop_learner.py` owns the four-arrival buffer and full
native/pending/rate/counter snapshot. It updates ordinary mean-BCE BP once over
each16row block, with the same cold seed17/LR.1/48total training rows.12wake
receipts are separate from3actual optimizer steps. Nine buffered diagnostics have
definition `scheduled_bp_buffered_no_optimizer_v1`: their0placeholder is NOT
task loss. Only three diagnostics are native BCE observations. Failed native
attempts retain costs/pending data and stop; restore cannot revive that instance.

[Evidence and retained failures](../artifacts/runs/r46-scheduled-bp-20261017/RESULTS.md):
96focused/scoped statics and independently reviewed corrected qualification.
Initial qualification failed from a wrapper snapshot-port misuse before any
SGD; its2arrival charges/4forecasts remain spent. Fake diagnosis and a separately
prospective corrected case exercise actual two-buffer pause/resume, preserving
full state/authority, then3native steps48rows. No baseline rerun or hidden retry.

New case18F144/6B96 including mandatory candidate-gradient validation, buffer
peak2176B plus concat2176B transient; pending forecastfeatures1536B. Total observed
RSS42,852,352B/parent.6852172s. Fewer optimizer calls do not establish faster runtime
or compute equivalence. Previous BP uses12steps with the same training-row exposure.

All observed accuracies1.0/no_deficit; this scheduled case also observes B before
its first SGD. Lastprearrival BCE.5417501706 at2steps vsprior onlineBP.4361568362;
third scheduled step occurs after final forecast and is not evaluated. No recovery,
retention, superiority or circadian claim. Failed case remains failed; sensor
physical native total27SGD144rows, with failed buffered/diagnostic work additional.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_scheduled_backprop_learner.py tests/test_scheduled_sensor_demo.py tests/test_scheduled_pause.py tests/test_sensor_monitoring_integration.py tests/test_sensor_monitoring.py tests/test_sensor_drift_demo.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check src/adapters/scheduled_backprop_learner.py src/adapters/sensor_drift_demo.py scripts/run_sensor_drift_demo.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check src/adapters/scheduled_backprop_learner.py src/adapters/sensor_drift_demo.py scripts/run_sensor_drift_demo.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL src/adapters/scheduled_backprop_learner.py src/adapters/sensor_drift_demo.py scripts/run_sensor_drift_demo.py
```

### Arrived-only recent replay

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_sensor_drift_demo --describe --replay --monitoring
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_sensor_drift_demo --replay --monitoring --output new-replayed-sensor-result.json
```

This opt-in mode reuses StreamRehearsal/RecentStreamReplayPolicy: after each fourth
arrived wake, select up to four recent content-unique rows. Retain at most8rows/
1088numericbytes, with a separate12row replay allowance. Charges precede native
calls. Scheduled and replay controls are mutually exclusive; old modes unchanged.
No final/probe data or promotion. No new framework/module/dependency is needed.

The repeated fixture yields2/4/4selected rows at events4/8/12 and retains at most
four unique contents/544B across its two patterns and brightness conditions.
Existing helper runs one SGD per selected row: actual12wake steps48rows plus
10replay steps10rows =22SGD58exposures,56F164/44B116 with candidate validation and
diagnostics. Exposure/update/compute are not matched to plain or scheduled BP.
Root/helper reports distinguish all charges/attempts/returns/commits/receipts.
Failed replay preserves emitted pending forecasts without releasing their targets.

[Evidence](../artifacts/runs/r46-sensor-replay-20261017/RESULTS.md):85focused/scoped
statics and one reviewed native qualification. Selected rows were already arrived,
first6forecasts exactly match plainBP before first replay, source/actor unchanged.
PeakRSS43,610,112B/parent.5224799s/JSON50,106B; these are not timing comparisons.
Allaccuracy1/no_deficit; lastprearrival BCE.3545036833 vsplain.4361568362 at9wake+
6replay vs9wake steps. Final3wake+4replay steps are not evaluated. Extra-work training
fit does not establish recovery, heldout retention, generalization or CPC benefit.
Return-window alarm1/4 and calibration null-denominator remain reported.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_sensor_replay.py tests/test_sensor_monitoring_integration.py tests/test_sensor_monitoring.py tests/test_scheduled_sensor_demo.py tests/test_scheduled_pause.py tests/test_sensor_drift_demo.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check src/adapters/sensor_drift_demo.py scripts/run_sensor_drift_demo.py tests/test_sensor_replay.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check src/adapters/sensor_drift_demo.py scripts/run_sensor_drift_demo.py tests/test_sensor_replay.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL src/adapters/sensor_drift_demo.py scripts/run_sensor_drift_demo.py tests/test_sensor_replay.py
```

The replay increment's next action was existing PC/neutral/active CPC method
composition. That qualification is now complete in the following section;
the old coupled pilot factory's2→8→8→1/LR.03 remains fixed. Full R4.6 actual
recovery/retention and matched scientific results remain open; preserve all
negative results and the easy fixture.

### Existing PC and CPC methods

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_sensor_drift_demo --describe --method pc --monitoring
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_sensor_drift_demo --method neutral_cpc --monitoring --output new-neutral-sensor-result.json
```

Choose bp(default), pc, neutral_cpc or gated_cpc(active). Each ordinary method
uses the common seed17/16→8→1 parameters and LR.1. Existing PC/CPC use fixed4step/
.2 latent inference, precision1 and complete chemistry-bearing snapshots/forks.
Neutral uses matched_pc_control, active uses unchanged default chemistry. The
older2→8→8→1/LR.03 pilot stays fixed. Scheduled/replay controls currently require
BP; unsupported combinations fail before work. All six old BP descriptors match.

PC/CPC train against coupled energy, not the recorded prequential BCE. Each
12update48row qualification actually performs72coupled energy/gradient evaluations
288rows,12training latent initializations48rows and12diagnostic forwards48rows.
Matched update/exposure counts do not imply compute equivalence to BP. Existing
`charged_training_updates` counts successfully completed wake receipts; failed
native attempts are separate physical work, retained by the qualification wrapper.
Failure output preserves every already-paid targetless pending forecast.

[Checks/results/costs](../artifacts/runs/r46-sensor-methods-20261017/RESULTS.md):
150focused/scoped statics pass; one native case per new method completes. Neutral
matches all12PC parameter/energy updates exactly while chemistry evolves. Active
gates evolve below1. Allaccuracy1/no_deficit; tick11BCE PC/neutral.49834125,
active.50061972 versus retained BP.43615684, all at9ordinary updates. Three final
updates per arm remain unscored. No recovery/retention/CPCadvantage/speed claim.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_sensor_learners.py tests/test_sensor_methods.py tests/test_sensor_replay.py tests/test_sensor_monitoring_integration.py tests/test_sensor_monitoring.py tests/test_scheduled_sensor_demo.py tests/test_scheduled_pause.py tests/test_sensor_drift_demo.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check src/adapters/sensor_learners.py src/adapters/sensor_drift_demo.py scripts/run_sensor_drift_demo.py tests/test_sensor_learners.py tests/test_sensor_methods.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check src/adapters/sensor_learners.py src/adapters/sensor_drift_demo.py scripts/run_sensor_drift_demo.py tests/test_sensor_learners.py tests/test_sensor_methods.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/adapters/sensor_learners.py src/adapters/sensor_drift_demo.py scripts/run_sensor_drift_demo.py tests/test_sensor_learners.py tests/test_sensor_methods.py
```

The method increment's next step, isolated checkpoint learnability, is qualified
below. Old R2.1 synthetic studies and all negative results remain preserved.

### Isolated fresh-task probes and activation health

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_sensor_drift_demo --describe --method pc --monitoring --learnability
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_sensor_drift_demo --method bp --monitoring --learnability --output new-sensor-probe-result.json
```

Ordinary BP/PC/neutral/active methods support --learnability; scheduled/replay
combinations fail before work. At wake0/4/8/12, existing run_fresh_task_probe
restores a detached full checkpoint on a disposable fork of an independent
template. It never updates the benchmark or actor. The reserved C task uses
sixteen uniform(-1,1) features with target x0*x1>=0, four8row probe_train batches
(RNG17031),32probe_test rows(17033),32label-free reference rows(17035). IDs/storage
are disjoint. Probe_test is used only for fixed scores at0/1/2/4; no score selects
training, a checkpoint, promotion, rate or horizon. Final benchmark evaluation
remains absent. Four distinct4update sessions are born at setup against the
original deadline/RSS sampler. No checkpoint renews authority.

The observer checks its template/policy, supplied checkpoint/arrived metadata
and reserved arrays unchanged. Reports retain four bounded slots, all paid partial
health/probe work and targetless pending forecasts on failure. Health uses existing
active/dead/saturation and centered singular-value entropy rank definitions;
importance/age are unavailable. Invocation timing includes preflight and copies.

[Evidence and exact commands](../artifacts/runs/r21-sensor-learnability-20261017/RESULTS.md):
193focused/scoped statics and four independently reviewed native cases pass.
Each method costs12benchmark updates48rows +16probe updates128rows,28prediction
forwards560rows,4health forwards128rows and4SVDs. BP56gradient evals352rows;
PC/CPC168coupled evals1056rows+28training initializers176rows. BP feedforward work
inside its gradient kernel is included there. Original reserved arrays12,800B,
snapshot arrays1,160B BP/PC or1,288B CPC, health activations2,048B/checkpoint;
RSS/time also cover signatures, copies, templates and reports. PeakRSS≤44,118,016B.

All benchmark forecasts/receipts/monitoring exactly match retained per-method
outputs. All16probe BCE gains are positive (.00249..00504), but all four cold
accuracy endpoints fall .5625→.53125. Neutral matches PC; active gains are smaller
than PC at every checkpoint. Its slightly lower return endpoint starts from a
lower initial BCE. All hidden units active, none dead/saturated, rank7.1288..7.1398.
These observations establish no CPC advantage, causal plasticity loss or general
novel-task learning. Benchmark accuracy1/no_deficit still leaves actual recovery
and, at this probe-only milestone, heldout old-context retention was unmeasured.
All curves/negative metrics are retained; the heldout increment follows below.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_sensor_learnability.py tests/test_sensor_probe_integration.py tests/test_sensor_learners.py tests/test_sensor_methods.py tests/test_sensor_replay.py tests/test_sensor_monitoring_integration.py tests/test_sensor_monitoring.py tests/test_scheduled_sensor_demo.py tests/test_scheduled_pause.py tests/test_sensor_drift_demo.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check src/adapters/sensor_learnability.py src/adapters/sensor_drift_demo.py scripts/run_sensor_drift_demo.py tests/test_sensor_learnability.py tests/test_sensor_probe_integration.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check src/adapters/sensor_learnability.py src/adapters/sensor_drift_demo.py scripts/run_sensor_drift_demo.py tests/test_sensor_learnability.py tests/test_sensor_probe_integration.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/adapters/sensor_learnability.py src/adapters/sensor_drift_demo.py scripts/run_sensor_drift_demo.py tests/test_sensor_learnability.py tests/test_sensor_probe_integration.py
```

The next action at the probe milestone was an opt-in heldout A/B readout through
existing AttributionStream.finish. This is now implemented and qualified below;
the frozen fixture, recipes, results and final-label isolation remain preserved.

### Heldout context readout

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_sensor_drift_demo --describe --heldout --learnability --monitoring
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_sensor_drift_demo --heldout --learnability --monitoring --output new-heldout-sensor-result.json
```

`--heldout` composes with all ordinary methods and the existing BP scheduled or
replay control. Without the flag, existing descriptors and call behavior remain
unchanged. The final A set has 32 balanced patterns with fixed seed27031 uniform
pixel noise in [-.025,.025]; B is paired A+.25 with independently owned arrays,
labels and IDs. Labels follow the unchanged right-half versus left-half mean
rule. Numeric arrays occupy8,704bytes; total RSS also includes native state,
snapshots, progress, copies and serialization. This narrow same-rule fixture is
not a new task or a broad robustness test.

After all 12 wake arrivals and requested C probes finish, the existing finish
scorer evaluates saved checkpoints0..12 on both final sets:26prediction forwards/
832rows per case. Final labels never reach training, promotion, tuning, observers
or probe selection. Summary endpoints are fixed at0/4/8/12. For BCE, initial A
gain=A0-A4; new B gain=B4-B8; old A loss=A8-A4; return A gain=A8-A12; return gap=
A12-A4. Accuracy reverses these signs. Positive gain is better; positive loss/gap
is worse. Preserve negative values. The summary validates all13points before
selecting endpoints.

`heldout_evaluation` records status/reason, attempted and returned calls/rows,
per-checkpoint partial metrics, original training receipts and final array bytes.
An invalid return, model mutation, incomplete requested probe or resource failure
refuses completion. Later failures preserve paid valid metrics; retention remains
null unless the evaluation completed. A completed checkpoint inside a failed run
is partial evidence, not an accepted full retention summary. Training failure can
leave scoring not_run. No budget or clock is renewed.

[Executed commands/native costs](../artifacts/runs/r46-sensor-heldout-20261017/RESULTS.md)
and [independent review](../artifacts/runs/r46-sensor-heldout-20261017/REVIEW.md):
375 unique focused passes, scoped Ruff and mypywin32/linux,26CLI previews with
20old descriptors exact, and six native cases pass. All156heldout accuracies are
1.0, including initialization. Every old-A BCE loss and return gap is negative;
there is further fitting without a demonstrated forgetting deficit or recovery.
PC and neutral trajectories match exactly; active CPC has no advantage. Replay
uses10extra updates/rows; scheduled makes3optimizer steps. Neither comparison is
equal-compute. Keep BP default and all prior cold-probe accuracy losses.

Six cases spend137native updates810rows including64isolated probe updates512rows;
156final forwards4992rows are additional evaluation work. All original caps and
SPENT markers remain. No sweep or further model execution is implied by examples.
The RESULTS JSON links preserve exact commands for the15-module375-test run,
Ruff format/check/format-check and scoped mypy on both target platforms; Linux
runtime, full suite, clean clone and release gates were not run.

Next R4.7: show these saved reports through the existing shared offline inspector,
with source JSON pointers, nulls/failures/inactive controls and optional actual
resource evidence. Missing measured costs stay unknown. Full R4.6 scientific
robustness, actual deficit/recovery and statistical false-positive rate remain
open; this increment does not justify tuning another fixture to produce a win.

### Incident-gated replay

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_sensor_drift_demo --describe --replay --monitoring --incident-replay --heldout
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_sensor_drift_demo --replay --monitoring --incident-replay --heldout --output new-incident-sensor-result.json
```

Requires explicit replay and monitoring, BP, no scheduled/learnability mode.
Feature channel is fixed for this physical brightness example. Existing periodic
mode remains the default. Declared training/replay counts in this new conditional
mode are upper bounds; actual attempts, returns, charges and skipped slots are
recorded separately. Final data is unavailable until all training/observations
complete; failed replay leaves final status not_run and zero final attempts.

```text
src/core/incident_replay.py      scalar temporal/transition gate
src/app/stream_rehearsal.py      optional admission +original costs/owned continuation
src/app/coupled_attribution.py  original-state/pre-wake progress check
src/adapters/sensor_drift_demo.py arrived incident composition, no final feedback
tests/test_incident_replay.py   pure temporal/bounded/copy/corruption behavior
tests/test_incident_rehearsal.py real stream with fake native work/failure/continuation
```

Why: existing callbacks observe incidentN after rehearsalN, so it can only affect
a later declared boundary. Every wake must be observed once. The gate validates
all channel transitions but admits only its birth-selected channel. One still-
active pending onset is consumed before returning admission; continued levels
cannot repeat it and clear cancels it. Skipped event retains null receipt and
zero charges/attempts/returns/commits. Native failure does not refund onset or
charges. No replay budget is renewed. The helper validates actual wake receipt;
stream refuses a missing prior observation before another paid wake. Gate data
is copied in the existing local pause token; replaced/corrupt state is refused.

437focused/scoped Ruff+mypywin32/linux checks pass;28CLI previews include26exact
old descriptors. Independent SOURCE+GO+ACTUAL accepted one16SGD52row qualification:
12wake48+4replay4 at boundary8, skip4/12. Onset6 observed through7 admits8, then
clear10/no new return onset. Original replayallowance12 spends4; actualretention
544numericB.26finalF832rows afteralltraining; total38predictionF880 plus32BPgradient
104rows. Allattempts/returns match and source/actor remain unchanged.

All heldout/prearrival accuracies1.0/no deficit. FinalA/B BCE.3451347078/.3586499753;
negativeoldA_loss/returngap mean furtherfit without forgetting/recovery. Four extra
replaycalls/rows versusplain or ten in periodic are different exposure/computation,
not matched benefit. Stationary monitoring denominator is0/ratesnull. Keep the
known feature noise/carryover, all negative results and BP/periodic defaults.

[Exact commands/checks/artifacts](../artifacts/runs/r27-incident-replay-20261017/RESULTS.md)
include13module pytest command, eight-path Ruff/mypy commands, previews and native
wrapper. The recorded parent memory sampler observed the Windows launcher only;
actualtrainerRSS49.95MB was measured/enforced internally. Combined process RSS and
external descendant preemption remain untested; see [RESOURCE_NOTE](../artifacts/runs/r27-incident-replay-20261017/RESOURCE_NOTE.md).
All native allowance SPENT; these examples do not renew it. Future native study
requires its own prospective protocol and actual trainer supervision. Fullsuite/
cleanclone/CI/Linuxruntime/GPU/statisticalFPR/recovery/scientific benefit remain open.
