# Arrived-only shift observations

The existing stream runner can report three bounded signals for each arrived
training batch, using a copied parameter state from the preceding update:

```powershell
.venv/Scripts/python.exe -m scripts.run_shift_signal_pilot --describe
.venv/Scripts/python.exe -m scripts.run_shift_signal_pilot --run
```

This is a fixed local diagnostic protocol: all four matched BP/PC/neutral-CPC/
active-CPC methods, stationary and recurring A/B/A streams, seed23, 18 updates
per arm/stream. It has no seed, threshold or rate tuning options. Signals stay
separate from training, promotion and probe selection in the original modes.
The explicit incident-replay mode below permits bounded training feedback.
Final evaluation remains
after all observations and training in each run.

## Meaning and availability

The callback runs **after update k**. It predicts the newly arrived training batch
with copied parameters from k−1, retained at the preceding callback. These are
pre-update parameters with retrospective computation; the report does not claim
a live actor prediction or a trigger available before update k. Labels have
already arrived. Predictor callbacks receive copied state/features and no labels.

| Channel | Per-batch measurement | Compared score | Fixed threshold |
| --- | --- | --- | ---: |
| Uncertainty | Mean binary entropy divided by log(2) | Window mean minus reference mean | 0.05 |
| Feature | Raw input column means | Mean absolute difference of window/reference column means | 0.25 |
| Residual | Mean absolute probability error | Window mean minus reference mean | 0.05 |

Uncertainty and input-feature means are target-free calculations. The residual
requires arrived labels; it is separate from label-conditioned latent energy.
Inputs are finite float64 matrices, at most128 rows/16 columns in[-1,1]; targets
and predictions have shape(B,1) and values in[0,1]. Normalized entropy is bounded
to[0,1] for numerical roundoff, with exact zero at probability endpoints.

The first three batches form a frozen reference. The next two post-reference
batches fill a moving window. Each batch has equal weight regardless of row
count. A channel alarms after its score strictly exceeds its threshold in two
consecutive windows. A nonexceedance resets that channel; sustained exceedance
keeps the alarm active. Overlapping windows and persistence are descriptive,
not independent statistical evidence or calibrated false-positive control.

Mean-feature drift cannot detect every distribution change. In the current
concept-only A/B/A stream, input distribution is unchanged, so it may correctly
miss the changed label rule. Predictor uncertainty and residuals can also miss
changes or respond to ordinary sampling/model variation. Report misses and
stationary false alerts without changing thresholds to obtain a desired result.

## Ownership, failures and bounds

Core stores only a bounded reference and window, at most32 batch summaries each,
with at most512 observations. The app requires consecutive update counts and
exact training roles. It retains one copied previous state, bounded row IDs, and
at most31 planned report rows. A current-state copy becomes the next previous
state only after a successful observation. Predictions receive a separate feature
copy so a mutating callback cannot alter the measured input means.

Invalid roles/counts/batch metadata refuse before state copies or predictions.
An admitted copy/prediction/measurement failure preserves its failed row, stops
the observer and propagates to the stream; all unreached rows remain present.
Reports contain detached measurements/decisions, no opaque state or model arrays.
This assumes the existing trusted owned-snapshot contract, not adversarial
serialization or concurrent mutation of borrowed inputs.

Timing covers accepted observation validation, copies, prediction and detector
work. Priming time is separate; total duration sums known measured windows.
Constructor/serialization work stays in whole-pilot/outer orchestration time.
Attempted and returned predictions are counted separately; returned invalid
predictions remain completed calls followed by a failed measurement.

The fixed pilot allows144 training updates/1152 rows,144 observation predictions,
304 final scoring forwards and144 detector steps. Its60-second, single-BLAS-
thread,256MiB process RSS and32MiB output caps are independent of algorithm
benefit. No probe training, replay, sleep, promotion, download or GPU work.

## Structure and extension

```text
src/core/shift_signals.py               bounded pure measurements and persistence
src/core/shift_calibration.py           bounded prefix fit and reachability
src/app/arrived_shift_observer.py       copied-state arrival/ownership boundary
src/adapters/shift_signal_pilot.py      fixed native binding and stream composition
scripts/run_shift_signal_pilot.py       describe/run JSON boundary
tests/test_shift_signals.py
tests/test_arrived_shift_observer.py
tests/test_shift_signal_pilot.py
tests/test_shift_calibration.py
tests/test_calibrated_shift_observer.py
tests/test_shift_calibration_pilot.py
tests/fixtures/shift_signal_protocol_v1.json
```

### Optional training-prefix calibration

```powershell
.venv/Scripts/python.exe -m scripts.run_shift_signal_pilot --describe --calibrated
.venv/Scripts/python.exe -m scripts.run_shift_signal_pilot --run --calibrated
```

This selects the separate `arrived_only_prefix_calibration_v1` protocol. The
original default descriptor/rule and retained negative result remain unchanged.
All methods use the same six-batch declared stationary training prefix, three-
batch reference, two-batch window and fixed margin1e-6. Two full prefix windows
fit each threshold to `max(0, prefix window scores) + margin`, capped only at
the channel domain upper bound (1,2,1). No alarms/streaks are emitted during
calibration. Thresholds freeze at update6 before monitoring7..18; there is no
refit on monitoring data and no signal feedback into training.

Core `shift_calibration` composes the existing detector. At fit it replays the
six scalar summaries once into a fresh detector, retaining the exact original
reference/window/update count and zero prefix streaks, then discards the prefix.
The replay costs48 additional detector steps across eight runs, not additional
model predictions, labels or training. Totals remain144 learner updates/1152
training rows/144 observation predictions; core detector steps become192.
Calibration uses48 batches/384 rows already included in those totals.

The fitted result reports uncertainty/residual ceilings `1-reference_mean` and
feature ceiling `mean(1+abs(reference_feature_mean))`. A strict-exceedance rule
is algebraically reachable only if its threshold is strictly below its ceiling.
Unreachable channels are explicitly reported; their thresholds are not lowered
to force reachability. Algebraic bounds do not prove the learner can generate
the corresponding predictions or that a shift will be detected.

The app adds detached calibration configuration/result and arrived row IDs.
Its required model identity is caller-declared method/stream/seed, with each
row's k−1 parameter-update index; it is not a parameter fingerprint or authority
check. Predictions evolve as the benchmark learns, so this is calibration of an
evolving previous-state population, not repeated evaluation of a fixed model.
Only arrived training data enter fitting. Prefix stationarity is a declared
experiment assumption, not something final evaluation verifies. Timing includes
fit/replay; final scoring still follows all observations.

This outcome-motivated exploratory rule has only two overlapping calibration
windows and supplies no calibrated false-positive guarantee. Any comparative
execution requires its separately frozen/reviewed local protocol. Full R2.7
requires useful discrimination, not merely reachable thresholds.

The native adapter supports exact matched BP/coupled snapshot parameters for
feedforward prediction; it does not attest complete model validity. Core knows
no model or data roles. The app receives only copied checkpoints and arrived
training metadata, never benchmark final data. The existing runner and learner
updates are unchanged. See ADR-0308.

For focused verification:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_shift_signals.py tests/test_arrived_shift_observer.py tests/test_shift_signal_pilot.py tests/test_attribution_checkpoint_observer.py
```

Keep thresholds, data, metrics and comparison directions frozen for a declared
run. A separate reviewed policy protocol is required before signals choose
training or promotion. Full R2.7 requires useful noise/shift discrimination;
implementation correctness alone does not establish deployable detection.

## Executed result and remaining work

The sole frozen v1 pilot completed all144 observations. All119 focused cases,
scoped static checks and independent source/actual review pass. Every channel
has zero alarms: no stationary false alerts, but all B and returning-A changes
are missed. The uncertainty reference is about0.9844–0.9847; since normalized
entropy is at most1, positive change cannot reach the fixed0.05 threshold.
This is an ineffective fixed rule, not demonstrated detection. The protocol and
negative result remain unchanged; full R2.7 is open.

The bounded training-prefix follow-up above is now implemented and separately
executed. Its187 focused cases and scoped statics pass. In its sole frozen run,
all channels are algebraically reachable, residuals alarm at8..13 only in the
recurring streams, feature alarms match stationary false alerts exactly, and
uncertainty has no alarms. Returning-A alarm13 carries the B streak, so it is
not fresh return detection. No signal controls training; full R2.7 remains open.

See the [original fixed-rule result](reports/arrived-shift-observations-20261016.md)
and [calibration result, costs and limits](reports/shift-prefix-calibration-20261016.md).
The later retention and capacity work is accepted in the roadmap. The remaining
detection limitations above still hold; no new threshold has been selected.

## Optional incident transitions

```powershell
.venv/Scripts/python.exe -m scripts.run_shift_signal_pilot --describe --calibrated --incidents
```

The existing `--run` action also accepts `--incidents`, with or without
`--calibrated`. Declare a local experiment envelope before another native run.
The accepted verification here uses fake learner integration and all 144 saved
observations from the original calibrated experiment; it does not rerun it.

`ArrivedShiftObserver(..., incidents=True)` adds an `incident` record to each
completed observation, and `None` to planned/unreached slots. Defaults retain
their original report schema. Channel order is uncertainty, feature, residual.
The pure `ShiftIncidentTracker.observe(ShiftDecision)` returns the batch index,
three states and three onset indices:

| Previous alarm | Current alarm | State | Recorded onset |
|---|---|---|---|
| false | false | inactive | none |
| false | true | onset | current batch |
| true | true | continued | original onset |
| true | false | cleared | original onset, released after this record |

Why this: persistence produces alarm levels, so successive alarm batches are not
successive new triggers. Clearing happens at the first non-alarm; no extra
threshold, cooldown or context-boundary reset is added. Reference/window/
calibration rows remain inactive. Inputs must be exact valid decisions with
consecutive indices1..512; after monitoring starts, prefix-phase regressions are
refused before mutation. The tracker retains only a count, monitoring flag and
three active onset indices. Its output records retain no model/data/history.

The observer records the decision and incident before its existing finish-clock
boundary, so admitted later failures retain that partial evidence and stop the
observer. Report copies are detached. Tracking/serialization stays inside the
existing measured observer duration; no additional predictions or updates are
performed. A full eight-arm run declares144 incident steps/432 channel
transitions in addition to the unchanged native work. Python object memory is
not estimated from scalar counts or called a measured byte saving.

The adapter preserves `signal_outcomes` as historical alarm-level annotations.
New `incident_outcomes` reports onset/continuation/clear lists, stationary false
incidents and onset-only B/return delays. Context labels annotate completed
reports; the tracker never receives them. In the original modes, no signal feeds training, replay,
promotion, calibration or probe selection. Final scoring remains after all
observations; probes/deployed state remain separate.

Saved-data reclassification confirms, in every method:

- Uncertainty: no incidents.
- Feature: onsets8,11 and clears9,16 in both stationary and recurring streams;
  the stationary incidents are false alarms.
- Recurring residual: onset8, continued9..13, clear14; no stationary incident.
  The old return alarm13 is continuation, with no new return onset.

Full R2.7 remains open: the new contract improves interpretation, not detection
efficacy. See [commands, failures and independent review](../artifacts/runs/r27-shift-incidents-20261016/RESULTS.md).

Focused checks for this addition:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_shift_incidents.py tests/test_shift_incident_integration.py
.venv/Scripts/python.exe -m ruff check src/core/shift_incidents.py src/app/arrived_shift_observer.py src/adapters/shift_signal_pilot.py scripts/run_shift_signal_pilot.py tests/test_shift_incidents.py tests/test_shift_incident_integration.py
.venv/Scripts/python.exe -m ruff format --check src/core/shift_incidents.py src/app/arrived_shift_observer.py src/adapters/shift_signal_pilot.py scripts/run_shift_signal_pilot.py tests/test_shift_incidents.py tests/test_shift_incident_integration.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/core/shift_incidents.py src/app/arrived_shift_observer.py src/adapters/shift_signal_pilot.py scripts/run_shift_signal_pilot.py tests/test_shift_incidents.py tests/test_shift_incident_integration.py
```

Repeat the last check with `--platform linux` for that target. Add future
consumers through the existing observer boundary and separately specify any
policy feedback; observation correctness does not authorize training triggers.

## Optional incident replay

```powershell
.venv/Scripts/python.exe -m scripts.run_shift_signal_pilot --describe --calibrated --incidents --incident-replay
.venv/Scripts/python.exe -m scripts.run_shift_signal_pilot --run --calibrated --incidents --incident-replay
```

This fixed opt-in routes completed arrived incidents into the existing rehearsal
helper. Only a prior active feature onset can admit a later boundary4/8/12/16;
clear cancels pending work and continuation cannot trigger repeatedly. Four
one-row updates per case are the total replay allowance, separately charged before
native work. Retention is eight arrived rows/192numeric bytes. Original clocks,
global deadline, learner recipes, data, seed and evaluation isolation stay in force.

`observations.rows[].prediction_replay_updates` names the preceding snapshot's
committed replay updates. Thus a replay at12 first affects diagnostic prediction13.
The observer remains a passive computation; the adapter explicitly reports policy
feedback. Reports retain helper admissions, skipped events, attempted/returned/
committed replay and partial final scores. Aggregate `training_costs_complete` is
false on incomplete cases; completed wakes plus charged replay cannot reveal
unobserved work inside a failed wake. Final scores never select policy or probes.

The saved eight-case execution spent176native updates/1184rows. Every stationary
and recurring case admitted12 from onset11, including costly false stationary
incidents. PC-neutral saved behavior matched exactly. Recurring replay improved
new-B gain but increased old-A retention loss in all methods; active CPC did not
establish an advantage. Comparisons have extra replay exposure, not equal compute.
The wrapper failed exporting its full tensor trace after all cases finished:
**native qualification incomplete, spent, no retry**. Reuse
[saved results and limitations](../artifacts/runs/r27-shift-incident-feedback-20261017/RESULTS.md)
instead of repeating it. Full R2.7 efficacy and repository readiness remain open.

## Isolated learnability after replay

```powershell
.venv/Scripts/python.exe -m scripts.run_shift_signal_pilot --describe --calibrated --incidents --incident-replay --learnability
.venv/Scripts/python.exe -m scripts.run_shift_signal_pilot --run --calibrated --incidents --incident-replay --learnability
```

Recurring cases reuse the existing C probe at benchmark checkpoints0/6/12/18,
with fixed probe points0/1/2/4/8/16. C uses two inputs and target x0*x1>=0, original
reserved data and an independent template. Four distinct16-update sessions are
prepared before learner work and retain their clock, quota, sampler and shared
deadline. Disposable probe forks cannot train the benchmark/deployed learner.
Stationary cases remain monitoring/replay controls without additional probes.

Each `learnability` slot contains initial C BCE, signed `fresh_gain`, full
accuracy/BCE curve, activity/dead/saturation/effective-rank diagnostics and
reserved/snapshot array bytes. Importance/age remain unavailable on these ordinary
models. Row/top costs distinguish probe attempts and commits, original session
counts, score calls, diagnostic calls/time and unknown failed delegate work.
Nested byte counts cover arrays, with full memory/time measured separately.
Final benchmark readouts follow all selected probes and observations; their
scores cannot select probes, thresholds or training policy.

PC-neutral learning/probe semantics must match. Full snapshot sizes must instead
be exactly840B for PC and968B for neutral CPC, whose additional128B chemistry/
plasticity state remains accounted. Any other size or counter/health mismatch
fails. This explicit check fixes a gate that incorrectly demanded equal storage.

The saved run completed432native updates/3232rows, including256isolated probe
updates. All benchmark results exactly match the retained no-probe replay run.
BP C gain is positive at all four checkpoints; PC/neutral/active CPC have negative
16-update gains at the three warm checkpoints. All reported units remain active,
without saturation: this does not establish a causal explanation or CPC benefit.
The original native command failed its old storage-equality gate; corrected source
and offline saved-data verification pass, with no retraining or rewriting the
failed run. [Results, failures and exact commands](../artifacts/runs/r21-incident-learnability-20261017/RESULTS.md).
