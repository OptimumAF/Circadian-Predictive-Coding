# ADR-0364: Observe current shadow predictions before label arrival

## Context

R4.6 requires predict-before-update evaluation with delayed corrections. Existing
ArrivedShiftObserver runs retrospectively after update k using copied k−1 state.
It cannot establish that a prediction preceded the corresponding label arrival.

## Decision

Add an optional feature-only callback immediately before the existing stream's
record-experience/record-label tick. Supply detached current CandidateState and
only tick, row IDs and feature arrays. Hold the existing candidate lease, compare
state/ownership afterward, and recheck original budget authority before recording
labels. Preserve defaults and existing retrospective callbacks. Count attempted/
completed callbacks separately from native prediction work; keep original costs
and counters across owner-local continuation. No prediction result drives training.

Why this: the ordinary stream already owns chronology, labels, updates and pauses;
a small callback provides the missing observation without another orchestrator.

## Alternatives

Retrospective prediction does not prove prearrival order. Predicting through the
stable actor would observe a frozen model and misrepresent shadow learning.

## Consequences

Observers can mutate their owned copies, but cannot change actual learner/actor/
candidate state or authority and then admit more training. Callbacks are trusted
local code, not a Python sandbox or data-provenance guarantee. Native inference
cost must be recorded by the observer. The new BP sensor CLI qualifies this path;
calibration, recovery, matched comparisons and deployment remain separate gates.

## Extension — prefix monitoring, 2026-10-17

An explicit --monitoring condition retains bounded prearrival pixel copies,
joins original forecasts to arrived labels, and only then fits/observes the
existing calibrated detector. Pure recovery timing counts phase labels received
strictly before the confirming prediction; no deficit is not zero-label recovery.
Reference2/window2 fits the existing four-A prefix (default3+2 would needfive),
with persistence2/defaultmargin fixed prospectively. Report return-window alarm
carryover and exclude calibration from false-alarm denominators. No feedback,
refitting on shift/return or native prediction is added by the monitor. Append
paid forecast/score before diagnostic work so failures preserve prior evidence.
Scheduled-finetuning/replay/matched comparisons and actual recovery remain open.

## Extension — scheduled BP baseline, 2026-10-17

The explicit --scheduled condition accumulates four arrived sensor batches in an
owned native adapter, then applies the existing16row mean-BCE BP update. Keep
12arrival receipts distinct from3optimizer steps and tag buffering as no observed
taskloss. Full native/pending/rate/counters survive fork/restore; failed attempts
remain stopped. Why: a simple lower-frequency control tests whether repeated
small updates are needed without a new stream/scheduler or changing data/seed/LR.
This is not update-matched; record buffer/compute costs explicitly. Replay/matched
PC/CPC and actual recovery remain separate requirements. First qualification's
wrapper snapshot-port error stays FAILED/SPENT; corrected case preserves the
original authority and complete buffer without changing the learner/protocol.

## Extension — arrived-only recent replay, 2026-10-17

Compose unchanged StreamRehearsal/RecentStreamReplayPolicy through explicit
--replay in the existing sensor adapter/CLI. The helper observes only committed
arrived wake batches and content-deduplicates them. Each retained selection row
is one native update; this fixture therefore yields2/4/4 calls, not three batch
updates. Use a separate original exposure allowance and the shared monotonic
function/RSS sampler; never renew sessions at events. Record retention numeric
bytes separately from total RSS. Why: reuse the implemented stream policy and
account for its actual work instead of changing semantics to match an expected
call count. Report preceding replay clocks on forecasts, and retain targetless
pending forecasts when a paid replay fails before the score callback. This
engineering qualification does not establish compute-matched benefit, recovery
or circadian advantage; matched methods and scientific criteria remain open.

## Extension — existing sensor native methods, 2026-10-17

Use a small sensor composition module for common16→8→1 seed17 parameters/LR.1
instead of changing the old2→8→8→1/LR.03 factory. Reuse existing CoupledLearner
and native PC/CPC, fixed4step/.2 inference/precision1. Full snapshots preserve
chemistry/policy; prearrival prediction checks PC/CPC state kind and CPC wake
clock. Explicit --method leaves all old BP descriptors unchanged; scheduled/
replay combinations remain BP-only pending separate qualification. Why: native
algorithms already exist; composition and snapshot admission were the concrete
gaps. Count6coupled evaluations/update plus training initialization and diagnostic
forward separately, and score prequential BCE independently from training energy.
Neutral must match the complete PC parameter trajectory, while real chemistry
still evolves. Preserve completed wake receipt semantics separately from failed
native attempts; external qualification records attempts and SPENT on failure.
No favorable-data/rate selection, recovery or comparative scientific claim.

## Extension — isolated sensor learnability, 2026-10-17

Compose existing run_fresh_task_probe and diagnose_activations at wake0/4/8/12
through explicit --learnability. An independent template creates disposable
forks restored from copied full checkpoints; separate reserved probe_train and
probe_test arrays cannot feed benchmark learning or selection. Four4update probe
sessions are allocated at setup against the original deadline and RSS sampler;
no checkpoint renews authority. Preserve template/policy/checkpoint/metadata/data
guards, bounded partial slots and all paid costs. Why: remaining learning capacity
requires a separate learning curve; the easy sensor fixture has no accuracy deficit.
Reuse accepted probe machinery instead of adding another framework or changing
the fixture to force a result. Fixed0/1/2/4 curves and initial/final metrics remain
visible even when accuracy worsens despite lower BCE. Heldout context retention,
actual recovery and broad comparative claims remain separate open requirements.

## Extension — finish-only sensor context readout, 2026-10-17

Use opt-in `--heldout` and the existing stream finish scorer after all training
and requested probes complete. Core owns fixed paired noisy A/B data and pure
signed summary validation; final-test labels never enter updates or observers.
Keep all 13 checkpoints and frozen endpoints 0/4/8/12. Why: the C learning curve
alone cannot measure old-context retention, while an existing isolated evaluation
boundary already supplies scoring. Add optional fresh progress with attempts,
returns and partial metrics using existing native-state checks; refuse invalid
returned values/mutations, and retain paid evidence if a later guard fails.
Scheduled BP admits exactly four or 32 prediction rows without changing training
or adding forwards. No budget renewal or new runner. Negative loss/gap values
stay signed; this easy fixture establishes neither forgetting nor recovery, and
one seed establishes no circadian or broad robustness benefit.
