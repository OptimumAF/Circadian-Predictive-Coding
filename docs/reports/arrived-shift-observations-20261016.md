# R2.7 arrived-only shift observations, 2026-10-16

## Working behavior and evidence status

The existing matched stream now supports bounded uncertainty, raw input-mean
drift and delayed supervised residual observations. Each observation runs after
update k using copied parameters from k−1 and the newly arrived training batch.
The predictor receives no labels. Measurements do not control training,
promotion or probe selection; benchmark final scoring follows all observations.
This is retrospective computation with previous parameters, not a live signal
available before update k.

The core measurement/persistence implementation, copied-state app observer and
fixed native CLI are implemented, executed and independently accepted within
their scoped contracts. All 119 focused tests pass; scoped Ruff, format and
Windows/Linux mypy checks pass. The sole prospectively frozen native pilot
completed all eight runs and 144 observations. This does **not** establish useful
shift detection: every recurring change was missed. Full R2.7 remains unchecked.

Why this increment: learnability/retention/return probes already exist, including
the preserved negative four- and sixteen-update findings. Observable training
signals are a concrete prerequisite for a future adaptation policy. Reusing the
stream observer supplies this boundary without another orchestration framework.

## Frozen exploratory protocol

`arrived_only_shift_observations_v1`: seed23, the existing matched BP, PC, neutral
CPC and active CPC settings, 18 updates per run, eight rows per batch, two-batch
label delay. Stationary runs train A throughout; recurring runs train A/B/A in
six-update phases. All 19 states are scored only after training/observations.
No learner, rate, seed, baseline or metric changed. This is one fixed-data
engineering observation, not independent confirmation.

The first three arrived batches freeze the reference. A two-batch post-reference
window produces batch-weighted scores; two consecutive strict exceedances are
required for an alarm. Thresholds are uncertainty-change 0.05, mean absolute
input-mean drift 0.25, residual-change 0.05. A nonexceedance resets the channel.
Uncertainty is normalized binary entropy; residual is mean absolute probability
error and uses labels only after arrival. No label-conditioned latent diagnostic
enters the detector. Overlapping windows do not give independent evidence or a
calibrated false-alarm rate.

## Negative detection result

| Method | Stationary alarms, every channel | First B alarm, every channel | Returning A alarm, every channel |
| --- | ---: | --- | --- |
| BP | 0 | missed | missed |
| PC | 0 | missed | missed |
| Neutral CPC | 0 | missed | missed |
| Active CPC | 0 | missed | missed |

All eight runs have zero alarms in all three channels. Detection delays remain
null. Zero stationary false alerts together with complete misses does not
demonstrate useful discrimination or absence of a shift. Phase annotations are
added after execution; even a phase alert could inherit an earlier streak and
would not establish causal detection.

The uncertainty rule has a stronger limitation. Normalized entropy is at most
one, so `uncertainty_change <= 1 - reference_uncertainty`. The references are
identical across the two streams:

| Method | Reference entropy | Maximum possible positive change | Fixed threshold |
| --- | ---: | ---: | ---: |
| BP | 0.9846782641 | 0.0153217359 | 0.05 |
| PC / neutral CPC | 0.9843920271 | 0.0156079729 | 0.05 |
| Active CPC | 0.9843924149 | 0.0156075851 | 0.05 |

The positive uncertainty threshold is therefore mathematically unreachable for
every reference in this run, even at maximum window entropy. This is a bounded-
domain failure of the chosen fixed rule, not just an empirical miss. The rule
and result are preserved; no threshold was changed and no pilot was repeated.

Each run has one isolated feature exceedance, maximum 0.2726492341. Only BP on
the recurring stream has an isolated residual exceedance, maximum 0.0557495868;
the other recurring residual maxima are about 0.04150–0.04172. Persistence
correctly suppresses these isolated events. Input draws/means match across
streams; unchanged input distribution makes this input-mean channel insufficient
for concept-only label changes. These observations do not select a new rule.

## Learning and preservation

The recurring benchmark checkpoint scores at 0/6/12/18 updates and all 18 applied-update
receipts match the retained original R2.1 pilot exactly for every method. Initial
parameters match exactly. PC and neutral CPC match across all 19 benchmark
checkpoint score reports and all observation measurements/decisions in both streams. Source-
learner preservation checks pass for all eight runs. The measured trajectory is
unchanged under the reviewed trusted owned-snapshot contract. Retained checkpoint
score reports are not restore-ready trained parameter states.

| Recurring method | B gain | A retention loss | A return gain |
| --- | ---: | ---: | ---: |
| BP | 0.056233286 | 0.009415478 | 0.053088664 |
| PC | 0.013821978 | 0.011633764 | 0.024221352 |
| Neutral CPC | 0.013821978 | 0.011633764 | 0.024221352 |
| Active CPC | 0.013030703 | 0.010985131 | 0.022036625 |

All recurring methods learn B, lose A performance during B and recover A on
return. Gains are BCE before minus after; retention loss is A BCE after B minus
after the first A phase. Stationary output uses the same numeric field names
(`new_B_gain`, `retention_loss`, `return_gain`), but trains only A: those fields
are phase-indexed score contrasts there, not B learning or forgetting evidence.
Earlier fresh-task negatives and R1.7 findings remain unchanged. There is no
circadian advantage, loss-of-plasticity, policy efficacy or release claim.

## Work, storage and timing

The sole pilot used 144 benchmark updates / 1,152 training-row exposures,
144 observation predictions and detector steps, eight primes, 304 final scoring
forwards, 448 total `predict_coupled` calls and 648 coupled / 72 BP evaluator
calls. No probe updates, SVDs, replay, sleep, promotion or environment steps.

Elapsed pilot time was 0.4973499 s, worker 0.5324055 s, parent 0.9448081 s.
Measured observation windows total 0.0767608 s; per-run benchmark/orchestration
time totals 0.4000599 s. Remaining global work stays in whole-pilot time.
Observation windows include admitted validation, copies, prediction, measurement
and detector work; priming is separately reported. Constructor/serialization
costs are not hidden as zero. This instrumented single execution is not a
matched-compute or latency comparison.

Sampled process RSS began at 38,117,376 B and peaked at 41,226,240 B; 1,898
samples include explicit boundary samples as well as the 5 ms sampler. This is
not a continuous hard memory guarantee. One sampler thread joined; final thread
count one. Output is 296,694 B, stderr empty, no resource violations.
Limits were 60 s body / 90 s parent, 256 MiB RSS, 32 MiB output, one BLAS/CPU
thread and no GPU/download work.

The observer retains one previous copied snapshot, bounded row IDs and planned
scalar report rows; core retains only the bounded reference/window, not a chain
of prior states. Snapshot arrays are 840 B for BP/PC and 968 B for CPC. These
array-only figures exclude Python objects and temporary copies; whole-process
RSS is reported separately. This pilot is bounded to 18 observations per run;
the app accepts at most 31, core at most 512.

## Checks, failures and next action

An initial pytest collection failed before executing any test phase: a poison
parameter fixture prevented pytest ID introspection, then an exact-count hook
masked that diagnostic. Its logs and receipt remain failed. A prospectively
declared repair constructs the same fixture inside its test and checks exact
count after successful pytest. The final combined 119 cases / 357 phases pass;
no native training was spent in either focused invocation. A pre-gate entropy
roundoff correction and author typing failure are disclosed in the evidence.

Independent source/protocol review preceded the sole native execution;
independent actual review recomputed all 144 detector decisions and verified
trajectory/parity/source-freeze checks. Whole-repository, clean-clone, release,
multi-seed/task, live-policy, latency and calibration gates were not run.

**Next implementation:** add bounded calibration from a declared stationary
arrived-training prefix, with explicit reachable-score checks per channel.
Freeze fitted thresholds before monitoring and account for calibration rows,
model identity and work using the same procedure/information budget for every
method. Keep final data and policy feedback unavailable. Why this: the fixed
uncertainty rule is unreachable; a training-only calibration/domain boundary is
a concrete prerequisite for another useful detector study. This does not
promise to resolve concept-shift misses. Preserve v1 unchanged and freeze/review
a separate small protocol before any new comparative execution.

## Evidence

- [Exact commands, checks and failures](../../artifacts/runs/r27-arrived-shift-20261016/RESULTS.md)
- [Prospective scope and repair amendments](../../artifacts/runs/r27-arrived-shift-20261016/SCOPE.md)
- [Independent source and actual review](../../artifacts/runs/r27-arrived-shift-20261016/REVIEW.md)
- [All eight runs and 144 raw observations](../../artifacts/runs/r27-arrived-shift-20261016/pilot.json)
- [Full observation CSV](../../artifacts/runs/r27-arrived-shift-20261016/observations.csv)
- [Readback summary](../../artifacts/runs/r27-arrived-shift-20261016/observations-summary.json)
- [Per-reference uncertainty reachability](../../artifacts/runs/r27-arrived-shift-20261016/reachability-observations.json)
