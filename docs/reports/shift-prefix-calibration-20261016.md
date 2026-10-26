# R2.7 training-prefix calibration, 2026-10-16

## Delivered and qualified

The existing shift observer and CLI now offer opt-in `--calibrated` mode. Pure
core calibration fits a bounded arrived-training prefix, freezes thresholds
before monitoring, reports algebraic reachability and discards the prefix after
replaying its scalar summaries into the existing detector. The app records
arrived row IDs and caller-declared run identity alongside k−1 parameter-update
indices. Labels/final data are unavailable to the predictor; only arrived
training labels enter residual measurements. Signals still control no training,
promotion or probe selection. Original v1 defaults and negative evidence remain.

All187 focused cases/561 phases pass, including the previous119 cases and68
new cases. Eleven-path Ruff/format and Windows/Linux mypy pass. Independent
source/protocol review preceded the sole frozen144-update native execution.
Implemented behavior and this scoped execution are qualified; broad deployment
efficacy, return detection and full R2.7 remain unestablished.

## Prospective exploratory rule

`arrived_only_prefix_calibration_v1` preserves seed23, data, matched BP/PC/neutral
CPC/active CPC settings and stationary/A→B→A streams. Each of eight runs has18
updates and eight rows per batch. The first six declared stationary A batches
calibrate the same rule for every method. Three batches freeze the reference;
two-batch windows give only two full calibration scores. Each fitted threshold
is `max(0, prefix window scores) + 1e-6`, capped only at domain upper bounds
(1,2,1). Monitoring starts at7; strict two-window persistence is unchanged.

No alarms or streaks are emitted during calibration. At6, six scalar summaries
are replayed once into a fresh detector with fitted thresholds, preserving its
reference/window/index and zero prefix streaks; the stored prefix is discarded.
Later observations cannot refit thresholds. Calibration sees an evolving k−1
parameter population, not a fixed model. Prefix stationarity is an experimental
assumption, not inferred using held-out scores. Two overlapping fit windows
provide no calibrated false-positive guarantee.

This rule was motivated by the fixed v1 rule's negative outcome, then frozen
before execution. It is not independent confirmation or per-method tuning.

## Actual fits and reachability

Fits are identical between streams for each method because their first six
training batches and parameters are identical.

| Method | Uncertainty threshold | Feature threshold | Residual threshold |
| --- | ---: | ---: | ---: |
| BP | 0.009294293793 | 0.109624232803 | 0.000001000000 |
| PC | 0.006922113969 | 0.109624232803 | 0.009756770768 |
| Neutral CPC | 0.006922113969 | 0.109624232803 | 0.009756770768 |
| Active CPC | 0.006895583889 | 0.109624232803 | 0.009926516620 |

All24 fitted channel thresholds are strictly below their algebraic ceilings.
Uncertainty headroom remains0.01532–0.01561, feature ceiling1.0887760420 and
residual headroom0.48035–0.48494. Reachability means a domain bound does not
exclude an exceedance; it does not prove attainable model predictions or useful
detection. No threshold was lowered to force this result.

## Observable outcomes, including false alerts

Every method has the same alarm update indices:

| Channel | Stationary A control | Recurring A/B/A |
| --- | --- | --- |
| Uncertainty | none | none |
| Input-mean drift | 8,11,12,13,14,15 | 8,11,12,13,14,15 |
| Supervised residual | none | 8,9,10,11,12,13 |

Residuals distinguish B arrival from the stationary controls in this fixed trace:
first alarm8, one update after B's first arrived batch7, with no stationary
alarms. This is a useful limited observation, not evidence of general efficacy.
All methods share the result; it gives no circadian advantage. Uncertainty still
misses changes despite being algebraically reachable.

Feature drift has six stationary false-alert updates per method, and the exact
same pattern in the recurring stream. Its apparent phase alerts do not
discriminate concept changes; input distribution is unchanged. Calibration's
small prefix does not control sampling noise here.

The raw phase annotation lists a first returning-A alert at13 with zero delay
for residual/feature. Both are continuing prior streaks, not new alarm onsets.
Residual alarms started at8 and end after13; feature's second streak started at11.
Neither establishes fresh return detection or recovery. Preserve these raw
annotations without treating them as successes. No threshold, margin, seed,
metric or endpoint changed after this execution; no retry occurred.

## Learning, isolation and costs

All eight complete benchmark results (all19 checkpoint score reports and applied
receipts), initial tensors and all144 raw batch measurements exactly match
retained uncalibrated v1. These checkpoint reports are not saved trained native
parameter states. Source-learner preservation checks pass; PC/neutral full score
reports and observation metrics/decisions match within each stream. Calibration
changes no measured benchmark trajectory. Original learnability negatives and
all earlier baseline outcomes remain unchanged. The recurring methods still
learn B, lose A during B and recover A on return. Stationary phase-indexed
learning fields remain score contrasts, not B-training/forgetting evidence.

Work:144 training updates/1152 rows,144 observation predictions/measurements,
eight primes,304 final scoring forwards,448 total `predict_coupled` calls,
648 coupled/72 BP evaluators. Eight fits add48 scalar-summary replays, giving192
underlying detector observations plus144 wrapper observations. Calibration's48
batches/384 rows are included in training totals, leaving96 monitored batches/
768 rows. No extra label exposure, probe, SVD, replay training, sleep, promotion
or environment steps.

Pilot0.4799335 s; worker0.5180116 s; parent0.9712259 s. Observation windows,
including fitting/replay, total0.0798881 s; per-run benchmark/orchestration
0.3772911 s. Global remainder stays in whole-pilot time. These instrumented
single-run times are not latency or equal-compute comparisons.

RSS start37,867,520 B / peak41,431,040 B,1896 samples including boundary reads,
5 ms interval; this is sampled observation, not a continuous hard guarantee.
Output362024 B/stderr0, one sampler joined/final1 thread, violations[]. Limits
60 s body/90 s parent/256 MiB RSS/32 MiB output/one BLAS CPU thread/no GPU.
The app retains one previous state plus bounded IDs/scalar rows; core retains
one reference/window and immutable result after dropping its prefix. Snapshot
arrays remain840 B BP/PC and968 B CPC, excluding Python/temporary copies; RSS
is separate.

## Remaining work and critical path

R2.7 remains unchecked: feature false alerts and lack of fresh return detection
prevent a broad deployable-signal claim. The original fixed-rule failure and
this limited residual result remain separate. Full R2.1 and repository release
requirements are also open. No further threshold/prefix/horizon sweep follows
automatically from this result.

Next substantial outcome is R2.3 retention selection. Inspection found existing
`SharedReplayBuffer` row/byte caps, deduplication, and separate FIFO/bottom-k
reservoir choices, used by the matched replay route; it lacks explicit combined
recent/reservoir allocations. **Next implementation:** add opt-in separate
recent/reservoir row-and-byte allocations to that buffer, preserving defaults,
content deduplication and detached selections. Verify bounded old-context
representation on a deterministic long arrived stream without future labels,
and use the existing matched replay consumer rather than another runner.
Why this: predictable memory coverage can improve retention independently of
the currently limited shift signal and supplies a simpler matched control before
renewal or automatic signal-driven training. Any comparative replay execution
requires a new prospective budget and frozen protocol.

## Evidence

- [Exact commands, failures and limits](../../artifacts/runs/r27-prefix-calibration-20261016/RESULTS.md)
- [Scope and amendments](../../artifacts/runs/r27-prefix-calibration-20261016/SCOPE.md)
- [Independent review](../../artifacts/runs/r27-prefix-calibration-20261016/REVIEW.md)
- [All eight runs and144 observations](../../artifacts/runs/r27-prefix-calibration-20261016/pilot.json)
- [Readback summary and fits](../../artifacts/runs/r27-prefix-calibration-20261016/observations-summary.json)
- [Full observation CSV](../../artifacts/runs/r27-prefix-calibration-20261016/observations.csv)
