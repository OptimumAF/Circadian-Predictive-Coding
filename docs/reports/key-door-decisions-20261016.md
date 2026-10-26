# Reserved key/door decision diagnostics — 2026-10-16

The existing learnability runner/CLI now exposes the accepted decision-report API.
One prospectively frozen local case recreates the original seed17/D8/H8/K5 native
checkpoints and captures previously unretained reserved E decisions. Original
mean-CE learner, data, checkpoints, replay exposure, metrics and directions remain
unchanged. Original BP/PC/neutral CPC/active CPC experiments are accepted references.

## What the additional evidence shows

All four cohorts end with the same argmax action: **left at every E step for
both cues**. The training label counts are `[16,0,16,64,32]`; left is the majority.
Each E test packet has support `[1,0,1,4,2]`, so the resulting accuracy is50%.
These are repeated deterministic training-domain traces. Their64 reported endpoint
decisions are not independent episodes of new-task generalization.

| Replay | Source wake | Prior native updates | Mean CE0 → CE16 | CE reduction | Accuracy0 →16 | Endpoint predictions per E packet |
|---|---:|---:|---:|---:|---:|---|
| off | 0 | 0 | 1.618711533 →1.325127978 | +0.293583554 | .125 →.500 | eight left |
| off | 8 | 8 | 1.660445268 →1.344937424 | +0.315507843 | .1875 →.500 | eight left |
| on | 0 | 0 | 1.618711533 →1.325127978 | +0.293583554 | .125 →.500 | eight left |
| on | 8 | 12 | 1.679548071 →1.340306064 | +0.339242007 | .1875 →.500 | eight left |

At each endpoint, left has recall1.0; up/down/interact each have recall0.0.
Right has no target support, so its accuracy/loss stay `None`. Correct ordered
indices are0,4,5,7; the other four decisions are wrong. This directly describes
the observed chosen actions. It does not prove that internal representations or
all predicted probabilities ignore the cue.

Intermediate curves are retained. At source0, both cues already choose all-left
by probe update8; blue does so by4. At source8, update4/8 predictions include
interact-heavy behavior before all-left at16. No curve, cohort, negative gain or
failed old experiment is discarded. This run adds class/decision evidence to
the earlier aggregate-only result; it establishes no useful E-task solution.

## Original acquisition, retention and return evidence

Every original saved field matches the accepted earlier learnability run after
removing only new diagnostic additions/protocol metadata, measured durations and
sampled RSS. This includes all training/native observations and gradient byte
counts, original demonstrations, aggregate scores, replay selection/receipts,
permanent charges, every live transition/outcome and original work accounting.
Original complete opaque state isolation is also executed by each probe guard.

| Replay | B CE gain | C CE gain | A retention loss | A return gain | Live successes |
|---|---:|---:|---:|---:|---:|
| off | +.0155780797 | -.0203605547 | -.0177739613 | +.0109111860 | 0/40 |
| on | +.0221212282 | -.0513932649 | -.0465369365 | +.0118024704 | 0/40 |

The original definitions remain B=CE0−CE4, C=CE0−CE6, retention=CE6−CE2,
return=CE6−CE8. All80 live episodes still fail illegal actions with zero success
and return. CE movement and a majority action do not demonstrate useful new-task
learning, old-task skill retention, context-return recovery or circadian benefit.
No final records reach training, promotion, tuning or probe selection.

## Actual costs

- 84 native SGD calls/672 training rows, including64 isolated probe calls/512 rows.
  Benchmark work is20 calls/160 rows; probes have four original16-update budgets.
- 112 classification calls/896 rows, including32 probe calls/256 rows;80 fresh live
  episodes use170 live predictions and90 legal steps. Total predictions282.
- 1738 recurrent timesteps:672 training +896 classification +170 live. Probe work
  contributes768; the original benchmark contributes970.
- 34 teacher packets/272 steps;114 environment constructions/362 total steps.
- Original replay permanent exposure32 transitions/on arm; final retained16
  transitions/4032B cap. No renewed exposure or source/deployed probe updates.
- Each probe owns10368B feature/target arrays. Python decision reports, temporary
  NumPy allocations, episode records and native/copy payloads are excluded;
  sampled whole-process RSS314388480B/1521 samples at requested5ms includes import
  and additional reports. Sampling is not a continuous peak measurement.
- Recipe5.0002264s, worker5.4759177s, parent6.0380481s. JSON851579B/stderr0.
  Invocation timing is observational; this case is not a causal overhead or
  relative-efficiency comparison to the old differently instrumented run.
- CPU/BLAS/Torch/interop1/GPU0; sampler joined/final Python thread1. All original
  work/90s body/120s parent/768MiB RSS/8MiB JSON caps pass. Exactly one invocation.

Native checkpoint recreation was necessary: archived JSON contains parameters
and aggregates, not complete restorable optimizer/gradient state or unretained
predictions. No old four-family baseline was retrained. No new seed, horizon,
objective, metric or post-result continuation was selected.

## Critical path

R2.1 scoped CLI/reporting/decision measurement is delivered. Existing mean-CE BC,
pure RL and demonstration-to-reward negatives remain preserved. Useful acquisition
is the next learner requirement; retention/return claims need an acquired skill.
Inspection confirms the recurrent supervised update still uses ordinary unweighted
mean action CE, alongside a separately typed reward objective.

Next implementation: add an explicit experimental class-weighted demonstration
update alongside the unchanged mean-CE baseline. Derive any weights solely from
released training labels and carry them explicitly with the update; keep complete
episode order. Verify neutral weights against existing CE and independently check
weighted BPTT/SGD, refusal/atomicity/owned continuation and final-role exclusion
before composing or experimenting. Training-label imbalance motivates this
hypothesis; improvement is unproven. Freeze a separate small comparison before
execution and preserve a negative outcome. No additional diagnostic/checkpoint
increment, larger sweep or tuning from final/probe-test scores is needed.

[Raw diagnostic run](../../artifacts/runs/r21-key-door-decision-route-20261016/diagnostic.json),
[all decision summaries/exact old projection](../../artifacts/runs/r21-key-door-decision-route-20261016/analysis.json),
[commands and engineering checks](../../artifacts/runs/r21-key-door-decision-route-20261016/RESULTS.md),
[independent review](../../artifacts/runs/r21-key-door-decision-route-20261016/REVIEW.md).
