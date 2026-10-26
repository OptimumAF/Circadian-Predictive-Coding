# Reserved key/door checkpoint learnability — 2026-10-16

One frozen local seed17 recurrent behavior-cloning case, replay off/on. All four
cohorts are retained; original BP/PC/neutral CPC/active CPC evidence was reused
without rerunning its experiments. The full repository goal remains active.

## Fresh-task diagnostic

At fixed wake0 and8, complete owned native checkpoints (including SGD momentum,
gradients and native counters) are restored on disposable forks. Each gets16
original probe updates on reserved E traces and two E test packets, with
checkpoints0/4/8/16. Probe-test traces repeat the deterministic training domain:
**this is supervised fit, not unseen-trajectory generalization.**

| Replay | Benchmark wake | Native prior updates | CE0 | CE16 | CE0−CE16 | Accuracy0→16 | Outer seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 0 | 0 | 1.6187115327 | 1.3251279784 | +0.2935835543 | 0.1250→0.5000 | 0.3125387 |
| off | 8 | 8 | 1.6604452679 | 1.3449374244 | +0.3155078435 | 0.1875→0.5000 | 0.3103217 |
| on | 0 | 0 | 1.6187115327 | 1.3251279784 | +0.2935835543 | 0.1250→0.5000 | 0.2804016 |
| on | 8 | 12 | 1.6795480707 | 1.3403060639 | +0.3392420067 | 0.1875→0.5000 | 0.3057669 |

Both initial curves are identical, retained explicitly. Endpoint CE improves
from every tested checkpoint and all endpoint accuracies reach0.5, with no
desired-quality acceptance gate. This narrow result shows some reserved-trace
fit remains possible after the fixed stream; it does not show full task learning,
learned cue use, causally superior replay or circadian benefit. It does not change
which model is deployed or justify further training of the final-scored case.

Posthoc training-label arithmetic gives counts `[16,0,16,64,32]` and empirical
prior `[.125,0,.125,.5,.25]`. A constant majority action achieves the same query
accuracy0.5; the fixed empirical-prior prediction has queryCE1.213007566,
lower than every recurrent endpoint1.325–1.345. This is an explicitly **posthoc
descriptive reference**, not a preregistered matched control, new model experiment
or tuning gate. It uses reserved training labels only to define the rule; no
native calls. It does not show the recurrent model itself predicts constantly,
whose reserved query probability traces were not retained. [Arithmetic receipt](../../artifacts/runs/r21-key-door-learnability-20261016/train-label-reference.json).

### Every mean probe point

| Replay | Benchmark wake | Probe updates | Mean CE | Mean accuracy |
|---|---:|---:|---:|---:|
| off | 0 | 0 | 1.6187115327 | 0.1250 |
| off | 0 | 4 | 1.5710179484 | 0.4375 |
| off | 0 | 8 | 1.4842748207 | 0.5000 |
| off | 0 | 16 | 1.3251279784 | 0.5000 |
| off | 8 | 0 | 1.6604452679 | 0.1875 |
| off | 8 | 4 | 1.6254478645 | 0.3125 |
| off | 8 | 8 | 1.5349878889 | 0.3125 |
| off | 8 | 16 | 1.3449374244 | 0.5000 |
| on | 0 | 0 | 1.6187115327 | 0.1250 |
| on | 0 | 4 | 1.5710179484 | 0.4375 |
| on | 0 | 8 | 1.4842748207 | 0.5000 |
| on | 0 | 16 | 1.3251279784 | 0.5000 |
| on | 8 | 0 | 1.6795480707 | 0.1875 |
| on | 8 | 4 | 1.6376827858 | 0.3125 |
| on | 8 | 8 | 1.5414834782 | 0.3125 |
| on | 8 | 16 | 1.3403060639 | 0.5000 |

## Old-task retention, return and live behavior

Exact saved-data comparison matches **every old run field** except the three
new diagnostic work fields; all original work fields also match exactly. This
includes every parameter observation/native counter, all classification
checkpoints, complete stream updates/diagnostics, replay receipts/selection/
retention/charges and every live transition/outcome. Original16demonstrations
and all unchanged protocol fields match. All13 frozen source/controller texts
are unchanged. Each probe reports original source/checkpoint/budget bindings
preserved, and benchmark source/serving actor guards pass.

| Replay | B CE gain | C CE gain | A retention loss | A return gain | Live successes |
|---|---:|---:|---:|---:|---:|
| off | +0.0155780797 | -0.0203605547 | -0.0177739613 | +0.0109111860 | 0/40 |
| on | +0.0221212282 | -0.0513932649 | -0.0465369365 | +0.0118024704 | 0/40 |

B gain=CE0−CE4; C gain=CE0−CE6; A retention loss=CE6−CE2; A return gain=CE6−CE8.
All80 live episodes still fail on illegal actions with zero success and return.
Changed CE and preserved trajectories do not establish useful acquisition,
old-room recall, return recovery or held-out-layout success. Previous320
feedforward failures and80 recurrent failures remain preserved, unmodified.

## Actual costs

- 84 native SGD calls/672 training rows:20 benchmark160rows +64 probe512rows.
- 112 cached classification calls/896rows:80 benchmark640rows +32 probe256rows.
- 80 fresh live sessions/episodes,170 live predictions/90 applied legal steps.
- 1738 unrolled recurrent timesteps =970 original benchmark +768 added probes.
- 34 teacher episodes/272 steps;114 environment constructions/362 total steps.
- Four probe slots total1.2090289s, including detachment and execution; final
  isolation readback is outside slot time and inside overall invocation time.
- 15.6646296s recipe/16.0900260s worker/16.6297894s parent. These are observed
  invocation times, not a paired causal speed or relative efficiency experiment.
- Sampled process RSS313,487,360B/2141 samples (requested5ms). JSON642,222B.
- Each probe owns10,368B feature/target arrays (128train+16test transitions);
 18 unique episode IDs. Excludes complete record/Python/native/copy/graph
 payloads. This per-cohort count is not total peak memory or free storage.
- Original replay cap16transitions/4032B and permanent32transition charge/on-arm.
- 181 parameters/724B; trained gradient and momentum tensor payload724B each,
 episode activation32B. Owned copies and temporary graphs excluded.

All prospective90s body/120s parent/768MiB RSS/8MiB JSON/work caps pass,
CPU/BLAS/Torch/interop1/GPU0, one sampler thread joined. No renewed budgets,
sweeps, post-result updates or final-data feedback. No new comparative CPC claim.

## Reassessed critical path

R2.1 reserved categorical checkpoint diagnosis is now measured; accepted binary
R2.1 and repeated-update runtime evidence are reused. Do not start another
checkpoint/provenance/capture or larger probe-horizon increment. Actual source
search in src/scripts finds no recurrent RL/policy-gradient implementation.
R4.2 explicitly requires a separately evaluated recurrent RL policy; only BC
currently exists. That is the next substantive learner dependency, while useful
acquisition/retention/context recovery, fullR2.7/R2.8 and release remain open.

**Next implementation:** add a bounded, explicitly reward-trajectory recurrent
policy-gradient update alongside the existing supervised native interface, with
owned optimizer/RNG continuation and episode-reset semantics. Validate independent
objective/gradient math, permanent costs and final-role exclusion before composing
it into the existing key/door runner. Define any training protocol using train-only
inputs; freeze/review before a new local experiment. Do not tune from these final
curves or replace failed baseline results. No additional experiment is authorized
by leftover time in this closed case.

[Raw report](../../artifacts/runs/r21-key-door-learnability-20261016/pilot.json),
[saved-data comparison](../../artifacts/runs/r21-key-door-learnability-20261016/analysis.json),
[commands/failures](../../artifacts/runs/r21-key-door-learnability-20261016/RESULTS.md),
[independent review](../../artifacts/runs/r21-key-door-learnability-20261016/REVIEW.md).
