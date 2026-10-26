# MiniGrid warm reward TRAIN endpoints — 2026-10-16

The existing warm runtime now executes all eight reward updates and two owned
native checkpoint readouts under unchanged caps. Independent ACTUAL acceptance
is recorded in the [case review](../../artifacts/runs/r21-minigrid-warm-readout-compose-20261016/guard-repair/REVIEW.md).
This is a negative local engineering result, not a circadian comparison.

## Observed endpoints

All training precedes both readouts. Native128 is the owned frozen warm learner;
native136 includes eight sampled reward updates. Sources are six original known
TRAIN traces, with raw unweighted scoring and two live cues per context. No final
layout, fresh-task probe, action mask, seed selection or teacher is introduced.

| TRAIN context | Mean CE (lower is better),128 → 136 | Raw accuracy,128 → 136 | Live goals,128 → 136 |
|---|---|---|---|
| A | 0.145778 → 1.904723 | 94.444% → 55.556% | 1/2 → 0/2 |
| B | 0.946894 → 1.837955 | 72.222% → 55.556% | 0/2 → 0/2 |
| C_train | 0.505898 → 1.843649 | 88.889% → 55.556% | 1/2 → 0/2 |

Every context's raw CE and accuracy worsen. The red-cue goal episodes on A and
C_train disappear; all six post-reward live episodes time out at16steps. All
joint both-cue acquisition conditions are false at both endpoints, so formal
task retention has no valid acquisition premise. Two endpoints do not establish
within-stream context-return recovery.

Mean return increases by0.253125 on A and C_train: the warm cue pair combines
one−1 wrong-door outcome with one+0.49375 goal, whereas the later pair has two
zero-return timeouts. Goal success falls from1/2 to0/2 despite that return gain.
Keep both metrics; do not interpret avoiding a negative return as task success.
B has no live goals at either endpoint.

## Paid work and resource limits

This invocation:136native SGD/1274trainingrows (warm128/1152 +reward8/122),
22rawforwards198rows,330nativeprediction/physicssteps,24environments/sessions,
1802observed recurrent positions. The added diagnostic work is12rawforwards/
108rows/12sessions/160live steps; it performs zero updates. Reward training
finishes in36.8321009s under45; its original charges/clock/resource observation
remain frozen. A distinct max0-update readout budget uses44.8632624s under60,
within original180whole/210parent limits. Pilot91.5007775s/parent94.5497263s;
peak sampled RSS368103424B under1GiB, warm arm cap768MiB. CLI import peak is
unmeasured. Strict JSON7180166B under16MiB. Source and serving actor stay unchanged.

All earlier failures remain: total across four warm invocations is540updates/
5032rows, including reconstruction each time. Each case allowance is spent;
none is renewed or erased. [Exact commands, tests, profile and costs](../../artifacts/runs/r21-minigrid-warm-readout-compose-20261016/guard-repair/RESULTS.md).

## Interpretation and next dependency

The warm128 endpoint is a frozen BC reference, not an equal-update control.
Carried optimizer momentum can move parameters even for zero-reward episodes;
this case includes one negative-reward episode and seven zero-return episodes.
These observations cannot attribute the degradation to reward gradients alone.

R2.1 isolated fresh-task probes and health diagnostics already exist and have
accepted evidence; repeating them does not remove this confound. Before a new
learning mechanism or reward benefit claim, reconcile and implement a distinct
matched additional-update/momentum control through the existing native/runtime
ports. It must keep actual observed rewards/probabilities intact, use a separate
learner from one shared warm fork, and explicitly account for its observations,
updates, replay exposure, memory and compute. Freeze/review a small protocol
before any comparative execution. All broad scientific/release goals remain open.
