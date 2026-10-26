# Actual key/door case — 2026-10-16

Actual local MiniGrid-like feature navigation, with a disappearing cue and
final-only changed layout D. This is not official MiniGrid integration or recurrent
RL. All16 scripted demonstrations reach the exit in eight actions. All320 learned
live episodes end on an illegal action, success0/return0, at every checkpoint and
both replay modes. No mask, fallback, retry or favorable checkpoint selection.

## Every declared CE outcome and live cost

Gains = initial paired-cue CE minus post-context CE; A loss=CE6−CE2 (lower);
A return gain=CE6−CE8. These signs and checkpoints were prospectively frozen.

| Method | Replay | B gain | C gain | A loss | A return gain | Live predictions | Applied live steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bp | False | 0.01103795087 | -0.01854898945 | -0.0134420243 | 0.009112964868 | 88 | 48 |
| pc | False | 0.00797907638 | 0.001774857201 | 0.007351447556 | 0.008649505335 | 67 | 27 |
| neutral_cpc | False | 0.00797907638 | 0.001774857201 | 0.007351447556 | 0.008649505335 | 67 | 27 |
| gated_cpc | False | 0.007978631436 | 0.001773874226 | 0.007351173637 | 0.008648196779 | 67 | 27 |
| bp | True | 0.01469234942 | -0.04595247957 | -0.03810016739 | 0.01055307064 | 88 | 48 |
| pc | True | 0.007067920443 | -4.208944093e-05 | 0.006499495647 | 0.005248023872 | 67 | 27 |
| neutral_cpc | True | 0.007067920443 | -4.208944093e-05 | 0.006499495647 | 0.005248023872 | 67 | 27 |
| gated_cpc | True | 0.007067198796 | -4.434104902e-05 | 0.006499082546 | 0.005246379902 | 67 | 27 |

CE and demonstration action accuracy change without successful game behavior.
BP has negative C gains; replay-on PC-family C gains are also negative. PC-family
A loss remains positive. Active CPC differences are small/mixed and every live
policy still fails. No useful acquisition, retention, recovery, D-layout
generalization or circadian advantage is established. All earlier negatives remain.

## All classification and live curves

Entries pair the two cues; raw JSON retains all320 individual CE/accuracy scores
and all320 full live traces. Every row below has0/2 live success and return0.
A/B/C are known rooms, D is final-only changed geometry; they remain separate.

| Method | Replay | Wake | Room | Mean CE | Mean action accuracy | Live predictions | Applied live steps |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| bp | False | 0 | A | 1.606395463 | 0.125 | 2 | 0 |
| bp | False | 0 | B | 1.60887531 | 0.375 | 2 | 0 |
| bp | False | 0 | C | 1.610091769 | 0.3125 | 2 | 0 |
| bp | False | 0 | D | 1.608856365 | 0.125 | 2 | 0 |
| bp | False | 2 | A | 1.593298835 | 0.5 | 8 | 6 |
| bp | False | 2 | B | 1.60973243 | 0.125 | 2 | 0 |
| bp | False | 2 | C | 1.619946439 | 0 | 2 | 0 |
| bp | False | 2 | D | 1.59581877 | 0.5 | 8 | 6 |
| bp | False | 4 | A | 1.579974413 | 0.5 | 8 | 6 |
| bp | False | 4 | B | 1.597837359 | 0.125 | 2 | 0 |
| bp | False | 4 | C | 1.632336701 | 0 | 2 | 0 |
| bp | False | 4 | D | 1.582590192 | 0.5 | 8 | 6 |
| bp | False | 6 | A | 1.579856811 | 0.5 | 8 | 6 |
| bp | False | 6 | B | 1.585566706 | 0.1875 | 2 | 0 |
| bp | False | 6 | C | 1.628640758 | 0 | 2 | 0 |
| bp | False | 6 | D | 1.582495617 | 0.5 | 8 | 6 |
| bp | False | 8 | A | 1.570743846 | 0.5 | 8 | 6 |
| bp | False | 8 | B | 1.577823681 | 0.125 | 2 | 0 |
| bp | False | 8 | C | 1.631369974 | 0 | 2 | 0 |
| bp | False | 8 | D | 1.573456496 | 0.5 | 8 | 6 |
| pc | False | 0 | A | 1.606395463 | 0.125 | 2 | 0 |
| pc | False | 0 | B | 1.60887531 | 0.375 | 2 | 0 |
| pc | False | 0 | C | 1.610091769 | 0.3125 | 2 | 0 |
| pc | False | 0 | D | 1.608856365 | 0.125 | 2 | 0 |
| pc | False | 2 | A | 1.597717745 | 0.5 | 8 | 6 |
| pc | False | 2 | B | 1.609609962 | 0.125 | 2 | 0 |
| pc | False | 2 | C | 1.616614493 | 0 | 2 | 0 |
| pc | False | 2 | D | 1.60020775 | 0.5 | 8 | 6 |
| pc | False | 4 | A | 1.598542808 | 0.375 | 2 | 0 |
| pc | False | 4 | B | 1.600896234 | 0.5625 | 8 | 6 |
| pc | False | 4 | C | 1.617318666 | 0.125 | 2 | 0 |
| pc | False | 4 | D | 1.601037703 | 0.375 | 5 | 3 |
| pc | False | 6 | A | 1.605069192 | 0.3125 | 2 | 0 |
| pc | False | 6 | B | 1.601695971 | 0.4375 | 5 | 3 |
| pc | False | 6 | C | 1.608316912 | 0.3125 | 2 | 0 |
| pc | False | 6 | D | 1.607545976 | 0.25 | 2 | 0 |
| pc | False | 8 | A | 1.596419687 | 0.4375 | 2 | 0 |
| pc | False | 8 | B | 1.602526013 | 0.3125 | 2 | 0 |
| pc | False | 8 | C | 1.614872979 | 0.125 | 2 | 0 |
| pc | False | 8 | D | 1.598925782 | 0.5 | 5 | 3 |
| neutral_cpc | False | 0 | A | 1.606395463 | 0.125 | 2 | 0 |
| neutral_cpc | False | 0 | B | 1.60887531 | 0.375 | 2 | 0 |
| neutral_cpc | False | 0 | C | 1.610091769 | 0.3125 | 2 | 0 |
| neutral_cpc | False | 0 | D | 1.608856365 | 0.125 | 2 | 0 |
| neutral_cpc | False | 2 | A | 1.597717745 | 0.5 | 8 | 6 |
| neutral_cpc | False | 2 | B | 1.609609962 | 0.125 | 2 | 0 |
| neutral_cpc | False | 2 | C | 1.616614493 | 0 | 2 | 0 |
| neutral_cpc | False | 2 | D | 1.60020775 | 0.5 | 8 | 6 |
| neutral_cpc | False | 4 | A | 1.598542808 | 0.375 | 2 | 0 |
| neutral_cpc | False | 4 | B | 1.600896234 | 0.5625 | 8 | 6 |
| neutral_cpc | False | 4 | C | 1.617318666 | 0.125 | 2 | 0 |
| neutral_cpc | False | 4 | D | 1.601037703 | 0.375 | 5 | 3 |
| neutral_cpc | False | 6 | A | 1.605069192 | 0.3125 | 2 | 0 |
| neutral_cpc | False | 6 | B | 1.601695971 | 0.4375 | 5 | 3 |
| neutral_cpc | False | 6 | C | 1.608316912 | 0.3125 | 2 | 0 |
| neutral_cpc | False | 6 | D | 1.607545976 | 0.25 | 2 | 0 |
| neutral_cpc | False | 8 | A | 1.596419687 | 0.4375 | 2 | 0 |
| neutral_cpc | False | 8 | B | 1.602526013 | 0.3125 | 2 | 0 |
| neutral_cpc | False | 8 | C | 1.614872979 | 0.125 | 2 | 0 |
| neutral_cpc | False | 8 | D | 1.598925782 | 0.5 | 5 | 3 |
| gated_cpc | False | 0 | A | 1.606395463 | 0.125 | 2 | 0 |
| gated_cpc | False | 0 | B | 1.60887531 | 0.375 | 2 | 0 |
| gated_cpc | False | 0 | C | 1.610091769 | 0.3125 | 2 | 0 |
| gated_cpc | False | 0 | D | 1.608856365 | 0.125 | 2 | 0 |
| gated_cpc | False | 2 | A | 1.597717982 | 0.5 | 8 | 6 |
| gated_cpc | False | 2 | B | 1.609610033 | 0.125 | 2 | 0 |
| gated_cpc | False | 2 | C | 1.616614387 | 0 | 2 | 0 |
| gated_cpc | False | 2 | D | 1.600207937 | 0.5 | 8 | 6 |
| gated_cpc | False | 4 | A | 1.598543234 | 0.375 | 2 | 0 |
| gated_cpc | False | 4 | B | 1.600896679 | 0.5625 | 8 | 6 |
| gated_cpc | False | 4 | C | 1.617318545 | 0.125 | 2 | 0 |
| gated_cpc | False | 4 | D | 1.601038016 | 0.375 | 5 | 3 |
| gated_cpc | False | 6 | A | 1.605069156 | 0.3125 | 2 | 0 |
| gated_cpc | False | 6 | B | 1.601696456 | 0.4375 | 5 | 3 |
| gated_cpc | False | 6 | C | 1.608317895 | 0.3125 | 2 | 0 |
| gated_cpc | False | 6 | D | 1.607545979 | 0.25 | 2 | 0 |
| gated_cpc | False | 8 | A | 1.596420959 | 0.4375 | 2 | 0 |
| gated_cpc | False | 8 | B | 1.602526809 | 0.3125 | 2 | 0 |
| gated_cpc | False | 8 | C | 1.614873319 | 0.125 | 2 | 0 |
| gated_cpc | False | 8 | D | 1.598926761 | 0.5 | 5 | 3 |
| bp | True | 0 | A | 1.606395463 | 0.125 | 2 | 0 |
| bp | True | 0 | B | 1.60887531 | 0.375 | 2 | 0 |
| bp | True | 0 | C | 1.610091769 | 0.3125 | 2 | 0 |
| bp | True | 0 | D | 1.608856365 | 0.125 | 2 | 0 |
| bp | True | 2 | A | 1.581353887 | 0.5 | 8 | 6 |
| bp | True | 2 | B | 1.610851255 | 0.125 | 2 | 0 |
| bp | True | 2 | C | 1.629292007 | 0 | 2 | 0 |
| bp | True | 2 | D | 1.583949923 | 0.5 | 8 | 6 |
| bp | True | 4 | A | 1.551138468 | 0.5 | 8 | 6 |
| bp | True | 4 | B | 1.594182961 | 0.125 | 2 | 0 |
| bp | True | 4 | C | 1.657870678 | 0 | 2 | 0 |
| bp | True | 4 | D | 1.553978108 | 0.5 | 8 | 6 |
| bp | True | 6 | A | 1.543253719 | 0.5 | 8 | 6 |
| bp | True | 6 | B | 1.581821376 | 0.125 | 2 | 0 |
| bp | True | 6 | C | 1.656044248 | 0 | 2 | 0 |
| bp | True | 6 | D | 1.546222991 | 0.5 | 8 | 6 |
| bp | True | 8 | A | 1.532700649 | 0.5 | 8 | 6 |
| bp | True | 8 | B | 1.577776929 | 0.125 | 2 | 0 |
| bp | True | 8 | C | 1.652648334 | 0 | 2 | 0 |
| bp | True | 8 | D | 1.535837515 | 0.5 | 8 | 6 |
| pc | True | 0 | A | 1.606395463 | 0.125 | 2 | 0 |
| pc | True | 0 | B | 1.60887531 | 0.375 | 2 | 0 |
| pc | True | 0 | C | 1.610091769 | 0.3125 | 2 | 0 |
| pc | True | 0 | D | 1.608856365 | 0.125 | 2 | 0 |
| pc | True | 2 | A | 1.593447243 | 0.5 | 8 | 6 |
| pc | True | 2 | B | 1.610037718 | 0.125 | 2 | 0 |
| pc | True | 2 | C | 1.619891528 | 0 | 2 | 0 |
| pc | True | 2 | D | 1.595962839 | 0.5 | 8 | 6 |
| pc | True | 4 | A | 1.590084383 | 0.5 | 8 | 6 |
| pc | True | 4 | B | 1.60180739 | 0.125 | 2 | 0 |
| pc | True | 4 | C | 1.623824944 | 0 | 2 | 0 |
| pc | True | 4 | D | 1.592630747 | 0.5 | 8 | 6 |
| pc | True | 6 | A | 1.599946738 | 0.25 | 2 | 0 |
| pc | True | 6 | B | 1.602905799 | 0.25 | 2 | 0 |
| pc | True | 6 | C | 1.610133858 | 0.1875 | 2 | 0 |
| pc | True | 6 | D | 1.60245961 | 0.1875 | 2 | 0 |
| pc | True | 8 | A | 1.594698715 | 0.4375 | 2 | 0 |
| pc | True | 8 | B | 1.604155197 | 0.25 | 2 | 0 |
| pc | True | 8 | C | 1.612238505 | 0.125 | 2 | 0 |
| pc | True | 8 | D | 1.597237757 | 0.5 | 5 | 3 |
| neutral_cpc | True | 0 | A | 1.606395463 | 0.125 | 2 | 0 |
| neutral_cpc | True | 0 | B | 1.60887531 | 0.375 | 2 | 0 |
| neutral_cpc | True | 0 | C | 1.610091769 | 0.3125 | 2 | 0 |
| neutral_cpc | True | 0 | D | 1.608856365 | 0.125 | 2 | 0 |
| neutral_cpc | True | 2 | A | 1.593447243 | 0.5 | 8 | 6 |
| neutral_cpc | True | 2 | B | 1.610037718 | 0.125 | 2 | 0 |
| neutral_cpc | True | 2 | C | 1.619891528 | 0 | 2 | 0 |
| neutral_cpc | True | 2 | D | 1.595962839 | 0.5 | 8 | 6 |
| neutral_cpc | True | 4 | A | 1.590084383 | 0.5 | 8 | 6 |
| neutral_cpc | True | 4 | B | 1.60180739 | 0.125 | 2 | 0 |
| neutral_cpc | True | 4 | C | 1.623824944 | 0 | 2 | 0 |
| neutral_cpc | True | 4 | D | 1.592630747 | 0.5 | 8 | 6 |
| neutral_cpc | True | 6 | A | 1.599946738 | 0.25 | 2 | 0 |
| neutral_cpc | True | 6 | B | 1.602905799 | 0.25 | 2 | 0 |
| neutral_cpc | True | 6 | C | 1.610133858 | 0.1875 | 2 | 0 |
| neutral_cpc | True | 6 | D | 1.60245961 | 0.1875 | 2 | 0 |
| neutral_cpc | True | 8 | A | 1.594698715 | 0.4375 | 2 | 0 |
| neutral_cpc | True | 8 | B | 1.604155197 | 0.25 | 2 | 0 |
| neutral_cpc | True | 8 | C | 1.612238505 | 0.125 | 2 | 0 |
| neutral_cpc | True | 8 | D | 1.597237757 | 0.5 | 5 | 3 |
| gated_cpc | True | 0 | A | 1.606395463 | 0.125 | 2 | 0 |
| gated_cpc | True | 0 | B | 1.60887531 | 0.375 | 2 | 0 |
| gated_cpc | True | 0 | C | 1.610091769 | 0.3125 | 2 | 0 |
| gated_cpc | True | 0 | D | 1.608856365 | 0.125 | 2 | 0 |
| gated_cpc | True | 2 | A | 1.593447863 | 0.5 | 8 | 6 |
| gated_cpc | True | 2 | B | 1.610037814 | 0.125 | 2 | 0 |
| gated_cpc | True | 2 | C | 1.619891195 | 0 | 2 | 0 |
| gated_cpc | True | 2 | D | 1.595963279 | 0.5 | 8 | 6 |
| gated_cpc | True | 4 | A | 1.590085908 | 0.5 | 8 | 6 |
| gated_cpc | True | 4 | B | 1.601808111 | 0.125 | 2 | 0 |
| gated_cpc | True | 4 | C | 1.623824209 | 0 | 2 | 0 |
| gated_cpc | True | 4 | D | 1.592631873 | 0.5 | 8 | 6 |
| gated_cpc | True | 6 | A | 1.599946946 | 0.25 | 2 | 0 |
| gated_cpc | True | 6 | B | 1.602906733 | 0.25 | 2 | 0 |
| gated_cpc | True | 6 | C | 1.61013611 | 0.1875 | 2 | 0 |
| gated_cpc | True | 6 | D | 1.602459828 | 0.1875 | 2 | 0 |
| gated_cpc | True | 8 | A | 1.594700566 | 0.4375 | 2 | 0 |
| gated_cpc | True | 8 | B | 1.604156419 | 0.25 | 2 | 0 |
| gated_cpc | True | 8 | C | 1.612240719 | 0.125 | 2 | 0 |
| gated_cpc | True | 8 | D | 1.59723909 | 0.5 | 5 | 3 |

## Cost, control and information limits

80 native training calls/640rows (64wake+16replay),60PC-family/20SGD;
320 classification forwards/2560rows plus578 live forwards =898 native predictions.
320 live episodes apply258 legal steps;128 demonstration steps give386 total.
All336 environments and both cue partitions are retained. Pilot21.795272s,
worker22.6135953s,parent23.2669227s; sampled process RSS316,186,624B,
JSON1,713,927B/stderr0. Within90/120s/768MiB/8MiB and all pre-entry caps;
one RSS sampler joined, no violations/final Python thread1. Timings include native
imports, validation, model copies and frozen-state checks, not an efficiency estimate.

| Method | Replay | Final head tensor B | Optimizer B | Gradient B | Encoded replay B |
| --- | --- | ---: | ---: | ---: | ---: |
| bp | False | 468 | 468 | 468 | 0 |
| pc | False | 500 | 0 | 0 | 0 |
| neutral_cpc | False | 5908 | 0 | 0 | 0 |
| gated_cpc | False | 5908 | 0 | 0 | 0 |
| bp | True | 468 | 468 | 468 | 4032 |
| pc | True | 500 | 0 | 0 | 4032 |
| neutral_cpc | True | 5908 | 0 | 0 | 4032 |
| gated_cpc | True | 5908 | 0 | 0 | 4032 |

Tensor/encoded bytes exclude Python histories, staging, full checkpoint copies
and runtime; process RSS is separate. Same-mode BP shares initializer, train data,
selection, replay quotas and exposure. Off/on work differs; original BP/PC objective
units differ. PC/neutral parameters, scores and complete live transitions agree.

Final data and layout D never enter training, promotion, replay, tuning or probes.
Every arm finishes training before classification/live scoring. Full owned native
checkpoint copies are checked unchanged around predictions; original learner and
actor stay exact. Local `eval:` versions convey no historical publication authority.

The two cue worlds share observations after the forced corridor prefix but require
different key choices. Feedforward deterministic policies cannot universally
remember that erased observation. This structural limit does not excuse their
observed illegal decisions, establish that a recurrent learner will succeed, or
justify tuning this fixed case. Teacher cue/layout access is for collection only.

## Evidence and next learner action

[Raw actual traces](../../artifacts/runs/r42-key-door-20261016/pilot.json),
[all scores and costs](../../artifacts/runs/r42-key-door-20261016/summary.json),
[commands/failures](../../artifacts/runs/r42-key-door-20261016/RESULTS.md),
[independent review](../../artifacts/runs/r42-key-door-20261016/REVIEW.md),
[usage](../key-door-navigation.md).

Next implement a trainable recurrent behavior-cloning control through existing
NativeLearner/complete-episode ports. Separate episode-local hidden state from
frozen learned state, reset it for every rollout, and admit only observed history.
Verify bounded updates and state ownership before one frozen local BC protocol.
Keep these feedforward negatives and matched native objectives unchanged.
This is not recurrent RL or full R4.2 completion; those requirements remain open.
