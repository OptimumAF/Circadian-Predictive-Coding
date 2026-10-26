# Categorical stream engineering case — 2026-10-16

Synthetic complete three-action records traverse the existing ordinary stream,
whole-packet replay and owner-local continuation. This is not an environment
rollout or measured reward experiment. All eight fixed method/replay arms complete;
same-mode BP shares data, initialization, memory caps, exposure and selection.
Off/on work and native objectives differ; no equal-compute or causal replay claim.

## Every declared outcome

B/C gain = initial CE minus post-context CE. A loss = CE3−CE1 (lower preferable);
A return gain = CE3−CE4. Signs/directions were frozen before execution.

| Method | Replay | B gain | C gain | A loss | A return gain |
| --- | --- | ---: | ---: | ---: | ---: |
| bp | False | 1.160674617e-06 | -6.407387393e-05 | -2.499787632e-05 | 3.267878978e-05 |
| pc | False | 3.302035231e-06 | 4.020622852e-06 | 1.17524304e-05 | 8.153295902e-06 |
| neutral_cpc | False | 3.302035231e-06 | 4.020622852e-06 | 1.17524304e-05 | 8.153295902e-06 |
| gated_cpc | False | 3.302035231e-06 | 4.020622852e-06 | 1.17524304e-05 | 8.153295902e-06 |
| bp | True | 1.340213045e-05 | -0.0001012553092 | -4.354491811e-05 | 8.996931933e-05 |
| pc | True | 1.013209698e-05 | 1.931623719e-06 | 1.63892152e-05 | 1.621698149e-05 |
| neutral_cpc | True | 1.013209698e-05 | 1.931623719e-06 | 1.63892152e-05 | 1.621698149e-05 |
| gated_cpc | True | 1.010220225e-05 | 1.901976703e-06 | 1.635934117e-05 | 1.615723502e-05 |

All120 accuracies remain unchanged: A0, B2/3, C1/3, for every checkpoint and arm.
A remains unsolved; tiny CE changes do not establish acquisition, retention or
context recovery. BP C gains are negative; PC-family A losses are positive.
Active/neutral/PC off outcomes are equal; active on differences are tiny and mixed.
No useful adaptation, circadian advantage or held-out-layout generalization follows.
No seed, threshold, rate, metric or baseline was tuned after these outcomes.

## All checkpoint scores

Each cell is CE / accuracy; JSON retains full precision and all120 points.

| Method | Replay | Wake | A | B | C |
| --- | --- | ---: | ---: | ---: | ---: |
| bp | False | 0 | 1.10020859017 / 0 | 1.09877679795 / 0.666667 | 1.09686017901 / 0.333333 |
| bp | False | 1 | 1.10016968149 / 0 | 1.09878043507 / 0.666667 | 1.0968948726 / 0.333333 |
| bp | False | 2 | 1.1001384871 / 0 | 1.09877563727 / 0.666667 | 1.09693057272 / 0.333333 |
| bp | False | 3 | 1.10014468362 / 0 | 1.09877565133 / 0.666667 | 1.09692425288 / 0.333333 |
| bp | False | 4 | 1.10011200483 / 0 | 1.09877928791 / 0.666667 | 1.09695286351 / 0.333333 |
| pc | False | 0 | 1.10020859017 / 0 | 1.09877679795 / 0.666667 | 1.09686017901 / 0.333333 |
| pc | False | 1 | 1.10020043658 / 0 | 1.09878029679 / 0.666667 | 1.09686466775 / 0.333333 |
| pc | False | 2 | 1.10020507297 / 0 | 1.09877349591 / 0.666667 | 1.09686678565 / 0.333333 |
| pc | False | 3 | 1.10021218901 / 0 | 1.09877709908 / 0.666667 | 1.09685615839 / 0.333333 |
| pc | False | 4 | 1.10020403572 / 0 | 1.09878062908 / 0.666667 | 1.09686058946 / 0.333333 |
| neutral_cpc | False | 0 | 1.10020859017 / 0 | 1.09877679795 / 0.666667 | 1.09686017901 / 0.333333 |
| neutral_cpc | False | 1 | 1.10020043658 / 0 | 1.09878029679 / 0.666667 | 1.09686466775 / 0.333333 |
| neutral_cpc | False | 2 | 1.10020507297 / 0 | 1.09877349591 / 0.666667 | 1.09686678565 / 0.333333 |
| neutral_cpc | False | 3 | 1.10021218901 / 0 | 1.09877709908 / 0.666667 | 1.09685615839 / 0.333333 |
| neutral_cpc | False | 4 | 1.10020403572 / 0 | 1.09878062908 / 0.666667 | 1.09686058946 / 0.333333 |
| gated_cpc | False | 0 | 1.10020859017 / 0 | 1.09877679795 / 0.666667 | 1.09686017901 / 0.333333 |
| gated_cpc | False | 1 | 1.10020043658 / 0 | 1.09878029679 / 0.666667 | 1.09686466775 / 0.333333 |
| gated_cpc | False | 2 | 1.10020507297 / 0 | 1.09877349591 / 0.666667 | 1.09686678565 / 0.333333 |
| gated_cpc | False | 3 | 1.10021218901 / 0 | 1.09877709908 / 0.666667 | 1.09685615839 / 0.333333 |
| gated_cpc | False | 4 | 1.10020403572 / 0 | 1.09878062908 / 0.666667 | 1.09686058946 / 0.333333 |
| bp | True | 0 | 1.10020859017 / 0 | 1.09877679795 / 0.666667 | 1.09686017901 / 0.333333 |
| bp | True | 1 | 1.10016968149 / 0 | 1.09878043507 / 0.666667 | 1.0968948726 / 0.333333 |
| bp | True | 2 | 1.10011401996 / 0 | 1.09876339582 / 0.666667 | 1.09696686619 / 0.333333 |
| bp | True | 3 | 1.10012613658 / 0 | 1.09875662421 / 0.666667 | 1.09696143432 / 0.333333 |
| bp | True | 4 | 1.10003616726 / 0 | 1.09875586756 / 0.666667 | 1.09705105352 / 0.333333 |
| pc | True | 0 | 1.10020859017 / 0 | 1.09877679795 / 0.666667 | 1.09686017901 / 0.333333 |
| pc | True | 1 | 1.10020043658 / 0 | 1.09878029679 / 0.666667 | 1.09686466775 / 0.333333 |
| pc | True | 2 | 1.1002097095 / 0 | 1.09876666585 / 0.666667 | 1.09686890415 / 0.333333 |
| pc | True | 3 | 1.1002168258 / 0 | 1.09877023977 / 0.666667 | 1.09685824739 / 0.333333 |
| pc | True | 4 | 1.10020060882 / 0 | 1.09877736058 / 0.666667 | 1.09686723072 / 0.333333 |
| neutral_cpc | True | 0 | 1.10020859017 / 0 | 1.09877679795 / 0.666667 | 1.09686017901 / 0.333333 |
| neutral_cpc | True | 1 | 1.10020043658 / 0 | 1.09878029679 / 0.666667 | 1.09686466775 / 0.333333 |
| neutral_cpc | True | 2 | 1.1002097095 / 0 | 1.09876666585 / 0.666667 | 1.09686890415 / 0.333333 |
| neutral_cpc | True | 3 | 1.1002168258 / 0 | 1.09877023977 / 0.666667 | 1.09685824739 / 0.333333 |
| neutral_cpc | True | 4 | 1.10020060882 / 0 | 1.09877736058 / 0.666667 | 1.09686723072 / 0.333333 |
| gated_cpc | True | 0 | 1.10020859017 / 0 | 1.09877679795 / 0.666667 | 1.09686017901 / 0.333333 |
| gated_cpc | True | 1 | 1.10020043658 / 0 | 1.09878029679 / 0.666667 | 1.09686466775 / 0.333333 |
| gated_cpc | True | 2 | 1.1002097095 / 0 | 1.09876669575 / 0.666667 | 1.09686890399 / 0.333333 |
| gated_cpc | True | 3 | 1.10021679592 / 0 | 1.09877026952 / 0.666667 | 1.09685827703 / 0.333333 |
| gated_cpc | True | 4 | 1.10020063869 / 0 | 1.09877736058 / 0.666667 | 1.09686723072 / 0.333333 |

## Observed cost and state payloads

32 wakes+8 replay =40 native calls,120 training-row exposures;120 final forwards/
360 scoring rows; eight resumes. Exact native profile:30 PC-family/10 SGD, one
joined RSS sampler, no violations/final Python thread1. Pilot8.4037506s,
worker8.8360946s,parent9.3425435s; observed sampled process RSS313,229,312B,
JSON209,754B/stderr0. Within60/90s,768MiB/2MiB and all pre-entry call limits.
No probe, sleep, promotion, backbone, renewal or nested process.

All four replay arms select training packet1 then packet3, charging three rows
each before native work. Each event retains968B/three transitions under the
1936B/six-transition cap. Recent/reservoir overlap is preserved; capacity is not
forced to fill. Header/dummy bytes count. No final packet is replayed.

| Method | Replay | Final head tensor B | Final optimizer B | Final gradient B | Encoded replay B |
| --- | --- | ---: | ---: | ---: | ---: |
| bp | False | 84 | 84 | 84 | 0 |
| pc | False | 96 | 0 | 0 | 0 |
| neutral_cpc | False | 5284 | 0 | 0 | 0 |
| gated_cpc | False | 5284 | 0 | 0 | 0 |
| bp | True | 84 | 84 | 84 | 968 |
| pc | True | 96 | 0 | 0 | 968 |
| neutral_cpc | True | 5284 | 0 | 0 | 968 |
| gated_cpc | True | 5284 | 0 | 0 | 968 |

Tensor/encoded payloads exclude Python records, staging/copies and runtime storage;
process RSS is separate and does not attribute an overhead or efficiency effect.
Final scoring begins after all training per arm on detached copies and never
feeds updates, replay, promotion or probe/metric selection. Original source and
deployed actor remain unchanged; continuation preserves original budgets and
permanent charges. No historical serialized migration or durable restart claim.

## Evidence and next implementation

[Raw case](../../artifacts/runs/r42-categorical-stream-20261016/pilot.json),
[all scores/costs](../../artifacts/runs/r42-categorical-stream-20261016/summary.json),
[commands/failures](../../artifacts/runs/r42-categorical-stream-20261016/RESULTS.md),
[independent review](../../artifacts/runs/r42-categorical-stream-20261016/REVIEW.md),
[usage](../categorical-trajectory-rehearsal.md).

Original R4.2 remains open. Next implement actual partial-observation key/door
demonstrations and complete live episodes through these existing ports. Preserve
hidden cue, A→B→C→A, changed layouts/visuals, old-room recall, held-out layouts and
separately evaluated recurrent RL. Do not relabel these synthetic packets as a game
or start another near-identical snapshot/provenance increment.
