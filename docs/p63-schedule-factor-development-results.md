# P6.3 matched schedule development result

The [prospective c6 contract](p63-schedule-factor-development.md) completed all
33 model cells and all 27 paired contrasts on seeds 79/83/89. The complete c5
train-only fact object and every after-A/after-B checkpoint matched globally
before the first outer value was accessed. Final-source values and ten
confirmation seeds remain unopened. These are exploratory development scores.

Two fresh bounded public processes saved exclusive request/result/audit bundles
under ignored `artifacts/runs/p63-schedule-factor-development/` and
`artifacts/runs/p63-schedule-factor-development-repeat/`, with no failure.
The deterministic results are byte-identical, SHA-256
`776df6a47efd452a9eb33aced8fa74b1938608637109bd716333fafb2edae689`.
Request hashes are
`f3b2bf5a92c99932b62679cafc6db9cf75fef7ade542400220951c7ca6c221ef` and
`1c4470dbaf747840462460f2f887e851836a85c746fbf5dac259837d957769e4`.
Independent readback verified finite JSON, frozen manifest/23-source/adapter
identities, exact c5 facts, all cells/metrics/contrasts and work/resource/hash
facts for both bundles.

## Every model cell

A-A = A after A; A-B/B-B = A/B after B, all on outer-selection roles.
Final mean and signed forgetting are the two primary metrics; negative
forgetting denotes improved A accuracy after B. Retention is the optional
zero-safe A-B/A-A ratio. Table values are rounded to six places; artifacts
retain exact float values. PC and neutral circadian are exact under every
policy, and adaptive/no-sleep outcomes are identical.

| Seed | Model | A-A | A-B | B-B | Final mean | Signed forgetting | Retention |
|---:|---|---:|---:|---:|---:|---:|---:|
| 79 | backprop_periodic | 0.958333 | 1.000000 | 1.000000 | 1.000000 | -0.041667 | 1.043478 |
| 79 | pc_periodic | 0.541667 | 0.916667 | 0.833333 | 0.875000 | -0.375000 | 1.692308 |
| 79 | neutral_periodic | 0.541667 | 0.916667 | 0.833333 | 0.875000 | -0.375000 | 1.692308 |
| 79 | backprop_adaptive | 0.958333 | 1.000000 | 1.000000 | 1.000000 | -0.041667 | 1.043478 |
| 79 | pc_adaptive | 0.541667 | 0.708333 | 0.416667 | 0.562500 | -0.166667 | 1.307692 |
| 79 | neutral_adaptive | 0.541667 | 0.708333 | 0.416667 | 0.562500 | -0.166667 | 1.307692 |
| 79 | backprop_no_sleep | 0.958333 | 1.000000 | 1.000000 | 1.000000 | -0.041667 | 1.043478 |
| 79 | pc_no_sleep | 0.541667 | 0.708333 | 0.416667 | 0.562500 | -0.166667 | 1.307692 |
| 79 | neutral_no_sleep | 0.541667 | 0.708333 | 0.416667 | 0.562500 | -0.166667 | 1.307692 |
| 79 | backprop_12_no_sleep | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 79 | pc_12_no_sleep | 0.583333 | 1.000000 | 0.916667 | 0.958333 | -0.416667 | 1.714286 |
| 83 | backprop_periodic | 1.000000 | 1.000000 | 0.833333 | 0.916667 | 0.000000 | 1.000000 |
| 83 | pc_periodic | 1.000000 | 1.000000 | 0.833333 | 0.916667 | 0.000000 | 1.000000 |
| 83 | neutral_periodic | 1.000000 | 1.000000 | 0.833333 | 0.916667 | 0.000000 | 1.000000 |
| 83 | backprop_adaptive | 1.000000 | 1.000000 | 0.833333 | 0.916667 | 0.000000 | 1.000000 |
| 83 | pc_adaptive | 0.958333 | 1.000000 | 0.750000 | 0.875000 | -0.041667 | 1.043478 |
| 83 | neutral_adaptive | 0.958333 | 1.000000 | 0.750000 | 0.875000 | -0.041667 | 1.043478 |
| 83 | backprop_no_sleep | 1.000000 | 1.000000 | 0.833333 | 0.916667 | 0.000000 | 1.000000 |
| 83 | pc_no_sleep | 0.958333 | 1.000000 | 0.750000 | 0.875000 | -0.041667 | 1.043478 |
| 83 | neutral_no_sleep | 0.958333 | 1.000000 | 0.750000 | 0.875000 | -0.041667 | 1.043478 |
| 83 | backprop_12_no_sleep | 1.000000 | 1.000000 | 0.833333 | 0.916667 | 0.000000 | 1.000000 |
| 83 | pc_12_no_sleep | 0.916667 | 1.000000 | 0.750000 | 0.875000 | -0.083333 | 1.090909 |
| 89 | backprop_periodic | 1.000000 | 1.000000 | 0.833333 | 0.916667 | 0.000000 | 1.000000 |
| 89 | pc_periodic | 0.708333 | 1.000000 | 0.750000 | 0.875000 | -0.291667 | 1.411765 |
| 89 | neutral_periodic | 0.708333 | 1.000000 | 0.750000 | 0.875000 | -0.291667 | 1.411765 |
| 89 | backprop_adaptive | 1.000000 | 1.000000 | 0.833333 | 0.916667 | 0.000000 | 1.000000 |
| 89 | pc_adaptive | 0.666667 | 1.000000 | 0.666667 | 0.833333 | -0.333333 | 1.500000 |
| 89 | neutral_adaptive | 0.666667 | 1.000000 | 0.666667 | 0.833333 | -0.333333 | 1.500000 |
| 89 | backprop_no_sleep | 1.000000 | 1.000000 | 0.833333 | 0.916667 | 0.000000 | 1.000000 |
| 89 | pc_no_sleep | 0.666667 | 1.000000 | 0.666667 | 0.833333 | -0.333333 | 1.500000 |
| 89 | neutral_no_sleep | 0.666667 | 1.000000 | 0.666667 | 0.833333 | -0.333333 | 1.500000 |
| 89 | backprop_12_no_sleep | 1.000000 | 1.000000 | 0.833333 | 0.916667 | 0.000000 | 1.000000 |
| 89 | pc_12_no_sleep | 1.000000 | 1.000000 | 0.750000 | 0.875000 | 0.000000 | 1.000000 |

## Every prespecified paired contrast

All columns are left minus right. Duplicate PC/neutral and zero adaptive
rows are retained. Equal wake work/capacity does not imply equal replay cost.

| Seed | Left − right | Δ A-A | Δ A-B | Δ B-B | Δ final mean | Δ signed forgetting |
|---:|---|---:|---:|---:|---:|---:|
| 79 | backprop_periodic − backprop_no_sleep | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 79 | backprop_adaptive − backprop_no_sleep | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 79 | backprop_adaptive − backprop_periodic | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 79 | pc_periodic − pc_no_sleep | 0.000000 | 0.208333 | 0.416667 | 0.312500 | -0.208333 |
| 79 | pc_adaptive − pc_no_sleep | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 79 | pc_adaptive − pc_periodic | 0.000000 | -0.208333 | -0.416667 | -0.312500 | 0.208333 |
| 79 | neutral_periodic − neutral_no_sleep | 0.000000 | 0.208333 | 0.416667 | 0.312500 | -0.208333 |
| 79 | neutral_adaptive − neutral_no_sleep | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 79 | neutral_adaptive − neutral_periodic | 0.000000 | -0.208333 | -0.416667 | -0.312500 | 0.208333 |
| 83 | backprop_periodic − backprop_no_sleep | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 83 | backprop_adaptive − backprop_no_sleep | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 83 | backprop_adaptive − backprop_periodic | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 83 | pc_periodic − pc_no_sleep | 0.041667 | 0.000000 | 0.083333 | 0.041667 | 0.041667 |
| 83 | pc_adaptive − pc_no_sleep | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 83 | pc_adaptive − pc_periodic | -0.041667 | 0.000000 | -0.083333 | -0.041667 | -0.041667 |
| 83 | neutral_periodic − neutral_no_sleep | 0.041667 | 0.000000 | 0.083333 | 0.041667 | 0.041667 |
| 83 | neutral_adaptive − neutral_no_sleep | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 83 | neutral_adaptive − neutral_periodic | -0.041667 | 0.000000 | -0.083333 | -0.041667 | -0.041667 |
| 89 | backprop_periodic − backprop_no_sleep | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 89 | backprop_adaptive − backprop_no_sleep | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 89 | backprop_adaptive − backprop_periodic | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 89 | pc_periodic − pc_no_sleep | 0.041667 | 0.000000 | 0.083333 | 0.041667 | 0.041667 |
| 89 | pc_adaptive − pc_no_sleep | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 89 | pc_adaptive − pc_periodic | -0.041667 | 0.000000 | -0.083333 | -0.041667 | -0.041667 |
| 89 | neutral_periodic − neutral_no_sleep | 0.041667 | 0.000000 | 0.083333 | 0.041667 | 0.041667 |
| 89 | neutral_adaptive − neutral_no_sleep | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 89 | neutral_adaptive − neutral_periodic | -0.041667 | 0.000000 | -0.083333 | -0.041667 | -0.041667 |

Descriptive paired mean ± sample SD across the three development seeds:

| Left − right | Δ A-A | Δ A-B | Δ B-B | Δ final mean | Δ signed forgetting |
|---|---:|---:|---:|---:|---:|
| backprop_periodic − backprop_no_sleep | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| backprop_adaptive − backprop_no_sleep | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| backprop_adaptive − backprop_periodic | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| pc_periodic − pc_no_sleep | 0.027778 ± 0.024056 | 0.069444 ± 0.120281 | 0.194444 ± 0.192450 | 0.131944 ± 0.156366 | -0.041667 ± 0.144338 |
| pc_adaptive − pc_no_sleep | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| pc_adaptive − pc_periodic | -0.027778 ± 0.024056 | -0.069444 ± 0.120281 | -0.194444 ± 0.192450 | -0.131944 ± 0.156366 | 0.041667 ± 0.144338 |
| neutral_periodic − neutral_no_sleep | 0.027778 ± 0.024056 | 0.069444 ± 0.120281 | 0.194444 ± 0.192450 | 0.131944 ± 0.156366 | -0.041667 ± 0.144338 |
| neutral_adaptive − neutral_no_sleep | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| neutral_adaptive − neutral_periodic | -0.027778 ± 0.024056 | -0.069444 ± 0.120281 | -0.194444 ± 0.192450 | -0.131944 ± 0.156366 | 0.041667 ± 0.144338 |

Periodic replay improves ordinary PC and neutral final mean by .3125,
.041667, .041667; backprop has zero accuracy differences on all seeds.
PC/neutral gains use the same replay rows, applied updates and outcomes.
There is no circadian-specific outcome advantage in this factor. The
inactive adaptive rule matches no-sleep, so adaptive-minus-periodic is
negative for PC/neutral final mean on every seed.

Signed forgetting for periodic-minus-no-sleep is −.208333, +.041667,
+.041667 in PC/neutral. On seeds 83/89, A-after-B is unchanged while periodic
A-after-A is .041667 higher; the positive forgetting difference reflects
that starting score. The seed 79 decrease accompanies higher final A and B
accuracy. The three-seed summaries are descriptive, not significance or
independent confirmation. Neither a new trigger nor a winning configuration
was selected.

## Work and capacity beside the outcomes

These per-model facts apply to each of the three seeds. Width/parameter
counts are unchanged at initial, final and peak. Every head receives 24 wake
updates/1,296 row presentations. PC/neutral use 48 wake latent loops/2,592
example-iterations. Each periodic PC/neutral adds 24 replay latent
loops/example-iterations. All rejected replay execution counts are zero in
the official runs; rejection accounting remains covered by c5 tests.

| Model | Width/parameters | Applied replay updates | Executed optimizer updates | Own sleep attempts | Inner guard evaluations | Replay array access |
|---|---:|---:|---:|---:|---:|---|
| backprop_periodic | 8/33 | 12 | 36 | 0 | 0 | shared_fifo |
| pc_periodic | 8/33 | 12 | 36 | 0 | 0 | shared_fifo |
| neutral_periodic | 8/33 | 12 | 36 | 6 | 12 | model_fifo |
| backprop_adaptive | 8/33 | 0 | 24 | 0 | 0 | shared_fifo |
| pc_adaptive | 8/33 | 0 | 24 | 0 | 0 | shared_fifo |
| neutral_adaptive | 8/33 | 0 | 24 | 0 | 0 | model_fifo |
| backprop_no_sleep | 8/33 | 0 | 24 | 0 | 0 | shared_fifo |
| pc_no_sleep | 8/33 | 0 | 24 | 0 | 0 | shared_fifo |
| neutral_no_sleep | 8/33 | 0 | 24 | 0 | 0 | model_fifo |
| backprop_12_no_sleep | 12/49 | 0 | 24 | 0 | 0 | none |
| pc_12_no_sleep | 12/49 | 0 | 24 | 0 | 0 | none |

The neutral controller committed all six periodic attempts per seed and
none for adaptive/no-sleep, with 36 inner evaluations/648 guard examples
across the study. Its guard also controls corresponding PC/backprop replay;
backprop did not use an independent guard. All periodic methods consumed
identical proposed/applied IDs. Policies had identical available FIFO supply
at every epoch, including epochs without a sleep. Wider references did not
replay and have more parameters/different per-update compute.

Total executed training work is **900 updates** (792 wake + 108 applied
replay), within the maximum 1,014/hard cap 1,100. Scoring adds **99 outer
forward evaluations/1,980 examples**, directly observed by the final-sealed
test and verified in artifacts. All 33 per-model and 216 decision facts,
including every role/checkpoint hash and replay ID, remain in `train_facts`
identical to c5 byte-bound reference SHA `87931c5f...4c813c41`.

Persistent labeled-array scope per seed is 192 shared FIFO bytes plus 192
in each of three neutral model buffers (768 total). Baselines consume private
row copies from the shared supply; metadata, temporary copies and parameters
are outside that array-byte count. Whole-worker RSS includes those allocations,
all held checkpoint copies and runtime overhead.

Workers took .645/.643 seconds with sampled whole-worker RSS peaks
44,539,904/44,630,016 bytes and 23/24 samples, under the 120-second/256-MiB
caps. Runtime: Windows 11, Intel Core i7-12700K, Python 3.14.7, NumPy 2.4.6,
CPU. Sampling can miss a brief peak; these measurements are not per-arm.

The related 67-test gate, repository Ruff/mypy (339 source files), four-file
format and diff checks passed. Full CPU/CUDA suites, broad sweeps, per-arm
wall/RSS attribution, confirmation and final-role release were skipped.
P6.3c6 is complete for this scored development factor. Full/full-minus-one,
scheduled/random growth and independent confirmation remain open. Next,
inspect existing combined-component and component-toggle paths and freeze a
budgeted train-only full/minus-one matrix with its missing matched controls;
no c6 score may tune that protocol or open confirmation/final roles.
