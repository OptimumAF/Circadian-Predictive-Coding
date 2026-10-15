# Complete nonlinear scaling matrix — 2026-10-15

## Execution and scope

The separate `nonlinear_raw_legacy_extension_v1` invocation completed all 48
legacy-Gaussian cases and 144 updates. It follows 40 new focused tests, 24
original-default regression cases, nine pure aggregation controls, scoped lint/
format/type checks and independent source review. No failed seeds, retries,
warmup, metric changes or baseline tuning occurred.

The [earlier report](nonlinear-scaling-measurement-20261015.md) retains the original
96 fan-in/mean-field cases. Those models were not rerun. A retained-only combiner
verified both frozen protocol descriptions, exact data/roles/IDs, each origin’s
prescribed keys and the complete union. `combined.json` preserves the full 144
rows, both protocol origins, all 672 summaries and 36 exact PC/neutral full-stream
comparisons. The original files and source identities remain intact.

All three schemes use widths 2/8, hidden depths 1/3 and seeds 23/41/59; BP, PC,
neutral CPC and active CPC each receive three batches of eight training rows.
The independent final batch has 64 rows. Data seeds are 7427/9233, uniform[-1,1],
with target x0 + .25*x1 >= 0. D=2, base rate .01, unit precisions, PC inference
2 steps/.05, delay 2 and checkpoints 0/1/2/3 are unchanged. Active CPC uses its
default policy and neutral minimum plasticity is 1. Legacy has raw standard
deviation .5, gamma 1 and unit forward/rate/bias factors. Mean-field gamma is
1.5. Within every condition all methods share exact initial raw/effective models.

All training finishes before any final checkpoint score in each case. Final
labels enter neither training probes, selection nor settings for later cases.
The additional scheme was required by the preserved plan, not selected from
earlier outcomes. Two complementary prospective protocols replace a redundant
combined rerun; all original scientific requirements and resource limits remain.

## Final common metrics

Primary BCE clips probabilities at 1e-12; lower is better. Accuracy uses >=.5.
Each cell is the mean of all three ordered initialization seeds, conditional on
the same fixed data. PC and neutral CPC are exactly equal at every checkpoint.

| Width | Hidden | Scheme | BP BCE | PC/neutral BCE | Active BCE | BP accuracy | PC/neutral accuracy | Active accuracy |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | 1 | fan_in | 0.6494737 | 0.6505702 | 0.6505836 | 0.6666667 | 0.6666667 | 0.6666667 |
| 2 | 1 | legacy_gaussian | 0.6586859 | 0.6595063 | 0.659512 | 0.6614583 | 0.6614583 | 0.6614583 |
| 2 | 1 | mean_field | 0.6608289 | 0.6614475 | 0.6614542 | 0.6666667 | 0.6666667 | 0.6666667 |
| 2 | 3 | fan_in | 0.6783167 | 0.6798728 | 0.6798861 | 0.7135417 | 0.7083333 | 0.7083333 |
| 2 | 3 | legacy_gaussian | 0.6873919 | 0.6877703 | 0.6877715 | 0.7083333 | 0.703125 | 0.703125 |
| 2 | 3 | mean_field | 0.6838324 | 0.6846406 | 0.6846472 | 0.7135417 | 0.7083333 | 0.7083333 |
| 8 | 1 | fan_in | 0.7319263 | 0.7325817 | 0.73266 | 0.2291667 | 0.2291667 | 0.2291667 |
| 8 | 1 | legacy_gaussian | 0.7336975 | 0.7352184 | 0.7352575 | 0.234375 | 0.234375 | 0.234375 |
| 8 | 1 | mean_field | 0.7014343 | 0.7017422 | 0.7017516 | 0.2239583 | 0.2239583 | 0.2239583 |
| 8 | 3 | fan_in | 0.7165251 | 0.7212873 | 0.7213053 | 0.3958333 | 0.390625 | 0.390625 |
| 8 | 3 | legacy_gaussian | 0.7484754 | 0.7615876 | 0.7616384 | 0.421875 | 0.390625 | 0.390625 |
| 8 | 3 | mean_field | 0.6985395 | 0.6992383 | 0.6992403 | 0.390625 | 0.390625 | 0.390625 |

Active CPC has slightly higher mean final BCE than neutral CPC in all 12
conditions; PC has higher mean final BCE than BP in all 12. This does not
demonstrate circadian benefit. Three updates and three fixed-data seeds do not
establish a broad ranking, convergence or generalization.

| Width | Hidden | Scheme | PC minus BP BCE | Nominal 95% interval | Active minus neutral BCE | Nominal 95% interval |
|---:|---:|---|---:|---|---:|---|
| 2 | 1 | fan_in | 0.0010965 | [5.779152e-05, 0.002135209] | 1.34433e-05 | [-1.613892e-05, 4.302552e-05] |
| 2 | 1 | legacy_gaussian | 0.0008203334 | [-0.0001127221, 0.001753389] | 5.779471e-06 | [-7.92371e-06, 1.948265e-05] |
| 2 | 1 | mean_field | 0.0006185954 | [-6.143136e-05, 0.001298622] | 6.66455e-06 | [-8.719232e-06, 2.204833e-05] |
| 2 | 3 | fan_in | 0.001556097 | [0.0003923685, 0.002719826] | 1.330876e-05 | [-2.715393e-05, 5.377145e-05] |
| 2 | 3 | legacy_gaussian | 0.000378338 | [-9.654314e-05, 0.0008532192] | 1.242079e-06 | [-3.115189e-06, 5.599347e-06] |
| 2 | 3 | mean_field | 0.0008081543 | [0.0002841789, 0.00133213] | 6.563475e-06 | [-1.61903e-05, 2.931725e-05] |
| 8 | 1 | fan_in | 0.0006553524 | [-0.0005197864, 0.001830491] | 7.834843e-05 | [1.462179e-05, 0.0001420751] |
| 8 | 1 | legacy_gaussian | 0.001520926 | [-0.00104119, 0.004083043] | 3.907208e-05 | [3.113587e-06, 7.503058e-05] |
| 8 | 1 | mean_field | 0.0003079398 | [-0.0002946961, 0.0009105757] | 9.346627e-06 | [2.057613e-06, 1.663564e-05] |
| 8 | 3 | fan_in | 0.004762232 | [-0.003767818, 0.01329228] | 1.800762e-05 | [-2.734321e-05, 6.335844e-05] |
| 8 | 3 | legacy_gaussian | 0.01311224 | [-0.009184872, 0.03540936] | 5.078067e-05 | [-6.847298e-05, 0.0001700343] |
| 8 | 3 | mean_field | 0.0006988305 | [-0.0004559032, 0.001853564] | 2.004867e-06 | [-2.774385e-06, 6.784119e-06] |

Positive differences mean higher loss for the left method. Intervals are
nominal marginal t intervals (df=2, critical 4.303), conditional on fixed data
and an unverified seed-effects model. They have no multiplicity adjustment or
population coverage guarantee. Neutral-minus-PC is exactly zero with no estimated
interval for that constant difference. All individual seed values and every
checkpoint/accuracy interval remain in the combined artifact.

## Parameter and activation scales

PC is the reference in these descriptive tables, covering every shape and scheme.
Norms combine all parameter arrays, including biases, then average across seeds.
Hidden and latent RMS refer to the last hidden layer. Initial/final feature
probes use first/last training batches respectively, so their differences include
data variation. All other layers, methods, actual factors and tensor bytes are
retained rather than inferred from these averages.

| Width | Hidden | Scheme | Raw norm initial → final | Effective norm initial → final | Hidden RMS initial → final | Logit RMS initial → final | Final relaxed latent RMS |
|---:|---:|---|---|---|---|---|---:|
| 2 | 1 | fan_in | 1.966731 → 1.966794 | 1.966731 → 1.966794 | 0.5373425 → 0.4174104 | 0.6665141 → 0.546376 | 0.4179179 |
| 2 | 1 | legacy_gaussian | 1.390689 → 1.391224 | 1.390689 → 1.391224 | 0.43669 → 0.3166087 | 0.3883857 → 0.2941006 | 0.3173682 |
| 2 | 1 | mean_field | 2.781378 → 2.782736 | 1.634238 → 1.634585 | 0.5373425 → 0.4174132 | 0.3141978 → 0.2578997 | 0.417951 |
| 2 | 3 | fan_in | 2.749847 → 2.749983 | 2.749847 → 2.749983 | 0.3810633 → 0.3326498 | 0.2732112 → 0.232553 | 0.3331889 |
| 2 | 3 | legacy_gaussian | 1.944436 → 1.944568 | 1.944436 → 1.944568 | 0.184321 → 0.1460026 | 0.0948522 → 0.07260125 | 0.1465064 |
| 2 | 3 | mean_field | 3.888871 → 3.889239 | 2.637419 → 2.637483 | 0.3810633 → 0.3326501 | 0.128793 → 0.1097779 | 0.3329746 |
| 8 | 1 | fan_in | 3.023982 → 3.023464 | 3.023982 → 3.023464 | 0.517454 → 0.4072786 | 0.2561549 → 0.2046534 | 0.4071249 |
| 8 | 1 | legacy_gaussian | 2.293465 → 2.292733 | 2.293465 → 2.292733 | 0.4170439 → 0.3121545 | 0.292288 → 0.2187567 | 0.311944 |
| 8 | 1 | mean_field | 4.58693 → 4.585732 | 2.950566 → 2.950542 | 0.517454 → 0.4072773 | 0.06037629 → 0.04898619 | 0.407246 |
| 8 | 3 | fan_in | 4.928187 → 4.927946 | 4.928187 → 4.927946 | 0.295191 → 0.2433167 | 0.2229292 → 0.1640659 | 0.2431247 |
| 8 | 3 | legacy_gaussian | 5.954301 → 5.953781 | 5.954301 → 5.953781 | 0.4211804 → 0.3404497 | 0.4987939 → 0.3478184 | 0.3400875 |
| 8 | 3 | mean_field | 11.9086 → 11.90823 | 4.845709 → 4.845701 | 0.295191 → 0.2433191 | 0.05254492 → 0.03907465 | 0.2432788 |

| Width | Hidden | Scheme | Final PC energy | Task component | Penalties sum | Ungated raw gradient norm | Ungated effective gradient norm | Parameter entries |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | 1 | fan_in | 0.6384544 | 0.6384293 | 2.516379e-05 | 0.2759962 | 0.2759962 | 9 |
| 2 | 1 | legacy_gaussian | 0.6544527 | 0.6544396 | 1.309314e-05 | 0.2659856 | 0.2659856 | 9 |
| 2 | 1 | mean_field | 0.658236 | 0.6582301 | 5.868176e-06 | 0.1252979 | 0.2951638 | 9 |
| 2 | 3 | fan_in | 0.6648179 | 0.6648039 | 1.401152e-05 | 0.2956118 | 0.2956118 | 21 |
| 2 | 3 | legacy_gaussian | 0.6831436 | 0.6831362 | 7.389047e-06 | 0.2601478 | 0.2601478 | 21 |
| 2 | 3 | mean_field | 0.6782711 | 0.6782679 | 3.231578e-06 | 0.1296686 | 0.2972028 | 21 |
| 8 | 1 | fan_in | 0.7248821 | 0.7248728 | 9.250568e-06 | 0.5034772 | 0.5034772 | 33 |
| 8 | 1 | legacy_gaussian | 0.7253236 | 0.7253052 | 1.84606e-05 | 0.4234325 | 0.4234325 | 33 |
| 8 | 1 | mean_field | 0.7004566 | 0.7004561 | 5.128498e-07 | 0.06945418 | 0.4917187 | 33 |
| 8 | 3 | fan_in | 0.7226852 | 0.7226667 | 1.844565e-05 | 0.3623968 | 0.3623968 | 177 |
| 8 | 3 | legacy_gaussian | 0.7620855 | 0.7620444 | 4.110464e-05 | 0.4498023 | 0.4498023 | 177 |
| 8 | 3 | mean_field | 0.700265 | 0.700264 | 9.528768e-07 | 0.06292534 | 0.353631 | 177 |

PC energy above is a new postupdate fixed two-step probe on arrived training
data, not equilibrium. BP uses full feedforward BCE with a distinct objective ID.
Native training receipts contain separate preupdate values. CPC diagnostics report
ungated objective partials; plasticity preconditions the applied update. Complete
raw states retain precisions, policies, chemical/gate traces and wake counts.
Finite measurements on this grid do not prove width/depth invariance.

## Work and resources

Legacy: terminal 0, full subprocess 1.2152433s, CLI 1.0910363s,
observed process peak RSS 44,793,856B and actual output 1,419,293B.
The newline-aware CLI byte receipt exactly matches the saved artifact. All are
within the frozen 60s/256MiB/16MiB/CPU1 limits. Import RSS transients remain
unobserved; sampled RSS is not an exact allocation ceiling or model-only memory.
Factories, probes, scoring, aggregation, actual encoding and stdout are measured
within the existing command interval. Report-stage time/RSS has an earlier endpoint.

Legacy work: 144 updates, 504 training + 360 probe objective calls, 192 final
forward calls, 1152 training-row exposures and 12288 scored rows. Together with
accepted original work: 432 updates, 1512 training + 1080 probe = 2592 nominal
objective calls, 576 score forwards, 3456 training-row exposures and 36864 scored
rows. PC-family training and probe latent steps total 1296 across both protocols.
Counts exclude copies/other arithmetic and are not FLOPs or equal-compute evidence.
Timings come from separate invocations; no comparative speed claim is made.

The retained-only combined artifact is 4,111,185B; aggregation ran no learners.
Combined data must match each original protocol exactly. Origin keys are checked
before union validation, so swapping rows between protocols is refused. No source
report, seed or score was changed. The earlier Windows byte-counter discrepancy
remains documented in its original report; that artifact also passed the actual cap.

## Acceptance assessment

| Original requirement | Evidence |
|---|---|
| Explicit activation/bias/BCE/precision/plasticity mapping | R1.8a core/guide/ADR0263 and 90 accepted derivative/state/extreme-product cases |
| Exact initialized state/config and owned continuation | Retained complete initial/raw/wake states; accepted R1.8a and 62-case matched-BP controls; current four-method manual-reference runtime test |
| Independent derivatives and genuine neutral control | Accepted all-parameter/bias/latent derivatives and native agreement; all 36 PC/neutral full streams exact in the complete grid |
| Matched baseline and isolated final evaluation | Shared raw initialization/rates; accepted copied-observer/role/failure controls, unchanged native runner; final scoring after all training |
| All schemes/seeds/failures with scales, quality and work | All 144 declared cases, 672 complete three-seed summaries and raw diagnostics; both frozen protocols and terminal/resource receipts |
| Preserve defaults, failures and negative evidence | 24 default regression cases; no original-cell rerun, parameter/seed/metric change or artifact rewrite; earlier failed budgets remain failed |

Independent final assessment accepts R1.8b and parent R1.8 for this explicit
local mapping and bounded measurement scope. Both canonical tasks are complete.
It does not complete broader R1.6 split/merge/scaled-topology transfer, the
failed R1.5b timing comparison or other unfinished roadmap gates. No nonlinear
asymptotic theorem, converged learner or general circadian benefit is claimed.
