# Nonlinear raw scaling measurements — 2026-10-15

**Subsequent completion:** the [three-scheme report](nonlinear-scaling-complete-20261015.md)
adds the separately frozen legacy extension and combines retained rows. This report
preserves the original two-scheme scope and its historical acceptance assessment.

## Scope and execution

One frozen `nonlinear_raw_scaling_fixed_pilot_v1` invocation completed all96cases
and288updates (72BP/216PC-family), without failure, retry or seed substitution.
The runner/observer, matched factories and deferred final scoring passed143distinct
pre-pilot engineering cases, scoped statics and independent source review.
A subsequent Windows output-byte correction passed one additional pure case;
no scientific calculation or experiment was repeated. Exact source pins, original
CLI, raw JSON and receipts are retained in
`artifacts/runs/nonlinear-scaling-measurement-20261015/`.

Grid: widths2/8, hidden depths1/3, fan-in/mean-field, seeds23/41/59,
BP/PC/neutralCPC/activeCPC. D2, three8row training batches, independent64row final
batch; fixed data seeds7427/9233, rate.01, precision1, mean-field gamma1.5.
PC training and diagnostic probes use two latent steps/.05. Active CPC retains
default policy; neutral minimum plasticity1. Label delay2, checkpoints0/1/2/3.
No final label enters training, diagnostics, thresholds, tuning or later settings.
Every case trains completely before its saved checkpoint scoring.

## Common final metrics

Primary probability BCE uses fixed1e-12 clipping; lower is better. Secondary
accuracy uses >=.5. Each mean contains all three ordered initialization seeds,
conditional on the same fixed train/final data. Intervals in the next table are
nominal marginal95% t intervals (df2/critical4.303); they are unadjusted for
multiple comparisons and do not establish population generalization. All checkpoint
values, individual seeds and448summaries remain in `pilot.json`.

| Width | Hidden | Scheme | BP BCE | PC/neutral BCE | Active CPC BCE | BP accuracy | PC/neutral accuracy | Active accuracy |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | 1 | fan_in | 0.6494737 | 0.6505702 | 0.6505836 | 0.6666667 | 0.6666667 | 0.6666667 |
| 2 | 1 | mean_field | 0.6608289 | 0.6614475 | 0.6614542 | 0.6666667 | 0.6666667 | 0.6666667 |
| 2 | 3 | fan_in | 0.6783167 | 0.6798728 | 0.6798861 | 0.7135417 | 0.7083333 | 0.7083333 |
| 2 | 3 | mean_field | 0.6838324 | 0.6846406 | 0.6846472 | 0.7135417 | 0.7083333 | 0.7083333 |
| 8 | 1 | fan_in | 0.7319263 | 0.7325817 | 0.73266 | 0.2291667 | 0.2291667 | 0.2291667 |
| 8 | 1 | mean_field | 0.7014343 | 0.7017422 | 0.7017516 | 0.2239583 | 0.2239583 | 0.2239583 |
| 8 | 3 | fan_in | 0.7165251 | 0.7212873 | 0.7213053 | 0.3958333 | 0.390625 | 0.390625 |
| 8 | 3 | mean_field | 0.6985395 | 0.6992383 | 0.6992403 | 0.390625 | 0.390625 | 0.390625 |

PC and neutral CPC have exactly equal initial/final raw parameter tensors in
all24shape/scheme/seed comparisons and identical checkpoint scores. Active CPC
has slightly higher mean final BCE in every one of eight conditions. This pilot
demonstrates no circadian benefit. PC also has higher mean final BCE than BP in
each condition. Small effect sizes, three updates and the fixed-data design do
not support a broad ranking or convergence claim.

| Width | Hidden | Scheme | PC minus BP BCE | Nominal interval | Active minus neutral BCE | Nominal interval |
|---:|---:|---|---:|---|---:|---|
| 2 | 1 | fan_in | 0.0010965 | [5.779152e-05, 0.002135209] | 1.34433e-05 | [-1.613892e-05, 4.302552e-05] |
| 2 | 1 | mean_field | 0.0006185954 | [-6.143136e-05, 0.001298622] | 6.66455e-06 | [-8.719232e-06, 2.204833e-05] |
| 2 | 3 | fan_in | 0.001556097 | [0.0003923685, 0.002719826] | 1.330876e-05 | [-2.715393e-05, 5.377145e-05] |
| 2 | 3 | mean_field | 0.0008081543 | [0.0002841789, 0.00133213] | 6.563475e-06 | [-1.61903e-05, 2.931725e-05] |
| 8 | 1 | fan_in | 0.0006553524 | [-0.0005197864, 0.001830491] | 7.834843e-05 | [1.462179e-05, 0.0001420751] |
| 8 | 1 | mean_field | 0.0003079398 | [-0.0002946961, 0.0009105757] | 9.346627e-06 | [2.057613e-06, 1.663564e-05] |
| 8 | 3 | fan_in | 0.004762232 | [-0.003767818, 0.01329228] | 1.800762e-05 | [-2.734321e-05, 6.335844e-05] |
| 8 | 3 | mean_field | 0.0006988305 | [-0.0004559032, 0.001853564] | 2.004867e-06 | [-2.774385e-06, 6.784119e-06] |

Positive paired differences mean higher loss for the left method. Neutral-minus-PC
is exactly zero; constant observed differences have no estimated uncertainty
interval. No favorable subset, seed replacement or post-result metric choice is used.

## Scaling diagnostics

The table below uses PC as the declared reference across every shape/scheme.
Norms are means across all three seeds of the Euclidean norm of all parameter
arrays including biases. Activation/latent columns are last-hidden-layer RMS,
averaged across seeds; logit RMS is retained at initial and final checkpoint.
PC final latent values follow a fresh fixed two-step training-data probe, not
equilibrium. All layer/method/checkpoint norms and actual factors remain in JSON.

| Width | Hidden | Scheme | Raw norm initial → final | Effective norm initial → final | Last hidden RMS initial → final | Logit RMS initial → final | Final relaxed latent RMS |
|---:|---:|---|---|---|---|---|---:|
| 2 | 1 | fan_in | 1.966731 → 1.966794 | 1.966731 → 1.966794 | 0.5373425 → 0.4174104 | 0.6665141 → 0.546376 | 0.4179179 |
| 2 | 1 | mean_field | 2.781378 → 2.782736 | 1.634238 → 1.634585 | 0.5373425 → 0.4174132 | 0.3141978 → 0.2578997 | 0.417951 |
| 2 | 3 | fan_in | 2.749847 → 2.749983 | 2.749847 → 2.749983 | 0.3810633 → 0.3326498 | 0.2732112 → 0.232553 | 0.3331889 |
| 2 | 3 | mean_field | 3.888871 → 3.889239 | 2.637419 → 2.637483 | 0.3810633 → 0.3326501 | 0.128793 → 0.1097779 | 0.3329746 |
| 8 | 1 | fan_in | 3.023982 → 3.023464 | 3.023982 → 3.023464 | 0.517454 → 0.4072786 | 0.2561549 → 0.2046534 | 0.4071249 |
| 8 | 1 | mean_field | 4.58693 → 4.585732 | 2.950566 → 2.950542 | 0.517454 → 0.4072773 | 0.06037629 → 0.04898619 | 0.407246 |
| 8 | 3 | fan_in | 4.928187 → 4.927946 | 4.928187 → 4.927946 | 0.295191 → 0.2433167 | 0.2229292 → 0.1640659 | 0.2431247 |
| 8 | 3 | mean_field | 11.9086 → 11.90823 | 4.845709 → 4.845701 | 0.295191 → 0.2433191 | 0.05254492 → 0.03907465 | 0.2432788 |

Initial and final feature batches differ by design (first versus last arrived
batch); changes in activation/logit RMS combine data and parameter effects.
Finite observed values across this tiny grid do not establish width/depth invariance.
Biases start at zero; actual effective bias factors and nonzero final norms are
retained rather than treating biases as unmeasured parameters.

| Width | Hidden | Scheme | Final PC energy | Task component | Hidden penalties sum | Ungated raw gradient norm | Ungated effective gradient norm | Parameter entries |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | 1 | fan_in | 0.6384544 | 0.6384293 | 2.516379e-05 | 0.2759962 | 0.2759962 | 9 |
| 2 | 1 | mean_field | 0.658236 | 0.6582301 | 5.868176e-06 | 0.1252979 | 0.2951638 | 9 |
| 2 | 3 | fan_in | 0.6648179 | 0.6648039 | 1.401152e-05 | 0.2956118 | 0.2956118 | 21 |
| 2 | 3 | mean_field | 0.6782711 | 0.6782679 | 3.231578e-06 | 0.1296686 | 0.2972028 | 21 |
| 8 | 1 | fan_in | 0.7248821 | 0.7248728 | 9.250568e-06 | 0.5034772 | 0.5034772 | 33 |
| 8 | 1 | mean_field | 0.7004566 | 0.7004561 | 5.128498e-07 | 0.06945418 | 0.4917187 | 33 |
| 8 | 3 | fan_in | 0.7226852 | 0.7226667 | 1.844565e-05 | 0.3623968 | 0.3623968 | 177 |
| 8 | 3 | mean_field | 0.700265 | 0.700264 | 9.528768e-07 | 0.06292534 | 0.353631 | 177 |

Native BP feedforward BCE and PC relaxed energy have distinct objective IDs.
Postupdate probe energies above are separate from preupdate training receipts.
CPC gradients are true ungated partials; gates precondition actual updates.
Complete raw state, precisions, policy, chemistry, gate and wake count are retained.

## Work, resources and interpretation

Training1008 + probes720 =1728nominal objective evaluations; initial probes0E.
BP uses9total nominal evaluations per three-update case; PC-family uses21.
Training exposure is equal (24rows/case;2304total), computation is different.
Training latent steps432 plus diagnostic latent steps432;384final checkpoint
forward calls score24576rows. Counts omit copies and other arithmetic; not FLOPs.
Report-stage elapsed1.9046951s/RSS43,028,480B.
Final CLI receipt elapsed1.9382164s/observed peakRSS48,676,864B;
full subprocess2.0623740s, actual saved output2,844,612B.
All are within frozen60s/256MiB/16MiB/CPU1/288update limits. No model download,
GPU, environment or replay work. JSON construction and stdout are inside the
live RSS interval. The command wall clock includes numerical imports; transient
import RSS is unobserved. RSS is sampled process memory, not an exact allocation
ceiling or native tensor footprint. Per-case times include probes/scoring and
should not be compared as training-only speed.

The original CLI receipt reports2,752,584output bytes because it counted LF
while Windows emitted CRLF. The external actual file count2,844,612B proves the
frozen cap independently. The future cap/receipt now charges every CRLF; its
prediction exactly matches the retained file, with a pure regression and scoped
statics. `OUTPUT_BYTE_CORRECTION.json` preserves the discrepancy and old CLI.
No artifact was rewritten and the pilot was not repeated.

## Assessment

R1.8b2 provides the working isolated runner and one fan-in/mean-field measurement,
with matched BP, all prespecified seeds/failures, norms/objectives and actual work.
The frozen protocol selected two schemes prospectively; supported
`legacy_gaussian` remains unmeasured. R1.8b's literal all-schemes requirement and
parent R1.8 therefore stay unchecked. This is an explicit split preserving the
original acceptance criterion, not an assertion of a three-scheme study.
R1.8a and R1.8b1 already establish derivatives, full state and neutral continuation.
This does not validate a nonlinear scaling theorem, convergence, general CPC
benefit, arbitrary geometry or raw/topology transfer. R1.6 broader structural
work, the failed R1.5b time comparison and other unfinished tasks remain open.
