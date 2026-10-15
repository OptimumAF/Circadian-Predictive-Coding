# Linear MLP scaling measurements — 2026-10-14

## Protocol and execution

One fixed invocation of `python -m scripts.run_linear_scaling_pilot` followed
correctness/statics and independent source review. All 81 declared cases
(three seeds × three widths × three hidden depths × three schemes) completed
six updates, with no refused, replaced, retried or omitted cases. The fixed
protocol and source identity are retained in
`artifacts/runs/linear-scaling-measurement-20261014/{PROTOCOL.md,SOURCE_VERSION.json}`.
Raw tensors, every checkpoint norm, timing, heldout score and ordered seed
summary are in `pilot.json` in the same directory. Prior negative R1.7 results
are unchanged. This is a linear regression measurement, not a CPC comparison.

The [runner guide](../linear-scaling-measurement.md) documents exact data, SGD,
normalizations and failure behavior. The protocol uses identity activation,
unit precision, no bias/skip, D=2, gamma0=1, learning rate .01; seeds 23/41/59,
widths 2/8/32, hidden layers 1/2/4. Each case trains all six steps before any
heldout scoring. Neither scores nor heldout targets influence later fixed cases.

## Complete heldout half-MSE summaries

All tables retain all three initialization seeds, conditional on the same fixed
16 training and 64 final-test rows. Lower half-MSE is better. Intervals are
nominal marginal 95% t (df 2, critical 4.303), with untested normal seed effects
and no simultaneous coverage. Negative interval endpoints are retained from
the declared untruncated t formula; losses themselves are nonnegative. No
significance, selected-winner, asymptotic or convergence claim follows.

### Before training

| Width | Hidden layers | Scheme | Mean | Seed range | Nominal 95% interval |
|---:|---:|---|---:|---|---|
| 2 | 1 | legacy_gaussian | 0.215676 | [0.0675828, 0.419804] | [-0.238149, 0.669501] |
| 2 | 1 | fan_in | 0.482816 | [0.0880503, 1.12677] | [-0.914395, 1.88003] |
| 2 | 1 | mean_field | 0.30246 | [0.0641511, 0.660174] | [-0.481206, 1.08613] |
| 2 | 2 | legacy_gaussian | 0.14564 | [0.113565, 0.171803] | [0.0721894, 0.21909] |
| 2 | 2 | fan_in | 0.246367 | [0.168563, 0.333086] | [0.041103, 0.451631] |
| 2 | 2 | mean_field | 0.184699 | [0.152187, 0.241944] | [0.0611571, 0.30824] |
| 2 | 4 | legacy_gaussian | 0.13016 | [0.0967136, 0.151683] | [0.0572188, 0.203102] |
| 2 | 4 | fan_in | 0.172666 | [0.130428, 0.239636] | [0.0269488, 0.318384] |
| 2 | 4 | mean_field | 0.136236 | [0.0773705, 0.197373] | [-0.0129072, 0.285379] |
| 8 | 1 | legacy_gaussian | 0.18496 | [0.0744336, 0.244539] | [-0.0530793, 0.423] |
| 8 | 1 | fan_in | 0.18496 | [0.0744336, 0.244539] | [-0.0530793, 0.423] |
| 8 | 1 | mean_field | 0.149825 | [0.103759, 0.176705] | [0.0502548, 0.249395] |
| 8 | 2 | legacy_gaussian | 0.131628 | [0.00606451, 0.36398] | [-0.368822, 0.632077] |
| 8 | 2 | fan_in | 0.109713 | [0.0254909, 0.258062] | [-0.210436, 0.429861] |
| 8 | 2 | mean_field | 0.118407 | [0.0911491, 0.162057] | [0.0235297, 0.213285] |
| 8 | 4 | legacy_gaussian | 0.596006 | [0.491784, 0.771579] | [0.216061, 0.975952] |
| 8 | 4 | fan_in | 0.199389 | [0.0643413, 0.300084] | [-0.102627, 0.501405] |
| 8 | 4 | mean_field | 0.151256 | [0.0912339, 0.191684] | [0.019551, 0.28296] |
| 32 | 1 | legacy_gaussian | 0.654599 | [0.425116, 1.10229] | [-0.308701, 1.6179] |
| 32 | 1 | fan_in | 0.255202 | [0.0732827, 0.446726] | [-0.209137, 0.719542] |
| 32 | 1 | mean_field | 0.14365 | [0.101714, 0.172803] | [0.0511623, 0.236137] |
| 32 | 2 | legacy_gaussian | 14.264 | [1.75553, 25.0009] | [-14.8612, 43.3893] |
| 32 | 2 | fan_in | 0.575306 | [0.297017, 0.851438] | [-0.113396, 1.26401] |
| 32 | 2 | mean_field | 0.156742 | [0.0978771, 0.204398] | [0.0222452, 0.291239] |
| 32 | 4 | legacy_gaussian | 260.212 | [4.61291, 412.478] | [-293.059, 813.483] |
| 32 | 4 | fan_in | 0.100115 | [0.00604637, 0.182646] | [-0.120652, 0.320881] |
| 32 | 4 | mean_field | 0.118251 | [0.0902486, 0.138901] | [0.0557803, 0.180722] |

### After six updates

| Width | Hidden layers | Scheme | Mean | Seed range | Nominal 95% interval |
|---:|---:|---|---:|---|---|
| 2 | 1 | legacy_gaussian | 0.21044 | [0.0661612, 0.40939] | [-0.231835, 0.652716] |
| 2 | 1 | fan_in | 0.46708 | [0.0858417, 1.08821] | [-0.880761, 1.81492] |
| 2 | 1 | mean_field | 0.294445 | [0.0629462, 0.641186] | [-0.465417, 1.05431] |
| 2 | 2 | legacy_gaussian | 0.144411 | [0.111716, 0.171612] | [0.0690765, 0.219745] |
| 2 | 2 | fan_in | 0.241233 | [0.164764, 0.328791] | [0.0360915, 0.446375] |
| 2 | 2 | mean_field | 0.182005 | [0.148686, 0.240364] | [0.0560232, 0.307986] |
| 2 | 4 | legacy_gaussian | 0.129501 | [0.0951748, 0.151379] | [0.054725, 0.204278] |
| 2 | 4 | fan_in | 0.169488 | [0.128999, 0.233956] | [0.0292761, 0.3097] |
| 2 | 4 | mean_field | 0.134576 | [0.0767738, 0.193823] | [-0.0108518, 0.280003] |
| 8 | 1 | legacy_gaussian | 0.174965 | [0.0709569, 0.23431] | [-0.0495502, 0.39948] |
| 8 | 1 | fan_in | 0.164531 | [0.0680241, 0.224137] | [-0.0450101, 0.374072] |
| 8 | 1 | mean_field | 0.146009 | [0.100425, 0.17323] | [0.0473199, 0.244698] |
| 8 | 2 | legacy_gaussian | 0.12567 | [0.00558827, 0.347584] | [-0.352316, 0.603656] |
| 8 | 2 | fan_in | 0.10098 | [0.0228646, 0.237611] | [-0.193987, 0.395948] |
| 8 | 2 | mean_field | 0.113306 | [0.0887333, 0.152635] | [0.0278144, 0.198797] |
| 8 | 4 | legacy_gaussian | 0.55868 | [0.453959, 0.721177] | [0.204239, 0.913121] |
| 8 | 4 | fan_in | 0.182666 | [0.0617163, 0.268385] | [-0.0850077, 0.45034] |
| 8 | 4 | mean_field | 0.142365 | [0.0878337, 0.177936] | [0.0232404, 0.261489] |
| 32 | 1 | legacy_gaussian | 0.613416 | [0.388302, 1.04112] | [-0.307215, 1.53405] |
| 32 | 1 | fan_in | 0.201857 | [0.0423244, 0.366395] | [-0.200837, 0.604551] |
| 32 | 1 | mean_field | 0.13922 | [0.0989066, 0.166276] | [0.0508085, 0.227631] |
| 32 | 2 | legacy_gaussian | 12.7716 | [1.64419, 22.6375] | [-13.4465, 38.9897] |
| 32 | 2 | fan_in | 0.358059 | [0.238055, 0.498657] | [0.0313292, 0.684788] |
| 32 | 2 | mean_field | 0.150339 | [0.0934034, 0.19568] | [0.0208683, 0.279811] |
| 32 | 4 | legacy_gaussian | 227.03 | [3.37636, 377.545] | [-263.666, 717.726] |
| 32 | 4 | fan_in | 0.0625079 | [0.00462057, 0.103423] | [-0.0655385, 0.190554] |
| 32 | 4 | mean_field | 0.112161 | [0.0836608, 0.134306] | [0.0477823, 0.17654] |

## Norms and finite behavior

All 567 training checkpoints have finite energy, half-MSE, denominator and
raw/effective weight/gradient norms, feedforward RMS and hidden equilibrium RMS.
All 81 final training energies are below their initial energies. This is an
observation over six steps at the fixed rate, not a monotonicity guarantee.
Per-layer vectors remain in the raw artifact; the following ranges cover **all**
feedforward layer RMS entries across all seeds, architectures and checkpoints.

| Scheme | Observed feedforward RMS range | Final training energy range |
|---|---|---|
| legacy_gaussian | [0.0140803, 31.9901] | [0.00178354, 0.336711] |
| fan_in | [0.0705727, 1.47105] | [0.00071169, 0.494055] |
| mean_field | [0.0124756, 1.04019] | [0.0246722, 0.465005] |

At width 32 and four hidden layers, legacy Gaussian final half-MSE mean is
227.030 (seed range 3.37636–377.545), fan-in is .0625079 (.00462057–.103423),
and mean-field is .112161 (.0836608–.134306). The full table also includes
settings where these orderings differ. Initialization output scales, effective
learning rates and equilibrium denominators differ by declared design. These
levels cannot isolate a universal optimization benefit. No setting is selected
for subsequent work from these final-test results.

## Observed work and resources

| Quantity | Observed |
|---|---:|
| Attempted / committed SGD calls | 486 / 486 |
| Completed step equilibrium/gradient evaluations | 972 |
| Extra diagnostic equilibrium/gradient evaluations | 567 |
| Heldout forward passes | 567 |
| Training row exposures | 7776 |
| Whole execution seconds | 0.3933692 |
| Sampled absolute process RSS bytes | 43847680 |
| JSON output bytes | 4123933 |

Timing totals across all cases (seconds):

| Window | Total seconds |
|---|---:|
| initialization_seconds | 0.0191701 |
| training_step_seconds | 0.1373192 |
| diagnostic_seconds | 0.1601543 |
| scoring_seconds | 0.0323163998 |
| elapsed_seconds | 0.3499078 |
| case_elapsed_seconds | 0.3905835 |

Case elapsed includes initialization, record assembly, data creation and input
preflight; application elapsed excludes preflight/report assembly. Step time
includes candidate validation, and diagnostic overhead is substantial. These
are descriptive timings on one local invocation, not an isolated speed benchmark.
All original ceilings passed: 512 updates, 60 execution seconds, 256 MiB sampled
RSS, one CPU thread, 32 MiB output; no GPU, downloads, replay or environment steps.
Execution time excludes imports and JSON encoding. Safe-boundary checks cannot
preempt native calls or establish exhaustive peak memory.

## Correctness and remaining scope

59 focused tests passed with zero failures/errors/skips; six-file Ruff, format
and scoped Windows/Linux mypy passed. Independent source review found no
remaining material issues before execution. Initial independent test failures
are retained: mutable predictor inputs contaminated later scores, and malformed
prediction arrays were accepted. Fresh copies and strict float64 shape checks
fixed these boundaries without weakening tests. Twelve engineering update calls
were used (six worker, six root); no experimental run was repeated.

R1.4b delivers the fixed measurement runner and this finite exploratory grid.
Parent R1.4 remains open: residual depth parameterization requires explicit
implementation and correctness controls. This study does not validate nonlinear
learners, CPC, paper figure reproduction, depth limits, convergence or general
learning benefit. Next implement and test residual depth factors independently
before a separately declared residual experiment; preserve these outcomes.
