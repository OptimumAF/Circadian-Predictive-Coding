# Residual linear scaling measurements — 2026-10-14

## Fixed scope and execution

One invocation of `python -m scripts.run_residual_scaling_pilot` followed
113 combined correctness tests, scoped static checks and independent source
review. All 81 declared cases completed all six updates, without refusal,
retry, replacement or omission. The fixed protocol and pre-execution source
identity are in `artifacts/runs/residual-scaling-measurement-20261014/` as
`PROTOCOL.md` and `SOURCE_VERSION.json`. The same directory contains `pilot.json`
with exact initialization tensors, every checkpoint norm, score, cost and seed
summary. This is scalar linear regression, not a CPC comparison.

The [measurement guide](../residual-scaling-measurement.md) and
[residual mapping](../residual-linear-pc.md) document native factors and isolation.
Fixed inputs: seeds 23/41/59, widths 2/8/32, hidden layers 1/2/4, three schemes,
D=2, gamma0=1, rate .01, unit precision, identity activation and no bias. All
schemes use fixed identities only in middle maps. L=H+1 counts learned matrices;
mean-field middle branches use 1/sqrt(N*L), raw SGD gamma0²*N. The grid/data/rate
are retained from the prospective MLP protocol; prior final scores select none
of them. All training per case precedes detached checkpoint scoring. No heldout
label enters training, diagnostics, stopping or settings for subsequent cases.

## Complete seed summaries

All cells retain the three ordered initialization seeds, conditional on the
same fixed 16 training and 64 final-test rows. Metric: heldout half-MSE, lower
better. Intervals are nominal marginal 95% t (df 2, critical 4.303), with untested
normal seed effects and no simultaneous coverage. Negative interval endpoints
are retained from the untruncated formula; individual losses are nonnegative.
No significance, selected-winner, convergence or asymptotic claim follows.

### Before training

| Width | Hidden layers | Scheme | Mean | Seed range | Nominal 95% interval |
|---:|---:|---|---:|---|---|
| 2 | 1 | legacy_gaussian | 0.215676 | [0.0675828, 0.419804] | [-0.238149, 0.669501] |
| 2 | 1 | fan_in | 0.482816 | [0.0880503, 1.12677] | [-0.914395, 1.88003] |
| 2 | 1 | mean_field | 0.30246 | [0.0641511, 0.660174] | [-0.481206, 1.08613] |
| 2 | 2 | legacy_gaussian | 0.176448 | [0.0697746, 0.271881] | [-0.0757639, 0.428661] |
| 2 | 2 | fan_in | 0.316046 | [0.030833, 0.593871] | [-0.383523, 1.01561] |
| 2 | 2 | mean_field | 0.222371 | [0.046006, 0.390575] | [-0.206002, 0.650744] |
| 2 | 4 | legacy_gaussian | 0.196247 | [0.0661829, 0.300876] | [-0.100374, 0.492867] |
| 2 | 4 | fan_in | 0.513128 | [0.306318, 0.911271] | [-0.343693, 1.36995] |
| 2 | 4 | mean_field | 0.256982 | [0.184834, 0.306631] | [0.0981344, 0.41583] |
| 8 | 1 | legacy_gaussian | 0.18496 | [0.0744336, 0.244539] | [-0.0530793, 0.423] |
| 8 | 1 | fan_in | 0.18496 | [0.0744336, 0.244539] | [-0.0530793, 0.423] |
| 8 | 1 | mean_field | 0.149825 | [0.103759, 0.176705] | [0.0502548, 0.249395] |
| 8 | 2 | legacy_gaussian | 0.225492 | [0.107878, 0.320542] | [-0.0430959, 0.49408] |
| 8 | 2 | fan_in | 0.227135 | [0.154767, 0.33086] | [-0.00176935, 0.456039] |
| 8 | 2 | mean_field | 0.142222 | [0.0462041, 0.218114] | [-0.0756654, 0.360109] |
| 8 | 4 | legacy_gaussian | 5.99913 | [2.96935, 7.59105] | [-0.522253, 12.5205] |
| 8 | 4 | fan_in | 2.87556 | [0.926479, 4.44073] | [-1.5672, 7.31831] |
| 8 | 4 | mean_field | 0.301557 | [0.042702, 0.504829] | [-0.284807, 0.887921] |
| 32 | 1 | legacy_gaussian | 0.654599 | [0.425116, 1.10229] | [-0.308701, 1.6179] |
| 32 | 1 | fan_in | 0.255202 | [0.0732827, 0.446726] | [-0.209137, 0.719542] |
| 32 | 1 | mean_field | 0.14365 | [0.101714, 0.172803] | [0.0511623, 0.236137] |
| 32 | 2 | legacy_gaussian | 16.7184 | [2.44287, 25.7014] | [-14.3357, 47.7725] |
| 32 | 2 | fan_in | 0.986352 | [0.301102, 2.12044] | [-1.47124, 3.44395] |
| 32 | 2 | mean_field | 0.171762 | [0.0952881, 0.2467] | [-0.0163462, 0.359871] |
| 32 | 4 | legacy_gaussian | 503.887 | [20.2428, 966.217] | [-672.067, 1679.84] |
| 32 | 4 | fan_in | 0.861483 | [0.140485, 2.28609] | [-2.20364, 3.92661] |
| 32 | 4 | mean_field | 0.14546 | [0.120373, 0.190947] | [0.0474227, 0.243498] |

### After six updates

| Width | Hidden layers | Scheme | Mean | Seed range | Nominal 95% interval |
|---:|---:|---|---:|---|---|
| 2 | 1 | legacy_gaussian | 0.21044 | [0.0661612, 0.40939] | [-0.231835, 0.652716] |
| 2 | 1 | fan_in | 0.46708 | [0.0858417, 1.08821] | [-0.880761, 1.81492] |
| 2 | 1 | mean_field | 0.294445 | [0.0629462, 0.641186] | [-0.465417, 1.05431] |
| 2 | 2 | legacy_gaussian | 0.172977 | [0.0686245, 0.265226] | [-0.0726197, 0.418575] |
| 2 | 2 | fan_in | 0.299137 | [0.0302857, 0.564078] | [-0.363977, 0.962251] |
| 2 | 2 | mean_field | 0.217669 | [0.0453869, 0.382367] | [-0.201236, 0.636573] |
| 2 | 4 | legacy_gaussian | 0.193527 | [0.0649395, 0.30054] | [-0.102787, 0.489841] |
| 2 | 4 | fan_in | 0.490389 | [0.283273, 0.874441] | [-0.33675, 1.31753] |
| 2 | 4 | mean_field | 0.254656 | [0.183895, 0.304307] | [0.0983398, 0.410972] |
| 8 | 1 | legacy_gaussian | 0.174965 | [0.0709569, 0.23431] | [-0.0495502, 0.39948] |
| 8 | 1 | fan_in | 0.164531 | [0.0680241, 0.224137] | [-0.0450101, 0.374072] |
| 8 | 1 | mean_field | 0.146009 | [0.100425, 0.17323] | [0.0473199, 0.244698] |
| 8 | 2 | legacy_gaussian | 0.211947 | [0.100597, 0.306154] | [-0.0460402, 0.469933] |
| 8 | 2 | fan_in | 0.200437 | [0.138121, 0.280911] | [0.0188176, 0.382057] |
| 8 | 2 | mean_field | 0.137362 | [0.0436242, 0.210698] | [-0.0747634, 0.349487] |
| 8 | 4 | legacy_gaussian | 5.68452 | [2.78333, 7.26834] | [-0.56616, 11.9352] |
| 8 | 4 | fan_in | 2.61231 | [0.83268, 3.99018] | [-1.40396, 6.62857] |
| 8 | 4 | mean_field | 0.289013 | [0.0402252, 0.48322] | [-0.273652, 0.851678] |
| 32 | 1 | legacy_gaussian | 0.613416 | [0.388302, 1.04112] | [-0.307215, 1.53405] |
| 32 | 1 | fan_in | 0.201857 | [0.0423244, 0.366395] | [-0.200837, 0.604551] |
| 32 | 1 | mean_field | 0.13922 | [0.0989066, 0.166276] | [0.0508085, 0.227631] |
| 32 | 2 | legacy_gaussian | 14.8292 | [2.29314, 22.6279] | [-12.4089, 42.0674] |
| 32 | 2 | fan_in | 0.54919 | [0.137954, 1.12088] | [-0.719633, 1.81801] |
| 32 | 2 | mean_field | 0.164532 | [0.0910863, 0.23632] | [-0.0159077, 0.344973] |
| 32 | 4 | legacy_gaussian | 423.872 | [16.1813, 772.248] | [-523.922, 1371.66] |
| 32 | 4 | fan_in | 0.253204 | [0.0130862, 0.656269] | [-0.619273, 1.12568] |
| 32 | 4 | mean_field | 0.138178 | [0.116193, 0.180303] | [0.0475155, 0.22884] |

## Finite behavior and norms

All 567 training checkpoints have finite diagnostics. All 81 final training
energies are strictly below their initial values; no descent guarantee is
inferred from this fixed six-step observation. The following ranges include
**every** learned feedforward output across all checkpoints and architectures;
raw/effective per-layer weight/gradient and hidden equilibrium norms remain in
`pilot.json`, without selecting favorable seeds.

| Scheme | Observed feedforward RMS range | Final training energy range |
|---|---|---|
| legacy_gaussian | [0.135897, 44.1336] | [0.007177, 0.303424] |
| fan_in | [0.139359, 3.23929] | [0.00362567, 0.494055] |
| mean_field | [0.0524925, 1.10255] | [0.0203683, 0.465005] |

At width 32/four hidden layers the final heldout half-MSE means are legacy
423.872, fan-in .253204 and mean-field .138178; their full ranges and intervals
appear above. Other cells have different orderings, and initial prediction
scales/effective learning rates differ by design. This does not isolate a
universal optimization benefit. No architecture/rate/seed is selected from the
final scores for subsequent work.

All 27 cases with H=1 have no middle block: every training checkpoint diagnostic
and heldout score exactly matches its archived MLP counterpart. This comparison
uses existing raw artifacts; no MLP grid or model update was rerun. Older
R1.4b results and the negative/inconclusive R1.7 CPC study remain unchanged.

## Work and resources

| Quantity | Observed |
|---|---:|
| Attempted / committed raw SGD calls | 486 / 486 |
| Completed step equilibrium/gradient evaluations | 972 |
| Extra diagnostic equilibrium/gradient evaluations | 567 |
| Heldout forward passes | 567 |
| Training row exposures | 7776 |
| Execution seconds | 0.4360792 |
| Sampled absolute process RSS bytes | 44343296 |
| JSON bytes | 4132196 |

Timing totals across all cases:

| Window | Total seconds |
|---|---:|
| initialization_seconds | 0.0222258999 |
| training_step_seconds | 0.1524171 |
| diagnostic_seconds | 0.1692429 |
| scoring_seconds | 0.0389154998 |
| elapsed_seconds | 0.3822287 |
| case_elapsed_seconds | 0.4330568 |

Step time includes candidate validation; diagnostics add gradient evaluations.
Application elapsed excludes copying/preflight/report assembly; full case elapsed
includes initialization, data construction and these boundaries. Global time
excludes imports and JSON encoding. These are descriptive, not isolated speed or
equal-compute measurements. Original ceilings pass: 512 updates, 60 seconds,
256 MiB sampled RSS, one CPU thread and 32 MiB output. There was no GPU, download,
replay or environment interaction. Safe-boundary checks cannot preempt native
operations or establish exhaustive peak RSS.

## Correctness and scope

113 combined tests passed with no failures/errors/skips, covering the shared
engine and existing MLP entry point, residual native routing, complete identities,
role/ID/storage validation, caller mutation from external callbacks, deferred
scoring, strict prediction arrays, partial failure records and failure-score
suppression. Six-file lint/format and scoped Windows/Linux type checks passed.
Independent source review found no material issues before execution. Eighteen
engineering update calls were used (six worker residual, six root residual, six
root MLP); no full grid was used by tests.

The result is finite width/depth behavior for an explicitly applicable linear
parameterization. It does not validate nonlinear/CPC scaling, theoretical
infinite-depth limits, hyperparameter transfer, convergence or learning benefit.
Any broader integration or experiment requires its own mathematical controls
and prospective protocol. Preserve all current outcomes, including unfavorable
cells and prior negative evidence.

Independent acceptance review assessed R1.4a–d against the original R1.4 criteria:
three controls, exact initialized tensors/configuration, repeated seeded width/
depth measurements, finite behavior, norms, cost and heldout outcomes. Those
criteria are satisfied for the applicable linear MLP/residual family. Previously
unfinished nonlinear/binary/CPC scaling is preserved as named unchecked R1.8;
these linear results do not establish that integration. Next is R1.5 bounded
latent computation, with fixed-step compatibility and stopping-check overhead
included in any future comparison.
