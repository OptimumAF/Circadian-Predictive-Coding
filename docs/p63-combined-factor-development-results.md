# P6.3c8 combined/full-minus-one development results

Date: 2026-09-30. Checkout: `master` at
`86cd5bff71b9c70da94ddcf69d8f62316f2d3382`, with preserved dirty c5/c6/c7
work and this c8 increment. Runtime: Windows 11, Intel Core i7-12700K CPU,
Python 3.14.7, NumPy 2.4.6.

## Protocol and decision

The [prospective contract](p63-combined-factor-development.md) and
[ADR-0148](adr/ADR-0148-verify-complete-combined-checkpoints-before-scoring.md)
fixed `continual_combined_factor_outer_development_v1`, all c7 cells/settings
on development seeds **263/269/271**, the saved all-seed reference, two
primary metrics, 22 ordered contrasts per seed and resource limits before
any outer score, including scored tests.

All **51 train cells and every after-A/after-B checkpoint** matched globally
before the first outer input/label access. Checkpoints bind parameters/width,
and complete circadian RNG/chemistry/memory/lineage/clocks. The full train-fact
object remains in each scored result. Two bounded public processes produced
identical deterministic bytes, including all **51 score rows, 66 contrasts,
153 outer evaluations/3,060 examples**. Ordinary-PC/neutral outcomes remain
exact in both off and full-controlled replay states. No final role was
released and no confirmation seed was used.

**Complete c8 only.** The three-seed development outcome does not establish
a full-model final-mean advantage over matched replay PC. Full-minus-matched
PC final-mean differences are **−.0625,+.020833,0** (mean **−.013889**, sample
SD **.043368**). Full is below backprop and the planned wider PC reference
on every seed. Keep all mixed, null and negative outcomes. No threshold,
metric, baseline, seed, stopping rule or configuration was changed from
scores, and no confirmation treatment was selected.

C9 scheduled/random parent controls, independent confirmation and the
original P6.3c/P6.3 matrix remain unfinished. This scored factor does not
close final-test accuracy/resource/statistical reporting milestones.

## Artifacts, identities and limits

Both ignored directories contain `combined-factor-development.request.json`,
`.result.json`, `.audit.json` and no failure:

- `artifacts/runs/p63-combined-factor-development/`
- `artifacts/runs/p63-combined-factor-development-repeat/`

Identical result SHA-256:
`2e32c5d7b98b5798f45f5e9b7734088950eec1ef0ae5f19420a282321f3f8714`.
Request hashes:
`ad334c090bb554ba5051366bf745aad6ae9127707397f8a33e9700592bea67b1`
and
`d0a841c73a065a63c277a467df8106eb6319825821319357bf795e59be5b7848`.
The unchanged c7 byte-bound reference is
`79a7d7e09f0ada01ff72d6e266ddee7576aca52316a0f9a8e2ab3dd402251d13`.
Manifest SHA-256 is
`729bf9df8472752f51696299373555339208af7b5add1892d14154d4f21ad04a`;
selected combined **29-source** map digest is
`ee0e2c8cb154f9aa427c0b27680841b64a0dfe3c6a90f254159114ca4ce4d6b5`;
scored adapter byte digest is
`edf68589d83033d0abdf39c737f78044938f84a97b49d86a25698edce42d8c40`.
The map extends c7's selected 26-source map with the new app, existing c4
scorer and c7 adapter. It is not a complete dependency-tree hash. No pinned
c7/core or older source/result bytes changed.

| Process | Elapsed seconds | Observed peak RSS bytes | RSS samples |
|---|---|---|---|
| First | 3.228156 | 66,109,440 | 172 |
| Repeat | 3.251908 | 64,065,536 | 168 |

Each is below **120 seconds/256 MiB**. Worker RSS is sampled every 5 ms
through training, retained copies, train/scored validation, scoring and
result serialization. Final stdout transport and parent artifact writing
are outside that interval; short peaks can be missed. These observations
include runtime and fact allocations and are not per-arm measurements.

## Every outer-development score

All three accuracies, final mean, signed forgetting and optional A retention
ratio follow `phase6_two_task_accuracy_v1`. A-after-A comes from the held
copy after all A decisions; A/B-after-B comes from the completed B model.
These are outer-development roles, not final-test or confirmation results.
Values are rounded to six decimals only in this report; artifacts retain
full float precision. Negative forgetting and ratios above one are valid
positive-transfer observations and do not alone rank treatments.

| Seed | Arm | A after A | A after B | B after B | Final mean | Signed A forgetting | A retention ratio |
|---|---|---|---|---|---|---|---|
| 263 | backprop_off | 0.958333 | 0.875000 | 0.833333 | 0.854167 | 0.083333 | 0.913043 |
| 263 | pc_off | 0.333333 | 0.958333 | 0.666667 | 0.812500 | -0.625000 | 2.875000 |
| 263 | neutral_off | 0.333333 | 0.958333 | 0.666667 | 0.812500 | -0.625000 | 2.875000 |
| 263 | backprop_full_replay | 0.958333 | 0.875000 | 0.833333 | 0.854167 | 0.083333 | 0.913043 |
| 263 | pc_full_replay | 0.333333 | 0.916667 | 0.833333 | 0.875000 | -0.583333 | 2.750000 |
| 263 | neutral_full_replay | 0.333333 | 0.916667 | 0.833333 | 0.875000 | -0.583333 | 2.750000 |
| 263 | full | 0.333333 | 0.958333 | 0.666667 | 0.812500 | -0.625000 | 2.875000 |
| 263 | minus_replay | 0.291667 | 0.958333 | 0.583333 | 0.770833 | -0.666667 | 3.285714 |
| 263 | minus_gating | 0.333333 | 0.958333 | 0.750000 | 0.854167 | -0.625000 | 2.875000 |
| 263 | minus_structure | 0.291667 | 0.958333 | 0.750000 | 0.854167 | -0.666667 | 3.285714 |
| 263 | minus_schedule | 0.250000 | 0.833333 | 0.250000 | 0.541667 | -0.583333 | 3.333333 |
| 263 | minus_difficulty | 0.333333 | 0.958333 | 0.750000 | 0.854167 | -0.625000 | 2.875000 |
| 263 | minus_homeostasis | 0.333333 | 0.958333 | 0.666667 | 0.812500 | -0.625000 | 2.875000 |
| 263 | minus_reset | 0.333333 | 0.958333 | 0.583333 | 0.770833 | -0.625000 | 2.875000 |
| 263 | periodic_structure_only | 0.375000 | 0.916667 | 0.833333 | 0.875000 | -0.541667 | 2.444444 |
| 263 | backprop_14_off | 0.958333 | 0.875000 | 0.833333 | 0.854167 | 0.083333 | 0.913043 |
| 263 | pc_14_off | 0.875000 | 0.875000 | 0.833333 | 0.854167 | 0.000000 | 1.000000 |
| 269 | backprop_off | 0.916667 | 0.916667 | 0.833333 | 0.875000 | 0.000000 | 1.000000 |
| 269 | pc_off | 0.583333 | 0.666667 | 0.666667 | 0.666667 | -0.083333 | 1.142857 |
| 269 | neutral_off | 0.583333 | 0.666667 | 0.666667 | 0.666667 | -0.083333 | 1.142857 |
| 269 | backprop_full_replay | 0.916667 | 0.916667 | 0.833333 | 0.875000 | 0.000000 | 1.000000 |
| 269 | pc_full_replay | 0.583333 | 0.666667 | 0.666667 | 0.666667 | -0.083333 | 1.142857 |
| 269 | neutral_full_replay | 0.583333 | 0.666667 | 0.666667 | 0.666667 | -0.083333 | 1.142857 |
| 269 | full | 0.583333 | 0.708333 | 0.666667 | 0.687500 | -0.125000 | 1.214286 |
| 269 | minus_replay | 0.541667 | 0.875000 | 0.833333 | 0.854167 | -0.333333 | 1.615385 |
| 269 | minus_gating | 0.583333 | 0.666667 | 0.666667 | 0.666667 | -0.083333 | 1.142857 |
| 269 | minus_structure | 0.583333 | 0.666667 | 0.666667 | 0.666667 | -0.083333 | 1.142857 |
| 269 | minus_schedule | 0.541667 | 0.583333 | 0.666667 | 0.625000 | -0.041667 | 1.076923 |
| 269 | minus_difficulty | 0.583333 | 0.708333 | 0.750000 | 0.729167 | -0.125000 | 1.214286 |
| 269 | minus_homeostasis | 0.583333 | 0.666667 | 0.666667 | 0.666667 | -0.083333 | 1.142857 |
| 269 | minus_reset | 0.583333 | 0.666667 | 0.666667 | 0.666667 | -0.083333 | 1.142857 |
| 269 | periodic_structure_only | 0.625000 | 0.708333 | 0.750000 | 0.729167 | -0.083333 | 1.133333 |
| 269 | backprop_14_off | 0.791667 | 0.833333 | 0.916667 | 0.875000 | -0.041667 | 1.052632 |
| 269 | pc_14_off | 0.625000 | 0.625000 | 0.833333 | 0.729167 | 0.000000 | 1.000000 |
| 271 | backprop_off | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 271 | pc_off | 0.916667 | 0.916667 | 1.000000 | 0.958333 | 0.000000 | 1.000000 |
| 271 | neutral_off | 0.916667 | 0.916667 | 1.000000 | 0.958333 | 0.000000 | 1.000000 |
| 271 | backprop_full_replay | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 271 | pc_full_replay | 0.916667 | 0.916667 | 1.000000 | 0.958333 | 0.000000 | 1.000000 |
| 271 | neutral_full_replay | 0.916667 | 0.916667 | 1.000000 | 0.958333 | 0.000000 | 1.000000 |
| 271 | full | 0.875000 | 0.916667 | 1.000000 | 0.958333 | -0.041667 | 1.047619 |
| 271 | minus_replay | 0.875000 | 0.916667 | 1.000000 | 0.958333 | -0.041667 | 1.047619 |
| 271 | minus_gating | 0.916667 | 0.916667 | 1.000000 | 0.958333 | 0.000000 | 1.000000 |
| 271 | minus_structure | 0.875000 | 0.916667 | 1.000000 | 0.958333 | -0.041667 | 1.047619 |
| 271 | minus_schedule | 0.875000 | 0.916667 | 1.000000 | 0.958333 | -0.041667 | 1.047619 |
| 271 | minus_difficulty | 0.916667 | 0.916667 | 1.000000 | 0.958333 | 0.000000 | 1.000000 |
| 271 | minus_homeostasis | 0.875000 | 0.916667 | 1.000000 | 0.958333 | -0.041667 | 1.047619 |
| 271 | minus_reset | 0.833333 | 0.916667 | 1.000000 | 0.958333 | -0.083333 | 1.100000 |
| 271 | periodic_structure_only | 0.916667 | 0.958333 | 1.000000 | 0.979167 | -0.041667 | 1.045455 |
| 271 | backprop_14_off | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 271 | pc_14_off | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Every paired contrast

Every row is **left minus right**. Positive final-mean differences mean
higher final task accuracy; signed forgetting must be interpreted together
with its A-after-A starting score and A-after-B. The first seven contrasts
are component removals, the next nine compare full to every reference, and
the remaining six compare replay, planned width and periodic structure.
Keep duplicate PC/neutral and zero rows.

| Seed | Left minus right | Delta A after A | Delta A after B | Delta B after B | Delta final mean | Delta signed forgetting |
|---|---|---|---|---|---|---|
| 263 | full - minus_replay | +0.041667 | +0.000000 | +0.083333 | +0.041667 | +0.041667 |
| 263 | full - minus_gating | +0.000000 | +0.000000 | -0.083333 | -0.041667 | +0.000000 |
| 263 | full - minus_structure | +0.041667 | +0.000000 | -0.083333 | -0.041667 | +0.041667 |
| 263 | full - minus_schedule | +0.083333 | +0.125000 | +0.416667 | +0.270833 | -0.041667 |
| 263 | full - minus_difficulty | +0.000000 | +0.000000 | -0.083333 | -0.041667 | +0.000000 |
| 263 | full - minus_homeostasis | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 263 | full - minus_reset | +0.000000 | +0.000000 | +0.083333 | +0.041667 | +0.000000 |
| 263 | full - backprop_off | -0.625000 | +0.083333 | -0.166667 | -0.041667 | -0.708333 |
| 263 | full - pc_off | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 263 | full - neutral_off | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 263 | full - backprop_full_replay | -0.625000 | +0.083333 | -0.166667 | -0.041667 | -0.708333 |
| 263 | full - pc_full_replay | +0.000000 | +0.041667 | -0.166667 | -0.062500 | -0.041667 |
| 263 | full - neutral_full_replay | +0.000000 | +0.041667 | -0.166667 | -0.062500 | -0.041667 |
| 263 | full - periodic_structure_only | -0.041667 | +0.041667 | -0.166667 | -0.062500 | -0.083333 |
| 263 | full - backprop_14_off | -0.625000 | +0.083333 | -0.166667 | -0.041667 | -0.708333 |
| 263 | full - pc_14_off | -0.541667 | +0.083333 | -0.166667 | -0.041667 | -0.625000 |
| 263 | backprop_full_replay - backprop_off | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 263 | pc_full_replay - pc_off | +0.000000 | -0.041667 | +0.166667 | +0.062500 | +0.041667 |
| 263 | neutral_full_replay - neutral_off | +0.000000 | -0.041667 | +0.166667 | +0.062500 | +0.041667 |
| 263 | backprop_14_off - backprop_off | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 263 | pc_14_off - pc_off | +0.541667 | -0.083333 | +0.166667 | +0.041667 | +0.625000 |
| 263 | periodic_structure_only - neutral_off | +0.041667 | -0.041667 | +0.166667 | +0.062500 | +0.083333 |
| 269 | full - minus_replay | +0.041667 | -0.166667 | -0.166667 | -0.166667 | +0.208333 |
| 269 | full - minus_gating | +0.000000 | +0.041667 | +0.000000 | +0.020833 | -0.041667 |
| 269 | full - minus_structure | +0.000000 | +0.041667 | +0.000000 | +0.020833 | -0.041667 |
| 269 | full - minus_schedule | +0.041667 | +0.125000 | +0.000000 | +0.062500 | -0.083333 |
| 269 | full - minus_difficulty | +0.000000 | +0.000000 | -0.083333 | -0.041667 | +0.000000 |
| 269 | full - minus_homeostasis | +0.000000 | +0.041667 | +0.000000 | +0.020833 | -0.041667 |
| 269 | full - minus_reset | +0.000000 | +0.041667 | +0.000000 | +0.020833 | -0.041667 |
| 269 | full - backprop_off | -0.333333 | -0.208333 | -0.166667 | -0.187500 | -0.125000 |
| 269 | full - pc_off | +0.000000 | +0.041667 | +0.000000 | +0.020833 | -0.041667 |
| 269 | full - neutral_off | +0.000000 | +0.041667 | +0.000000 | +0.020833 | -0.041667 |
| 269 | full - backprop_full_replay | -0.333333 | -0.208333 | -0.166667 | -0.187500 | -0.125000 |
| 269 | full - pc_full_replay | +0.000000 | +0.041667 | +0.000000 | +0.020833 | -0.041667 |
| 269 | full - neutral_full_replay | +0.000000 | +0.041667 | +0.000000 | +0.020833 | -0.041667 |
| 269 | full - periodic_structure_only | -0.041667 | +0.000000 | -0.083333 | -0.041667 | -0.041667 |
| 269 | full - backprop_14_off | -0.208333 | -0.125000 | -0.250000 | -0.187500 | -0.083333 |
| 269 | full - pc_14_off | -0.041667 | +0.083333 | -0.166667 | -0.041667 | -0.125000 |
| 269 | backprop_full_replay - backprop_off | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 269 | pc_full_replay - pc_off | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 269 | neutral_full_replay - neutral_off | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 269 | backprop_14_off - backprop_off | -0.125000 | -0.083333 | +0.083333 | +0.000000 | -0.041667 |
| 269 | pc_14_off - pc_off | +0.041667 | -0.041667 | +0.166667 | +0.062500 | +0.083333 |
| 269 | periodic_structure_only - neutral_off | +0.041667 | +0.041667 | +0.083333 | +0.062500 | -0.000000 |
| 271 | full - minus_replay | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 271 | full - minus_gating | -0.041667 | +0.000000 | +0.000000 | +0.000000 | -0.041667 |
| 271 | full - minus_structure | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 271 | full - minus_schedule | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 271 | full - minus_difficulty | -0.041667 | +0.000000 | +0.000000 | +0.000000 | -0.041667 |
| 271 | full - minus_homeostasis | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 271 | full - minus_reset | +0.041667 | +0.000000 | +0.000000 | +0.000000 | +0.041667 |
| 271 | full - backprop_off | -0.125000 | -0.083333 | +0.000000 | -0.041667 | -0.041667 |
| 271 | full - pc_off | -0.041667 | +0.000000 | +0.000000 | +0.000000 | -0.041667 |
| 271 | full - neutral_off | -0.041667 | +0.000000 | +0.000000 | +0.000000 | -0.041667 |
| 271 | full - backprop_full_replay | -0.125000 | -0.083333 | +0.000000 | -0.041667 | -0.041667 |
| 271 | full - pc_full_replay | -0.041667 | +0.000000 | +0.000000 | +0.000000 | -0.041667 |
| 271 | full - neutral_full_replay | -0.041667 | +0.000000 | +0.000000 | +0.000000 | -0.041667 |
| 271 | full - periodic_structure_only | -0.041667 | -0.041667 | +0.000000 | -0.020833 | +0.000000 |
| 271 | full - backprop_14_off | -0.125000 | -0.083333 | +0.000000 | -0.041667 | -0.041667 |
| 271 | full - pc_14_off | -0.125000 | -0.083333 | +0.000000 | -0.041667 | -0.041667 |
| 271 | backprop_full_replay - backprop_off | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 271 | pc_full_replay - pc_off | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 271 | neutral_full_replay - neutral_off | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 271 | backprop_14_off - backprop_off | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 271 | pc_14_off - pc_off | +0.083333 | +0.083333 | +0.000000 | +0.041667 | +0.000000 |
| 271 | periodic_structure_only - neutral_off | +0.000000 | +0.041667 | +0.000000 | +0.020833 | -0.041667 |

### Descriptive means and sample SD

The replication unit is the three shared seeds. Each entry is
`mean +/- sample SD` for the same five differences. These are exploratory
descriptions, with no selected multiple-comparison winner or confirmatory
significance test.

| Left minus right | Delta A after A | Delta A after B | Delta B after B | Delta final mean | Delta signed forgetting |
|---|---|---|---|---|---|
| full - minus_replay | +0.027778 +/- 0.024056 | -0.055556 +/- 0.096225 | -0.027778 +/- 0.127294 | -0.041667 +/- 0.110240 | +0.083333 +/- 0.110240 |
| full - minus_gating | -0.013889 +/- 0.024056 | +0.013889 +/- 0.024056 | -0.027778 +/- 0.048113 | -0.006944 +/- 0.031823 | -0.027778 +/- 0.024056 |
| full - minus_structure | +0.013889 +/- 0.024056 | +0.013889 +/- 0.024056 | -0.027778 +/- 0.048113 | -0.006944 +/- 0.031823 | +0.000000 +/- 0.041667 |
| full - minus_schedule | +0.041667 +/- 0.041667 | +0.083333 +/- 0.072169 | +0.138889 +/- 0.240563 | +0.111111 +/- 0.141810 | -0.041667 +/- 0.041667 |
| full - minus_difficulty | -0.013889 +/- 0.024056 | +0.000000 +/- 0.000000 | -0.055556 +/- 0.048113 | -0.027778 +/- 0.024056 | -0.013889 +/- 0.024056 |
| full - minus_homeostasis | +0.000000 +/- 0.000000 | +0.013889 +/- 0.024056 | +0.000000 +/- 0.000000 | +0.006944 +/- 0.012028 | -0.013889 +/- 0.024056 |
| full - minus_reset | +0.013889 +/- 0.024056 | +0.013889 +/- 0.024056 | +0.027778 +/- 0.048113 | +0.020833 +/- 0.020833 | -0.000000 +/- 0.041667 |
| full - backprop_off | -0.361111 +/- 0.251155 | -0.069444 +/- 0.146329 | -0.111111 +/- 0.096225 | -0.090278 +/- 0.084197 | -0.291667 +/- 0.363242 |
| full - pc_off | -0.013889 +/- 0.024056 | +0.013889 +/- 0.024056 | +0.000000 +/- 0.000000 | +0.006944 +/- 0.012028 | -0.027778 +/- 0.024056 |
| full - neutral_off | -0.013889 +/- 0.024056 | +0.013889 +/- 0.024056 | +0.000000 +/- 0.000000 | +0.006944 +/- 0.012028 | -0.027778 +/- 0.024056 |
| full - backprop_full_replay | -0.361111 +/- 0.251155 | -0.069444 +/- 0.146329 | -0.111111 +/- 0.096225 | -0.090278 +/- 0.084197 | -0.291667 +/- 0.363242 |
| full - pc_full_replay | -0.013889 +/- 0.024056 | +0.027778 +/- 0.024056 | -0.055556 +/- 0.096225 | -0.013889 +/- 0.043368 | -0.041667 +/- 0.000000 |
| full - neutral_full_replay | -0.013889 +/- 0.024056 | +0.027778 +/- 0.024056 | -0.055556 +/- 0.096225 | -0.013889 +/- 0.043368 | -0.041667 +/- 0.000000 |
| full - periodic_structure_only | -0.041667 +/- 0.000000 | +0.000000 +/- 0.041667 | -0.083333 +/- 0.083333 | -0.041667 +/- 0.020833 | -0.041667 +/- 0.041667 |
| full - backprop_14_off | -0.319444 +/- 0.267879 | -0.041667 +/- 0.110240 | -0.138889 +/- 0.127294 | -0.090278 +/- 0.084197 | -0.277778 +/- 0.373454 |
| full - pc_14_off | -0.236111 +/- 0.267879 | +0.027778 +/- 0.096225 | -0.111111 +/- 0.096225 | -0.041667 +/- 0.000000 | -0.263889 +/- 0.315495 |
| backprop_full_replay - backprop_off | +0.000000 +/- 0.000000 | +0.000000 +/- 0.000000 | +0.000000 +/- 0.000000 | +0.000000 +/- 0.000000 | +0.000000 +/- 0.000000 |
| pc_full_replay - pc_off | +0.000000 +/- 0.000000 | -0.013889 +/- 0.024056 | +0.055556 +/- 0.096225 | +0.020833 +/- 0.036084 | +0.013889 +/- 0.024056 |
| neutral_full_replay - neutral_off | +0.000000 +/- 0.000000 | -0.013889 +/- 0.024056 | +0.055556 +/- 0.096225 | +0.020833 +/- 0.036084 | +0.013889 +/- 0.024056 |
| backprop_14_off - backprop_off | -0.041667 +/- 0.072169 | -0.027778 +/- 0.048113 | +0.027778 +/- 0.048113 | +0.000000 +/- 0.000000 | -0.013889 +/- 0.024056 |
| pc_14_off - pc_off | +0.222222 +/- 0.277430 | -0.013889 +/- 0.086736 | +0.111111 +/- 0.096225 | +0.048611 +/- 0.012028 | +0.236111 +/- 0.339355 |
| periodic_structure_only - neutral_off | +0.027778 +/- 0.024056 | +0.013889 +/- 0.048113 | +0.083333 +/- 0.083333 | +0.048611 +/- 0.024056 | +0.013889 +/- 0.063647 |

## Costs and interpretation

Training is exactly the c7 gate: **1,530 executed updates = 1,224 wake + 280
applied replay + 26 rejected replay**, below maximum **1,548** and hard cap
**1,600**. Every cell executes 24 wake updates/1,296 wake presentations;
PC/circadian cells use 48 wake inference loops/2,592 example-inference
iterations, while backprop uses zero latent loops. There are 144 own
guarded attempts, 288 inner calls/5,184 examples, 126 own commits and 18
rollbacks. Scoring adds forward evaluations without optimizer updates.

The shared eight-row/192-byte FIFO and eleven matching circadian buffers
have **2,304 persistent labeled-array bytes/seed**, excluding parameters,
metadata and temporary copies. Only full commits supply the three matched
consumers: 10/8/12 applied replay updates each across the three seeds. Their
accepted IDs/work match exactly, but full also pays rejected replay,
guard/topology/chemistry/difficulty costs. The neutral replay consumer's
commits have zero own guard attempts. Wider references were declared at
width 14 before observing full width; these are prospective capacity
references with different per-update compute.

| Seed | Arm | Width final/peak | Parameters final/peak | Wake updates | Replay applied/rejected | Sleep commits/own attempts | Guard rows |
|---|---|---|---|---|---|---|---|
| 263 | backprop_off | 8/8 | 33/33 | 24 | 0/0 | 0/0 | 0 |
| 263 | pc_off | 8/8 | 33/33 | 24 | 0/0 | 0/0 | 0 |
| 263 | neutral_off | 8/8 | 33/33 | 24 | 0/0 | 0/0 | 0 |
| 263 | backprop_full_replay | 8/8 | 33/33 | 24 | 10/0 | 0/0 | 0 |
| 263 | pc_full_replay | 8/8 | 33/33 | 24 | 10/0 | 0/0 | 0 |
| 263 | neutral_full_replay | 8/8 | 33/33 | 24 | 10/0 | 5/0 | 0 |
| 263 | full | 6/8 | 25/33 | 24 | 10/2 | 5/6 | 216 |
| 263 | minus_replay | 5/8 | 21/33 | 24 | 0/0 | 5/6 | 216 |
| 263 | minus_gating | 6/8 | 25/33 | 24 | 10/2 | 5/6 | 216 |
| 263 | minus_structure | 8/8 | 33/33 | 24 | 12/0 | 6/6 | 216 |
| 263 | minus_schedule | 8/8 | 33/33 | 24 | 0/0 | 0/0 | 0 |
| 263 | minus_difficulty | 6/8 | 25/33 | 24 | 10/2 | 5/6 | 216 |
| 263 | minus_homeostasis | 6/8 | 25/33 | 24 | 10/2 | 5/6 | 216 |
| 263 | minus_reset | 6/8 | 25/33 | 24 | 12/0 | 6/6 | 216 |
| 263 | periodic_structure_only | 7/9 | 29/37 | 24 | 0/0 | 6/6 | 216 |
| 263 | backprop_14_off | 14/14 | 57/57 | 24 | 0/0 | 0/0 | 0 |
| 263 | pc_14_off | 14/14 | 57/57 | 24 | 0/0 | 0/0 | 0 |
| 269 | backprop_off | 8/8 | 33/33 | 24 | 0/0 | 0/0 | 0 |
| 269 | pc_off | 8/8 | 33/33 | 24 | 0/0 | 0/0 | 0 |
| 269 | neutral_off | 8/8 | 33/33 | 24 | 0/0 | 0/0 | 0 |
| 269 | backprop_full_replay | 8/8 | 33/33 | 24 | 8/0 | 0/0 | 0 |
| 269 | pc_full_replay | 8/8 | 33/33 | 24 | 8/0 | 0/0 | 0 |
| 269 | neutral_full_replay | 8/8 | 33/33 | 24 | 8/0 | 4/0 | 0 |
| 269 | full | 6/8 | 25/33 | 24 | 8/4 | 4/6 | 216 |
| 269 | minus_replay | 5/8 | 21/33 | 24 | 0/0 | 5/6 | 216 |
| 269 | minus_gating | 7/8 | 29/33 | 24 | 8/4 | 4/6 | 216 |
| 269 | minus_structure | 8/8 | 33/33 | 24 | 12/0 | 6/6 | 216 |
| 269 | minus_schedule | 8/8 | 33/33 | 24 | 0/0 | 0/0 | 0 |
| 269 | minus_difficulty | 6/8 | 25/33 | 24 | 8/4 | 4/6 | 216 |
| 269 | minus_homeostasis | 6/8 | 25/33 | 24 | 8/4 | 4/6 | 216 |
| 269 | minus_reset | 7/8 | 29/33 | 24 | 12/0 | 6/6 | 216 |
| 269 | periodic_structure_only | 8/9 | 33/37 | 24 | 0/0 | 3/6 | 216 |
| 269 | backprop_14_off | 14/14 | 57/57 | 24 | 0/0 | 0/0 | 0 |
| 269 | pc_14_off | 14/14 | 57/57 | 24 | 0/0 | 0/0 | 0 |
| 271 | backprop_off | 8/8 | 33/33 | 24 | 0/0 | 0/0 | 0 |
| 271 | pc_off | 8/8 | 33/33 | 24 | 0/0 | 0/0 | 0 |
| 271 | neutral_off | 8/8 | 33/33 | 24 | 0/0 | 0/0 | 0 |
| 271 | backprop_full_replay | 8/8 | 33/33 | 24 | 12/0 | 0/0 | 0 |
| 271 | pc_full_replay | 8/8 | 33/33 | 24 | 12/0 | 0/0 | 0 |
| 271 | neutral_full_replay | 8/8 | 33/33 | 24 | 12/0 | 6/0 | 0 |
| 271 | full | 6/8 | 25/33 | 24 | 12/0 | 6/6 | 216 |
| 271 | minus_replay | 6/8 | 25/33 | 24 | 0/0 | 6/6 | 216 |
| 271 | minus_gating | 6/8 | 25/33 | 24 | 12/0 | 6/6 | 216 |
| 271 | minus_structure | 8/8 | 33/33 | 24 | 12/0 | 6/6 | 216 |
| 271 | minus_schedule | 8/8 | 33/33 | 24 | 0/0 | 0/0 | 0 |
| 271 | minus_difficulty | 6/8 | 25/33 | 24 | 12/0 | 6/6 | 216 |
| 271 | minus_homeostasis | 6/8 | 25/33 | 24 | 10/2 | 5/6 | 216 |
| 271 | minus_reset | 7/8 | 29/33 | 24 | 12/0 | 6/6 | 216 |
| 271 | periodic_structure_only | 7/9 | 29/37 | 24 | 0/0 | 6/6 | 216 |
| 271 | backprop_14_off | 14/14 | 57/57 | 24 | 0/0 | 0/0 | 0 |
| 271 | pc_14_off | 14/14 | 57/57 | 24 | 0/0 | 0/0 | 0 |

The [c7 report](p63-combined-factor-preflight-results.md) publishes all 18
rejections, proposed/applied split/prune IDs and every circadian scale
range. They remain unchanged and embedded beside the c8 scores. Full/removal
cells proposed no split under frozen defaults. All controllers together
proposed nine splits/62 removals and committed seven/44; the splits are
entirely in the separate periodic structure-only control. Full ends at width
six on all seeds, versus width-eight matched replay PC. Costs and topology
are part of the treatment, so no equal-compute/capacity claim follows.

Observed interpretation:

- Full versus matched replay PC has mixed final-mean differences and a
  negative mean. Signed forgetting is lower by .041667 on every seed,
  from higher A-after-B on seeds 263/269 and lower A-after-A with unchanged
  A-after-B on seed 271. On seed 263, full's higher A-after-B is accompanied
  by lower B-after-B, lowering the primary final mean. The complete matrix
  exposes that retention/adaptation tradeoff.
- Full versus backprop (off or full-controlled replay) has final-mean
  differences −.041667,−.1875,−.041667, mean −.090278. Backprop replay is
  null in all five contrast fields. Full versus planned-width PC is
  −.041667 on every seed.
- Full versus periodic structure-only is negative on every seed:
  −.0625,−.041667,−.020833. Structure-only versus neutral off has positive
  final-mean differences .0625,.0625,.020833, while planned-width PC versus
  PC off has .041667,.0625,.041667. These are different bounds/settings/
  costs, and neither compares usage-aware selection to scheduled/random
  parent selection.
- Full-minus-removal final-mean means, in the declared order, are
  **−.041667,−.006944,−.006944,+.111111,−.027778,+.006944,+.020833**.
  These are component interventions inside the fixed composition, with
  independently accepted/rejected sleep histories. Schedule removal
  disables downstream sleep; difficulty removal changes importance
  weighting as well as wake scale. Mixed and tiny changes at this
  accuracy resolution do not select a policy.
- Matched replay PC/neutral versus off has .0625,0,0 final-mean differences,
  mean .020833, with extra replay work and exact PC/neutral outcomes.
  No circadian-specific gain follows from those neutral pairs.

## Validation and reproduction

Five pre-score negative cases passed before identities were frozen: changed
manifest/seal before data, valid-shaped last-seed reference mismatch under
raising outer/final sentinels, changed late replay facts and corrupted late
after-A parameters/RNG. A scored positive test observes all seeds trained
before the scorer, all 153 actual accuracy calls in order (24/24/12 rows
per cell), all 17 A models before B arrival, unchanged complete checkpoints,
exact neutral parity and every metric/contrast with final sources sealed.

Five public artifact cases verify the complete c7 bundle and changed byte/
audit/source refusal, all score/metric/pair/fact identities, exclusive
duplicate paths, and timeout/nonfinite failure artifacts without a false
result/audit. Read-only artifact checks independently rederived all facts,
metrics and contrast arithmetic, checked source/manifest/adapter/request/
result/audit/reference identities and work/resource limits, and compared
saved deterministic bytes. Tables above were generated from those verified
facts.

Commands from the repository root:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_p63_combined_factor_development --output-dir artifacts/runs/p63-combined-factor-development
.\.venv\Scripts\python.exe -m scripts.run_p63_combined_factor_development --output-dir artifacts/runs/p63-combined-factor-development-repeat
.\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_continual_combined_factor_development.py tests/test_p63_combined_factor_development_cli.py
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy
```

Both public commands exited 0. New tests passed **11 in 18.02 seconds**;
the final related gate in the development log passed **87 in 47.30 seconds**
with no skips. Ruff, mypy (349 source files), four-file formatting and diff
checks passed. Full CPU suite, CUDA, broad sweeps, independent confirmation,
final release and per-arm resource attribution were skipped.

**Exact next action:** inspect existing structural eligibility and split
selection plus adaptation-policy interfaces for c9. Prospectively design the
smallest explicit scheduled/random parent control preserving budgets,
stable lineage, deterministic RNG rollback and earlier pinned protocols.
Freeze its source/seed/train-only work/capacity/guard contract and verify
paired control schedules before separately frozen scoring. Keep all
confirmation reservations and final roles unopened; do not choose c9
settings from these c8 development scores.
