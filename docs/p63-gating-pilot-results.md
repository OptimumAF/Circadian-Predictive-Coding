# P6.3 first mechanism factor: development-only chemical gating

The predeclared [pilot contract](p63-gating-pilot.md) ran on all three
development seeds `41,43,59` under
`continual_mechanism_gating_dev_v1`. Each method received the same arrived
train rows, 24 full-batch wake updates, 1,296 row presentations, 48 latent
iterations, 2,592 example-inference iterations, and width eight/33
parameters. No sleep or replay occurred. Ordinary PC and the neutral
circadian control started from identical tensors and remained byte-identical
after every update, including their development scores. The gating arm
changed only the existing wake plasticity floor from 1.0 to 0.2. Its
observed minimum plasticity was below one on every seed, so the treatment
was active. Six train/inner-guard/outer-selection source role hashes per seed
passed the audit. The final roles were never released or scored.

All values below are **outer-selection development** accuracy, not final
test results. `A/A` is A after A training; `A/B` and `B/B` are after B.
Final mean is `(A/B + B/B)/2`; signed forgetting is `A/A - A/B`.

| Seed | Arm | A/A | A/B | B/B | Final mean | Signed forgetting | Minimum plasticity |
|---:|---|---:|---:|---:|---:|---:|---:|
| 41 | ordinary PC / neutral circadian | 0.833333 | 0.916667 | 0.833333 | 0.875000 | -0.083333 | 1.000000 neutral |
| 41 | chemical gating | 0.750000 | 0.916667 | 0.833333 | 0.875000 | -0.166667 | 0.767757 |
| 43 | ordinary PC / neutral circadian | 0.750000 | 0.791667 | 1.000000 | 0.895833 | -0.041667 | 1.000000 neutral |
| 43 | chemical gating | 0.708333 | 0.791667 | 1.000000 | 0.895833 | -0.083333 | 0.809348 |
| 59 | ordinary PC / neutral circadian | 0.916667 | 0.916667 | 0.833333 | 0.875000 | 0.000000 | 1.000000 neutral |
| 59 | chemical gating | 0.875000 | 0.916667 | 0.833333 | 0.875000 | -0.041667 | 0.742955 |

The paired gating-minus-neutral final-mean differences are exactly
`0, 0, 0` (mean `0`). The paired signed-forgetting differences are
`-0.083333, -0.041667, -0.041667` (mean `-0.055556`). That lower forgetting
number is **not a retention benefit** here: gating reduced `A/A` by
`0.083333, 0.041667, 0.041667`, while `A/B` and `B/B` remained exactly
equal for each seed. Across these three seeds, both arms have mean final
task accuracy `0.881944`. The 24-row A and 12-row B outer roles have coarse
accuracy steps; these development observations do not establish a final
or cross-setting advantage. No configuration, seed, metric, or stopping
rule was changed after seeing them.

## Artifacts and reproducibility

The adapter wrote exclusive request, result, and audit files in ignored
`artifacts/runs/p63-gating-pilot/`, then ran the same worker in a fresh
process into `artifacts/runs/p63-gating-pilot-repeat/`. Both result JSON
files are byte-identical at SHA-256
`ede5ebcd618c663c6cc34a142f81ea6fef8ead627069edc0019ac8e5dda516f7`.
The first and repeat request hashes differ because they record distinct
UTC start times; each audit binds its own request and the common result.
Observed worker-plus-audit elapsed times were about 0.355 and 0.349
seconds against the 120-second ceiling. No failure file was created.
The frozen manifest digest is
`3aec33aba455ff3fa04b0d20993ed97f6e0aedc8e64f150fbe0040bf0a44dfe9`;
the eight-source map and adapter identities are in the prospective
contract. A read-only second pass verified both requests, manifests,
source hashes, audits, role/work/capacity rows, finite metrics, and exact
result bytes. The source-sentinel test proves this pilot does not request
final-test arrays. It does not prove a broader strict-online source
construction boundary, and no confirmation seed was run.

The rest of the staged matrix remains open. Existing fixed v14 periodic
versus no-sleep outcomes include added replay work and changed width;
they cannot substitute for a matched single-factor control. Next, freeze
the replay/no-replay and planned-width references before running them,
while retaining the ten independent confirmation seeds and final roles.
