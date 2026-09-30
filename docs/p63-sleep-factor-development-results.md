# P6.3 guarded sleep-factor outer-development result

The [prospectively frozen contract](p63-sleep-factor-development.md) scored all nine no-replay arms on seeds 67/71/73. The complete c3 train-only fact object matched the saved preflight across all 27 cells **before any outer-selection value was read**. Final roles and the ten reserved confirmation seeds remain unopened. These are exploratory development scores, not independent confirmation.

Two fresh bounded public processes saved exclusive request/result/audit sets under ignored `artifacts/runs/p63-sleep-factor-development/` and `artifacts/runs/p63-sleep-factor-development-repeat/`. Both deterministic result files are byte-identical, SHA-256 `e0795b279346054d9c4e4c680e3aea103e347e3dd49960c44b93f8b642eaa9b6`; neither has a failure sidecar. The request SHA-256 values are `8bba0112aac729afb8ddc959b4be99a5bbdf0c706b93c4872471404f3ce4a553` and `16b327f095d509f7fded69f5ced3526060f624af426776471a6911dc18ff397b`. An independent read-only pass verified the frozen 16-source map, scored adapter, c3 reference, both manifest/request/result/audit hashes, all 27 scored cells, metrics and all nine paired contrast cells.

The source was the v14 arrived A→B two-cluster development geometry. Each seed has A train/inner/outer counts 72/24/24 and B counts 36/12/12; the complete six role hashes per seed and parameter/guard facts remain in each scored result's `train_facts`. All arms share source rows, 12+12 wake updates, and initialization within width. Five circadian arms use one guarded A-boundary sleep, and every trained arm has zero replay. The worker completed 648 wake updates (700 maximum), 81 outer evaluations and 1,620 outer evaluated examples. The first and second workers took 0.485 and 0.473 seconds end to end, with sampled whole-worker RSS peaks 43,401,216 and 43,446,272 bytes, below the 256-MiB cap. These are whole-process observations, not per-arm memory; sampling can miss a shorter peak. The environment was Windows 11, Intel Core i7-12700K, Python 3.14.7, NumPy 2.4.6, NumPy CPU execution.

## Complete outer-selection accuracy matrix

`A/A` is A outer accuracy after A wake and its boundary sleep; `A/B` and `B/B` are A and B outer accuracies after B wake. `Mean` is equal-task final mean; `Forget` is signed `A/A − A/B` (negative means A improved). `Retain` is `A/B ÷ A/A` and is undefined at zero denominator. Values are rounded to six decimals for display; result JSON retains full precision.

| Seed | Arm | A/A | A/B | B/B | Mean | Forget | Retain |
|---:|---|---:|---:|---:|---:|---:|---:|
| 67 | `backprop_8` | .916667 | .916667 | 1.000000 | .958333 | .000000 | 1.000000 |
| 67 | `pc_8` | .416667 | .916667 | .750000 | .833333 | -.500000 | 2.200000 |
| 67 | `neutral_sham` | .416667 | .916667 | .750000 | .833333 | -.500000 | 2.200000 |
| 67 | `structure_only` | .500000 | .916667 | .750000 | .833333 | -.416667 | 1.833333 |
| 67 | `homeostasis_only` | .416667 | .916667 | .750000 | .833333 | -.500000 | 2.200000 |
| 67 | `gating_sham` | .416667 | .916667 | .750000 | .833333 | -.500000 | 2.200000 |
| 67 | `gating_reset` | .416667 | .916667 | .750000 | .833333 | -.500000 | 2.200000 |
| 67 | `backprop_12` | .916667 | .916667 | 1.000000 | .958333 | .000000 | 1.000000 |
| 67 | `pc_12` | .000000 | .708333 | .916667 | .812500 | -.708333 | null |
| 71 | `backprop_8` | 1.000000 | .958333 | .916667 | .937500 | .041667 | .958333 |
| 71 | `pc_8` | .958333 | .958333 | 1.000000 | .979167 | .000000 | 1.000000 |
| 71 | `neutral_sham` | .958333 | .958333 | 1.000000 | .979167 | .000000 | 1.000000 |
| 71 | `structure_only` | .958333 | .958333 | .916667 | .937500 | .000000 | 1.000000 |
| 71 | `homeostasis_only` | .958333 | .958333 | 1.000000 | .979167 | .000000 | 1.000000 |
| 71 | `gating_sham` | .958333 | .958333 | 1.000000 | .979167 | .000000 | 1.000000 |
| 71 | `gating_reset` | .958333 | .958333 | 1.000000 | .979167 | .000000 | 1.000000 |
| 71 | `backprop_12` | 1.000000 | 1.000000 | .916667 | .958333 | .000000 | 1.000000 |
| 71 | `pc_12` | 1.000000 | 1.000000 | .916667 | .958333 | .000000 | 1.000000 |
| 73 | `backprop_8` | .958333 | .958333 | .916667 | .937500 | .000000 | 1.000000 |
| 73 | `pc_8` | 1.000000 | 1.000000 | .916667 | .958333 | .000000 | 1.000000 |
| 73 | `neutral_sham` | 1.000000 | 1.000000 | .916667 | .958333 | .000000 | 1.000000 |
| 73 | `structure_only` | 1.000000 | 1.000000 | .916667 | .958333 | .000000 | 1.000000 |
| 73 | `homeostasis_only` | 1.000000 | 1.000000 | .916667 | .958333 | .000000 | 1.000000 |
| 73 | `gating_sham` | 1.000000 | 1.000000 | .916667 | .958333 | .000000 | 1.000000 |
| 73 | `gating_reset` | 1.000000 | 1.000000 | .916667 | .958333 | .000000 | 1.000000 |
| 73 | `backprop_12` | .916667 | .958333 | .916667 | .937500 | -.041667 | 1.045455 |
| 73 | `pc_12` | .916667 | .916667 | .916667 | .916667 | .000000 | 1.000000 |

## Prespecified paired contrasts

Each value is left arm minus right arm within one source seed. `ΔMean` and `ΔForget` are the two primary contrasts. The table retains all three accuracy differences so a forgetting change cannot hide a changed A-after-A starting point.

| Seed | Left − right | ΔA/A | ΔA/B | ΔB/B | ΔMean | ΔForget |
|---:|---|---:|---:|---:|---:|---:|
| 67 | structure − neutral | +.083333 | .000000 | .000000 | .000000 | +.083333 |
| 71 | structure − neutral | .000000 | .000000 | -.083333 | -.041667 | .000000 |
| 73 | structure − neutral | .000000 | .000000 | .000000 | .000000 | .000000 |
| 67 | homeostasis − neutral | .000000 | .000000 | .000000 | .000000 | .000000 |
| 71 | homeostasis − neutral | .000000 | .000000 | .000000 | .000000 | .000000 |
| 73 | homeostasis − neutral | .000000 | .000000 | .000000 | .000000 | .000000 |
| 67 | gated reset − gated sham | .000000 | .000000 | .000000 | .000000 | .000000 |
| 71 | gated reset − gated sham | .000000 | .000000 | .000000 | .000000 | .000000 |
| 73 | gated reset − gated sham | .000000 | .000000 | .000000 | .000000 | .000000 |

Across three seeds, structure minus neutral has mean ΔMean **-.013889** (sample SD .024056) and mean ΔForget **+.027778** (sample SD .048113). Homeostasis minus neutral and gated reset minus gated sham have exactly zero observed differences in both primary metrics and all three accuracy fields on all three seeds. Ordinary PC and neutral circadian also match exactly on every accuracy field. These are descriptive pilot values; no significance or superiority claim follows.

The structure arm accepted one split and one prune per seed, transiently reaching width nine/37 parameters before returning to width eight/33. Its seed-67 A-after-A gain did not persist as a final A or B difference, and the resulting higher signed forgetting is **not** a retention benefit or loss caused solely by B. On seed 71, it lost one of 12 B outer examples against neutral. Homeostasis changed parameters after the accepted A sleep; conditional chemical reset reduced chemical state under active gating. Neither changed an outer accuracy at this resolution. The planned width-12 controls have 49 parameters versus 33 for width eight, and unequal per-update compute. No arm, threshold, seed, metric, or stopping rule was changed after viewing these outcomes.

This result closes only P6.3c4's development factor. Schedule, full-minus-one, measured per-arm costs, the broader mechanism matrix, and independent confirmation remain open under P6.3c/P6.3 and later cost tasks. The next implementation step is to inspect the remaining schedule/full-minus-one dependency and freeze its own train-only matched control before any scored run; these three-seed outcomes cannot choose its settings.
