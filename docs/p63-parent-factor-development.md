# P6.3c9c paired parent-control development scoring contract

## Frozen training and evaluation boundary

Protocol: `continual_parent_factor_outer_development_v1`. Preserve the
complete [c9b train-only protocol](p63-parent-factor-preflight.md): all eight
cells on development seeds **347/349/353**, every arrived role, initialization,
selector seed/cursor, wake, explicit count, eligibility, guard, topology,
memory and work setting. No training setting is selected from a score.
Confirmation seeds **359/367/373/379/383/389/397/401/409/419**, previous
reservations and final sources remain unopened.

Bind the canonical local bundle at
`artifacts/runs/p63-parent-factor-preflight/parent-factor-preflight.*.json`:

| File | Exact byte SHA-256 |
|---|---|
| Request | `e5c7f0c49d34896409614383db728798bb33cb1b50de239a2a98cad036cfb385` |
| Result | `555fc2fe5dfd86981d87af2d0c15bd8ab925417d9ad748783d95cee95f5a1874` |
| Audit | `c872792099a2ab5c772df69fac532b0205b2c33000112fb976219115b7a366d7` |

The independent c9b result repeats exactly. Before source construction,
require all three canonical bytes, no failure, finite JSON, independent
complete result validation, unchanged manifest/source/adapter identities,
all 24 cells/216 decisions, actual work and elapsed/RSS checks. Preserve the
original files. Missing, changed or failed references prevent execution.

Reuse pinned c9b model/wake/guard/fact helpers without modifying them.
After every A wake/decision completes, deep-copy all eight models before
constructing B. Complete every seed and independently validate the full
train object; require exact all-seed equality with the reference. Globally
check every held after-A/after-B parameter hash and width, plus complete
circadian snapshot hashes (including selector settings, cursor, calls,
last decision and both noise/selection RNG state), against the saved epoch
facts **before the first outer input or label**. Repeat checkpoint checks
after scoring. Late-seed fact, supply, parameter or selector drift must
prevent even the first scorer. Scoring performs no retraining or selection.

## Every outcome and contrast

Evaluate after-A copies on A outer selection and after-B models on A/B
outer selection. Keep all 24 cells: **72 outer forward calls / 1,440
examples** (24 × (24+24+12)). Reuse `phase6_two_task_accuracy_v1` with
primary final mean `(A_after_B+B_after_B)/2` and signed A forgetting
`A_after_A-A_after_B`. Publish all three accuracies, both primary metrics,
optional zero-safe A retention, and the complete cost/capacity vector.
Ordinary `pc_off` and `neutral_off` outcomes must be exact.

Freeze these **20 ordered left-minus-right pairs per seed, 60 total**:

| Group | Ordered left | Ordered right |
|---|---|---|
| Parent ranking | `usage_growth`, `usage_growth`, `scheduled_growth` | `scheduled_growth`, `random_growth`, `random_growth` |
| Growth/reference | Each of `usage_growth`, `scheduled_growth`, `random_growth`, in that order | Each of `backprop_off`, `pc_off`, `neutral_off`, `backprop_13_off`, `pc_13_off`, in that order |
| Planned capacity | `backprop_13_off`, `pc_13_off` | Corresponding `backprop_off`, `pc_off` |

Publish all five differences (three accuracies/two primary metrics), all
seed values, descriptive means and sample SD for every pair. Preserve
duplicate PC/neutral comparisons, null/negative outcomes and all guard
decisions. A lower forgetting value alone does not establish retention
when A-after-A was lower. No composite winner, confirmatory significance
claim, new metric, selected seed or confirmation configuration results
from this three-seed development pilot.

The three growth cells have matched planned counts, wake/source exposure,
memory availability, neutral chemistry, guard rule and width ceiling;
actual accepted widths and costs remain observable outcomes. Fixed-eight
references have fewer parameters; planned-thirteen references are wider
from initialization, with different initialization tensors and per-update
compute. Backprop has no latent relaxation. Growth adds guard evaluations;
baselines have no independent guards. Splitting includes unchanged core
noise. Parent contrasts therefore isolate the selector within this
growth-only policy; they do not test the full adaptive circadian model or
equalize total FLOPs. No reference is a retrospective final-width oracle.

## Local budget and artifacts

Keep **576 executed/planned wake updates / 600 hard cap**, zero replay,
54 guarded attempts/108 calls/1,944 inner examples. Scoring adds only the
declared 72 outer calls. Keep a **120-second** child wall limit and
**256 MiB** observed whole-worker RSS cap, sampled every 5 ms through
training, retained model copies, independent validation, scoring and
deterministic serialization. Stdout transport and parent writes are outside
the interval; brief peaks may be missed. Per-arm time/RSS/FLOPs are
unmeasured. Retained replay/supply arrays are 960 bytes/seed before copies,
excluding source/role arrays, parameters, metadata and temporary copies.

Pin new scored source/adapter identities before any scored test or public
outer evaluation. Write exclusive request/result/audit or failure records
in ignored local directories; reject occupied outputs before launching
work. Repeat a complete fresh process and require identical deterministic
result bytes. Independently rederive every metric/pair and artifact identity.
Any failure or partial run leaves c9c unchecked. The original c9/P6.3c/P6.3
matrix, independent confirmation and final release retain their criteria.

**Why this order:** complete saved unscored evidence fixes the training path
before results are known. Holding small after-A copies permits global state
checks without another training pass. New app/adapter modules compose pinned
c9b training and existing c4 score arithmetic; earlier protocols stay valid.
See [ADR-0151](adr/ADR-0151-verify-parent-selector-checkpoints-before-scoring.md).

## Implementation identities and execution

Before any outer score, eleven unscored contract/late-fact/full-checkpoint
sentinel cases passed (one scored case deselected), with no skipped test.
Repository Ruff/mypy (361 files), four-file format and diff checks pass.
The canonical reference validates all three byte pins and independent
facts/resources. Manifest SHA-256 remains
`a7938028ed3c9279ef74a5f9a2550012927e4bb626b72672861ae64aa71497c9`.
Freeze these additions to the unchanged c9b 31-source map:

| Additional source | Byte SHA-256 |
|---|---|
| `src/app/continual_parent_factor_development.py` | `53aab115493ba2ca2dd34787e07b4deb2f246bed257aad4eb97baf9e35af1530` |
| `src/app/continual_sleep_factor_development.py` | `e6a498c107d0071b822309907691e225b1a1ae828c93f8435c94ba0cfa94bf88` |
| `scripts/run_p63_parent_factor_preflight.py` | `39bfd61bd50a1abf0a00308ca40028152270b09f99eb401b4176924dd4c6a8b1` |

Selected **34-source** map SHA-256:
`b0a86792167a9965e38007b0a0a3cd5a7ee3d229495df4950201198b545be55b`.
Scored adapter byte SHA-256:
`be57cf809149af570876d75d6f924ec073ee1a35a1d96b68ae4fc7b59129e9c0`.
This is a selected map, not a full transitive dependency hash. Unit/worker
boundary tests construct fresh unscored temporary c9b bundles and substitute
only their request/audit byte pins in isolated test contexts; no fixture is
official experimental evidence, and tests need no ignored canonical files.
The public CLI retains the canonical pins above. Execute:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_p63_parent_factor_development --output-dir artifacts/runs/p63-parent-factor-development
.\.venv\Scripts\python.exe -m scripts.run_p63_parent_factor_development --output-dir artifacts/runs/p63-parent-factor-development-repeat
```

Both official bounded runs complete all 24 cells/60 pairs and repeat exactly
at result SHA-256
`7f76e793eaf123b30116f57d68aa755962d63a56e4bba4b29703ae37b16e2040`.
Independent readback validates the complete embedded train facts, every
metric/contrast, all request/audit/source/reference hashes and resource caps.
All 31 new/260 related tests pass, zero skipped. The
[complete outcome and cost report](p63-parent-factor-development-results.md)
retains the null/negative usage results and mixed planned-width references.
No outcome changed this contract. Independent confirmation and original
matrix/final requirements remain open.

The scientific CLI requires the original recorded canonical c9b bundle;
its request/audit observations cannot be regenerated byte for byte on a
clean clone. Fresh isolated bundles make unit/worker tests runnable without
ignored artifacts. Portable reference ingestion remains a separate artifact
workflow concern; this gate does not claim cross-version/device portability.
