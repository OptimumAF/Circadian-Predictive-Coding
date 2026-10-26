# Saved native calibration and rank summary

`scripts/summarize_native_consolidation.py` reads four fixed accepted native
development reports and writes one fresh JSON summary. It loads no model and
makes no predictions, training updates, random draws or simulator interactions.
The lightweight repository environment is sufficient; no native SDK is needed.

```powershell
python -m scripts.summarize_native_consolidation --output native-calibration.json
```

Use `--runs-dir <saved-runs-directory>` when the four original artifact folders
are elsewhere. The accepted files are warm recurring, optimizer reset,
projection stream and consolidation stream actual.json reports from2026-10-22.
Each input is bounded to1MiB and its existing accepted SHA256; output is fresh
and bounded to256KiB. Missing/replaced inputs fail without model work. This CLI
does not reconstruct missing artifacts or admit arbitrary experimental arms.

## What it reports

- Unchanged baseline, warm after-B, after-A reset/B6, after-A rank4 projection/
  reset/B6, and terminal rank8 half/half consolidation/reset after warm B6.
- Overall and A/B/anchor Brier, raw NLL, accuracy, candidate recall and the same
  five fixed ECE bins used by `decision_reference_evaluation`. Confidence for
  ECE is maximum probability, separate from provider margin confidence. Empty
  bins/support remain visible. Zero target probability remains infinite NLL;
  undefined signed NLL comparisons remain unavailable.
- Original acquisition, retention and anchor criteria; all signed contrasts
  against unchanged, consolidation-minus-warm, projection-minus-reset and
  reset-minus-warm. Higher accuracy/recall and lower Brier/NLL/ECE are favorable.
- Per-module recorded spectra, discarded tail, allocated/requested rank and
  reported float64 pretruncation numerical rank. Installed float32 numerical
  rank remains unmeasured. No SVD or counterfactual rank evaluation is performed.
- Original total native/intervention ledgers and final observed model/scorer/
  token/initializer costs, elapsed time and recorded peakRSS, separately from
  derivation's zero model work. Missing observations mean unknown, not zero.
  Warm total source costs include its return phase beyond the selected B12 state.

Raw captured responses must agree with typed saved probabilities and original
correctness/NLL. Fixed corpus, request/target, source group and role admission
prevents silently recanonicalizing changed evidence. The pure function owns its
output and leaves inputs unchanged. Final/probe data are not admitted.

## Interpretation

These are six synthetic development cases, two per context. ECE and Brier here
describe saved predictions; they do not establish calibrated quality or justify
temperature tuning. Keep original negative results and metric directions.
The selected arms differ in optimizer history, timing and extra computation.
They are historical controls, not a matched-compute rank ablation. Terminal
consolidation combines correlated complete operators under the current head.
Spectra alone do not establish rank adequacy, task success at other ranks or
memory savings. No recovery, generalization or CPC benefit follows.

## Recorded result

The one saved derivation completes with30 scored rows/120 module spectra and
zero model work. All original combined task criteria fail in at least one
context. Consolidation minus warm increases overall Brier+.0127136461 and
NLL+.0352059645, while ECE decreases-.0006591768. This small ECE difference
does not reverse the failed acquisition criterion or establish useful calibration.
[Full fixed-arm metrics, commands and limits](../artifacts/runs/native-calibration-summary-20261022/RESULTS.md).

## Structure and extension

```text
scripts/summarize_native_consolidation.py  bounded IO and saved-evidence assembly
src/app/decision_reference_evaluation.py existing shared probability metrics
tests/test_native_calibration_summary.py scalar behavior and boundary checks
```

No new core/app interface, dependency or environment variable. To extend the
comparison, first freeze the task/roles/metrics/budget and independently review
new source evidence. Do not turn this fixed summarizer into a model loader,
tuner, training controller or archive-admission framework.

```powershell
python -m pytest -q tests/test_native_calibration_summary.py
python -m ruff check scripts/summarize_native_consolidation.py tests/test_native_calibration_summary.py
python -m ruff format --check scripts/summarize_native_consolidation.py tests/test_native_calibration_summary.py
python -m mypy --explicit-package-bases --follow-imports=silent scripts/summarize_native_consolidation.py tests/test_native_calibration_summary.py
```

[Prospective protocol](../artifacts/runs/native-calibration-summary-20261022/PROTOCOL.md) ·
[Decision](adr/ADR-0530-describe-saved-native-calibration-and-rank.md).
