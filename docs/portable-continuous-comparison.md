# Portable saved-control input for the continuous comparison

Validate the accepted bundled controls without training:

```powershell
python -m scripts.continuous_replay_example --example-baseline --validate-baseline-only
```

This works from a checkout or installed package and needs only the CPU dependency.
It checks the original saved-control contract, prints a preflight JSON record and
returns before the experiment path. It is not a completed new study or source
authentication. The original validator constructs fixed episode/plan metadata;
no learner, fitted statistics, optimizer, predictions or simulator execute.

To validate your explicit saved controls instead:

```powershell
python -m scripts.continuous_replay_example --baseline controls.json --validate-baseline-only
```

`--baseline` and `--example-baseline` are mutually exclusive. Explicit missing or
invalid files are refused; no fallback occurs. Without either flag the historical
source-relative remaining-arms.log path is unchanged. Inputs are bounded to6MiB
UTF-8 JSON objects; duplicate keys, nonfinite numbers and numeric overflow refuse
before model work. The existing detailed report/role/receipt/metric admission is
preserved. The bundled1,647,144B controls are exact original accepted JSON, with
FAILED/SPENT history, scalar state summaries and costs; they are not model weights
or a restorable parameter checkpoint. No reconstruction or baseline retraining.

## Separately budgeted fresh execution

Omitting `--validate-baseline-only` enters the existing fixed comparison. This
still checks original Python3.14.7, NumPy2.4.6 and OMP_NUM_THREADS,
MKL_NUM_THREADS, OPENBLAS_NUM_THREADS each exactly1 before loading controls or
allocating experiment work. The producer was Windows; there is no new OS
isolation claim. A different interpreter can validate saved metadata but cannot
bypass the original execution checks using preflight.

Under a separately frozen local protocol, `--example-baseline` can supply the
same controls without ignored research archives. An explicit `--baseline PATH`
provides an alternative source subject to identical original admission. The
normal path keeps four families, fixedseed17/H64/settings, A2/B3/A2, disposable
probes and all directions. New work is56updates116row exposures, four A-only fits/
8rows/16values, four initializers/1024normaldraws plus predictions/gradients/latent/
chemistry/copies/validation costs already declared in the original protocol.
Replay adds12updates/24rows versus saved controls, so computation is not matched.
No controls are retrained. Plan and review resource limits before execution.

This change validates input portability without rerunning that cohort. Existing
scientific negatives and comparisons remain unchanged; no CPC advantage follows.
Final evaluation, tuning/promotion authority and live hardware remain unavailable.
Clean installation, GPU/native/asset terms and full release are separate gates.

## Files and checks

The existing CLI owns file/resource IO; the existing `_validate_baseline` and
`run_study` retain scientific admission. `importlib.resources.as_file` stays open
through use and records a stable package URI rather than a deleted temporary path.
The existing comparison data package contains both saved summary and controls.
Why this: remove a demonstrated archive dependency with ordinary CLI composition,
without a new loader framework or experiment runner.

```powershell
python -m pytest -q tests/test_continuous_replay_input.py tests/test_continuous_replay_example.py tests/test_continuous_comparison_inspector.py
python -m ruff check scripts/continuous_replay_example.py tests/test_continuous_replay_input.py
python -m ruff format --check scripts/continuous_replay_example.py tests/test_continuous_replay_input.py
python -m mypy --follow-imports=silent scripts/continuous_replay_example.py tests/test_continuous_replay_input.py
```

These tests use metadata/stub callbacks and do not retrain controls or run a cohort.
Extend source selection explicitly; preserve strict execution checks, original
receipts and all failed work when adding other comparison inputs.
