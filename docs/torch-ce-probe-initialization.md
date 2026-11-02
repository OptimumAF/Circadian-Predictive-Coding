# CE-initialized native probe state

`initialize_categorical_ce_probe(actual_source, fit, check_boundary=guard)`
binds an accepted categorical CE fit to the actual unchanged BP source and
returns a detached `CategoricalCEInitialization`. It constructs no models and
performs no fits, forward passes, random draws or native updates. Derivation
copies tensors on CPU. The caller serializes access to the source and owns
TRAIN arrival, actor/receipts, clocks, budgets and paid failure records.

Only output weight/bias values change. Source hidden parameters, complete SGD
momentum, gradient buffers, policy and native update/row counts are preserved.
Why: this is auxiliary CE initialization followed by original SGD continuation;
resetting momentum or rescaling logits would define a different experiment.
Preserved gradients are stale relative to the fitted output and are deliberately
not relabeled as CE gradients. The next original native update clears gradients
as usual. This continuation policy may fail; no outcome-based correction follows.

The record keeps the full source fit and derived snapshot separately.
`copy_ce_initialization` validates that the derived state differs only by its
exact float32 output coefficients. `ce_initialization_record` reports the fixed
continuation, coefficients, fit losses/work, original native counters and logical
copied-output-value count. That count is the initialization geometry, not a claim
to count every tensor allocation/deepcopy. Revalidation may copy those values
again. Auxiliary fit exposure, construction, probe updates, readouts, time, memory
and storage must all be charged separately. Zero additional native updates does
not mean zero auxiliary training cost or zero old-TRAIN rehearsal.

The existing `scripts.ticket_native_probe.run_owned_probe` accepts the optional
`ce_initialization` keyword. Its `state` and source report remain the ORIGINAL
checkpoint12; do not substitute the derived snapshot or edit that checkpoint's
record. The helper separately admits the derived state and installs it only via
the existing independent `ProbeLearner` factory. Source-counted forks are never
called. The existing probe receives the derived state at its restore boundary,
preserving its original F16/return16 roles, counters, endpoints and budgets.
No CLI flag or default behavior changes. This optional composition needs a
separately frozen/reviewed native protocol before scientific execution.

At the existing guarded source handoff, the Python composition is:

```python
initialization = initialize_categorical_ce_probe(
    actual_source, fit, check_boundary=require_original_source
)
probe_row = run_owned_probe(
    original_state, spec, original_report, original_budget,
    check_boundary=require_original_source, memory=original_memory,
    ce_initialization=initialization,
)
```

The caller must finish qualifying the original source and fit before that
handoff, and retain the fit's paid work even if initialization or probing fails.

Input values and ownership are guarded around caller callbacks. Fit/source
changes and equal-value replacement of fit work are refused, never rolled back.
Source actual state is checked despite spoofed public snapshots. These are
ordinary Python ownership contracts, not a security sandbox or proof that
arbitrary hand-created fit records came from legitimate TRAIN data. No scalar
JSON source loader, final selection, deployment or promotion is added.

Focused software verification:

```powershell
$env:TEMP='D:\Dev\_Temp'; $env:TMP='D:\Dev\_Temp'
.venv/Scripts/python.exe -B -m pytest -q tests/test_torch_categorical_ce_initialization.py -p no:cacheprovider
.venv/Scripts/python.exe -B -m ruff check src/adapters/torch_categorical_ce_initialization.py scripts/ticket_native_probe.py tests/test_torch_categorical_ce_initialization.py
```

Budget model fixtures before running them. [Executed commands/costs/limits](reports/torch-ce-probe-initialization-20261030.md)
distinguish the tiny actual owned probe from scripted source12 handoff seams.
Neither is a native task-acquisition, retention, recovery, comparative or release
result. [The accepted known-task control](reports/ticket-ce-readout-native-20261030.md)
remains separate and paid; its old-TRAIN auxiliary rehearsal was408gradient rows.

The first [native adaptation attempt](reports/ticket-ce-probe-native-20261030.md)
stopped before CE fitting or private training: its observation wrapper compared
copied public receipts by identity. Its allowance is spent and all adaptation
outcomes remain unknown. A metadata-only correction passes eight checks but is
unapplied to the frozen producer; it supplies no native adaptation evidence.
