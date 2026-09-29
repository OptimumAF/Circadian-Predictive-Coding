# Measured v14 wake observations (P5.2b)

The opt-in `v14_wake_diagnostics_v1` sidecar records the diagnostic
returned by each existing successful NumPy `train_epoch` wake update.
There is exactly one row per method, phase, epoch, seed, and trigger
arm: 2 seeds × 3 arms × 24 epochs × 3 methods = 432 rows. Capturing
the return value adds no model update or extra data evaluation. The
default v14 runner still discards the return, and its fixed train-only
and scored JSON protocols and bytes remain unchanged.

| Method | Field | Definition | Measurement stage |
|---|---|---|---|
| Backprop | `loss` | `numpy_binary_bce_preupdate_v1`: binary cross-entropy on the train batch | Before the gradient parameter update; no latent inference |
| Ordinary PC | `energy` | `numpy_pc_bce_plus_half_mean_all_hidden_error_sq_v1` | After latent relaxation, before the gradient parameter update |
| Circadian PC | `energy` | `numpy_circadian_bce_plus_half_mean_final_hidden_error_sq_v1` | After latent relaxation, before the gradient parameter update; a pending prune decay may have run at the start of this call |

Loss and energy have different definitions and must not be pooled as
one metric. The values describe the same train-only batch used for the
update, not inner-guard, outer-selection, or final-test data. They
are computed before the gradient parameter mutation and returned
after a successful update. `metric_name`, `metric_definition`, and
`measurement_stage` travel with every value. The metrics are finite
and serialized in fixed seed/arm/phase/epoch/method order.

## Local workflow

From the repository root, with the documented NumPy environment:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --run-id p52-measured-local --capture-wake-diagnostics
.\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --verify-run artifacts/runs/p52-measured-local
.\.venv\Scripts\python.exe -m scripts.project_v14_observations --run-measured artifacts/runs/p52-measured-local
.\.venv\Scripts\python.exe -m scripts.project_v14_observations --verify-measured-run artifacts/runs/p52-measured-local
```

The runner trains the same six v14 trials once. It preflights all
train-only role, replay, work, and capacity facts before any final
label. It serializes raw training and diagnostic rows before the
global final gate, then scores all six trials on common final roles.
Only after the complete P5.1 bundle verifies does it write
`measurements-v1/wake-diagnostics.jsonl` and a manifest last. That
manifest binds the run ID, completed source-manifest SHA-256, raw
training/outcome SHA-256 values, diagnostic SHA-256, and row count.
The sidecar verifier checks all saved row identities, metric
definitions, finiteness, matched update counts, canonical JSONL, and
source hashes. The manifest is not a cryptographic signature; values
cannot be independently reconstructed from the historical v14 JSON.

`--run-measured` writes an exclusive `observations-measured-v1/`
directory. It retains the eight P5.2a data files byte for byte and
adds 432-row `wake-metrics.jsonl` and `wake-metrics.csv`. The CSV is a
direct view of all measured method/epoch rows. The existing 18-row
`summary.csv` is still derived only from final method results. The
P5.2a `wake-epochs.jsonl` status remains accurate for the *raw v14
payload*, which lacks metrics; measured values live in the separately
identified files. Verification re-derives all ten projection files
from the completed source and verified sidecar, rejecting changed,
missing, or rehashed projection output.

Two bounded local captures, `p52-measured-a/b`, repeat the 432-row
diagnostic file and all ten measured projection data-file SHA-256
values byte for byte. The run-bound manifest hashes differ by run ID.
These checks establish same-environment repeatability, not a
cross-platform bitwise guarantee. P5.3a atomically publishes the
sidecar and measured projection directories; checked training resume
and interrupted-state recovery remain P5.3b.

Why this: a distinct sidecar preserves the fixed negative/mixed v14
baseline evidence while allowing true wake-update diagnostics to be
audited alongside sleep, replay, topology, guard, and final outcomes.
