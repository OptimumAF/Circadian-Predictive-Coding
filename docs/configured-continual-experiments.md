# Typed continual-shift configuration (P5.4)

The existing `run_continual_shift_benchmark` CLI has three named
presets: `baseline`, `strength-case`, and `hardest-case`. It remains
the configurable descriptive continual route. The fixed v14 matched
comparison keeps its original manifest, seeds, protocol IDs, and
raw result bytes.

From the repository root, a small local example is:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_continual_shift_benchmark --profile baseline --seeds 13 --sample-count-phase-a 80 --sample-count-phase-b 80 --phase-b-train-fraction 0.5 --override phase_a_epochs=2 --override phase_b_epochs=2 --override hidden_dim=4 --override circadian_sleep_interval_phase_a=2 --override circadian_sleep_interval_phase_b=1 --json-result artifacts/runs/example-continual-result.json --resolved-config artifacts/runs/example-continual-config.json
```

Each `--override` uses `FIELD=JSON`. Values must be JSON numbers,
`hidden_dims` as an integer array or `null`, and the field must be in
the allowlist below. For PowerShell array values, quote the whole
argument, for example `--override 'hidden_dims=[4,4]'`. Repeating a
field is an error. Overrides apply after the selected preset and
legacy individual flags, so the final value is explicit. An override
requires `--json-result` or `--resolved-config`; paths must be new and
distinct. Unknown keys, baseline learning rates, `protocol_id`,
`model_order`, and whole nested `circadian_config` changes are rejected.

| Type | Allowed fields |
|---|---|
| Integer | `sample_count_phase_a`, `sample_count_phase_b`, `phase_a_epochs`, `phase_b_epochs`, `hidden_dim`, `circadian_sleep_interval_phase_a`, `circadian_sleep_interval_phase_b` |
| Finite number | `validation_fraction`, `phase_b_train_fraction`, `phase_a_noise_scale`, `phase_b_noise_scale`, `phase_b_rotation_degrees`, `phase_b_translation_x`, `phase_b_translation_y` |
| Integer array or `null` | `hidden_dims` |

The existing `ContinualShiftConfig` validator checks applicable
ranges and protocol-specific replay constraints before training.
`hidden_dim` must equal the final `hidden_dims` width when an array
is set. The `continual_resolved_config_v1` artifact records the preset,
seeds, explicit overrides, and every field of the exact typed config
passed to training. The completed JSON result embeds the same
`config`. For example, a verification script can compare both objects
before analyzing scores. This artifact is an input record, not a
versioned v14 run manifest or a scientific claim about the model.
