# Figures

This folder stores visual assets used by `README.md`.

## Generated Artifacts

- `benchmark_accuracy.png`
- `benchmark_train_speed.png`
- `benchmark_inference_latency_p95.png`
- `circadian_sleep_dynamics.gif`

## Regeneration

Run:

```powershell
python scripts/generate_readme_figures.py --summary-csv benchmark_multiseed_cifar100_summary.csv --output-dir docs/figures
```

The GIF is illustrative and intended for communication in the README.
