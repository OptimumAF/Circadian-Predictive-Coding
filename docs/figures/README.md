# Figures

This folder stores visual assets used by `README.md`.

## Generated Artifacts

- `benchmark_overview_compact.png`
- `benchmark_accuracy.png`
- `benchmark_train_speed.png`
- `benchmark_inference_latency_p95.png`
- `circadian_sleep_dynamics.gif`
- `interactive_benchmark_overview.html`
- `interactive_benchmark_accuracy.html`
- `interactive_benchmark_train_speed.html`
- `interactive_benchmark_inference_latency_p95.html`

## Regeneration

Run:

```powershell
python scripts/generate_readme_figures.py --summary-csv benchmark_multiseed_cifar100_summary.csv --output-dir docs/figures
```

The GIF is illustrative and intended for communication in the README.
Plotly HTML charts are interactive when opened in a normal browser context
(local file, static host, or GitHub Pages).

The combined dashboard page is at `docs/index.html`.
When Pages is enabled, the live dashboard URL is:
`https://optimumaf.github.io/Circadian-Predictive-Coding/`.
