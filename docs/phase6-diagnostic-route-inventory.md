# Phase 6 diagnostic and visualization route inventory

P6.1f covers public output families left after matrix rows A–E. This is an
artifact and interface check, not a new model ranking. Historical files and
their scores remain read only. The current checkout is CPU-only, has an
extracted CIFAR-10 cache, lacks an extracted CIFAR-100 cache, and has no paired
`benchmark_multiseed_cifar100_summary.csv`/result JSON. No weight download or
large sweep is needed for this inventory.

| Public route | Declared outputs | Smallest safe check and current boundary |
|---|---|---|
| `scripts.generate_readme_figures` | Four PNG charts, four interactive HTML pages, one illustrative GIF, and `provenance.json`. | **P6.1f2 covered:** a temporary, explicitly synthetic three-model paired CSV/JSON source exercised the public CLI and all ten files, with source SHA, finite values, protocol labels, illustrative GIF, and occupied-path checks. Nonfinite source input now rejects before rendering. No saved paired CIFAR-100 source exists locally; this fixture supplies no measured CIFAR-100 result. |
| `scripts.generate_hardest_mode_dynamics` | One dynamics GIF and one interactive HTML page; stdout names protocol, comparison scope, and final accuracy. | **P6.1f3 covered:** opt-in `--tiny-smoke` fixes 40 rows/phase, two epochs/phase, an 8-point grid, and one latency repeat. Both validation and explicit legacy test-informed CLI choices produced/read four-frame GIFs and finite HTML payloads with fixture/config/split labels. The default 120/180-epoch route was not run. |
| `scripts.run_multiseed_resnet_benchmark` | Completed JSON, per-seed CSV, summary CSV, plus stdout. | **P6.1f4 covered:** a real public two-seed, eight-train-row-per-seed CPU synthetic run with no downloaded weights produced all three finite files in about five seconds; rows, resolved configs, role hashes, summary aggregates, and three occupied-path refusals passed. These unmatched fixture scores are not a CIFAR-100 figure source. |
| `scripts.run_isolated_head_memory_smoke` | Finite JSON stdout with one CPU child per matched head and observed process RSS/capacity fields. | **P6.1f4 covered:** the fixed tiny CPU public process returned three distinct PIDs, common development feature/split hashes and fixed capacity. Stdout now includes the full fixture config and benchmark track alongside RSS observations. RSS is observed, not a reproducible score. |
| `scripts.verify_cifar_loader_order` | Finite JSON stdout with 0/2-worker role IDs, train-batch/view hashes, and optional training-seal record. | **P6.1f4 covered locally:** the real public default CLI read the already extracted CIFAR-10 cache, emitted both worker reports, matching role split hashes, 8/4/4/4 counts, and no final-test iteration. Existing bounded fake-source tests also check seeded stochastic views and optional final-test seal. The real CLI took about 39 seconds; an uncached clean clone needs the verified local CIFAR-10 cache to repeat it. |
| `scripts.run_circadian_policy_sweep` | Estimate JSON stdout; if launched, validation-only ranking JSON and text summaries. | **P6.1f5 covered for interface/artifacts:** the real public estimate and default 1,000-update refusal ran before data work. The 18-cell ordered policy grid is SHA-pinned. An in-process test with equal synthetic development reports exercised its complete temporary JSON/top-ten/stdout writer using a test-only cap; no real Torch/CUDA trial ran. The default full path still requires 14,400 planned updates/900,000 exposures plus CUDA/ImageNet. Historical unversioned `benchmark_circadian_policy_sweep_results.json` remains distinct. |
| `scripts.run_pareto_hard_tuning` | Estimate JSON stdout; if launched, validation-only three-family ranking JSON and text summaries. | **P6.1f5 covered for interface/artifacts:** the real public estimate and default 1,000-update refusal ran before data work. Existing SHA pins preserve all 10/12/12 ordered grids. Equal synthetic report stubs for all 34 cells and three seeds exercised complete temporary three-family JSON/top-ten/Pareto/stdout output with a test-only cap. The default full path still requires 81,600 planned updates/5,100,000 exposures plus CUDA/ImageNet. Historical unversioned `benchmark_pareto_hard_results.json` remains distinct. |
| `scripts.run_continual_shift_benchmark`, toy/indepth, v6–v14, and `scripts.build_v14_dashboard` | Their configured text/JSON/CSV/HTML/PNG or checkpoint outputs. | Already covered through public routes and readers in P6.1a–d. Their explicit legacy protocol branches were checked separately; repeating them under F would add no output family. |

The two sweep estimate-only CLIs and default refusals passed in empty
temporary directories: 18/1 and 34/3 candidates/seeds, with the work counts
above. Complete fixture-only JSON writers used equal synthetic reports and
test-local caps; the default live launch cap was not raised for a model run.
Representative CUDA runtime/device skips are recorded separately in
`phase6-torch-route-inventory.md`.
