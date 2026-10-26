# R2.1 fixed sixteen-update probe, 2026-10-16

## Scope and findings

One prospectively frozen exposure follow-up used the same seed23, matched BP/PC/neutral-CPC/active-CPC settings, delayed A/B/A stream and reserved C task as the original four-update pilot. Its sixteen probe batches preserve the original four as an exact prefix; C evaluation and label-free reference data are unchanged. This outcome-motivated exploratory follow-up is not independent confirmation or task generalization. No baseline, learning rate, seed or metric was tuned.

All four arms and sixteen isolated probes completed. The sixteen-update endpoint improved C BCE for BP at all four stream checkpoints. PC, neutral CPC and active CPC each improved slightly at initialization but had negative endpoint gains at all three trained checkpoints: seven positive and nine negative endpoints overall. Every probe had positive gain at eight updates, followed by nine PC-family rebounds at sixteen. The full curve is retained; eight is not selected as a replacement endpoint. The original fifteen negative four-update endpoints remain unchanged.

All methods still improved B, lost some A performance during B, and recovered A on return. All benchmark results and initial tensors exactly match the retained original run; every original 0/1/2/4 probe prefix matches exactly. PC and neutral CPC match over the entire new curve. Probe work therefore left the measured benchmark trajectory unchanged within this execution.

These observations favor the simpler matched BP control for the frozen sixteen-update C endpoint. They establish no circadian advantage, general superiority, causal plasticity loss, or permanent inability to learn: the observed curves themselves contain intermediate improvements. No further horizon extension or tuning is part of this result.

## Prespecified endpoint gains

Gain = initial C BCE minus BCE after sixteen updates; positive means improvement. All initial levels and scoring points remain in raw JSON and CSV.

| Method | Initially (0) | After A (6) | After B (12) | Return A (18) |
| --- | ---: | ---: | ---: | ---: |
| bp | 0.009477998 | 0.006066047 | 0.004961249 | 0.004035151 |
| pc | 0.000029299 | -0.000153421 | -0.000229245 | -0.000298415 |
| neutral_cpc | 0.000029299 | -0.000153421 | -0.000229245 | -0.000298415 |
| gated_cpc | 0.000011976 | -0.000170075 | -0.000239758 | -0.000303728 |

Full prespecified C curve after B (no point selected as a new policy):

| Method | 0 | 1 | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bp | 0.713544224 | 0.713119836 | 0.711439880 | 0.713640911 | 0.709272527 | 0.708582975 |
| pc | 0.728981160 | 0.728895863 | 0.728701318 | 0.729721558 | 0.728688929 | 0.729210405 |
| neutral_cpc | 0.728981160 | 0.728895863 | 0.728701318 | 0.729721558 | 0.728688929 | 0.729210405 |
| gated_cpc | 0.729200799 | 0.729127723 | 0.728918553 | 0.729885832 | 0.728929513 | 0.729440557 |

Benchmark outcomes, unchanged from the original run:

| Method | B gain | A retention loss | A return gain |
| --- | ---: | ---: | ---: |
| bp | 0.056233286 | 0.009415478 | 0.053088664 |
| pc | 0.013821978 | 0.011633764 | 0.024221352 |
| neutral_cpc | 0.013821978 | 0.011633764 | 0.024221352 |
| gated_cpc | 0.013030703 | 0.010985131 | 0.022036625 |

Positive retention loss means A worsened during B. Retention is separate from learning reserved C; final benchmark outcomes are released after every probe finishes, with no feedback to training or probe selection.

## Costs and limits

Observed72 benchmark +256 isolated probe =328 attempted/committed adapter updates,576+2048=2624 training-row exposures. Sixteen disposable probe clones used96 C scoring forwards; benchmark scoring used32 forwards. Diagnostics used16 feedforward passes and32 SVDs. Native evaluator counts1476 coupled across three arms (492 each) and164 BP correspond to different computations, not FLOPs or matched compute. Equal update counts do not imply equal work.

Instrumented pilot time 0.4872798s; worker0.5195137s, parent0.8764303s. Diagnostic windows total0.0086345s; complete probe-call windows0.3349315s; per-arm benchmark/orchestration remainder0.1266592s. Global setup/measurement remainder stays in total time. These are one instrumented run, not latency comparisons.

Whole-process RSS peak41,197,568B, start37,859,328B,1147 samples; one sampler thread joined, final one thread, no budget violations. CPU/BLAS threads1. Output162974B. No replay, sleep, promotion, environment interaction, download or GPU work. Reserved array storage3840B per probe clone, reference512B, diagnostic activations4096B per checkpoint. Retained snapshot arrays840B for BP/PC,968B for CPC. Array counts exclude Python objects and temporary allocations; they are not substituted for measured RSS.

Feature observations remain those of the unchanged benchmark checkpoints: active1/dead0/saturated0 at fixed thresholds, with centered ranks about2.74–2.81 despite width8. They do not establish why a probe curve rebounds. Coupled importance/age remain explicitly unavailable. Full R2.1 and broader roadmap requirements remain open.

## Execution and evidence

Final combined four-path Ruff/format and Windows/Linux mypy passed. Initial58 focused tests produced57 passes and one test-controller constructor-cap failure after four PC updates; owned model reconstructions had been omitted from the forecast. No production or oracle change was needed. A prospectively separate single-case validation passed with the source-derived nine initializer entries and eight native updates. All58 distinct focused nodes are qualified; initial failure and profiler cutoff remain disclosed. No comparative training started before those gates and independent frozen-protocol/source review passed.

Artifacts: `artifacts/runs/r21-sixteen-probe-20261016/` contains frozen `protocol.json` and twelve source texts in `source-snapshot.json`, actual `pilot.json`, all96 points in `curves.csv`, `observations-summary.json`, focused/validation/pilot receipts, exact commands in `RESULTS.md`, and independent `REVIEW.md`. Standard-library readback verified protocol/counts/old-new equality. Original evidence under `r21-learnability-20261016/` was read, never rerun or modified.

Skipped whole-repository/clean-clone/release gates, large sweeps, independent multi-seed/task confirmation, latency benchmarks and algorithm changes. This is useful scoped evidence; no repository-wide readiness or scientific advantage is claimed.
