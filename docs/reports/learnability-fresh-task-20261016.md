# R2.1 learnability: fixed fresh-task probe, 2026-10-16

## Scope and outcome

This is one prespecified engineering/exploratory seed (23), using the existing
matched BP/PC/neutral-CPC/active-CPC models and delayed A/B/A runner. It is a
same-data diagnostic extension of the accepted R1.7 pilot, not independent
confirmation or a new claim of circadian advantage. Eighteen benchmark updates
per method were needed to obtain copied checkpoint states; the older outcome
report did not retain those restore-ready states. Baseline settings and the
original R1.7 seeds/results/protocol were not changed or retuned.

All four arms and sixteen isolated probes completed. At the checkpoint after B,
every method reduced C evaluation BCE after one/two probe updates but ended
slightly worse after the fixed fourth update. Across all sixteen curves, every
probe improved at an earlier checkpoint and fifteen had negative fourth-update
gain. Only BP at the untrained checkpoint had positive fourth-update gain.
**This is not evidence of absent learning or loss of plasticity.** PC/active CPC
already have negative fourth-update gains at initialization. A fixed small C task
and short exposure cannot establish degradation or a general ranking. Neither
updates nor metrics were changed to make an endpoint improve.

The benchmark showed positive B gain, positive A retention loss during B, and
positive A recovery on return for all methods. PC and neutral CPC benchmark
outputs and entire fresh-task curves matched exactly. No confidence interval,
causal claim, equal-compute comparison or generalized advantage is supported.

## Prespecified outcomes

BCE directions: positive B gain/return gain/fresh gain means improvement;
positive retention loss means A worsened during B. Values are raw differences.

| Method | B gain | A retention loss | A return gain | C initial after B | C after 4 | C gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bp | 0.056233286 | 0.009415478 | 0.053088664 | 0.713544224 | 0.713640911 | -0.000096687 |
| pc | 0.013821978 | 0.011633764 | 0.024221352 | 0.728981160 | 0.729721558 | -0.000740398 |
| neutral_cpc | 0.013821978 | 0.011633764 | 0.024221352 | 0.728981160 | 0.729721558 | -0.000740398 |
| gated_cpc | 0.013030703 | 0.010985131 | 0.022036625 | 0.729200799 | 0.729885832 | -0.000685033 |

All prespecified C evaluation checkpoints after B (none selected as a new rule):

| Method | 0 updates | 1 update | 2 updates | 4 updates |
| --- | ---: | ---: | ---: | ---: |
| bp | 0.713544224 | 0.713119836 | 0.711439880 | 0.713640911 |
| pc | 0.728981160 | 0.728895863 | 0.728701318 | 0.729721558 |
| neutral_cpc | 0.728981160 | 0.728895863 | 0.728701318 | 0.729721558 |
| gated_cpc | 0.729200799 | 0.729127723 | 0.728918553 | 0.729885832 |

Fourth-update fresh gains at every original benchmark checkpoint:

| Method | Initially (0) | After A (6) | After B (12) | Return A (18) |
| --- | ---: | ---: | ---: | ---: |
| bp | 0.000977335 | -0.000489929 | -0.000096687 | -0.000810736 |
| pc | -0.000659032 | -0.000844782 | -0.000740398 | -0.000874813 |
| neutral_cpc | -0.000659032 | -0.000844782 | -0.000740398 | -0.000874813 |
| gated_cpc | -0.000645906 | -0.000801287 | -0.000685033 | -0.000778405 |

`artifacts/runs/r21-learnability-20261016/pilot.json` retains all 64 probe scoring points with BCE/accuracy, all initial
levels, all benchmark scores, diagnostics, original initialized tensors and
complete status vectors. The protocol/source were frozen before execution in
`artifacts/runs/r21-learnability-20261016/protocol.json` and `artifacts/runs/r21-learnability-20261016/source-snapshot.json`; no source changed afterward.

## Costs and diagnostics

Observed 72 benchmark +64 probe =136 committed/attempted adapter updates;
576+512=1088 training-row exposures; 16 disposable probe clones; 64 probe and
32 benchmark score forwards. Diagnostics used16 feedforward passes and32 SVDs.
Pre-entry counters observed612 coupled-energy and68 BP-gradient evaluator calls,
which are different computations and are not FLOPs or matched compute.

Instrumented pilot time0.274876s (worker0.3099109s, process0.7635722s).
Completed diagnostic windows total0.008955s; complete probe calls0.1077954s,
including preflight and copies. Per-arm benchmark/orchestration remainder totals
0.1366556s; other global setup/measurement time stays in total time. These are
one instrumented execution, not reliable latency comparisons. Peak whole-process
RSS40,960,000B, initial37,384,192B,687 samples; one RSS sampler thread joined,
final one thread. CPU/BLAS threads1; no GPU/download/replay/sleep/promotion.
Output126304B, below32MiB. No budget/profile violation or failed pilot row.

Reserved probe array data1536B per clone and label-free reference512B; diagnostic
activations4096B per checkpoint. Checkpoint native arrays840B for BP/PC and968B
for CPC controls. These retained-array observations exclude Python and temporary
allocations and are not summed into a false peak-memory claim; RSS is separate.

All32 layer observations had active fraction1, dead0 and saturated0 at the fixed
thresholds; centered effective ranks were roughly2.74–2.81 despite width8. Such
health summaries did not by themselves predict the fourth-update rebound.
Native importance and neuron ages remain explicitly unavailable in coupled
models. Chemistry/wake counts are not substituted. **Full R2.1 remains open.**

## Verification and limits

101 focused cases/303 phases passed on the final combined tree: analytic rank/
activity/bounds, reserved roles/IDs/storage, owned-fork/state/batch isolation,
mutating failures, attempted/committed work, invalid predictions/timers/budgets,
actual native PC-neutral probe parity and existing stream-observer isolation.
Ruff, format and Windows/Linux mypy passed on seven changed paths; an initial
root JSON-row inference annotation failure was fixed and retained separately.
Source review corrected pre-scan bounds, detachment failure rows and sampler
constructor failure accounting before runtime. Independent actual review
recalculated all16 curves and64 scoring points, costs and parity.

No whole-repository/clean-clone/release, long-run/large sweep, multi-seed/task
confirmation, causal plasticity, native importance/age, or circadian-benefit
claim. Existing final benchmark data/scores never reached training, promotion,
probe selection or observer controls. All probes use separate templates/budgets
and finish before benchmark final scoring. Source checkpoint/template and original
benchmark template remained unchanged. No tuning, seed replacement or retry.

Historical next action: implement a read-only diagnostic adapter for existing legacy CPC native
importance/age fields with snapshot/RNG preservation tests. Keep unsupported
coupled fields explicit and this frozen pilot unchanged. Broader fresh-task
confirmation requires its own reviewed protocol, rather than choosing an earlier
checkpoint or adding exposure until a model wins.

Follow-up: the legacy reader and opt-in native stream CLI are now implemented
and independently accepted; see [native observations](../learnability-diagnostics.md#native-importance-and-age-observations)
and `artifacts/runs/r21-native-units-20261016/RESULTS.md`. The original pilot's
protocol, archived source snapshot, outcomes and feature-diagnostic functions
remain unchanged; no additional comparative run was performed.
