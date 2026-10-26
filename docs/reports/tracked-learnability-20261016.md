# Tracked histories in isolated learnability probes — 2026-10-16

## Delivered behavior and acceptance

The existing rehearsal CLI now supports the fixed `tracked-learnability`
scenario. It uses tracked PC/neutral CPC/active CPC models and independent tracked
probe templates. Ordinary matched BP remains unchanged and reports its missing
native histories as unavailable. The existing observer validates native history
before feature/SVD work, reports the measured gradient and age summaries, and
accounts its full snapshot arrays. Every disposable probe restores complete
tracked state; the source/template/checkpoint remain unchanged.

```powershell
.venv/Scripts/python.exe -m scripts.run_stream_rehearsal_pilot --describe --scenario tracked-learnability
.venv/Scripts/python.exe -m scripts.run_stream_rehearsal_pilot --run --scenario tracked-learnability
```

The scenario fixes rotating replay and sixteen-step probes. It rejects the
single-case `--recent-only` and `--diagnose-learnability` flags. All original
single and paired modes remain available with their original descriptors and
call behavior. No new dependency, native objective or orchestration framework.

R2.1's original bounded diagnostic implementation is accepted after independent
SOURCE/GO/ACTUAL review: activation/activity/importance/age/rank/probe fields are
implemented, costs measured, probes isolated, and retention reported separately.
The retained runtime field `full_R21_complete=False` is conservative per-run
metadata: a run does not certify project task acceptance. Canonical acceptance
belongs to the plans and independent evidence, not a self-certifying runtime flag.
Persistence, differentiated maturity, causal utility, renewal and scientific
benefit are separate unfinished requirements.

## Original R2.1 criterion mapping

| Requirement | Current authoritative evidence |
|---|---|
| Active/dead/saturated fractions and activity | 32 feature-health layer records across16 observations; full values exact retained reference; bounded core implementation and focused tests |
| Importance/age summaries |12 measured native histories, IDs0..7, pre-gate batch-aggregate outgoing-gradient EMA and native ages0/10/20/30; BP explicitly unavailable; legacy native reader preserved |
| Feature effective rank |32 actual SVDs and32 per-layer rank records, unchanged definitions/data/values |
| Reserved fresh-task probe |16 disposable independent probes,16 updates each,96 complete BCE/accuracy score points at0/1/2/4/8/16 |
| Diagnostic and probe overhead measured |.0169421001s diagnostics,.4611213999s probe time; calls/rows/arrays/RSS accounted below |
| Probe isolation |Complete tracked fork/restore tests including2→18 ages, pre/post source/template/checkpoint pickle checks; all16 observation flags and all4 source/template flags pass |
| Retention separate from ability to learn |All4 full benchmark stream reports/outcomes retained separately from C curves, exactly equal accepted rehearsal reference |

This is bounded last-hidden, fixed-width native history support. The gradient EMA
is not causal importance or mean absolute per-example sensitivity. Ages are
successful native calls, including replay, not rows or time. There is no renewal
or unit replacement yet; homogeneous ages do not establish differentiated maturity.
Old exact format1 disk codecs/managed capture continue to refuse tracked format2.

## Executed verification

This is a verification of the NEW instrumentation path against retained results,
not a new independent learning study or an untracked baseline rerun. Fixed seed23,
original RNGs/data, ABA18×8/delay2, dimensions2→8→8→1, rate.03/infer4/.2,
rotating16-row/384-byte replay every3wakes/two rows, and all old probe settings.
The new history decay is fixed.95. No rates, endpoints, seeds or metrics changed.

All4 ENTIRE benchmark reports/receipts/plans and all16 probe curves/feature-health
records are exactly equal the accepted rehearsal/probe reference. PC-neutral
benchmark, probe curves AND native histories match exactly. Native histories are
read12times and advance282times including benchmark/replay/probe work. All source
learners, templates and input checkpoints remain unchanged.

| Method | Wake checkpoint | Mean gradient EMA | Mean native age | Snapshot array bytes | Initial C BCE | C gain at16 |
|---|---:|---:|---:|---:|---:|---:|
| bp | 0 | unavailable | unavailable | 840 | 0.73441467 | +0.00947800 |
| bp | 6 | unavailable | unavailable | 840 | 0.71811637 | +0.00538653 |
| bp | 12 | unavailable | unavailable | 840 | 0.69761237 | +0.00178348 |
| bp | 18 | unavailable | unavailable | 840 | 0.69912700 | +0.00214041 |
| pc | 0 | 0.00000000 | 0 | 1032 | 0.73441467 | +0.00002930 |
| pc | 6 | 0.04202064 | 10 | 1032 | 0.73051742 | -0.00021568 |
| pc | 12 | 0.07324279 | 20 | 1032 | 0.72175357 | -0.00052889 |
| pc | 18 | 0.08832100 | 30 | 1032 | 0.72128243 | -0.00059157 |
| neutral_cpc | 0 | 0.00000000 | 0 | 1160 | 0.73441467 | +0.00002930 |
| neutral_cpc | 6 | 0.04202064 | 10 | 1160 | 0.73051742 | -0.00021568 |
| neutral_cpc | 12 | 0.07324279 | 20 | 1160 | 0.72175357 | -0.00052889 |
| neutral_cpc | 18 | 0.08832100 | 30 | 1160 | 0.72128243 | -0.00059157 |
| gated_cpc | 0 | 0.00000000 | 0 | 1160 | 0.73441467 | +0.00001198 |
| gated_cpc | 6 | 0.04203569 | 10 | 1160 | 0.73049159 | -0.00022490 |
| gated_cpc | 12 | 0.07322267 | 20 | 1160 | 0.72230011 | -0.00049497 |
| gated_cpc | 18 | 0.08834882 | 30 | 1160 | 0.72192093 | -0.00053885 |

All96 BCE/accuracy points are preserved in pilot.json/summary.json. Endpoint signs
remain7positive/9negative, including the nine trained PC-family negatives; every
original intermediate improvement and reversal remains. These observations establish
no circadian advantage, general plasticity loss, causal replay benefit or task
transfer/generalization. No endpoint selection, promotion or tuning occurred.

## Costs and checks

Sole native case:376 entered/returned calls=72wake+48replay+256probe,2672labeled-row
exposures=576wake+48replay+2048probe,128score forwards,1692coupled/188BP energy
checks,16feature passes,32SVDs,12history readouts and282history transitions. One
joined RSS sampler, no violations, finalthread1. Pilot.8831015s, worker.9244655s,
parent1.3680155s; observed peak RSS42,110,976bytes;516627outputbytes/empty stderr.

Tracked last-hidden histories retain192arraybytes per width8 state. Complete
snapshot arrays are1032bytes PC and1160bytes CPC, versus840bytes BP. Across the12
reported tracked checkpoints the added history arrays sum2304bytes; this is not
peak total memory. Native/fork/temporary/metadata storage and process RSS are
additional. Aggregate profiled timing is not an isolated causal overhead or
production latency estimate.

116focused cases/348phases pass first attempt:56native training entries,
224energy checks,19predictions,1642fake calls,303forks,8feature passes/14SVD,
no violations/skips/files/finalthread1. Six-path Ruff/format/Windows+Linuxmypy
pass first attempt in8.732659s. Both authors ran only bounded serial statics;
no author tests/native work. Independent review accepted source and execution.
25frozen source texts unchanged;14native-history implementation texts exact;
original describe/paired helper bodies preserved. Readback used stdlib JSON/AST
only, no model work or regenerated data.

No new gate failure this increment. Prior failures/negative results/unrelated
changes remain. Whole-repository, clean-clone, release, Torch, legacy consumer,
persistence migration, fresh comparative studies and sweeps were skipped.
R2.2/R2.3/R2.7/R2.8 and release/full goal remain open.

Next substantial outcome is R2.2: a bounded fixed-capacity mature/low-utility
renewal control with explicit eligibility and complete native reset/identity/state
semantics. Keep protected high-importance units unchanged, preserve old tracked
format2 semantics, and freeze matched no-renewal controls before comparative use.
Utility/age thresholds must not be selected from final or probe outcomes.

[Exact commands and acceptance evidence](../../artifacts/runs/r21-tracked-observer-20261016/RESULTS.md).
[Independent review](../../artifacts/runs/r21-tracked-observer-20261016/REVIEW.md).
