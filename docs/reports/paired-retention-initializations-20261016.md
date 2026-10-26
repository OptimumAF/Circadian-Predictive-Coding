# Fixed paired retention initializations — 2026-10-16

## Status and protocol

Implemented and executed the existing rehearsal adapter/CLI's fixed
`--scenario paired-retention`. Independent SOURCE/GO accepted before execution;
ACTUAL review is recorded in the linked stage REVIEW. Full R2.8/R2.3/R2.1/R2.7
and release/goal remain open. Existing repeated-update/checkpoint work and
seed23 positive/negative results are reused, not rerun.

Protocol `coherent_paired_retention_initializations_v1`: original remaining
initialization seeds41/59 × rotating/recent-only × BP/PC/neutral CPC/active CPC.
Same fixed data RNG7301/9101,2→8→8→1 geometry, rates .03/infer4/.2, ABA18×8,
delay2, checkpoints0/6/12/18. Each seed shares one initial parameter bundle across
all eight policy/method rows. Each row trains18wake +12replay updates, six events
at3/6/9/12/15/18 with two rows/event. Rotating recent8/reservoir8 seed53 versus
recent FIFO16/latest-two-in-arrival-order, global array cap16rows/384bytes.

Both policies use the existing native consolidation helper, separate initially
planned wake/replay budgets, permanent attempt charges, private learner forks,
complete native state commits and deferred final scoring. The matrix runs once
in declared seed→policy→method order. All16slots/eightpairs survive failures;
incomplete differences are unavailable with both reasons, never zero/pooled.
Known partial work is distinguished from potentially unobserved failed calls.
No probes/SVD, tuning, promotion, seed replacement or retries occurred. Existing
single APIs, all four descriptors and default flag dispatch remain supported.

## Observed outcomes

All16rows/eightpairs completed; all16 learned B, lost some A during B and
improved A when it returned. In ALL eight pairs, recent-only has LOWER absolute
A6/A12/A18 BCE and HIGHER B12/B18 BCE. Its B gain is smaller, A forgetting larger
from a better A6 starting level, and return gain larger. Seed23's retained BP
return-gain direction differs; do not replace or pool it away.

Differences below are **recent minus rotating**. Larger B/return gain and smaller
forgetting are favorable for those individual metrics, with no scalar winner.

| Seed | Method | B gain difference | A forgetting difference | A return-gain difference |
|---|---|---:|---:|---:|
| 41 | bp | -0.00793614 | +0.00765016 | +0.00313745 |
| 41 | pc | -0.00503361 | +0.00507172 | +0.00129097 |
| 41 | neutral_cpc | -0.00503361 | +0.00507172 | +0.00129097 |
| 41 | gated_cpc | -0.00464017 | +0.00481331 | +0.00095039 |
| 59 | bp | -0.00542668 | +0.01035176 | +0.00092091 |
| 59 | pc | -0.00292948 | +0.00436161 | +0.00096953 |
| 59 | neutral_cpc | -0.00292948 | +0.00436161 | +0.00096953 |
| 59 | gated_cpc | -0.00269170 | +0.00417904 | +0.00065517 |

Primary absolute BCE levels (full0/6/12/18 BCE **and accuracy** for both A/B are
retained in pilot.json/pairs and summary.json, without endpoint selection):

| Seed | Policy | Method | A6 | A12 | A18 | B12 | B18 |
|---|---|---|---:|---:|---:|---:|---:|
| 41 | rotating | bp | 0.56668309 | 0.58217868 | 0.55904678 | 0.65341795 | 0.63667098 |
| 41 | rotating | pc | 0.57790872 | 0.58032511 | 0.56639643 | 0.68353155 | 0.66707033 |
| 41 | rotating | neutral_cpc | 0.57790872 | 0.58032511 | 0.56639643 | 0.68353155 | 0.66707033 |
| 41 | rotating | gated_cpc | 0.57841027 | 0.58059175 | 0.56794439 | 0.68535311 | 0.67018916 |
| 41 | recent_only | bp | 0.55485151 | 0.57799726 | 0.55172792 | 0.66135409 | 0.67639932 |
| 41 | recent_only | pc | 0.57067451 | 0.57816262 | 0.56294296 | 0.68856516 | 0.69402674 |
| 41 | recent_only | neutral_cpc | 0.57067451 | 0.57816262 | 0.56294296 | 0.68856516 | 0.69402674 |
| 41 | recent_only | gated_cpc | 0.57131509 | 0.57830987 | 0.56471213 | 0.68999328 | 0.69525937 |
| 59 | rotating | bp | 0.56534419 | 0.58140727 | 0.55207799 | 0.70043091 | 0.67638166 |
| 59 | rotating | pc | 0.58745949 | 0.59170600 | 0.58268553 | 0.73754744 | 0.72257667 |
| 59 | rotating | neutral_cpc | 0.58745949 | 0.59170600 | 0.58268553 | 0.73754744 | 0.72257667 |
| 59 | rotating | gated_cpc | 0.58769826 | 0.59159942 | 0.58309052 | 0.73877035 | 0.72463844 |
| 59 | recent_only | bp | 0.55051095 | 0.57692579 | 0.54667561 | 0.70585759 | 0.71870298 |
| 59 | recent_only | pc | 0.58008746 | 0.58869558 | 0.57870557 | 0.74047692 | 0.74716597 |
| 59 | recent_only | neutral_cpc | 0.58008746 | 0.58869558 | 0.57870557 | 0.74047692 | 0.74716597 |
| 59 | recent_only | gated_cpc | 0.58045504 | 0.58853526 | 0.57937119 | 0.74146205 | 0.74779935 |

PC and neutral CPC complete stream reports match exactly in all four seed/policy
groups. All methods and both initializations have the same arrived-only replay
plans per policy, identical to accepted seed23 data-only plans/retention. Initial
scores match across policies for each seed/method; the two seed bundles differ.
All source learners remain unchanged. BP has larger B/return gains and lower
A18/B12 BCE in this small matrix, while also losing more A relative to its own
A6. Do not infer a universal method ranking or matched BP/PC computation.

Accuracy differences are mixed; BCE directions do not imply uniform accuracy gains.

This repeats the policy tradeoff on two fixed-data initializations. It is not
task/data generalization, statistical confirmation, equivalence or a circadian
advantage. Retention composition AND within-event order change together. Same
memory caps are not equal occupancy: rotating14–16rows/336–384arraybytes versus
recent16/384. Metadata, learner states, fork copies and process RSS are extra.
No fresh-task C observations were collected here; prior negative C endpoints
remain unchanged and cannot be generalized to these new initializations.

## Costs and verification

SOLE native case:480entered+returned calls=288wake+192replay,2496labeled-row
exposures=2304wake+192replay,96committed events,128score forwards,
2160coupled/240BP evaluations; zero probes/feature diagnostics/SVD. All attempted,
returned and committed replay counts192. One joined RSS sampler; no violations,
finalthread1. Pilot1.1581120s, worker1.1955598s, parent1.6346574s; observed peak
RSS41766912bytes;1159058outputbytes/stderr0. These are aggregate profiled local
measurements, not per-policy latency or deployment efficiency.

Scoped combined Ruff/format/Win+Linuxmypy5paths pass9.2809998s. Focused70distinct
nodes qualified via69currentpasses plus one repaired pass and prior unchanged
six-case guarded evidence. Initial fake-function-entry cap FAILED at2049/2048:
subclass and superclass entries were both counted. Original69PASS/1FAIL and
all logs remain. After that exception Python disables profiling, so six following
old adapter cases' work counts were unobserved in the failed batch; their prior
guarded evidence is reused with exact test text/all8old adapter function AST
bodies unchanged. New22paired cases preceded cutoff and passed. Only the failed
existing recent+probe node reran under a separate prospectively512entry allowance:
376entries/52fork,3PASSphases/no violations,2.3808013s. No production/test repair
or whole-batch rerun. Across71current executions:212PASSphases/1preservedFAIL.

Pre-entry source freeze includes23relevant texts (review added the directly
imported seed/method controls before execution). Saved-result stdlib arithmetic
passes: all23unchanged,19common old texts exact, all8old adapter bodies unchanged,
all primary/delta/receipt/work/parity/retention assertions. No repository import,
regenerated data, model update, diagnostic or extra native experiment in readback.

Whole-repository/clean-clone/release/Torch/legacy-consumer/durable-resume checks,
long-stream coverage, additional tasks/seeds, fresh probes, intervals and sweeps
were not run. Existing unrelated ARCHITECTURE/CHANGELOG EOF whitespace debt is
preserved. No full task ID closed; only this fixed paired increment is delivered.

[Protocol, exact commands, raw failures and receipts](../../artifacts/runs/r28-paired-retention-20261016/RESULTS.md).
[Independent review](../../artifacts/runs/r28-paired-retention-20261016/REVIEW.md).
