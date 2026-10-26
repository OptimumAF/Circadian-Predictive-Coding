# Recent-only rehearsal: a matched-exposure control

`--recent-only` selects an explicit recent FIFO policy through the existing
rehearsal CLI. Combined with `--diagnose-learnability`, it uses the same declared
memory cap and per-method native work as the accepted rotating-memory run.
132focused cases, scoped statics and independent SOURCE/GO/ACTUAL review pass.

## A tradeoff, not an unconditional winner

The fixed seed23 A→B→A run compares two bundles: recent FIFO/latest-two in arrival
order, versus recent/reservoir rotation. Retention composition AND order change.
Both execute72wake+48replay+256probe updates; there is no isolated reservoir
effect or general claim from this one sequence/initialization.

| Recent minus rotating | B gain ↑ | A forgetting ↓ | Return gain ↑ |
| --- | ---: | ---: | ---: |
| BP |-.00517829|+.01622242|-.00446634|
| PC / neutral CPC |-.00428941|+.00618584|+.00209533|
| active CPC |-.00379034|+.00570041|+.00101003|

Absolute levels prevent misreading these differences. Recent-only has LOWER A
loss at every trained checkpoint for every method, despite larger forgetting
relative to its much better A6 level. BP A12/A18=.66280614/.60109441 recent versus
.67701313/.61083506 rotating. Conversely, every B12/B18 loss is HIGHERrecent.
This observation favors recent on absolute A levels and rotation on B.

Recent has9positive/7negative sixteen-step fresh C gains, versus rotating7/9.
Yet all12trained recent C initial AND final losses are HIGHER. PC wake6 gains
slightly from .73662543→.73659712, while rotating worsens from .73051742→.73073310
but still finishes better. Larger gains/sign flips do not establish better
learnability or plasticity; starting states/headroom differ. All96new+96reference
curve points and primary A/B/C levels are preserved. PC and neutral match exactly.
No circadian advantage, equivalence, isolated reservoir effect or universal winner.

## Costs and scope

Observed376calls/returns,2672rows,128predictions,1692coupled/188BP evaluations,
16health passes/32SVDs. Pilot0.6853550s, sampled peak processRSS41586688B; no timing
or process-memory efficiency conclusion. Recent stores16rows/384arrayB at each
boundary, rotating14–16/336–384B; only their384B caps match. Metadata, states,
copies and native forks add memory. Sources/templates remain unchanged and
probe labels never train benchmark/deployed models. Final scoring stays deferred.

All rates/data/seeds/metrics remained frozen; no baseline rerun or tuning. The
source author's initial type error and repair remain recorded; root runtime
gates and native case passed firstattempt. Full R2.3/R2.8/release stay open.
Exact commands/protocol/raw results/independent review:
`artifacts/runs/r23-recent-control-20261016/{SCOPE,RESULTS,REVIEW}.md` and JSON.
See [usage](../stream-rehearsal.md).

Next: one fixed two-initialization paired-policy scenario using original seeds
41/59, both policies/all methods,480updates without probes. Freeze/review its
small local protocol first; preserve seed23 and all tradeoffs rather than choose
favorable runs. This checks initialization reproducibility, not new-task
generalization or confirmation.
