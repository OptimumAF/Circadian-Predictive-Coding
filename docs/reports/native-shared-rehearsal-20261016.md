# Native shared rehearsal on the coherent stream

The existing delayed stream now executes bounded, shared rehearsal for BP, PC,
neutral CPC and active CPC. Focused boundary tests, scoped static checks and
independent review qualify this opt-in integration. No-replay behavior and the
separate legacy native/shared-buffer equality gate remain unchanged.

## Fixed local observation

One predeclared seed23 A→B→A stream, eighteen eight-row wake batches per arm,
delay2 and checkpoints0/6/12/18. Every third wake update rehearses two arrived
rows from recent/reservoir quotas8+8,192+192float64 array bytes, seed53 and the
accepted rotating selector. All arms use identical content selections and
initial tensors. Final scoring begins only after that arm's training finishes.

| Method | New B BCE gain ↑ | Old A loss during B ↓ | A recovery gain ↑ |
| --- | ---: | ---: | ---: |
| BP |0.07407838|0.00263935|0.06617807|
| PC |0.01627108|0.00998856|0.02915016|
| neutral CPC |0.01627108|0.00998856|0.02915016|
| active CPC |0.01443337|0.00895049|0.02594464|

All four learn B and recover on A's return; all retain a positive A loss during
B. BP has the strongest three outcomes in this small observation. PC and
neutral CPC reports match exactly. Active CPC has slightly less A loss than PC,
with smaller B and return gains. This establishes no circadian advantage.

The retained no-replay R2.1 outcomes are descriptively worse on these three
differences for each arm. That reference used72 optimizer updates/576rows;
this run used120/624. Unequal extra work prevents attributing the difference
to replay rather than exposure/compute. No rates, seeds, metrics or baselines
were changed in response. Earlier negative probe/shift outcomes remain intact.

## Cost and coverage

Observed72wake+48replay updates and624labeled-row exposures,24complete
consolidations,32final predictions,540coupled/60BP evaluations. No probe or
promotion. All charged replay calls returned and committed in this native run;
tests separately cover failed charges and postcommit failures. Replay advances
native internal clocks/gates. Pilot0.3178311s, peak sampled processRSS40939520B;
not a model-attributed or equal-compute timing comparison.

Stored replay arrays peak16rows/384bytes, excluding metadata, reports, states
and temporary copies. At B boundaries9/12, four/three first-A content IDs remain,
but one/zero are proposed. Coverage is therefore limited even in this short
stream. Do not infer general long-stream context survival or tune this negative
exposure fact away. Posthoc annotation regenerated only the fixed NumPy data;
no additional native training or evaluation was performed.

## Acceptance and remaining work

70current focused cases qualified via69initial passes and one corrected fake
RSS fixture case;71executions retain the original failure. A test annotation
type failure and all corrected gate receipts are also preserved. No production
repair was needed. Independent SOURCE/GO and ACTUAL assessments, raw results,
all checkpoint scores/accuracies and exact commands live in
`artifacts/runs/r23-native-rehearsal-20261016/{RESULTS,REVIEW}.md` and JSON files.

FullR2.3/R2.1/R2.7 and repository release requirements remain open. Next useful
integration is the existing isolated reserved sixteen-step fresh-task observer
on these rehearsed candidates, under a separate frozen budget. It can distinguish
retention/recovery from ability to learn a fresh task without training the
deployed/benchmark learner on probe data. See [usage](../stream-rehearsal.md).
