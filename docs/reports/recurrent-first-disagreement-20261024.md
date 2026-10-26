# Why the saved recurrent blue-cue game fails

The first supported failure is a learned choice on the original TRAIN trajectory,
not an observed raw-to-live input or action mismatch. At decision2 (the third
decision), both fixed recurrent controls choose **Right** where the blue teacher
calls for **Left**. Their teacher-forced and live seven-action probabilities
match exactly until this divergence. No model was restored, retrained or executed.

| Existing control at64paired updates | P(Left), target | P(Right), chosen | Target minus best other | Native blue outcome |
| --- | ---: | ---: | ---: | --- |
| Raw148 recurrent |0.2643829882|0.4620306492|−0.1976476610|7steps, return−1, closed |
| Public-history296 recurrent |0.2429322451|0.4322894216|−0.1893571764|7steps, return−1, closed |

Seven probabilities, in original Left/Right/Forward/Pickup/Drop/Toggle/Done order:

```
raw:     [0.2643829882, 0.4620306492, 0.1165192574, 0.1194128171, 0.0068001789, 0.0251858365, 0.0056683295]
history: [0.2429322451, 0.4322894216, 0.1211344451, 0.1675563753, 0.0068280152, 0.0237134360, 0.0055461735]
```

[Full-precision readout and source pointers](../../artifacts/runs/recurrent-first-disagreement-20261024/readout.json).
Raw pointer: `/evaluations/1/episodes/0/predictions/2`; history pointer:
`/report/evaluations/1/episodes/0/predictions/2`. Teacher pointers end in
`/reference/training_sources/0/transitions/2` in their respective source files.

Both controls match the first two teacher actions and public observations through
this decision. Chosen action equals the saved first argmax and applied native
action; this is not a numerical tie. After divergence, teacher labels and raw
teacher comparisons are null: a later matching image cannot restore a recurrent
causal prefix. The recorded later wrong-key termination remains negative.
Trained red matches all9teacher actions and reaches its goal in9steps/return0.49375
for both controls. Cold0blue/red first choose Right rather than Forward, then
time out at16steps/return0. All eight selected episodes closed; neither cue nor
an unfavorable checkpoint is dropped.

## Supported interpretation and limits

The initial public pixels contain one cue ball with color code2forblue or0forred;
the initial vectors differ by L2distance0.8. Every recorded history feature is
exactly the actual initial148frame plus current148frame. Thus the history
condition has the cue available at the failing decision. Merely appending the
initial frame did not produce the correct blue branch under the frozen recipe.

Source inspection agrees with the contracts: the cue is removed before the first
native action, pair transport retains original TRAIN targets, episodes start
with zero hidden activation, and two episode losses are averaged before SGD.
No alignment or native-action defect is evidenced at the first failure. This
does not identify a gradient, saturation, optimization, memory or capacity cause,
or show that more updates, normalization, replay or circadian mechanisms help.

Both controls still have17/18teacher actions and only one of two native goals.
The simpler raw control retains better total CE improvement and fewer parameters.
History's slightly less negative single-decision margin establishes no task
advantage. Raw remains an **incomplete full run**, with separately validated0/64
endpoints and incomplete/unknown later32readout work/final preservation. History's
declared two-endpoint case is complete. This diagnosis completes neither old
protocol, retries no case, and does not equate their compute or information.

## Validation and next hypothesis

Three stdlib source-read passes and a detached self-review completed with0models/
gradients/optimizer/teacher/physics work. Primary-reader peak RSS34,086,912B;
charged parse/readout time1.1706s of30s includes a conservative1s derived-summary
print whose RSS is unknown. No retroactive all-process memory claim. The third
referenced archive was not opened; TRAIN references are already embedded.
[Commands, costs, failures, skips and self-review](../../artifacts/runs/recurrent-first-disagreement-20261024/RESULTS.md).
New review is self-review; original endpoint acceptance was independently recorded
under its historical protocols. No new native replication or scientific benefit.

Next conventional hypothesis: **public-initial-cue specialization** with two
existing recurrent heads selected only by the first observed cue, retaining each
head's raw148/H8/K7 recurrence, CE and SGD. This explicit public inductive bias is
already useful in affine control; it is not a demonstrated repair or explanation
of shared-head failure. Capacity, initialization, update schedule and costs differ
and must stay explicit. Preserve old controls/seeds/doses/metrics; qualify routing,
full owned state and genuine episode activation through existing ports before
freezing one new engineering case. No hidden cue metadata, teacher fallback,
action mask, final/probe selection or failed-case retry. Full recurrent RL,
retention/return, heldout reliability, scientific and release goals remain open.
