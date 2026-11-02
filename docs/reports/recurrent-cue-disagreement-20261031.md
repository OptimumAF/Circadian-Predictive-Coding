# Two-head recurrent acquisition error — #375 / R4.2-RCS.D

[Canonical scoped completion and handoff](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/375#issuecomment-6548284876).
375closed; the full repository goal remains active.

Both saved A32 failures start with a learned **Forward** choice on the original
teacher trajectory, where blue requires **Left** and red requires **Right**.
At that third decision (index2), teacher-forced and live probabilities match
exactly. The saved public observations and earlier applied actions match the
teacher prefix. No input/temporal/action-application defect is evidenced at
this first error. No learner or simulator was executed to reach this finding.

| A32 cue | P(required turn) | P(Forward), chosen/applied | Target minus best other | Saved native outcome |
|---|---:|---:|---:|---|
| Blue / Left |.1255059838294983|.6153771877288818|-.48987120389938354|16steps/0return/truncated/closed|
| Red / Right |.1205846443772316|.6936737298965454|-.5730890855193138|16steps/0return/truncated/closed|

Full seven probabilities in Left/Right/Forward/Pickup/Drop/Toggle/Done order:

```text
Blue [.1255059838294983,.058179620653390884,.6153771877288818,.0679636225104332,.034201111644506454,.0728144645690918,.025957945734262466]
Red  [.03401907905936241,.1205846443772316,.6936737298965454,.03408467769622803,.02965732105076313,.06333302706480026,.024647574871778488]
```

Both teacher-forced traces disagree with their teacher at indices2/3/4/6.
Only comparisons on the original causal prefix carry labels or teacher/live
probability differences in this diagnosis. After the first wrong action, all
teacher labels and comparisons are null, even if a later image happens to
match a teacher-index image. Later native actions are still preserved as facts.
Applied action is the saved first argmax over all seven actions, without masks,
teacher fallback, new games or a changed metric. The first wrong choices are
not ties.

Pointers into original native-result.json:
`/evaluations/1/episodes/0/predictions/2` and
`/evaluations/1/episodes/1/predictions/2`; matching teacher-forced records are
`/evaluations/1/scores/0/probabilities/2` and
`/evaluations/1/scores/1/probabilities/2`.
Teachers in the packaged TRAIN JSON are `/training_sources/0/transitions/2`
and `/training_sources/1/transitions/2`. All first-error observations, full
probabilities and32live-step facts are retained in stage readout.json.

## Scope and scientific limits

This is the accepted132two-head raw148/H8/K7 public-cue recurring case, not
129's older raw/history shared-head control. The saved32checkpoint was read
after all128original fits under that protocol. This is not a new trained model,
rescoring pass, current-run native observation or recovery of a lost endpoint.
Original132full status/final preservation and selected checkpoint/game
state-preservation facts are true in the accepted saved report. The original
case had single-agent self-review, not independent numerical replication.

132remains mixed: final C-blue/red and A-red reach their goals; A-blue and both
B games fail. Native A64/A96 were unmeasured. Do not describe all recurrent
behavior as failing, or claim acquired-A retention/recovery when A32fails.
The simpler accepted373cold joint control solves its fixed8/8tasks; no reason
to replace it or add minimum-change replay follows from this diagnosis.

374's separate tiny step trace establishes nonzero initial gradients/updates,
not32trained health or a cause here. Forward dominance suggests investigating
the learning objective/readout/representation, but does not prove class imbalance
as a cause, dead features, capacity shortage, saturation or harmful momentum.
Existing class-balanced TRAIN138/stream139 already implement that alternative;
140's original A32native games also failed and its later readouts remain
FAILED-SPENT. Do not add the same weighting mechanism again, tune a dose/seed,
recover140/143weights, or infer that more updates/replay/CPC will solve this.
No final/probe labels, promotion, controller selection or generalization claim.

## Execution, costs and review

Read authority and the original two-pass verifier were inspected before parsing.
Its spent2passes/32s stay spent. This new semantic first-error question has a
separate prospective <=2passes/30s/128MiB/8MiBaggregate allowance, frozen in
issue375body/start6548156967 and stage PROTOCOL.md. It does not rerun its score/
cost reconciliation or renew the native case. Exactly two declared inputs:

- artifacts/runs/recurrent-context-stream-20261025/native-result.json,
  1487883B/SHA2560b76d534e37b4c13aee5bd61bfdd48477e7ff82d20b8ace65c9aeb7ca1725763.
- examples/minigrid_affine_train/episodes.json,
  94705B/SHA2565a368cfa8d4d7326d5d3ab4d6663a0e8e35e1e660968a68212e627ba9203d508.

ONE original pass completed, exit0/errornull: body.045117100002244115s,
actual-self peak26984448B,1582588inputbytes,50argmaxrows (18teacher+32live).
No NumPy/Torch/model/learner/simulator imports or numerical/model/forward/
gradient/SGD/random/teacher/physics work. The existing RSS measurement helper
is expressly allowed. No failed parser or budget amendment. Full source
parameter matrices were neither exported nor reconstructed. Physical D:
HEADd9119e8/inherited dirty checkout and all prior evidence remain untouched.

Exact command, TEMP/TMP D:\Dev\_Temp, physical repository cwd:

```text
.venv\Scripts\python.exe -B artifacts/runs/recurrent-cue-disagreement-20261031/diagnose.py first
```

Host receipt and all observations/costs are in stage
`artifacts/runs/recurrent-cue-disagreement-20261031/`.
Procedure reuses the essential existing first-disagreement causal-prefix loop,
adapted to fixed132schema. No product module changed; no new parser framework
or artificial production interface. Focused tests/statics/model suites are
skipped because no production behavior changed; source/procedure/result
self-review checked prefix irreversibility, fixed cue order, all seven
probabilities, role/clock/pointer/outcome scope and actual costs. Independent
corrected-evidence interpretation review ACCEPT: three read-only calls/.4130089s,
no original-source parsing, models/tests or numerical replication. Scope/failure/
correction are in stage REVIEW.md/timing ledger; finite exception ended.

## Derived pointer correction

Independent review first required a suffix-provenance correction: the original
derived output retained teacher pointers off-prefix, including14nonexistent
indices9–15. The original readout-before-pointer-fix.json is preserved. Corrected
producer code emits null teacher pointers off-prefix; the existing derived
readout was corrected without re-opening either original input or recomputing
argmax/first-error facts. All26off-prefix pointers are now null. Every other
trace field and each complete first-disagreement record is equal to its preimage.

Exact correction command:
`.venv\Scripts\python.exe -B artifacts/runs/recurrent-cue-disagreement-20261031/correct_pointers.py`.
It completed in.03771880001295358s/21180416B,29131derived bytes read,0original
input bytes/argmax/model work. Total measured producer+correction body
.0828359000151977s remains within original30s; failed review and correction costs
are retained. No metrics/budgets/acceptance criteria were weakened. A no-op
mixed documentation patch anchor failed before editing; the smaller known
source patch then succeeded. Observed producer/correction command outer times
.2400629s/.1708463s total.4109092s<60, transcribed from exec chunks70eef9/13d08d
in host-timing-scope.json; producer shell separately measured.1374475s. These
are two command timings, not an all-session time claim. Final corrected-evidence
review ACCEPT: three read-only calls/.4130089s, no original-source parse,
models/tests or numerical replication. The subsequent host-timing transcription
is lead evidence outside that independent review; no review allowance is reopened.

## Next useful dependency

Tracked as planned [376](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/376),
with no numerical allowance yet. Before changing hidden learning, use the EXISTING recurrent activation reader
and categorical linear-fit core to test whether a fixed readout can separate
the original A teacher actions on a fresh frozen recurrent representation.
The BP readout diagnostic currently binds a feedforward owner, whereas the
activation reader already supports native recurrent snapshots. Direct existing
APIs may suffice; do not build another learner/runner/codec just for the check.
A small separately frozen TRAIN-only feasibility control can distinguish a
readout-optimization limitation from an observed representational limitation
for that frozen source. It cannot diagnose unavailable132trained features or
prove native success from known-row fit. No new numerical allowance here;
freeze data/seed/fits/feature-forward/solve/memory/time costs before any model.
If feasible, choose the smallest explicit readout-learning composition through
existing ports, with native task qualification only after its correctness gates.
Keep all ordinary meanCE/weighted/default/control results and negative findings.
Full acquisition/retention/recovery/neuralRL/heldout/scientific/release goal open.
