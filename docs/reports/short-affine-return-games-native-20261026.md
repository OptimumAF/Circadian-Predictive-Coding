# Native task acquisition, forgetting and return

Issue [224](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/224),
R4.2-A4.R.N. The one frozen ordinary native case completed. The private learner
acquired F and recovered old A3/A4 tasks when they returned, but forgot the
other task set each time. This is a negative joint-retention result.

| Same private learner | F goals | Old A3/A4 goals | Private game steps |
| --- | ---: | ---: | ---: |
| After two F fits:6updates/72rows |2/2|0/4|82|
| After two old-TRAIN return fits:8/108 |0/2|4/4|68|

Each success took9steps and returned.49375; each failure reached16steps with
return0. All12private episodes completed and closed. Games used the fixed order
Fblue/Fred probe_test, original A3blue/red TRAIN, A4blue/red TRAIN; seed17,
max16, public296/history encoding and raw K7 argmax. No outcome selected the
dose, seed, labels, replay, controller, promotion or probe. These are fixed known
teacher-supported tasks, not evidence of held-out-context generalization.

The original fresh source learned A4 while retaining A3: source goals2/0→2/2,
then the logical third readout reused the same admitted4/54 state. Its source
and actor remained unchanged during private work. The private fit diagnostics
matched the earlier scoped observation: old TRAIN36→34→31/36 during F, then
31→33→36/36 on return; F correct-equivalent14→16→18/18, then18→16→14/18.
Fit recovery now has corresponding native old-task recovery evidence; F loss
is an actual task failure, not just a weaker fit score.

## Procedure and costs

Ordinary command, executed exactly once:

```text
.venv/Scripts/python.exe -B artifacts/runs/short-affine-return-games-native-20261026/execute.py
```

The existing222 finite producer and checked parent were copied into the new
stage. Changes opt into223's native games and add an explicit scalar task/cost
projection. No product, solver, learner, dataset, stream or controller changed.
The new fixed12readouts have a distinct prospective allowance; this did not
retry217/220, renew any prior budget or refit a comparison control.

Child and checked parent exit0. Body51.5843563001s, parent52.0820911999s;
sampled RSS98,336,768bytes from before optional imports through completed raw
and scalar serialization/writes and sampler exit. Only the final<=16KiB work
receipt/console follows the RSS scope, inside parent time. No instantaneous,
whole-process or process-tree peak claim.

| Native work, all attempts = returns | Observed |
| --- | ---: |
| Empty owners / snapshots |19 /10325|
| Fits / training rows |8 /108|
| LAPACK solves / rows |8 /108|
| Predictions / rows |266 /650|
| Environments / closes / physics steps |26 /26 /290|
| Packaged TRAIN reads / bytes |1 /94705|

Source fits4/54 include36current and18prior rows. F fits2/18 and old-TRAIN
return fits2/36 remain separate; all36return rows are old-task learning.
Original four owners retain limits0/0/150,6/None/90 with actual replay0,
F2/0/90,return2/0/90, all512MiB. Starts, bindings, clocks, charges, source/actor,
global RNG and restored native ports all passed preservation checks. The return
owner began before F; no owner was renewed. No random model initialization,
consolidation replay, final access or benchmark/deployed updates occurred.

After-F game call:17.7117246999s,82predictions/steps,4541snapshots. After-return
game call:21.3545756000s,68predictions/steps,3827snapshots. Each used6environments
and6closes,0fits/solves/source reads. Cold readout constructors outside these
call spans remain included in the global totals. The total helper diagnostic
42.5471339000s includes both game phases and return; F budget elapsed42.4813209s
also spans that work and is not isolated F-learning time. The heavy snapshot
and preservation costs are included, not claimed as efficient computation.
Returned projection arrays account for195488+162112numeric bytes; these are
not total allocations or retained Python/native storage. Return-only RSS and
learning-time isolation remain unavailable.

Frozen caps150sbody/180sparent/512MiB,8fits108rows,256owners40000snapshots,
512predictions2048rows,30environments/closes480physics and1TRAINread94705B
all passed. Raw1865287bytes is strict finite compact UTF8LF, exclusive writeonce
and NEVER reopened. Emitted scalar24638bytes, no observations, teacher traces,
parameter/prediction arrays or labels; tiny final work/parent receipts retain
costs. Result postflight read only those emitted records, qualifications and
source hashes, with raw-file metadata only.

## Verification and interpretation

Reused223's51software cases/platform after all6pins matched, and fourteen
unchanged222core/learner/solver/simulator/budget/sampler/stream/resource pins.
Four new pure scalar projection/cost checks passed twice per Windows/cached
Linux platform: positive zero-goal, partial/unknown prefix, payload exclusion,
fixed caps/no marker. No models, labels, physics, optional imports or native
markers in those checks. Windows bodies.0053110002s/parents.7716328998s;
Linux bodies.008237188s/parents8.7344562002s. No unchanged baseline reruns.

Four-file Ruff format/check/default+Linux-platform mypy pass. Static13.0217586s
includes initial one elapsed-field typing diagnostic and formatting; one atomic
patch-context miss made no changes. Both are preserved in gate receipts, fixed
without ignores or weakened conditions. No native execution failure or retry.
The negative task result is retained. Source/procedure/result SELF-REVIEW,
not independent acceptance. The scalar-only postflight passed in.0193845s
internal/.3542734s observed wrapper time, checking25source pins, all original
owners, role/order/geometry/closure/costs, caps, payload exclusion and footprints.

Local evidence: `artifacts/runs/short-affine-return-games-native-20261026/`:
frozen `PROTOCOL.md`, `qualification.json`, `case-changes.patch`,
`gate-receipts.json`, projection/parent receipts, `native-work.json`,
`native-parent.json`, emitted `terminal-summary.json` and `result-review.json`.
Exact postflight command/output is in `result-review-command.json`. The raw
artifact's in-memory hash and byte count are recorded without reparsing it.
No full repository suite, release/install/wheel/browser/CI/GPU/agent/sweep or
new neural/CPC comparison was run. Scoped checks do not establish readiness.

Earlier217failed/spent,220unqualified capture/projection and222accepted fit
tradeoff remain unchanged and separately charged. Their native totals plus
this case are22fits288rows56environments588physics,4TRAINreads378820bytes;
bodies73.7451750000s/parents75.6478796000s. This is historical cost accounting,
not pooled replication or a matched comparison. No old raw data was reread.

The learner replaces one cue head with each batch's fit; the observed mutual
forgetting gives a concrete reason to test existing joint rehearsal rather
than add a circadian mechanism or another capture/count increment. Next precise
implementation action: extend the existing equal-row private rehearsal
composition with an explicit retained-A3/A4 recipe:27F-only rows per cue versus
9F+18old-TRAIN rows, using its existing two arms, source/input witnesses, budgets
and stream. Preserve original ABC/8/180 controls and isolation, test the new
composition before freezing any native comparison. Equal rows/fit calls are
not equal computation. No new native allowance is granted by this report.

Full goal, parents21/23/33, reliable joint retention, generalization,
comparative/CPC/resource benefit and release requirements remain open. Live
issue/tracker handoff replaces duplicate historical plan/log status per AGENTS.
