# R2.1-CTX.LS.M.T.N: ordinary native case and legacy diagnosis

Scoped [#312](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/312),
following accepted #311 ordinary implementation. ONE native Linux case executed;
no native retry or production edit. Full goal/#21/#18/science/release stay open;
#270 closed scoped dependency. [Frozen protocol](../context-minimum-change-ordinary-native.md).

## Result: preserve the negative outcome

| Fixed ordinary readout | Saved #300 cold | Minimum-change |
|---|---:|---:|
| Known tasks /4 |4|4|
| Legacy requests /2 |2|1|
| v1 legacy |success|success|
| v2 legacy |success|wrong_workflow|
| Original live actor successes /24 |15|14|

Both native commands completed; allsix terminal tasks completed in3steps each,
with v2legacy committed to the wrong workflow in the new run. The raw legacy
delta is -1 and remains in the comparison. No threshold/tie-break/metric/seed/
task substitution, favorable-run selection or additional fit. This does not
establish an ordinary benefit from minimum-change; cold remains the default.
The positive #310 private transfer case is a different task/training protocol.

## What the saved failure actually shows

The failed request is `ready legacy policy_v2 feedback applied please`. Its returned
scores rank `inspect`0.3636363636363641 just above `defer`0.36363636363636387;
probabilities0.23709353474499778 versus0.23709353474499775. The next action commits
`inspect`, ending `wrong_workflow`. Saved cold scores instead rank `defer`
0.36363636363636437 above `inspect`0.3636363636363638, and commit `defer` successfully.
These are already-paid native scores, not recomputed predictions. The observed
winning gaps are about2.22e-16 and5.55e-16 respectively. They support numerical
sensitivity of this decision; they do not by themselves prove a learning defect.

Coverage explains why this is not demonstrated forgetting during retained learning:

- Stored source phases are v1invoice/password, v2invoice/parcel, then v1 return;
  `ticket_controller_streams.build_context_return_stream` selects exactly those
  teacher cases. There is no source v2legacy training phase.
- Stored child contexts are old-v1,new-v1,new-v1,old-v1,old-v1,new-v1. The actual
  `context_training_streams.build_context_stream` non-paired compiler validates
  new-v1 as legacy and old-v1 as invoice/password, all v1.
- The final v2head is exactly equal to its source checkpoint12 head, still4updates/
  24rows and rank4. It receives no child update. Its legacy-feature coefficients
  are small roundoff-scale values (up to about1.22e-15), not a learned legacy signal.

Thus v1legacy was taught and succeeded, while v2legacy is an untrained context-task
readout in this recipe. The actual failure occurs on the unchanged v2head; attributing
it to retained forgetting is unsupported. Its source-time response was not separately
executed for this legacy query, so unchanged-head attribution is a source/state
inference, not an extra observed before/after episode. No new model/encoder/forward
was run for diagnosis. All raw six-task outcomes and both denominators remain;
coverage annotations do not remove the negative task from the frozen metric.

Earlier #300's native legacy2/2 observation is preserved, but interpreting both as
newly acquired skills is too strong: its v2legacy also lacked TRAIN exposure and
its successful decision had the tiny recorded gap above. Both algorithms learned
the arrived v1 task in this case; neither readout proves reliable v2legacy learning
or unseen generalization. No after-the-fact tie policy was introduced to make either
model win. This scope correction is additive; frozen prior reports/results unchanged.

## Exact execution, cost and preservation

All commands physical `D:\Dev\Repositories\Circadian`, process TEMP/TMP
`D:\Dev\_Temp`; Linux `/mnt/d` equivalents/TMPDIR. Existing Linux3.12.3/NumPy1.26.4,
no install. HEADd9119e8 plus accepted dirty #308/#309/#311 source preserved.

```powershell
$env:TEMP='D:\Dev\_Temp'; $env:TMP='D:\Dev\_Temp'
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/context-minimum-change-ordinary-native-20261029/preflight.py
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/context-minimum-change-ordinary-native-20261029/native.py
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/context-minimum-change-ordinary-native-20261029/compare.py
```

The final compare command was invoked twice (one failure, one narrow repair), native
exactly once. Producer argv is the existing public CLI `--method linear_fit
--context-return --separate-context-heads --paired-serving retained_joint
--train-update minimum_change --output <new-stage>/actual.json`, exact argv in
native-work.json. Never rerun native.py to inspect saved data; exclusive
native.attempted remains. The new native budget is fully spent.

| Invocation | Outcome | Seconds | Peak RSS B | New model work |
|---|---|---:|---:|---|
| preflight |0,10synthetic metadata checks|0.106248038|23,572,480|0|
| native |0, root/child complete|3.249430676|64,151,552|23solves144rows|
| initial comparison |1, source.stream mismatch|unavailable|unavailable|0|
| corrected comparison/trace extraction |0,84exact checks|0.029008303|22,835,200|0|

First comparison erroneously compared duplicate nested source.stream receipt
diagnostic IDs/values, despite the frozen protocol explicitly allowing treatment
diagnostic differences and already filtering top-level receipt diagnostics. Preserve
`compare-initial.py` and `comparison-failure.json`/terminal exception. Failure
runtime/RSS were not written; do not invent them from tool latency. Synthetic
preflight changed only top-level fake diagnostics, so it missed the duplicate.

The repair applies the existing `receipt_metadata` projection only to that duplicate
list: every other stream field and receipt ID/actor/label/logical time/counter still
matches exactly. Selected diagnostic intent is separately validated for all23updates.
No outcome metric, data, resource, model or comparison direction changed; this fixes
an overly broad implementation of the predeclared comparison, not a new acceptance
waiver. Repair recorded prospectively6513775692 before the last allowed third
stdlib gate invocation. No native rerun or extra engineering/script budget granted.

Corrected84matches cover common initial unfitted state/tasks; source input metadata/
geometry/counters/receipt metadata; declared child features/targets/rowcounts,
reference/recipe/replay policy and allselected episodes/IDs/charges; task initial
states/limits and original owner limits. Post-training states are treatment outcomes,
not required equal. Full raw source arrays were not serialized; applicability also
assumes unchanged deterministic source/TRAIN builders and exact original teacher/
role/packet guards. This is not complete raw-array identity or independent replication.

Physical work: source12minimum-operation fits72rows +child6wakes27rows and5jointfits
45rows =23sharedsolves144rows23real NumPy. First two unfitted heads use cold core
2calls12rows; warm minimum core21calls132rows. Minimum contextual+generic ports
46calls288rows are delegation counts, cold operation ports0. Source/child/diagnostic/
validation/task work all included. Other native counts:158affine158rows,6probability
27rows,268routes506rows,194typed388delegated6predictports,54sandbox episodes269
dispatches,9direct teacher traces+4builders (old profiler counted4per builder,
25combined). No Torch/neural/GPU/gradient/optimizer/latent/modelrandom/probe/final
calls. Equal23fits144rows does not match all residual/prior arithmetic or copying;
new named observer differs from old profiler, so3.25s versus11.80s is not a speedup.

Original source30s age2.343746039, child60s age1.327843872, no reset/renewal. Final
candidate23/144; all12source/6wake/5joint returns known. Source/actor/RNG/owned-source
guards passed within each run; cross-arm trained actor weights differ as treatment,
consistent with live14/24 versus15/24. Action snapshot:24decisions,6wake admissions,
11native attempts/completions,5replay attempts/completions/commits,45permanent replay
charges, unknown flags false. Sharing snapshot:12paid reservations/max12, one
serving_active deferral, final idle. Paid reservation count is not returned-fit count.
No publication or probe state moved into deployed/benchmark weights.

All native23/144/60s512MiB caps held. Three original stdlib preflight/comparison
invocations used, including failed comparison, <=30s64MiB measured on successful
ones; failed comparison lacks a process resource receipt. No cap amendment. Stage
10files1,396,187B including raw actual1,347,972B, below8MiB. Sampling/terminal resource
assertions do not establish asynchronous termination or exact process-tree peaks.
Read-only JSON/source inspection and report writing perform no fits/readouts.
No install/download/delete/global change/Git commit/push; unrelated dirty/ignored/
frozen data and #285/#287FAILED-SPENT/#300/#303/#305/#306/#310/#311 outcomes preserved.

## Review and next useful implementation

Unchanged #311 focused Windows/Linux source/state/role/intent/failure checks and
scoped Ruff/format/both-target mypy reused. New producer AST checked by preflight.
No paid fixture/native control/simulator/neural/physics/CI/current wheel/clean-clone/
release suite repeated. Whitespace/link checks apply to new docs only; no global
readiness claim. Lead reviewed actual procedure, repaired projection, all budgets/
scalars/trajectories and source exposure alignment. Independent prelaunch review
GO2readonlycalls; third read accepted actual counters/negative trace/projection
repair and actual action/sharing permanent-charge fields. Fourth final read accepted
report/source-coverage/unchanged-head inference and comparative interpretation.
Total4readonlycommands0.5083038s,0project imports/tests/models/network/writes; finite
exception ended. Independent scoped review is not experimental replication, native
benefit, reliable adaptation, generalization, CPC or release/fullgoal acceptance.

Do not tune a tie-break or add another legacy capture. The useful next route already
exists: ordinary paired TRAIN uses explicit v1parcel/v2password prototypes and
`PAIRED_TASKS` matching those taught tasks. Reuse #270's accepted Perceptron paired
training/publication evidence; do not rerun it or follow its superseded next action.
The LS minimum-change selector currently admits only source-only or single-head
retained recipe; the paired joint/publication path exists but rejects this selector.

Next: extend the explicit selector through the existing ordinary paired-joint
TRAIN/publication path, using both arrived-head prototypes and its corresponding
PAIRED_TASKS, preserving cold defaults, exact original guards/roles/counters/leases/
publication/rollback and no probe-state ingress. Start scoped admission/fork/causal
learning/guard-refusal tests under a new software budget; inspect existing paired
source first, add no runner or recipe if present. Any later native qualification
requires a separately frozen genuinely changed algorithm protocol, not #287's spent
comparison/cohort retry. This targets taught-task adaptation and serving, rather
than interpreting an untrained legacy near-tie as forgetting or forcing a win.
Broader reliable adaptation/transfer/CPC/scientific/release/fullgoal remain open.
