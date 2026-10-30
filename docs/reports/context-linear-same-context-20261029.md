# R2.1-CTX.LS.K — same-context joint LS learning serves all six known tasks

[Issue296](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/296)
owns live status. Full goal and parents18/21/23/26/33 remain open. Previous295
was progress: its chooser selected waiting after updates learned legacy-v1 and
lost two old tasks. Its study, metrics and actions remain frozen.287FAILED/SPENT
dynamic-rehearsal comparison remains UNKNOWN; no retry or replacement claim.

## Delivered and executed behavior

The normal fresh LS workflow now accepts --paired-serving same_context_joint.
It uses the existing context_joint9_return recipe and original legacy tasks,
without the paired recipe's parcel-v1/password-v2 substitutions. The two learning
packets each contain legacy-v1 three rows plus old invoice/password-v1 six rows
in the SAME final LS fit. The return packet is the existing old-only six rows.
No new controller, runner, solver, labels, loss or replay law was introduced.

Existing FollowupTrainingPlan(2,12) replaces paired4/12 only in this explicit
owned LS mode. Actual observed/arrived/applied clocks:

| Packet | Rows | Observed | Arrived | Applied |
|---|---:|---:|---:|---:|
| joint-v1 learning1 |9|0|1|4|
| joint-v1 learning2 |9|1|2|6|
| old-v1 return |6|12|13|14|

The unchanged guard accepted learning and publication at tick6. Guard new-row
accuracy rose5/6→6/6; old-row accuracy stayed12/12. This is row accuracy, not
six-task success. Separately, published frozen actor generation1/c0/history14
passed ALL six complete three-step task episodes after the whole stream:

| Known TRAIN task | Published actor | Exact source rollback |
|---|---:|---:|
| v1 invoice |PASS|PASS|
| v1 password |PASS|PASS|
| v2 invoice |PASS|PASS|
| v2 parcel |PASS|PASS|
| v1 legacy |PASS|FAIL|
| v2 legacy |PASS|PASS|

Rollback generation2/context-ls-12/history12 equals the original complete source.
Root/source/owned-source/Python+NumPy RNG checks passed. Root actor equality is
post-rollback evidence; the actor changed during serving. Child has no
actor_unchanged field. All published task reads used history14, rollback reads12.

The live24opportunity trace completed19successful episodes. Before publication,
legacy-v1 failed at1 and5; it passed at11,13,21,23. Three stale cases recovered
(ticks4/12/20); three revoked cases refused permission (6/14/22), counted as
unsuccessful task episodes rather than silently excluded. No external actions.
These observations support useful learning and retained old tasks in the served
snapshot under this finite known-template protocol, not held-out generalization.

The return fit completed: private candidate15updates/96rows. It was NOT
republished, and no final-candidate task episodes were run. Saved JSON equality
shows both final candidate head policies/models exactly equal the original
source's predictive policies/models; update histories differ. This is a
saved-state observation, not an executed candidate forgetting readout. Do not
conflate the final candidate with the deployed actor, which still passed6/6.
The schedule does not attempt return publication after a successful first
publication. No return guard rejection or measured recovery duration is claimed.

## Implementation and isolation

Changed scripts/context_head_live.py and scripts/recurring_ticket_workflow_example.py;
added tests/test_context_linear_same_context.py and docs/context-linear-same-context.md.
API/CLI/control rejects non-LS and adaptive/incident/catalog/private combinations
before source/data/output access. Existing current/joint recipes, paired4/12,
Perceptron paths and default30second sources are unchanged. No new dependencies.

Owned source12/72 is prepared by the normal workflow, not loaded from saved
probe/final weights. Its original12update/30second/512MiB budget, sampler, guard,
birth clocks and source-progress checks stay binding until the child returns.
Child original3update/60second/512MiB budget is created once. Guards/prepare/commit,
candidate ownership, feedback invalidation, live worker and full rollback reuse
existing components. Final/probe labels never enter TRAIN/promotion/selection.
The existing stream's ordinary TRAIN packets account for all24child row
exposures: new6, repeated old12 during learning, old6 on return. No separate replay
executor or replay-budget charge is claimed. Source72rows are additionally paid.

Why this scope: the catalog core explicitly requires replay-before-wake. A new
joint arm would change that frozen contract. Existing same-context joint packets
were already available through historical Perceptron CLI265, but fresh owned LS
publication only admitted the different paired task family. This minimal owned
composition exercises the learner repair without weakening catalog assumptions
or manufacturing a new comparative result.

## Frozen native observation and costs

Stage artifacts/runs/context-linear-same-context-20261029 retains PROTOCOL.md,
native.py/native.attempted,actual.json,native-work.json and pure receipts. Protocol
and source review preceded model work. ONE Linux normal CLI invocation EXIT0,
no retry, no post-result tuning or extra model calls. Native measured costs:

| Quantity | Observed |
|---|---:|
| LS fits/optimized row exposures/lstsq |15/96/15|
| Source fits/rows |12/72|
| Child fits/rows |3/24|
| Teacher traces/derived steps |16/48|
| Task episodes including source and workers |60|
| Sandbox dispatches |260|
| Outer typed/delegate calls |212/424|
| Typed affine calls/rows |176/176|
| Full probability calls/rows |11/60|
| Context routing calls/rows |227/308|
| Unfitted reads |36|
| Recorder body seconds |12.579713255|
| Recorder sampled peak RSS bytes |61218816|
| Child body seconds/sampled peak bytes |6.123919105999999/58142720|
| Raw stage bytes at execution closeout |1455331|

All frozen15fits96rows/60seconds512MiB/stage8MiB/output4MiB and other bounds held.
0Torch/GPU/modelrandomdraw/probe/final. Shared locked native counters include main
and newly created Python worker threads. Counts are API work, not all-in FLOPs,
SVD/copies/validation/softmax overhead. Boundary/post checks do not guarantee
preemption of an in-flight native call or hard never-overrun memory limits.

Exact invocation, physical D checkout with TEMP/TMP on D:

```text
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/context-linear-same-context-20261029/native.py
```

Recorder invokes --method linear_fit --context-return --separate-context-heads
--paired-serving same_context_joint --output stage/actual.json. Attempt marker
and outputs are permanent; the allowance is spent. This is a conventional product
observation, not matched compute or BP/PC/neutral/active CPC comparison. No
circadian benefit or superior controller claim is supported.

## Focused verification and review

105PASS/0FAIL/0SKIP per Windows/Linux:20new pure API/CLI/dispatch/source lifetime/
causal-plan checks plus85existing catalog-study/live/incident checks. Actual
FollowupTrainingPlan preserves the no-arrival gap and tick12/13 return; exact
recipe/source/sampler forwarding and before-data refusals are covered. Pure
receipts report zero native fit/forward/teacher/task work and no Torch. Inherited
scripted study cases do execute scalar table fitting and literal prediction
ports; observer scripted counters do not cover those fixtures. Source-derived
fourteen completed scalar rows/two fits and six literal monitoring calls24rows
per cohort are not native model work or scientific qualification.

Windows body4.620740800004569seconds/83517440B; Linux33.311722968000005seconds/
80232448B. Exact commands (TEMP/TMP and pytest outputs on D):

```text
.venv/Scripts/python.exe -B artifacts/runs/context-linear-same-context-20261029/check.py artifacts/runs/context-linear-same-context-20261029/win-1-work.json tests/test_context_linear_same_context.py tests/test_context_linear_catalog_study.py tests/test_context_linear_live_catalog.py tests/test_context_incident_serving.py --basetemp=D:/Dev/_Temp/circadian-lssame-win1
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/context-linear-same-context-20261029/check.py artifacts/runs/context-linear-same-context-20261029/linux-1-work.json --capture=sys tests/test_context_linear_same_context.py tests/test_context_linear_catalog_study.py tests/test_context_linear_live_catalog.py tests/test_context_incident_serving.py --basetemp=/mnt/d/Dev/_Temp/circadian-lssame-linux1
.venv/Scripts/python.exe -B -m ruff check scripts/context_head_live.py scripts/recurring_ticket_workflow_example.py tests/test_context_linear_same_context.py
.venv/Scripts/python.exe -B -m ruff format --check scripts/context_head_live.py scripts/recurring_ticket_workflow_example.py tests/test_context_linear_same_context.py
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent scripts/context_head_live.py scripts/recurring_ticket_workflow_example.py tests/test_context_linear_same_context.py
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent --platform linux scripts/context_head_live.py scripts/recurring_ticket_workflow_example.py tests/test_context_linear_same_context.py
```

All statics pass, final body6.204866seconds; static RSS uninstrumented. Incorrect
field name learning_updates was corrected to existing first_batches before the
first checks; no observed pure/native failure in this scope. One source read used
a nonexistent filename; several oversized orientation outputs were truncated,
then simplified to bounded projections. No failed/spent budget was renewed.

Lead self-review reconciled source/guards/plans, all endpoint tasks and live
episodes, candidate-vs-deployed histories, costs, clocks and unchanged protocols.
Finite independent source/protocol and visible native-result review accepted;
four readonly calls, approximately0.77seconds command wall. Initial partial
source truncation was followed by bounded critical-region reads. Reviewer
independently saw published6/6, rollback5/6, source/RNG and all native caps, and
explicitly declined a final-candidate task claim. No reviewer imports/tests/
models/network/writes; exception ended. No independent scientific replication.

Skipped broader unchanged math/native/raw-backed suites, current wheel/clean
clone, Windows native duplicate, remote CI, GPU/physics/release and comparative
baseline reruns outside scope.279wheel predates current changes. No installs,
downloads, global changes, deletion, commit or push. Inherited dirty work,
295negative study,287UNKNOWN and all other historical failures remain intact.

## Remaining goal and precise next implementation action

Known-task useful adaptation now works in this fresh same-context LS path: the
served snapshot learns legacy-v1, retains four old tasks and survives the old
return while private candidate evolves separately. Reliable ongoing candidate
retention, novel tasks/contexts, held-out generalization, fair comparative cost,
BP/PC/CPC adoption and release readiness remain open. Existing R2.1 isolated
diagnostics/probes284/285 and adaptive error/ACK serving289 already exist.

Exact next implementation action: extend the existing arrived-error/current-
candidate ACK schedule to this same-context owned recipe while preserving its
two predeclared windows. Correct old-return packets should be acknowledged
without another destructive narrow fit. Reuse current guarded publication,
original budgets and diagnostic cost ledgers; preserve the periodic mode and
this observation. First add focused causal/ACK/retention tests, then freeze any
distinct native observation before work. Do not add a new controller, retry287,
rerun296, or choose thresholds/metrics from the successful endpoint.
