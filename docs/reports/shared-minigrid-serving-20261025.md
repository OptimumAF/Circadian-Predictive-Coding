# R4.2 / R7.7 — shared game actor requests

[Issue181](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/181)
connects MiniGrid predictions to the existing sharing lease. A ready private
update deferred during a real request, committed once after release before
first action, and both known A games succeeded with unchanged deployed weights.
This is cooperative serving integration, not new-task learning, retention,
recovery, latency benefit, independent acceptance or release readiness.

## Implementation

Optional `shared_runtime` on existing bind/episode APIs requires an exact existing
ResourceSharedRuntime/ordinary TRAIN runtime/priority gate/same actor. New
`src/adapters/minigrid_shared_prediction.py` pins controls and stores start/latest
observations. Real shared predict calls real actor.predict; original full actor/
model/counter/policy/version/generation/config/cache/backend+slot pins and action
gate remain. No app/core/CLI/native math changes. Absent keyword retains old
direct route/report shape. [Usage](../promotable-minigrid.md#shared-prediction-requests).

Pins precede callbacks. Finite native controls and monotonic admission/deferral/
budget observations reject replacements, edits, backwards counters and foreign
roles. Legitimate private progress/pause stays visible, never reset/refunded.
Failure retains last valid metadata/readback uncertainty and paid prediction/
action/episode prefix; normal cleanup releases requests. No private weight read,
history chain, training/promotion/consent/restore authority or arbitrary Python
isolation. Caller boundary checks original consent/preceding paid work. Request
lease covers prediction only; no hard preemption/p95/physics-wide lease.

## Software verification

Stage `artifacts/runs/shared-minigrid-serving-20261025/` contains exact commands,
XML/stdout/counters/parents/statics/protocol/SELF-REVIEW/source/resource receipts.
**39distinct checks qualify/platform:**33new +6 explicitly selected ordinary
direct/publication/cleanup regressions. Actual runtime/gate/public shared/public
actor/literal loop execute; Event tests cover partial private mutation, ready
update deferral/subsequent charge. Capacity/error cleanup, pause/quota/resource,
owner/limits/budget/role/counter changes, first/post-callback mutation and default
behavior pass. No fixture failures/skips/foreign collection or genuine numeric/
physics/teacher/probe/final/historical-reader work.

| Cohort | Windows body / parent seconds | Linux body / parent seconds |
| --- | ---: | ---: |
|Initial36 |3.966808600 /5.810239500;36PASS |Final full cohort reused |
|Affected5 (3new +2repeat) |1.621457200 /2.893725900;5PASS |Final full cohort reused |
|Final39 |Unchanged passing evidence reused |7.268147434 /14.084798200;39PASS |

Cumulative Windows121empty metadata owners593native snapshots/2scripted train
18rows/28scripted predicts28rows/9literal episodes13steps. Linux115owners535
snapshots/2scripted train18rows/26predicts26rows/8episodes11steps. Slot upper
<30M is snapshot-record scope, not allocation/copy/equality traffic. Max sampled
body RSS97,542,144B/92,254,208B after imports, not process-tree/import peak.
Original cumulative caps pass. No real random initialization/fit/forward/physics.

Self-review found first-read controls/slot pinned after a gate callback. Moved
pins before that port and rechecked actor before its public snapshot; directed
and positive cases passed. Static unused-import and native-helper list annotation
failures retained/repaired before launch. Final7-file Ruff/format/default and
Linux-target mypy pass; production/tests unchanged semantically afterward.
SELF-REVIEW is not independent. Full suite/clean clone/installed wheel/CLI rerun/
other simulator/math/baseline/scientific/release checks skipped for this scope.

Exact executed commands include:

```console
.venv/Scripts/python.exe -B artifacts/runs/shared-minigrid-serving-20261025/checks.py windows windows-1
.venv/Scripts/python.exe -B artifacts/runs/shared-minigrid-serving-20261025/checks.py windows windows-final -k "pin_controls_before_first or both_real_public_ports or post_snapshot_control"
.venv/Scripts/python.exe -B artifacts/runs/shared-minigrid-serving-20261025/checks.py linux linux-final
.venv/Scripts/python.exe -B artifacts/runs/shared-minigrid-serving-20261025/checks.py statics statics-live
.venv/Scripts/python.exe -B artifacts/runs/shared-minigrid-serving-20261025/checks.py native native
```

## Native engineering result

ONE protocol after gates, existing Windows pins/CPU1. Strict bundled94705B/
SHA5a368cfa8d4d7326d5d3ab4d6663a0e8e35e1e660968a68212e627ba9203d508
read54rows: A18unique fitted, B18innerguard-only, C18unused. Total3fits27rows
includes9explicit retained-Ablue re-exposure, not novel feedback. No new teacher/
final/probe/replay buffer/sleep/SGD/random model init. Replay-like exposure is
counted, not hidden as zero replay cost.

Before launch, source review corrected an unsupported unexecuted draft driver:
AttributionStream pins an actor throughout an open stream. The case instead
uses existing live ActorShadowRuntime/max3experiences/consolidations0 from birth
and unchanged ordinary _record_tick at0/1/2 with train(max1), then same runtime/
clock/helper at3 after guarded publication. No partial-stream extraction,
counter bypass/resume/falsefinish or new production runner. Correction was
prospective protocol appendix/comment6408766664, final statics before launch;
data/order/guards/seed/metrics/quotas unchanged. No native retry/old allowance renewal.

Original3update90s512MiB native and0update0replay150s512MiB readout owners start
before data/models. Post-publication gate allows1request/1additional admission/
1poll once; original native ceiling3 remains. Shared public prediction observes
active request1; one real ready train poll returns serving_active/0updates/
0admissions with no fit. After original lease release, caller boundary makes
one real shared train poll: batch-2, observed2/arrived3/applied3, update3 before
first physics action. Original2receipt prefix and new receipt verified before
more work. Existing B guards/promotion fixed; actorc0/gen1 stays2fits18rows,
private state3fits27rows/basea0 cannot authorize a second promotion.

Both A3blue/red games succeed,9steps/.49375return each/closed. Gate0→1admission,
0→1serving_active deferral,0active at end; first game native progress2→3, second
3→3. Source/actor/candidate/paid evidence/original controls preserved; both global
RNGs unchanged after optional imports. This is cooperative work between requests,
not native preemption, parallel throughput or extra-wake task-learning benefit.

Costs:1source read/6empty owners1096model snapshots687gate snapshots/3fits27rows/
3LAPACK/22native predicts54rows INCLUDING4guard36rows/18publicactor calls18rows/
2resets18physicssteps2closes; every attempt returned. Body10.461999000s/
parent11.495118900s/peak75,034,624B after mandatory imports, including optional
runtime. Snapshot-record upper<12M excludes total copies/equality/allocations.
Preprocessing/guard/state/serialization costs included in time/RSS. Result and
resource sizes in final metadata. All original caps pass, one allowance spent
successfully, no retries or parameter selection.

## Remaining goal and next action

All scoped criteria complete; full R4.2/R7.7/#18/#23/#26/repository goal open.
New-task learning, learned-old-task retention, recovery, held-out performance,
matched compute, efficiency, CPC benefit and release remain separate. Prior
failed/inconclusive/negative evidence unchanged; existing R2.1 diagnostics/probes
and A4 collector/catalogue already work.

Next implementation: connect existing fresh A4 TRAIN collection and arrived-cue
catalogue to an explicit live published-affine continuation with the same
original runtime/sharing gate and prospectively sized lifetime budgets. Observe
private old-A3/new-A4 utility/cost while old actor serves; preserve stale-base
refusal for a second promotion. Reconcile existing lifetime/retired-ledger/
current-actor candidate facilities before further publication; no new checkpoint
audit or repeated A-only update. Historical plans/log frozen; live handoff181.
