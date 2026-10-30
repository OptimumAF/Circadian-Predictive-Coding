# R7-CTX.LS.A — adaptive LS serving consumes TRAIN by fits or verified ACK

[Issue #289](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/289).
Full goal and parents18/21/23/24/26/33 remain OPEN. Previous goal turn PROGRESS:
288qualified the fixed LS serving path. Reuse that evidence; do not rerun its
fixed control or287's FAILED/SPENT dynamic-retention case. No old scientific
raw/model reconstruction occurred here.

## Decision and working behavior

Inspection reconciled two existing observer paths. ArrivedShiftObserver is
retrospective after-update observation, and its original frozen pilot missed
every recurring change. ArrivedCandidateShift observes before wake, but uses
the fixed detector policy without a calibrated live stationary prefix, bounded
to8batches48rows. Merely exposing it on a short six-packet recipe would not
establish deployable detection. Those negative results and thresholds remain
unchanged; full R2.7 stays open.

Instead, this increment connects the **existing** arrival-only candidate-error
and correct-ACK baseline to the working owned LS route. No new control law,
optimizer, runner, calibration fit or threshold tuning. Fixed drift threshold.1,
FIFO, candidate-training diagnostic and acknowledge_ready verification are the
existing implementations. This is a justified reorder toward useful adaptation,
not a claim that shift detection is solved.

```text
scripts/recurring_ticket_workflow_example.py  explicit adaptive API/CLI validation/dispatch
scripts/context_head_live.py                 owned LS drift/ACK composition and receipt report
tests/test_context_linear_adaptive_serving.py pure gates plus actual adaptive CLI case
```

[Usage](../context-linear-adaptive-serving.md). Explicit adaptive mode is allowed
only with ordinary owned LS context-return/separate-heads/paired-serving. Default
Perceptron and fixed-periodic LS paths preserve their arguments and guards. No
private/replay/catalog/other-feedback/arrived-shift mixture. Exact paired recipe
is required even on internal owned transport before state copying/data access.

Source12fits72rows, head8/48+4/24, schema, original lifetime clock/budget/sampler,
report/receipt owners and caller guard are unchanged. The child observes only
already-arrived FIFO-front TRAIN labels and detached current candidate state.
Correct ACKs use original label receipts/revision clocks, do not fit or increment
native/wake counts, and cannot renew budget. All original receipts must drain
before publication; unchanged inner guards decide publication. Task outcomes
are collected afterwards and never feed triggering/fitting/guard selection.

## Executed integration — both Windows and Linux

One full **actual CLI** joint adaptive integration/platform passes, with no retry.
Fresh source12LS72rows, child **2LS18rows**, four verified correct ACKs, all6TRAIN
packets consumed. Child wake limit remains6; actual charged wake count2 and four
unused credits remain unused. `training_complete=True` is explicitly scoped to
complete fit+ACK consumption, not six fits. Native candidate14updates90rows equals
12source+2child fits; no fabricated/inferred native updates from ACK.

The static TRAIN stream remains9+9+9+9+6+6rows,1tick label delay. Only the first
two9row packets fit: each includes3new+6old TRAIN rows. These18optimized rows
include12old rows of supervised rehearsal; `replay_examples=0` denotes no separate
replay-budget operation, not an absence of old-row reuse. Four ACKs consume the
remaining30TRAINrows (6new+24old) without optimization. The full observed stream
is48rows (12new+36old).

Diagnostics actually run **7times/57rows**, not6/48. Packet1 is observed at both
ticks2and3 while its fitting admission is deferred; its original label arrived
at2. Native/wake clocks remain13/1 for both reads. Every extra read is charged;
it is not an unobserved update or a free repeated score. Read indices0,1,1,2,3,4,5;
mistakes1,1,1,0,0,0,0. The existing.1threshold admits each1/9error; unchanged.
ACKs for packets2..5 record original observed/arrived times and actual wake2.
All diagnostics precede publication and contain ordinary TRAIN roles only.

At tick7 the unchanged guard accepts: new accuracy2/3→1, old1→1, full record
2316B Windows/2336B Linux under4096. Guard prediction maxima.0011654s/
.001014224s under the existing.25s bound; observations do not establish a broad
latency guarantee. All six published known workflows complete in3steps.
Live24opportunities/78typed worker requests complete, observing generations0/1.
Those are typed requests, not78closed-loop environment episodes. Candidate and
published model are14/90; rollback generation2 restores exact source12/72.
Four old workflows still succeed after rollback and both new head/task pairings
fail. Source/actor guards and original source charges/RNG remain unchanged.
No old source task was lost, so source return recovery is null, not a new gain.

This demonstrates this supervised adaptation control on known TRAIN templates
with public context cues and explicit old TRAIN reuse. No unseen-task discovery,
unlabeled drift capability, long-stream retention, CPC advantage or release.

## Qualification, fixes and review

27distinct checks/platform qualified through25pure+1new-refusal+1full-CLI affected
checks, not one final27case cohort.25new cases and2selected existing pure default
CLI cases. All pass,0failure/error/skip. Initial pure checks execute0fits/teachers/
rollouts. The sole full case pays all work below. Default fixed LS and Perceptron
argument dispatch checks pass; accepted288unchanged source/adapter assumptions
are reused, not numerically reconstructed.

Independent source/state/isolation review3readonly calls found two pre-run gaps:
adaptive disabled the previous published-workflow gate, and missing paired recipe
could broaden internal transport to default data. Fixed published readout/tick
gates after complete TRAIN consumption, and added strict no-recipe refusal before
copy/producer. Repair re-read accepted. No imports/models/tests/old raw/writes in
review. Lead actual source/procedure/results self-review is separate. Final
visible-result review is bounded to1readonly call60s, not replication/rescoring.
No pytest/static failure, budget increase, case retry, threshold or seed selection.

## Complete measured and reconciled costs

Stage `artifacts/runs/context-linear-adaptive-serving-20261029/` holds SCOPE.md,
check.py, Win/Linux1..3work receipts and win32/linux integration JSON. Stage
2,882,220B at initial closeout, under8MiB; no overwritten capture.

| All engineering/platform, incl pure checks | Windows | Linux |
|---|---:|---:|
| pytest invocations |3|3|
| observer pytest-body seconds |12.3223674|27.580250917|
| sampled process peak RSS bytes |78,000,128|85,188,608|
| LS attempts / physical solves / optimized rows |14 /14 /90|14 /14 /90|
| original TRAIN teacher traces / steps |16 /48|16 /48|
| closed-loop workflow episodes / sandbox dispatches incl teachers |36 /182|36 /182|
| candidate diagnostics calls / rows |7 /57|7 /57|
| diagnostic body seconds |.1317655|.139582664|
| main-thread typed delegates / affine rows |268 /98|268 /98|
| worker typed delegates / affine rows |156 /78|156 /78|
| all LS outer+selected delegates / affine rows |424 /176|424 /176|
| unfitted typed reads |36|36|
| full-class prediction calls / rows (diagnostics+guards) |21 /141|21 /141|
| main-thread homogeneous route calls / rows |148 /224|148 /224|
| worker homogeneous route calls / rows |78 /78|78 /78|
| all homogeneous route calls / rows |226 /302|226 /302|

The current-thread sys.setprofile observer does not include worker calls. Owned
serving-native receipts114returns include36main published/rollback reads plus
78worker requests. Add78affines/routes and156outer+selected delegates to the
immutable observer totals. All fits/full-class diagnostics/guards/sandbox effects
occur on the main thread. Worker requests do not dispatch the environment.
Source-inspected projection rows add141guard/diagnostic head rows to302homogeneous
rows =>443total; source/guard models are fitted, so no extra uniform full-class
guard rows. This is derived row accounting, not full projection/FLOP instrumentation.
All original time/RSS/fit/teacher/dispatch/reconciled read/storage limits held.

Diagnostic reads/copied state,57forward rows, readout114requests/5472nominal
affine terms/1218token visits, buffers/guards/state evidence/serialization and
Python allocations add work.768coefficient bytes are not full-record/process
memory. Original source12/72 is not free initialization. All28fits/180rows across
the two platform integrations count; they are platform engineering qualification,
not two independent scientific repetitions.

Windows2worker intervals intersect model calls; Linux1. No forced rendezvous,
parallel CPU, speedup, deterministic scheduling or exact snapshot parity claim.
Worker and model floats/times differ by platform; successful task outcomes agree.
The observer body excludes launch and final observer output; sampled RSS is not
an exact peak proof. Linux initially yielded a confirmed live process, then the
same handle completed; no restart after an observation yield.

Prior288fixed case is a descriptive implementation reference only: different
schedule/optimization/diagnostic work, no matched-compute or efficiency study.
No fixed arm retraining or287dynamic-comparison retry. Shift/calibration failures
and all scientific negative/SPENT evidence remain unchanged.0Torch/neural/GPU/
randommodeldraw/probe/final/oldscientificraw/newcomparativecohort work.

## Commands and skipped checks

Physical cwd D:/Dev/Repositories/Circadian; Windows TEMP/TMP=D:/Dev/_Temp;
Linux TEMP/TMP/TMPDIR=/mnt/d/Dev/_Temp, Python-B. Existing Windows3.14.7 and Linux
3.12.3/NumPy1.26.4 environments. Exact invocation argv/results in work receipts.
Initial selection: new test file plus existing pure CLI default forwarding,
`-k "not full_adaptive"`; next selection only the new no-paired-recipe refusal.

Final full actual CLI check, exit0/1PASS each:

```text
.venv/Scripts/python.exe -B artifacts/runs/context-linear-adaptive-serving-20261029/check.py artifacts/runs/context-linear-adaptive-serving-20261029/win-3-work.json tests/test_context_linear_adaptive_serving.py::test_should_execute_one_full_adaptive_ls_cli_with_real_fit_and_ack_receipts --basetemp=D:/Dev/_Temp/circadian-ls-adaptive-win3
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/context-linear-adaptive-serving-20261029/check.py artifacts/runs/context-linear-adaptive-serving-20261029/linux-3-work.json --capture=sys tests/test_context_linear_adaptive_serving.py::test_should_execute_one_full_adaptive_ls_cli_with_real_fit_and_ack_receipts --basetemp=/mnt/d/Dev/_Temp/circadian-ls-adaptive-linux3
```

Three-file `python -B -m ruff check`, `ruff format --check`,
`mypy --explicit-package-bases --follow-imports=silent` (default and--platform
linux) pass. Final whitespace check `git diff --check`; inherited CRLF advisories
retained. No new dependency/install/download/delete/global change/commit/push.
HEAD0aae1d6 and inherited dirty270–288/user changes preserved.

Skipped fixed288model rerun, current recipe full adaptive integration, broad
unchanged math/native/BP-PC-CPC baselines, old-raw-backed suites, full repository/
clean clone/current wheel/remote CI/GPU/physics/release checks. Original source/
binding evidence reused where unchanged; scoped checks are not repository readiness.

## Open requirements and precise next action

Working arrival-only supervised adaptation is delivered; calibrated unlabeled
or predictive shift detection, robust recurring recovery, broader R2.1/R2.7,
CPC applicability, generalization, persistent deployment and release remain open.
287comparison stays UNKNOWN/SPENT. No full-goal completion claim.

Exact next action: inspect CalibratedShiftDetector's prefix/model-identity
contract against ArrivedCandidateShift and add pure tests for a declared
stationary arrived-TRAIN calibration prefix before introducing that optional
binding. Keep model/candidate changes, prefix costs, score reachability and
onset/continued/clear semantics explicit; preserve old misses/false alerts and
keep task readouts/final data out of calibration/selection. Do not tune a threshold
or rerun a spent study. Any new model/diagnostic comparison needs its own small
frozen protocol after correctness gates; this closeout grants no allowance.

GitHub remains live tracking; plans/log receive requested archival pointers only.


Final independent visible-result review accepted the scoped interpretation and
receipt fields, fit/ACK separation, paid repeated diagnostic and worker-cost
reconciliation. One readonly call; no execution/writes. This is not independent
replication or episode-by-episode rescoring. No blocking correction; exception
ended. Lead combined source/procedure/results self-review remains separately named.
