# #383 Stage A: joint frozen-recurrent TRAIN readout

The fixed joint least-squares control does NOT fit all original TRAIN actions:
41/54 correct, despite both designs having full column rank9. This is a valid
negative learnability result, not a reason to tune seeds, metrics or baselines
to force a win. It does not prove that every readout objective or representation
is incapable, or that native games would fail. No native games were run.

| TRAIN context | Blue correct | Red correct |
|---|---:|---:|
|A|7/9|8/9|
|B|6/9|7/9|
|C_train|7/9|6/9|
|All|20/27|21/27|

| Cue | Rank/columns | Squared error | Coefficient L2 | Maximum absolute coefficient | Minimum target margin |
|---|---:|---:|---:|---:|---:|
|Blue|9/9|8.936691244240174|4.040829017728622|1.9162073484477495|-.43290739754119745|
|Red|9/9|10.833833560796531|6.436441377899759|3.6787963036650724|-.49212582782506975|

All54 logits/target/prediction/margin rows, all18singular values and all6context
groups are retained in native-result.json.41positive/13negative margins, zero
ties. Full column rank is not interpolation of27one-hot rows or a proof of
classification separability. Previous accepted376A-only18/18 interpolation is
reused without rerunning; this joint pool is different data/exposure/computation,
not matched compute or causal forgetting. Current380/381sequential game outcomes
and all10old-retention failures remain unchanged; no CPC/retention/generalization/
reliability/efficiency/release benefit established.

## Why a diagnostic before another owner

Previous goal turn PROGRESS382 (recorded inspector software/export, visual gate
open). Goal objective/current AGENTS/entrypoint/tracker18/383/latestcomments,
handoff/plans and relevant actual source were read. Current readout snapshot
deliberately limits each fit to<=16rows and failedwork to one bounded episode.
A27row joint fit cannot be placed there by silently widening that contract.
Existing recurrent replay is nativeSGD, not this readout control. Existing376
public activation+core LS APIs suffice for a disposable joint diagnosis; no
production edit/module quota, new math or orchestration framework was needed.

Stage A therefore precedes online/rehearsal expansion. It is COMPLETE executed
diagnosis, with final independent result acceptance. Original383online joint
state/late-label/consent/retained-bank requirements remain DEFERRED/UNCHECKED;
offline upfront TRAIN is not online adaptation or a substitute for those gates.
Current owner/stream/evaluator/policy/ordinarydefaults unchanged. Existing simple
joint-affine native fresh/old successes and R2.1/matched controls reused.

Original6A/B/C_train blue/red9stepTRAIN episodes explicitly released upfront,
read ONCE94705B under original SHA
5a368cfa8d4d7326d5d3ab4d6663a0e8e35e1e660968a68212e627ba9203d508.
Original typed_sources reuses complete TRAIN/raw148/K7/context/ID/public-cue
admission. Unreleased/foreign role/partial/reordered input refused before model
work. Fresh disposable fit uses each original row once, not another learner's
replay history/consent. No heldout/probe/final labels, old captures or models.

One original PublicCueRecurrentLearner/two cold H8 heads, private seed17 native
policy unchanged; real StableActor fork never predicts. Per cue, the existing
activation reader receives A/B/C episodes in three SEPARATE9row calls, each
zero-reset by accepted recurrence code. Only returned hidden features/labels
are concatenated into27x8; raw episodes never joined across hidden-state reset.
Two existing fixed RCOND1e-12 minimum-norm one-hot LS fits append their usual
intercept, followed by two existing core TRAIN logit products/all7firstargmax.
No optimizer/native SGD/hidden representation update or model capacity change.

Original source/actor state+policy owners/parameters/gradients/momentum/counters
and Torch/NumPy/Python RNG guarded around every feature/fit/readout boundary.
Original clockbirth/budget and exact expected paid fit/replay counters checked
before AND after native/clock/snapshot/record callbacks. All returned arrays,
including logits, immediately pinned by exactdtype/shape/bytes; earlier values
rechecked before more work. Existing paid_report_guard seals at most2success
roots/containers/values; no unlimited record chain/new framework. Fit charge
is permanent BEFORE call; failures would retain attempted/returned counters and
SPENT marker. No restoration, promotion or silent counter forgiveness.

## Executed work and limits

PhysicalD:/Dev/Repositories/Circadian, HEADd9119e80926a139a2f38460564d7eab1b6d2a2b0
plus preserved inherited dirty/untracked source; TEMP/TMP/TMPDIR D:/Dev/_Temp.
All code/artifacts in new artifacts/runs/recurrent-readout-joint-20261031/.
Frozen PROTOCOL/comment6550486176; source/static/fixture gates and independent
PRE6550578072 precede ONE native case, no retry or cap amendment.

    & ./artifacts/runs/recurrent-readout-joint-20261031/run-step.ps1 -Step checks
    & ./artifacts/runs/recurrent-readout-joint-20261031/run-step.ps1 -Step checks -Attempt final
    & ./artifacts/runs/recurrent-readout-joint-20261031/run-step.ps1 -Step checks -Attempt combined
    & ./artifacts/runs/recurrent-readout-joint-20261031/run-step.ps1 -Step native

| Process | Outcome | Body seconds | Observed RSS bytes | Host seconds/PID |
|---|---|---:|---:|---|
|Initial literal composition|15PASS|1.6440626|75,071,488|1.7780138 /53780|
|Counter/logit/source fixes|20PASS|1.4835673|75,792,384|1.6069455 /69420|
|Final combined tree|20PASS|1.5102546|75,231,232|1.6402002 /5816|
|ONE genuine jointTRAIN diagnosis|complete/negative|2.8330434|319,787,008|3.210454 /67320|

Every same owned hidden process terminal0/no host expiry.55passing fixture
executions/20distinct/cumulative4.6378845s, all native/Torch/solver/sourcepayload
work0 in fixtures. They check grouping/separate9rowreads/release/role/complete
geometry/old-input mutation/expected charges/returned-logit signedzero mutation;
scripted reader isn't native numerical replication. Existing activation/LS math
evidence reused. No test failures/errors/skips. Third/original last fixture slot
used for final guard ordering, all within20distinct/60executions/60s512MiB.

Initial5type annotations/private-interface diagnostics3.3518472s retained; fixed
.7269176s, combined.7195791s, final.6711641s allPASS. Totalstatics5.469508s<90s.
Ruff/format4scripts, mypy3composition/check scripts, all4AST/PowerShell parse0.
Dynamic observer instrumenter typecheck skipped (accepted376thin adaptation);
syntax/lint/source/procedure review used. One preparation patch-anchor mismatch
was corrected by exact source read before any fixture/model execution.

Native work ALL attempted calls returned, caps held:
-2native constructors,6private randn calls2608values plus30derivedzero biases,
  2optimizer constructions. Native source0updates/0rows, actor never predicted.
-6zero-reset activation calls54rows,54single-row native forwards including
  discarded original output projections;2LS/design/LAPACK54rows;2TRAINlogit
  readouts54rows. Derived2postfit products54rows/2prediction products54rows,
  not separately measured kernels. No old baseline or sequential rerun.
-114native snapshot-port calls,21Tensorclones10332scalars,1TRAINbundle read,
  0physicsimports. Native train/backward/SGD/autograd/predict/teacher/probe/final/
  replay/promotion/physics0. Named calls aren't all copies/FLOPs/allocations/IO.
-24bounded pin-pairs149472B two-sided numeric reference payload, not deduplicated
  memory/Python/native state/fullreport size. No unlimited retainedsuccesschain.
-One original2fit60s512MiB budget birth242948.1215254→242949.1766191,
  1.0550937s,2charged LS attempts/0replay. Observer2.8330434s/319787008B includes
  imports/capture under90s512MiB; host3.210454s<105s. No speed/efficiency comparison.
-Source/actor/TorchNumPyPythonRNG preservation alltrue, complete report/errornull.
  Python3.14.7/NumPy2.4.6/Torch2.14.0+cpu. NewJSON9608B/work1868B/log1436B/
  stderr0. Stage38files17211592B before final report/review, below64MiB including
  localcache32MiB. No limit revision or forgotten failures.

## Review, unfinished work and next action

Independent source/procedure/actualPOST ACCEPT4reads/.5143820s resolved logits/
charge pins and post-callback ordering; finite exceptionENDED/no carryover.
Reviewer guessed one nonexistent static receipt name, source/fixture read
succeeded; actual statics-final-tree.json reconciled at POST with all0.
No independent numeric execution by peer. Lead source/
procedure/result selfreview separate; final archival prose may be selfreviewed.

Original383online joint-owner/state/replay eligibility/consent/delayed-source
integration remains unfinished and is deferred after this negative prerequisite;
do not mark it completed by the offline diagnosis. Native joint gameplay unknown,
not zero or failure. Same-viewer visual382/Linux378/heldout/recurrentRL/science/
release/fullgoal remainOPEN. No current native task advantage claimed.

Next unblocked #384PLANNED/no model/fixture allowance: reconcile existing native
recurrent CE complete-episode joint-TRAIN baseline (targeted source has sequential
A32/B32/C32/A32 only). Inspect existing AttributionStream/factory/evaluator and
freeze smallest modelZERO composition checks for fixed original128update1152row
joint reference IF supported. Same seed/architecture/lr/momentum/CE; explicit
TRAIN upfront schedule, no raw episode concatenation, future/final/probe access
or original budget renewal. A differently trained conventional control is not
matched compute/exposure to LS or affine and must not be mislabeled as old study.
Do not add online LS rehearsal/controller/CPC machinery from this failedfit.

Skipped native games, newonline owner/stream features, oldcohort refits, fullsuite,
currentLinux/WSL/newwheel/install/browser/GPU/CI/final/probe studies. No dependency/
download/globalchange/cleanup/push/delete or other repository work. All original
378/96FAILED-SPENT, other negatives/unknowns/consents/clocks/costs retained.
