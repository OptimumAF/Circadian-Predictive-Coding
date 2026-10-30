# R2.1-CTX.LS.K.A — learn once and acknowledge the correct return

[Issue297](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/297)
owns live status. Full goal and parents18/21/23/26/33 remain open.296 useful
periodic serving and its private return-state regression were reused, not rerun.
295negative wait policy and287FAILED/SPENT UNKNOWN remain unchanged.

## Delivered and observed behavior

The normal same_context_joint LS workflow now optionally accepts
--adaptive-serving through the existing drift/current-candidate ACK schedule.
One joint9 TRAIN fit learned the missing legacy-v1 task while retaining the old
tasks. The second joint packet and old-only return packet were verified correct
and acknowledged without further fitting. The two original windows, source
budget/sampler/guard and publication/rollback ports remain binding.

Final repaired-tree engineering case EXIT0:

| Packet | Observed/arrived | Actual consumption | Served source version |
|---|---|---|---|
| joint learning1,9rows |0/1|1fit/9rows|context-ls-12|
| joint learning2,9rows |1/2|ACK at2, no fit|context-ls-12|
| old-only return,6rows |12/13|ACK at13, no fit|c0|

All three packets consumed; actual update count1, ACK count2. Existing guard
accepted publication at tick2 after first-window consumption. Candidate origin
remained context-ls-12; the changed served version had its original verified
publication proof. No candidate rebasing, update credit renewal or old-source
authority restoration. Source12fits72rows plus child1fit9rows =13fits81rows.
The child ceiling remains3updates/60seconds/512MiB; unused credits remain unused.
The original source12update/30second/512MiB budget is unchanged and born once.

Both final private candidate and published actor passed ALL six complete
three-step known TRAIN task episodes. Exact source rollback restored5/6:

| Task | Final private candidate | Published actor | Source rollback |
|---|---:|---:|---:|
| v1 invoice |PASS|PASS|PASS|
| v1 password |PASS|PASS|PASS|
| v2 invoice |PASS|PASS|PASS|
| v2 parcel |PASS|PASS|PASS|
| v1 legacy |PASS|PASS|FAIL|
| v2 legacy |PASS|PASS|PASS|

The private candidate readout occurs only after stream/worker completion and
cannot select updates, ACKs or promotion. It uses the existing frozen LS TRAIN
task-readout port and64typed-attempt ceiling; failed readout work is retained.
Source/owned-source, NumPy/Python RNG and full rollback checks passed. Root actor
equality is post-rollback evidence, not an unchanged actor throughout serving.

Live worker24episodes completed20successfully: one initial legacy-v1 failure and
three permission refusals are retained. Three stale cases recovered. Candidate,
served actor, before-publication source and rollback outcomes are distinct.
Known public-context TRAIN templates do not establish held-out generalization,
long-run retention, population recovery duration, safety or circadian benefit.

## Source changes and preserved boundaries

Five production files changed: scripts/context_head_live.py,
scripts/recurring_ticket_workflow_example.py,scripts/run_bp_ticket_serving.py,
scripts/candidate_training_feedback.py,src/app/actor_shadow.py. Added
tests/test_context_same_context_ack.py; existing same-context refusal tests were
updated for the newly supported explicit adaptive option. No learner math,
labels, loss, rcond, margin, task metric, threshold, recipe, seed or dependency
changed. No new controller or orchestration framework.

Candidate feedback now accepts exact declared label clocks, matching original
source/arrival and payload inside the ACK lease. Default consecutive-clock
behavior is unchanged. Default ACK still requires base actor=current actor.
Post-publication ACK requires an explicit trusted original boundary guard before
and after proof, pinning actor model/configuration/metadata/generation/version.
Candidate source version stays fixed. A changed receipt/candidate/actor, early
label or failed proof refuses consumption and retains diagnostic cost.

Followup admits only the narrow drift+candidate+verifiedACK composition, retaining
FollowupTrainingPlan(2,12). Receipt completion counts fits+ACKs; native counters
and budgets count only actual fits. Serving leases can defer publication within
the original horizon. Publication pins paid ACK history as well as applied fit
receipts and original versions. One bounded full candidate snapshot is checked
before future ticks and inside ACK, advancing only with matching observed poll
receipts/native+budget deltas. There is no growing success-record chain.
Periodic296, paired4/12, default strict ACK and frozen catalog semantics remain.
Final/probe labels never enter fit, promotion, replay or controller selection.

## Preserved first run and prospective engineering amendment

Original source/protocol acceptance preceded the first launch. Focused follow-up
review findings arrived AFTER launch: boundary budget proof omitted ACK history,
base origin was pinned only within an ACK, and a new ACK-entry snapshot did not
prove continuity across the return gap. Original run completed EXIT0 with6/6
candidate/served tasks,13LS81rows,14.432494561seconds/62971904B. It is explicitly
PRE-STRENGTHENING evidence, not qualification of the later guards.

No source changed while it executed. Original native.py/native.attempted/
actual.json/native-work.json/PROTOCOL.md are unchanged. The changed source file
preimage, five-source hashes and original capture hashes are retained in stage.
Three new scalar mutation cases exercise ACK-history corruption during guard,
changed base origin and changed weights across return. Full final161case
Windows/Linux cohort and seven-file statics passed before the second launch.

The original once-only allowance remains SPENT. Before reviewer read4 or any
second model work, [amendment6508241554](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/297#issuecomment-6508241554)
and AMENDMENT.md permitted EXACTLYONE final-tree engineering qualification.
Measured14.43seconds/~60MiB left room under the SAME cumulative60seconds/512MiB/
stage8MiB. Only fit caps increased prospectively to28LS177rows28lstsq
(spent13/81 +new<=15/96), two invocations total. All other native caps remained.
The repair changed rejection checks, not learner inputs/configuration/endpoints.
This is not an advantage study, independent scientific replication, refund or
revival of a failed scientific budget. No third case or further grant.

native-final.py uses distinct exclusive actual-final/native-final-work/
native-final.attempted files, authenticates original captures before/after work,
and adds all prior counts/time/peak. Both source budgets are born separately once
for fresh cases; the closed first source is never resumed or extended. Repaired
source/amended protocol independent review preceded final case. Final source
hashes match current production files. Scalar negative cases qualify refusal
paths; the final normal native case qualifies their integrated ordinary path.

## Costs and artifacts

Stage artifacts/runs/context-same-context-ack-20261029 retains both native
captures, original protocol/amendment, source preimage/hash manifests and all
pure receipts. Raw stage3295088B at closeout, below8MiB. Native costs:

| Quantity | Final case | Both paid cases |
|---|---:|---:|
| LS fits/optimized rows/lstsq |13/81/13|26/162/26|
| Teacher traces/derived steps |16/48|32/96|
| Task episodes incl source/worker/private/served/rollback |66|132|
| Sandbox dispatches |278|556|
| Outer typed/delegate calls |230/460|460/920|
| Typed affine calls/rows |194/194|388/388|
| Full probability calls/rows |14/84|28/168|
| Context routing calls/rows |243/311|486/622|
| Unfitted reads |36|72|
| Recorder body seconds |14.026231064|28.458725625|
| Sampled peak RSS bytes |62525440|62971904|

Final case diagnostics:3reads24rows (joint9,joint9,old6), separate from1fit9rows.
Acknowledged packet rows are diagnostic computation, not optimizer/replay row
exposures. Native counters include main and newly created Python worker threads
under a shared lock. Original controls/data were not retrained.0Torch/GPU/model
random draws/probe/final. API counters are not all-in FLOPs, SVD/copy/validation or
matched compute. ACK uses fewer native fits than periodic296, but adds diagnostic
and validation work; no overall efficiency or causal timing advantage is claimed.

The final receipt's cumulative_seconds28.45872013 was sampled~5.5microseconds
before its final seconds field. The conservative sum of both final body fields
is28.458725625 and is used here; postcheck enforces that later sum<=60. All
amended cumulative caps held. Boundary/post checks do not forcibly preempt an
in-flight native call or guarantee hard never-overrun peaks.

Exact native commands (physical D checkout, TEMP/TMP on D):

```text
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/context-same-context-ack-20261029/native.py
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/context-same-context-ack-20261029/native-final.py
```

Both invoke normal CLI --method linear_fit --context-return
--separate-context-heads --paired-serving same_context_joint --adaptive-serving.
Do not rerun spent recorders or overwrite outputs/markers.

## Focused checks, failures, review and skips

Final161PASS/0FAIL/0SKIP per platform. Seven selected files include new real
scalar runtime/lease/two-window tests, same-context/adaptive dispatch, ordinary
ACK defaults, guard-boundary pinning, incident compatibility and catalog-study
defaults. Eight explicitly named native/raw-backed cases were deselected to avoid
duplicating their paid model work. Pure native fit/forward/teacher/task counters
were zero; Torch absent. Literal predictions and scalar learner updates are
fixture work, not task usefulness.

Windows final body6.355343seconds/92811264B,55observed literal helper calls55rows/
45scalar updates45rows. Linux final45.684191602seconds/79302656B, same observed
counts. Across all pure invocations: Windows4calls22.5372062000024seconds/
92811264B,120observed helpers/91scalar updates; Linux2calls92.447901701seconds/
79306752B,96helpers/87updates. These observer helper counts cover selected paths,
not every fixture operation. Inherited scalar catalog fitting/literal monitoring
is separately executed and not covered by those counters: each completed study
cohort has fourteen completed scalar rows/two table fits and six monitoring
calls24rows (two completed cohorts per platform in this scope).

Preserved Windows failures: win1 andwin2 each1PASS/1fixtureFAIL,
2.1344876seconds/70500352B and6.9259407seconds/88731648B, each12literal helper calls/
2scalar updates. Initially guessed contention ticks; inspection then showed a
context-bound fixture lease ended before the publication check. The correction
models a real asynchronous lease spanning the poll, releases it next opportunity,
and verifies deferred publication. No production guard was weakened. Production
mypy annotation errors, one fixture append-return annotation and unused import
were fixed. One failed documentation patch anchor was reread/corrected.

Exact final pure commands:

```text
.venv/Scripts/python.exe -B artifacts/runs/context-same-context-ack-20261029/check.py artifacts/runs/context-same-context-ack-20261029/win-4-work.json tests/test_context_same_context_ack.py tests/test_context_linear_same_context.py tests/test_context_linear_adaptive_serving.py tests/test_training_acknowledgement.py tests/test_context_learning_boundary.py tests/test_context_incident_serving.py tests/test_context_linear_catalog_study.py -k "not execute_one_full_adaptive and not settle_recurring and not retain_failed_proof and not deploy_paired_learning and not keep_early_publication" --basetemp=D:/Dev/_Temp/circadian-sameack-win4
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/context-same-context-ack-20261029/check.py artifacts/runs/context-same-context-ack-20261029/linux-2-work.json --capture=sys tests/test_context_same_context_ack.py tests/test_context_linear_same_context.py tests/test_context_linear_adaptive_serving.py tests/test_training_acknowledgement.py tests/test_context_learning_boundary.py tests/test_context_incident_serving.py tests/test_context_linear_catalog_study.py -k "not execute_one_full_adaptive and not settle_recurring and not retain_failed_proof and not deploy_paired_learning and not keep_early_publication" --basetemp=/mnt/d/Dev/_Temp/circadian-sameack-linux2
```

Seven-file static commands use production files listed above plus both changed
test files. Exact scope/commands in PowerShell:

```powershell
$checked = @('scripts/candidate_training_feedback.py', 'scripts/context_head_live.py', 'scripts/run_bp_ticket_serving.py', 'scripts/recurring_ticket_workflow_example.py', 'src/app/actor_shadow.py', 'tests/test_context_same_context_ack.py', 'tests/test_context_linear_same_context.py')
.venv/Scripts/python.exe -B -m ruff check @checked
.venv/Scripts/python.exe -B -m ruff format --check @checked
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent @checked
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent --platform linux @checked
```

All pass; final static body10.428629800000635seconds, RSS
uninstrumented. Earlier passing pre-strengthening cohort158/platform remains
distinct from final161. Lead self-reviewed source/procedure/final JSON task and
receipt vectors/source hashes/all costs; no production edits followed final gates.

Finite independent review found and checked the three integration gaps. Original
four-read cap is retained; the prospective amendment allowed exactly five reads
under same120seconds/max2agents. Repaired source/protocol accepted in read4 before
final native work, visible final/cumulative results accepted in read5. Total
readonly command wall~1.18seconds; no imports/tests/models/network/writes. The
exception ended. This is scoped independent review, not scientific replication.

Skipped broader unchanged native/math/raw-backed/baseline suites, current wheel/
clean clone, Windows native duplicate, remoteCI/GPU/physics/release and comparative
BP/PC/CPC reruns outside scope.279wheel predates current changes. No installs,
downloads, global changes, deletion, commit or push. Inherited dirty work and all
earlier failures, scientific negatives and SPENT outcomes remain preserved.

## Full-goal reassessment and precise next action

This increment answers the known-task case: a current candidate can acquire the
missing task, retain old tasks and consume a returning old context without
destroying that learning, while its served snapshot and exact rollback remain
separate. Existing R2.1 private diagnostics/probes already exist284/285. No further
checkpoint, count, ACK-proof or nearly identical numeric sequel is the next
default. Held-out stream/task generalization, fair comparison, broader learners,
long-duration operation and release readiness remain open.

The existing context_stream_comparison already has reference modes and withheld
recipes, but its source/fit reuse is bound to historical Perceptron artifacts.
Exact next implementation action: add an explicit owned-LS route through that
EXISTING comparison runner, reusing the original reference controls and declared
withheld recipes while keeping validation outcomes unavailable until candidates
finish. Reconcile reuse assumptions first; do not reload a saved fit as current
LS authority or rebuild a runner. Freeze a small local validation protocol before
any comparative model work, with all source/diagnostic/ACK/worker costs. Preserve
the conventional learner, thresholds, metrics and every existing control result;
do not tune baselines or select seeds to manufacture a circadian win.
