# CE post-repair original-clock qualification: FAILED/SPENT

R2.1-TW.CE.QUAL / #348. One genuine repeated-data model workload after accepted
#346/#347 validation-cost repairs. **The native qualification failed.** Those
component improvements did not establish complete ordinary CE execution under
the original continuous source30s clock. No retry, continuation, budget renewal,
clock extension, new seed, changed horizon/metric or algorithm tuning occurred.
#344 remains separately FAILED/SPENT with all original costs/artifacts unchanged.
This is post-repair software qualification, not an independent scientific trial.

## Observed behavior and missing outcomes

| Qualified closed-loop task phase | Old /4 | Fresh /2 |
| --- | ---: | ---: |
| Before private probe | 4 | 0 |
| Completed fresh16 prefix | 4 | 2 |
| After complete return16 | unknown | unknown |

Twelve complete role-qualified task episodes (six before/six after fresh) are
retained with all actions/transitions, zero task-time training/external actions.
The six after-return task episodes were not executed. The source, deployed actor
and both reported NumPy/Torch RNG flags are true; overall/private complete=false.
No observed old task was lost at the fresh endpoint, so old recovery after
forgetting cannot be credited. Full return/recovery stays unverified.

Fresh phase completed16/16 applications. Return completed7/16; the eighth
return callback was admitted and refused before hidden/optimizer work, with
known zero kernel counts and applied=false in its failed receipt. The previous
seven successful return applications remain permanently charged. Totals:
24attempted applications/96current rows,23returned+verified applications,
zero private native SGD. Source12SGD/72rows remains unchanged.

Important negative partial evidence: at the fixed return4 classification
readout, BOTH fresh tasks fell from accuracy1.0 to2/3, with CE9.210340372 and
8.738769538. Old TRAIN accuracy stayed1.0. These are saved classification rows,
not executed after-return workflow outcomes, and do not establish a terminal
task success/failure rate. Return8/16 readouts are absent. No fresh-label replay
bank or use of final labels was added to repair this result.

All fixed available curves remain in analysis.json/native.json. Fresh F0 is2/3
accuracy per task; F4/8/16 is1.0. Old TRAIN is1.0 throughout available F0/4/8/16
and return0/4. Direction and denominator did not change; high CE/partial fresh
retention loss is not hidden by the valid F16 six-task result.

## Failure boundary and complete paid-cost accounting

Native error: `ToyExecutionStopped: ... max_wall_seconds after 12 wake updates
(30.041s); no resumable checkpoint`. Original source budget started4.543911573,
last clock34.584960876, original source12update/30s/512MiB limits intact. Private
fresh and return budgets both retain16updates/30s/512MiB and their original
births8.937148904/8.937150223. No clock reset or allowance transfer.

Sole invocation host EXIT1/36.394283100s; native32.299492443s, sampled RSS
350711808B. Outer capture `budget_verified=true` means only its60s/768MiB/
per-counter caps held. It does NOT pass the failed original app clock or make
the case complete. RSS is sampled, not OS-enforced hard memory admission.

| Paid gradient phase | Fits | Actual backward calls | Gradient-row attempts |
| --- | ---: | ---: | ---: |
| Source SGD | 12 | 12 | 72 |
| Auxiliary CE initialization | 1 | 34 | 408 |
| Fresh CE updates | 16 | 136 | 2040 |
| Completed return CE fits | 7 | 52 | 936 |
| Total new qualification | 36 optimization calls | 234 | **3456** |

The CE optimizer is LBFGS; its24steps/222backwards must not be called24SGD
updates. Each backward is charged by actual fit rows, including retained rows.
Private role allocation:408probe_gradient_rows +312current_train_gradient_rows
+2256retained_train_gradient_rows=2976; plus initialization408=3384CE rows,
plus source72=3456. Completed F16 prefix incl source/init costs2520gradient rows.
All24completed fit loss trajectories,222losses and final coefficients/readouts
are retained. No interrupted/unobserved CE closure loss is imputed in this run.
The last eighth-return receipt has zero kernel work because it failed before
hidden/optimizer construction; it still has admitted attempt/wall costs.

Observer totals:57components/114named seeded draws2565values, source12SGD/
backwards72rows;24CEoptimizers/steps/returns,222CEforwards+backwards3384rows,
24hidden and24finalCEreads378rows each;213native head forwards357rows,
177typed calls,36workflow episode entries/249dispatches. No private SGD,
affine/LS/SVD/autograd.grad/random-mutation/PC latent/physics/GPU/remote/final
work. These named counts are not complete FLOPs, allocator/copy traffic or
all-in historical cost. New work is genuine model computation despite its
engineering-qualification label; no charge is removed because the run failed.

Previous #3443186gradient-row attempts and failed clock remain permanent,
separate charges. The two workloads together consumed6642gradient-row attempts;
this excludes older controls, engineering fixtures and all other historical
work, so it is not an all-in development-cost claim. More completed return
applications at the same exhausted clock is not a matched-work latency study.

## Reuse, procedure and exact commands

Saved-only analysis compared all12 available common source/checkpoint/data/
training-work/limits/initialization/logit/before-task/F0/auxiliary/private-limit
fields with #344 BEFORE exposing its historical prefix task outcomes. All12
equal. This supports reuse of that known prefix, not complete-case acceptance
or byte identity of unsaved feature matrices. No archived model was restored,
no controls retrained and no retired owner/clock resumed. Existing unchanged
#341/#342/#343/#346/#347 math/composition/preservation evidence was reused.

The ordinary CLI uses original BP9/3/6seed17/lr.03/momentum.9 source12/72,
TRAIN-only CE initialization,16fresh/48probe_train rows and16planned return/
96TRAIN rows, original12TRAIN anchors per fit. Same fixed float32LBFGS settings,
six known templates, CE/accuracy checkpoints0/4/8/16. Probe_test labels remain
metrics-only, no final-label/training/promotion/tuning path. These data were
previously observed; no generalization, CPC advantage, independent replication,
matched-compute, causal bundle or release claim follows.

Physical cwd D:/Dev/Repositories/Circadian, process TEMP/TMP=D:/Dev/_Temp.
Existing producer Python3.12.3/NumPy1.26.4/Torch2.14CPU, threads1, WSL
TEMP/TMP/TMPDIR=/mnt/d/Dev/_Temp, offline/local. Stage
artifacts/runs/ce-postrepair-qualification-20261030/:

```text
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/ce-postrepair-qualification-20261030/preflight.py
.venv/Scripts/python.exe -B artifacts/runs/ce-postrepair-qualification-20261030/launch.py
.venv/Scripts/python.exe -B artifacts/runs/ce-postrepair-qualification-20261030/analyze.py
```

launch.py's recorded sole command uses WSL env as above plus
`timeout --signal=KILL 60s .../python -B .../observe.py .../native-work.json`,
host subprocess timeout75s. Its actual.py calls existing ordinary main with
`--context-return --method bp --reserved-probe --probe-initialization ce
--probe-update ce_rehearsal --output NEWSTAGE/native.json`. No new runner.
Exclusive LAUNCH_SPENT/NATIVE_SPENT precede launch/imports; both retained.
Raw native.json/native-work.json/native.log/shell.json/analysis.json and saved
cost-consistency checks retained. Native observer timing includes imports/model
work through its final sample/report preparation, excludes final serialization;
external host includes WSL startup/stdio. All original app clocks remain separate.

Five artifact scripts passed scoped Ruffformat/check/Linux-targetmypy,
13.2814589s/scopedcache46403829B<64MiB. Non-numerical installed-metadata/AST
preflight EXIT0/.064796027s/24489984B (external3.0881373s). PRE source review
prompted stricter saved case_complete qualification for all18tasks plus32event/
144row/33fit/loss consistency checks before launch; two scoped analyzer statics
passes(.5731286/.55764s) follow that change. No new model fixtures or baseline
suites repeated. Final full five-script statics and saved metadata gates recorded
before handoff; all static/preparation work below90s. Initial learned64MiB cache
allowance is prospective; old32MiB cache failure from #344 remains unchanged.

No production/default/algorithm/data/seed/metric/clock changes this increment,
no install/dependency/download/delete/global settings/commit/push. Broad/clean-
clone/release and unchanged mathematical/baseline suites skipped; this actual
case qualifies the changed runtime path, and failed, rather than substituting
unchanged broad suites for useful behavior. Stage cap4MiB retained.

A saved-summary PowerShell projection failed before writing because array
division parsed differently than intended (tool shell EXIT0 despite the reported
error). Its0.3427229s/zero-model failure is retained; the simpler observed-value
projection passed0.1938053s. No model or native observation was repeated for it.

## Review, unfinished criteria and critical path

PRE independent review ALLOW SOLE INVOCATION,4read commands0.2956273s;
lead source/procedure/result self-review distinct. POST ACCEPT of the saved
FAILED/SPENT capture, accounting and qualified partial interpretation; native
qualification is NOT accepted. Eight PRE+POST reads,1.1181125s measured read
bodies, exact command/exit/timing receipts in PRE_REVIEW.md/REVIEW.md. No
imports/tests/models/numerical replication/network/children; finite exception
ended. Product-delivery next-action inspection remains lead evidence, outside
that native result review. Full native completion
criterion remains unchecked; failure is not promoted to software acceptance by
independent review of its evidence. #348 remains OPEN/FAILED-SPENT/no retry.
Broader reliable adaptation/R2/R4/R5/R7/matched science/release/full goal open.

The fixed-clock CE case is not a complete usable path. Do not automatically
launch another repair microprobe or another model retry. Reassessment reuses the
already accepted conventional minimum-change paired workflow (#314): it has
known-template acquisition/retention through return, while #312's legacy negative
result remains. Its protocol differs; do not rank comparative efficiency or
claim recovery after forgetting from that case.

Delivery inspection confirms the existing module entry point is packaged through
`scripts`; a new runner/console command is unnecessary. The installed setup guide
`docs/native-ticket-setup.md` still says paired serving is perceptron-only,
contradicting the accepted linear_fit paired command in #314. The actual command
is `--method linear_fit --context-return --separate-context-heads --paired-serving
joint --train-update minimum_change`. One ancillary discovery command also named
an absent guessed guide file (EXIT1); it performed no model work and was not retried.

Next concrete product action: correct the installed CPU setup guide for that
existing paired minimum-change command, then freeze a small offline wheel/
retained-trajectory/fresh-install qualification through the existing module
entry point. Verify existing installed/replay coverage before duplicating it;
implement only the missing R4.8 delivery behavior under a new bounded software
protocol. This need is in product delivery, not another learner or CE microprobe.
Reuse accepted task evidence; no new algorithm, favorable task selection, matched
scientific claim, continuation of spent CE budgets or extra native cohort is
authorized by this failed result. This moves toward a usable conventional demo
instead of treating more CE capture infrastructure as the full repository goal.

Evidence receipts for R2.1-TW.CE.QUAL: [canonical FAILED/SPENT handoff](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/348#issuecomment-6538180635); [parent #18](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/18#issuecomment-6538185397), [parent #21](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/21#issuecomment-6538185612), [parent #23](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/23#issuecomment-6538185844), [parent #31](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/31#issuecomment-6538186018).
Machine-readable handoff: artifacts/runs/ce-postrepair-qualification-20261030/handoff.json. Native completion stays unchecked; issue OPEN.
