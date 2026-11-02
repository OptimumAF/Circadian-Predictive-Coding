# Native output-only BP probe: retention without acquisition

R2.1-BP.OUTPUT.N / issue337. Previous336 was progress: the explicit output-only
port passed55 focused checks per platform and independent source/evidence review.
This increment executes one prospectively frozen native variant. It does not
renew either spent323 full-update attempt or change the original task protocol.

## Observed behavior

The ordinary run completed with source, actor, NumPy RNG and Torch RNG unchanged.
The auxiliary initialization and private probe also completed. Output-only SGD
kept all four old tasks successful, but neither fresh task was acquired.

| Fixed typed tasks | Before F | After F16 | After old TRAIN return16 |
| --- | ---: | ---: | ---: |
| Old source-trained tasks | 4/4 | 4/4 | 4/4 |
| Fresh source-untaught tasks | 0/2 | 0/2 | 0/2 |

All18 task episodes completed in three applied actions. At every endpoint:
v1invoice uses open/route/commit; v1password and v2invoice use
open/inspect/commit; v2parcel uses open/defer/commit. Fresh v1parcel still uses
inspect and fresh v2password still uses defer, then commits the wrong workflow.
There was no observed old-task loss, so recovery is inapplicable, not successful.
Intermediate typed behavior and unseen generalization were not measured.

All fresh checkpoint accuracies remain2/3 for each task. Original CE curves:

| Phase / updates | v1parcel CE | v2password CE |
| --- | ---: | ---: |
| F0 | 7.780184903400713 | 0.3153515197823921 |
| F4 | 7.762105116112886 | 0.303540179942421 |
| F8 | 7.718148331655196 | 0.29314513850881835 |
| F16 / return0 | 7.584016174362033 | 0.27650456894262015 |
| return4 | 7.532249193890646 | 0.26760673287928277 |
| return8 | 7.498877449328429 | 0.26163209371444013 |
| return16 | 7.464528259502667 | 0.2547880942304904 |

Native F losses begin7.780185222625732 and0.3133639991283417, matching335's fixed
output-only first-step reference anchors. They end7.6217803955078125 and
0.2794506847858429 on alternating batches. There is no earlier thousands-scale
jump. Lower losses did not yield a correct READY decision for either fresh task.
Old TRAIN monitor accuracy stays1. Both monitor CEs increase during return:
F16 values0.004717851304043525/0.0017797566184879993 become
0.005571758756655645/0.0025386048817943973. Preserve this adverse change too.
Native return losses and every0/4/8/16 monitor/readout remain in native.json.
No metric clipping, learning rate, momentum, teacher, tie handling, action mask,
seed, horizon or outcome selection changed.

## Historical comparison and isolation

Before computing a historical task comparison, a saved-JSON-only check passed
all11 common-initial-state checks: all four complete source checkpoints
(including policy, parameters, SGD buffers, gradients and counters), training
data/source-training metadata, the entire CE initialization including coefficients
and closure history, CE training logits, all six raw pre-F episodes, fresh and
old-TRAIN zero checkpoints, and source/auxiliary/private budget limits.
Evidence: stage/common-initial-state.json. No model reconstruction or refit was
used to establish equality. All before-episode logits and transitions matched.

The accepted historical323 full-update result remains old4->0->0/fresh0->0->0;
its first failed capture remains FAILED/SPENT with its original null flags and
costs. The new fixed update scope avoids that observed forgetting under this
recipe while still failing fresh acquisition. This is a scoped conventional
diagnostic, not general BP/CPC superiority or a matched-compute experiment.
Diagnostic overhead differs, and failed versus successful task lengths change
forward/dispatch work. A shorter measured wall time is not an efficiency result.

The source is original D9/H3/K6 seed17/lr.03/momentum.9/inference2, prepared live
again through A4/B4/A4=12SGD72rows. One original old-TRAIN12 CE fit is paid again.
Its output24 values initialize only a detached descendant, retaining source
hidden state, momentum, gradients, policy and native counters. Private F16 uses
48probe_train rows; return16 uses96old TRAIN rows. Old monitors score only during
F; task observations never become training data. Final examples and replay0;
no probe_test/final labels enter training, replay, promotion or selection.
The private output-only learner and diagnostic records never become the source,
benchmark or deployed actor. Original guards, receipts and live clocks remain.

All32 update measurements completed. Hidden parameter delta norms are0 and the
stored hidden momentum norm stays0.33419998299231357, despite computed hidden
gradient norms ranging2.8860615522865887..1863.2232961480165. Output parameters
move on every step, delta norms0.008895036451846403..0.05593515146442672.
The hidden residual uses preserved stored momentum; it is descriptive, not a
counterfactual full-SGD direction or validation that hidden SGD occurred.
Bitwise preservation semantics are qualified by336's tiny native fixtures;
these scalar records do not serialize every private tensor. Full backward is
still performed and paid. No backward-work saving is claimed.

## Execution and costs

Stage: artifacts/runs/ticket-output-only-native-20261030/.
Protocol frozen in issue337 before model work; local PROTOCOL.md retains it.
New stage/report paths were absent before creation. No production source changed.
actual.py calls the existing ordinary main with only two flags added to323:
`--probe-output-only --probe-update-diagnostics`. Existing323 observe.py is
reused unchanged; no new learner, runner, loader, optimizer or hashing system.

All commands use physical D:/Dev/Repositories/Circadian and PowerShell
TEMP/TMP=D:/Dev/_Temp. WSL uses /mnt/d/Dev/Repositories/Circadian and
TEMP/TMP/TMPDIR=/mnt/d/Dev/_Temp. Existing producer Python3.12.3,
NumPy1.26.4/Torch2.14CPU, threads1. One invocation, EXIT0:

```text
.venv/Scripts/python.exe -B artifacts/runs/ticket-output-only-native-20261030/launch.py
```

Exact WSL argv and terminal status are in shell.json. Exclusive launch/native
SPENT markers were created before work; session67308 was polled twice to terminal
EXIT0 and never restarted. Native process16.624779692s/349442048B RSS;
host19.640147500002058s. Native60s768MiB/host75s limits held.

Original app clocks/limits were not reborn or extended:

| Session | Birth | Last clock | Updates | Wall / RSS limit |
| --- | ---: | ---: | ---: | --- |
| Source | 4.651044934 | 18.979815643 | 12 | 30s / 512MiB |
| Auxiliary CE | 7.750212339 | 9.029172016 | 0 | 15s / 512MiB |
| F | 9.074454798 | 18.936109629 | 16 | 30s / 512MiB |
| Return | 9.074455895 | 18.936150608 | 16 | 30s / 512MiB |

Return clock includes F; source clock continues through auxiliary/private work.
Physical counters in native-work.json, all prospective caps verified:

- 44native SGD/backwards216rows: source12/72 + F16/48 + return16/96.
- One CE optimizer/step,34forward/backward closures408rows; final read1/12 and
  hidden read1/12. Old TRAIN reuse is paid rehearsal. Total gradient rows624.
- 68components136named seeded draws3060values;271native head forwards555rows;
  195typed calls,42episodes,267sandbox dispatches; no robot physics.
- 64diagnostic snapshot attempts/returns,32summary attempts/returns and
  8640scalar reads, exactly the frozen diagnostic ceilings. Snapshot validation,
  guard copies and reductions are extra work, not extra native optimizer steps.
- Zero standalone affine/LS/LP/SVD/autograd.grad/random mutations/GPU/final/
  remote work. Copy counts/bytes and total FLOPs were not separately measured;
  their overhead contributes to elapsed time/RSS. No equal-compute certificate.

Across323's original failed capture, its accepted second full-update attempt
and this variant:100native SGD504rows +816auxiliary gradient rows =1320gradient
rows. Their separately preserved reported native process times sum39.567276773s;
RSS peaks are per process, not additive. None of the old charges is erased.

Metadata-only preflight EXIT0/.061969557s/24432640B; no numerical/project imports.
Ruff check/format and Linux-target mypy PASS on all four new stage scripts.
Exact argv/output/times: statics.json and statics-final.json; total engineering
commands6.469313900s including WSL preflight startup, below90s. Linux-target
mypy ran on Windows. No fixture or native failure/retry/budget extension.
One early Get-Item/Get-Content review-receipt read preceded the reviewer file
write and reported missing-file errors. That read did not prove a receipt;
issue closure waits for the actual reviewer write. No model work was restarted.
Saved-only analysis.py EXIT0; analysis.json processes all18 raw task endpoints,
curves, source flags, norms and physical work, without learner imports or work.

Accepted336 tests and unchanged mathematical/family controls are reused at their
original scope. No new Windows numerical cohort, four-family comparison, wheel,
clean-clone/install, browser, GPU, remoteCI, release or sweep was run. No
dependencies, global settings changes, deletes, commit/push or unrelated reset.
Stage remains below4MiB; static cache D:/Dev/_Temp/circadian337-mypy below32MiB.
If a hard kill had preempted capture, missing work would have remained unknown
and conservatively charged; no such failure occurred in this sole invocation.

## Acceptance and next action

Lead self-reviewed the actual ordinary path, common state before comparison,
all18 task traces/curves, guarded roles/flags, original clocks and paid work.
Independent prelaunch and final actual-source/results/report ACCEPT: five
read-only commands in total, all EXIT0, 1.0247298s within six-command/30s limits.
Reviewer checked all18 task traces, all32 norms, all11 common-state equalities,
flags, original clocks, costs and qualifications; wrote only stage/REVIEW.md.
No project imports, tests, models, solves, network or child agents. Exception ended.
No independent numerical replication is requested. Self-review is separate.

Acquisition fails at this fixed dose; retention succeeds on sampled known
tasks; recovery is untested because no old success was lost. This completes
the scoped observation without establishing the full learner/release goal.
328's simpler LS control (old4->4->4/fresh1->2->2, near-tie and return-CE caveats),
333's joint TRAIN reference capacity and all earlier negatives remain unchanged.

Critical path: do not extend the SGD dose or tune a learning rate from these
endpoints.333 already establishes joint affine separation on reference hidden
TRAIN rows; repeating that LP would add no useful evidence. Existing
joint_old_fresh_teacher and fit_categorical_ce_readout already accept the needed
TRAIN data, so no new orchestration or algorithm is required.
Next precise action: freeze/review one TRAIN-only native joint old+fresh CE
readout qualification through those existing ports and typed task readouts on a
disposable owner, explicitly charging old-data rehearsal. It should test native
readout optimization and six-task usefulness, not reuse333's witness coefficients
or claim sequential adaptation. This isolates the remaining learning gap before
another mechanism or comparison. Broader R2/R4/R5/R7/science/release remain open.

## Tracking receipts

Scoped337 closed after physical review receipt verification; canonical
https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/337#issuecomment-6535569323 .
Parents remain open: #18 comment6535575745, #21 comment6535576037,
#23 comment6535576262. Original scientific requirements remain unfinished.
