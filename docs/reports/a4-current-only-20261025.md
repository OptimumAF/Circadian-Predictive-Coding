# R2.1-A4.C — native current-only control, 2026-10-25

[#172](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/172),
single `/root` implementation/tests/source-procedure-result SELF-REVIEW, not
independent acceptance. Reuse171joint scalar evidence without rerunning its
fit/game cohort or reading raw171model/actual/work. Full goal ACTIVE, broad
parents OPEN; historical plans/log frozen underAGENTS. HEAD09ff811f/unrelated
dirty work preserved. [Usage](../minigrid-a4-adaptation.md).

## Same-condition result

Each cell shows A3goals ; A4goals, two fixed cues/task. Same data versions/order,
representation/solver, seed17/max16/K7firstargmax/checkpoints/metrics. Games use
the actual causal states after all fixed fits finish; no readout selects updates.

| Recipe | After initial A3 | After A4 | After A3 return | Fit exposures |
| --- | --- | --- | --- | ---: |
| Joint previous/current TRAIN, accepted171 | 2/2 ; 0/2 | 2/2 ; 2/2 | 2/2 ; 2/2 | 90 |
| Current episode only, new172 | 2/2 ; 0/2 | 0/2 ; 2/2 | 2/2 ; 0/2 | 54 |

Current-only acquired A4, lost A3, and recovered A3 after18explicit returning A3
TRAIN rows, while losing A4. Joint replay preserved both known tasks after A4
training and return, using36additional joint row exposures. This is a local
retention/relearning tradeoff, not an overall winner, matched-compute efficiency,
statistical generalization or CPC benefit. Joint results were known beforehand:
exploratory same-condition control, not independent confirmation/preregistered
joint arm. This is an affine head-replacement learner, not evidence about all
gradient-based/recurrent learners.

| Frozen metric | Joint171 | Current-only172 |
| --- | ---: | ---: |
| A4 acquisition gain | +1.0 | +1.0 |
| A3 retention delta after A4 | 0.0 | -1.0 |
| A3 return gain | 0.0 | +1.0 |
| Old-cue recovery flags | false, false | true, true |

Joint recovery is false because A3 was never lost. Current recovery required
old-task supervised retraining; its18return rows/cost stay. Return gain does not
rank a retained joint arm below one that first forgot the task. Current-only
has six successes9steps/return0.49375 and six complete failures16steps/return0;
all12closed. Joint has ten successes9steps/.49375 and two initial A4 failures
16steps/0. All fixed outcomes/negative results stay, raw end flags/probabilities/
public states/full parameters/receipts retained locally, no extra raw reader.

## Implementation and verification

Separate exactbool `a4_current_only` in existing stream/evaluator selects the
original nonretained path: six9row batches, native2/18/head9,4/36/head9,
6/54/head9, no catalogue. Original `a4_training` retention-required90row mode,
definitions/defaults unchanged. Same A3/A4 source IDs/contextA/296K7/nine rows,
sourcewitness/labels delayed1/6receipts. Tests inspect the actual x/y arrays of
every fit: exactly the current arrived episode, no jointly fitted prior rows.

Existing composition/CLI adds `--context-linear-a4-current-only` and strict
keyword `current_only=True`. Same pinned from-birth source/real full new-state
transfer/three-stage game loop/branchIDs/source/scorer/state/budget/clock/paid
prefix value+identity/effect/cleanup guards. One original150s512MiB0update/replay
owner spans admission/train/allplays; inner6update90s512MiB retained. Bounded3
phase storage, complete negatives/partialUNKNOWN output, exclusive8MiB/describe
beforeIO remain. No learner math/framework/model reset/hyperparameter change.

26/26PASS/platform0skips, firstcohort/no new failed checks. One actualCLI6literal
LSE54row capture/platform drives12scripted two-step games, reused by remaining
tests. Exactcurrent rows/18→36→54counters/zerojointreplay/delay1receipts plus
conflict/role/shape/clock/paidprefix/output/metric refusals. Joint descriptor/
typed2/4/6plan/originalretention-required regressions0fit;171numerical evidence
reused. Scripted success is not native evidence.

Windows body3.905882899998687s/parent4.7290088s/RSS104017920B; Linux body
12.301691684s/parent17.5762824s/RSS97005568B. Each6LSE54rows/40predict168rows/
222projections1426copies4064998typed scalar bounds, JSON6937152B Windows/
6937767B Linux/countUTF8same,0forbidden/RNGsame. Original120sbody150sparent/
512MiB/operation caps held; no nativephysics/actual170payload in qualification.
Seven-file Ruff/format/default+Linux-targetmypy first/final allPASS;
21.8300822s inclformat<150, touchedcache115728802B<160MiB/global603885932B
separately. No profiler, extra cohort, joint numerical repeat or native retry.

## Data and observed costs

New CONTROL payload read once each:170TRAINactual428191B/SHA
`bce498c34c23d9ef128f9344bf7164a70edc6afa156d42e7534bb708b2e3c26f`,
bundled94705B/SHA
`5a368cfa8d4d7326d5d3ab4d6663a0e8e35e1e660968a68212e627ba9203d508`.
No new hash preparation. Shared171prep522896B/.8282145s and170collection18teacher
steps/2env/2.5406233s remain sunk setup/provenance, not renewed authority.
Pre-existing bundled examples' original generation cost was not remeasured.
Decoder prepared54bundledTRAINrows, selects18A3; unselected36neverfit/forward.
A4 was TRAIN from birth; no168probe_test/private/final/D/E/F/old127/bank/model/
171rawparameter inputs. Both newpayload/case allowances nowSPENT.

| Observed new-case work | Joint171 | Current-only172 |
| --- | ---: | ---: |
| Fits | 6 | 6 |
| Arrival exposures, incl18return repeats | 54 | 54 |
| Within-update replay exposures | 36 | 0 |
| Total fitted row exposures | 90 | 54 |
| TRAIN diagnostics calls / rows | 16 / 144 | 16 / 144 |
| Native game steps / predictions | 122 | 150 |
| Total prediction calls / rows | 138 / 266 | 166 / 294 |
| Port projections | 356 | 348 |
| Copies / typed scalar bound | 5031 / 15196617 | 6067 / 18352942 |
| Owners / environments / closes | 9 / 12 / 12 | 9 / 12 / 12 |
| Body seconds | 10.512506900005974 | 9.495146699948236 |
| Parent seconds | 10.89451310003642 | 10.010503400000744 |
| Sampled RSS bytes | 85368832 | 89899008 |

Equal updates with54versus90rows is **not matched computation**. Fewer fit rows
did not mean fewer game steps or lower sampledRSS. Single-run times/peaks do not
prove universal efficiency; counters/reserves are not full allocation/FLOPs.
Both use36distinct source records;18old return rows are repeated exposures,
not new independent observations or zero rehearsal.

Current child/parent EXIT0, directly metered original fit/readout/physics/clock/
RSS/JSON caps held; source/scorer/paid-prefix guards/RNG equality passed. No
optimizer/backward/Torch/randommodeltraining/teacher/newcollection. LogicalJSON/
countUTF83913188B each incl16KiBterminalreserve, physicalactual3677939B/log2705B/
stage3727807B beforefinalreview, within8MiB/256KiB/16MiB limits. Same-process
summary only, no extra raw reader/rerun/extension. Forced OS-timeout cleanup
source-reviewed only; normal path passed. No deployed publication. Fullsuite/
cleanclone/newwheel/independent/multiseed/statistical/release checks skipped.
No install/download/GPU/browser/CI/commit/push/delete/unrelated reset. All
171initial negatives/170engineeringfailures/168missingRNGcriterion/167timeout/
166copyFAIL/olderCPCnegatives remain, not retroactively passed.

## Commands, critical path and exact next action

Root `.venv/Scripts/python.exe -B` plus stage
`artifacts/runs/a4-current-only-20261025/`:

- `qualify.py windows-1` →26PASS; cachedWSL CPU1 `timeout140s .../bin/python -B
  .../qualify.py linux-1` →26PASS; expanded commands/times in parent JSON.
- Scoped `ruff check`, `ruff format --check`, `mypy --follow-imports=silent`
  default and `--platform linux`; exact commands/files in statics*.json and
  format-1.json. AllPASS. PROTOCOL/REVIEW freeze source/metric/work limits/GO.
- `execute.py` →first/sole child EXIT0; execution/work/native/actual preserve
  complete procedure/cost/terminal/raw result evidence. No new source prep.

A combined documentation patch missed a usage-file context after native and
wrote nothing; simplified to separate Add-only report and exact-context edits.
No model/test/protocol work or source changes resulted from that documentation
failure. Product/tests remain unchanged since passing cohorts/native.

Critical path reassessed: no further context/seed/control/capture/audit clone.
The proven mode still requires historical170exactSHA via read_a4_training and
CLI --source. artifacts/runs is ignored and not packaged; current packaging
ships bundled ABC only. Existing first-party run_workflow has play/privateF/
recurringG modes, while its collector already accepts fresh A4 TRAIN. This is
a concrete user-workflow dependency, not a reason to weaken scientific pins or
invent a loader/bootstrap framework.

**Next implementation:** factor existing A4 composition's source handoff into a
producer/typed-source seam and connect fresh A4 collection through existing
minigrid_affine_workflow/CLI. Keep fixed-source research commands strict and
compatible; fresh provenance carries its own records/costs/clocks, never claims
historical170SHA. Reuse collector/stream/evaluator/cumulative guards; literal
integration first. Later bounded first-party qualification is product behavior,
not independent scientific replication or permission to repeat171/172arms.
Broad R2/R4/R5/G5/R7/P9 requirements/fullgoal remain OPEN/ACTIVE.
