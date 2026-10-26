# Isolated cold linear probes through the existing stream

2026-10-25, [issue160](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/160),
R2.1-CLM.LP. Implemented/executed scoped software. Source/procedure/results
self-review, not independent acceptance. Full repository goal remains active.

The existing temporal API/CLI now supports `--probe-linear-fit-control` alongside
`--catalog-training --reserved-task-probes`. It completes the original twelve
benchmark updates and four isolated two-row fits, then scores all32cached probe
rows. Marked literal public-CLI captures run on Windows/Linux. This increment
runs no pretrained experiment and establishes no new native learning benefit.

## Behavior and interpretation

Reuse the original16-text frozen-feature union, canonical tavrix/vemlot roles/
targets/catalogs, four source checkpoints0/4/8/12, benchmark receipts/clocks and
existing metrics. Each control makes one fixed minimum-norm one-hot fit of two
`probe_train` rows, then reads TRAIN2/suffix2/old-task4. Four solves/eight fitting
rows/twelve reads/thirty-two readout rows,0probe optimizer/backward/head updates.
No fake six-update native receipts or unfitted-before prediction. The original
six-update native probes remain the default; there is no replay or final access.

This control uses **cold raw features**, not the native source head's weights.
Old-task scores measure compatibility of a newly fitted classifier, not that
native learner's retention/forgetting. A fixed-encoder control cannot measure
changes in warm-head plasticity. All groups/endpoints remain, without favorable
seed/metric/case selection. No matched-compute, calibrated-confidence, CPC or
generalization claim follows from these fixtures.

Read-only `outer_selection` is now explicitly admitted for a probe-fitted linear
control, matching the original old-task port's declared role. The fitting role
stays `probe_train`; final roles and serving remain refused. The serving test
matches the specific probe-role refusal before accessing encoder/actions.
Original numeric core files match accepted157hashes; no solver/data/tolerance
change. The expanded role admission is deliberately new, not claimed unchanged.

New adapter composes the existing fit/readout and checkpoint ports; the original
stream, native learner and default update rules remain. Exact feature values,
owners/storage/roles/catalog/targets, source state/work/owners/RNG, prior success
records and local fitted-model values/owners are guarded around callbacks.
Each returned operation is checked before more work. Failed fits/reads retain
their paid/uncertain records and trusted prefix; no retry after uncertain work.
Storage is fixed to four completed records plus one current fit/pending operation
and a bounded preserved prefix. No exported probe, deployment or promotion.

All four records and original benchmark completion are admitted before any metric
is scored. Corrupted later probabilities, roles, work or phase reject scoring of
earlier groups too. Existing native/replay flags cannot be added later to this
control. API/CLI conflicts and non-boolean selections fail before loading.

Files: new `src/adapters/clm_linear_checkpoint_probes.py` and focused test,
narrow changes to `clm_checkpoint_probes.py`, `clm_linear_fit_control.py`, existing
temporal script; README and [usage/structure guide](../clm-linear-probes.md).
No new dependencies, environment variables, loader, framework or core math.

## Executed checks and retained failures

Stage `artifacts/runs/clm-linear-probe-20261025/` contains prospective PROTOCOL,
reused narrow engineering harness, expanded commands/XML/stdout/work/parent
receipts, marked CLI captures and rollup/source pins.

| Run | Cases/outcome | Body seconds | Parent seconds | Peak RSS bytes |
| --- | --- | ---: | ---: | ---: |
| windows-1 |1setup error; tail unrun|8.009537200|8.627802600|327000064|
| windows-2 |35pass|16.477226900|17.030033800|329265152|
| windows-final |7pass|13.970226900|14.574292200|329261056|
| windows-cli-final |1pass|9.256887700|9.786080700|329109504|
| windows-model-guard |3pass|11.415723600|12.006950900|330145792|
| windows-role |1pass|3.733939800|4.167408000|247885824|
| linux-1 |39pass|22.187316787|27.648122800|349249536|
| linux-cli |1pass|17.363852892|22.841526200|347774976|
| linux-cli-final |1pass|17.521450646|22.381368700|347742208|
| linux-model-guard |4pass|16.771656840|21.673951400|347312128|

Final **41distinct qualified cases/platform** combine unchanged cohort evidence
with targeted late guards/role/capture checks. Neither platform has one final
41case full rerun. Windows exhausted its two full-cohort attempts, then targeted
only affected/new cases. Linux had one39case cohort plus targets. Initial error
was the test wrapping an existing `(batches,choices)` result twice; the original
stream refused it before any fit/TRAIN/backward. Fixture unpacking alone fixed
it. Its4heads/1scheduler constructor/10112RNG elements remain charged.

Self-review added late native-mode/boolean-clock refusal, then fitted-model
value/owner guards across readouts; qualified with focused new failures and full
normal CLI paths. Each corresponding source change was followed by tests. The
last test-only `captured:list[Any]` annotation changes no executed behavior.
The original default four-native-probe regression and two original linear-role
cases pass. Unchanged4096math and broader baseline suites were reused, not rerun.

Initial scoped mypy found5test typing errors (None instead of Path; module base
alias), then1new capture-list annotation error after the guard tests. Both failed
receipts remain. Final five-file Ruff/format/default+Linux-targetmypy allPASS.
Four statics attempts total104.306131300s, plus bounded formatting/metadata within
180s. Actual Linux Ruff/mypy unavailable/uninstalled; Linux-target runs on Windows.
Expected existing scheduler-wrapper warnings remain, without algorithm change.

One orchestration mistake is retained: after a poll still showed Linux CLI live,
the lead started statics1. Receipt timestamps estimate2.408s parent-interval
overlap; this is not CPU-overlap profiling. This attempt does not demonstrate
the intended sequential procedure or fair timing. Later final CLI checks ran
sequentially after verified terminal handles, including the final model guards.
The earlier attempt/charges remain, and no cap was raised or model work refunded.
There is no comparative timing claim.

Public main/run_case executes with only the loader replaced by literal2D heads/
backbone; the fixture records requested2threads and forces1 for its allowance.
Final marked artifacts:

- `windows-model-guard-literal-cli.json` —358371B.
- `linux-model-guard-literal-cli.json` —348689B.

Both are `literal_cli_fixture_not_native_v1`: complete original12update stream,
4fits/32cached rows, all role groups/source guards and occupied-output refusal.
This is actual CLI/software execution, not pretrained quality or two-thread
hardware qualification. Synthetic feature fit is not useful semantic adaptation.

## Cumulative engineering costs

| All attempts including failures/repetitions | Windows | Linux |
| --- | ---: | ---: |
| Actual least-squares solves / fitting rows |40 /80|36 /72|
| Linear readouts / rows |85 /224|73 /192|
| Profiled linear/design/copy cells |6446|5750|
| Literal oracle-cell reserve |393216|262144|
| Literal backbones / head objects |4 /438|4 /396|
| Fake encoder forwards / sequences / token IDs / slots |4 /64 /64 /64|4 /64 /64 /64|
| Native Tiny TRAIN / backward / optimizer / scheduler |108each|96each|
| TRAIN query / candidate / logit exposures |300 /300 /852|264 /264 /744|
| Head forwards / projected rows / initial scheduler steps |350 /1037 /26|316 /930 /24|
| Returned native capture + Torch RNG elements |12181638|10742168|
| Metered random initializers |0|0|
| Cumulative body / parent seconds |62.863542100 /66.192568200|73.844277165 /94.544969100|

The108/96Tiny updates are paid software/default-regression work, not pretrained
updates. Cold probes themselves use no native head update. Parameter construction
uses explicit literal tensors; primitive initializer meter0 has source-backed
literal construction, not a claim of universal initializer profiling. Model/RNG
work performed by fault tests is included. Logical counters/reserves are not all
allocations, FLOPs, comparisons, serialized bytes or IO. Source captures and
other-copy reserve64MiB are distinct; whole sampled RSS includes overhead.

Original numerical/time/RSS/storage limits held:96fits256rows,256reads1024rows,
2Mcells including reserves,256Tiny updates640query320candidate3200logits,
512heads1024headfw4096projectedrows,128Mreturned tensor/RNG elements+64MiBreserve,
120sbody150sparent/1GiBRSS/CPU1perfixture. The sequential-procedure exception above
is not erased. No retrospective engineering cap amendment. Conservative cache
cutoff18:30UTC (predates issue160) gives109799808Btouched whole files<160MiB;
global inherited+task603295958B separately exceeds that task reservation. Stage
2197115B/stdout20066B before final rollup, below8MiB/512KiB. NumPy2.4.6Windows/
1.26.4Linux and Torch2.14.0+cpu; no cross-platform bitwise claim.

## Commands, limits and next action

From repository root:

```powershell
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-probe-20261025/launch.py windows-1
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-probe-20261025/launch.py windows-2
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-probe-20261025/launch.py linux-1
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-probe-20261025/statics.py statics-1
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-probe-20261025/statics.py statics-2
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-probe-20261025/statics.py statics-3
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-probe-20261025/statics.py statics-4
```

Exact targeted node arrays/expanded WSL commands for windows-final, linux-cli,
windows/linux-cli-final, windows/linux-model-guard and windows-role are retained
in their `*-parent.json`; the harness lists selected files/regressions and XML
retains each attempt's exact cases. Scoped
Ruff formatting ran after relevant edits. No timeout/skip; only initial setup
error and the two typing failures. One atomic patch context failed before changing
files and was corrected; no model work resulted. No full unchanged suite,
cleanclone/package/install/browser/GPU/remoteCI/pretrained/native study,
independent review or release checks. No installs/downloads/commits/pushes/
deletion/unrelated reset. Local evidence paths do not imply published files.

Historical plans/log remain frozen under AGENTS; live issue/tracker owns status.
Prior159 known native task result remains accepted within its scope; native and
its two result readers remain spent. No159actual/ledger/fit or older scientific
raw/detached review was read/reconstructed. Prior153failed-spent/154manualUI/
151partial acquisition and interference, and all earlier negatives/charges stand.
This software does not complete R2/R4/R5.1/R5.3/G5/R7/release or the full goal.

**One next action:** freeze a small native raw-feature reserved-marker diagnostic
at the existing baseline hook: one original frozen16-text extraction, one cold
two-row fit, and TRAIN/suffix/old-task outputs with all fit/encode/copy/time/RSS
costs. The control is independent of native-head weights: do not repeat four
identical fits or twelve benchmark updates solely to manufacture checkpoint
work. Use the existing role-bound ports and declared source, retain every result,
and label old-task compatibility honestly. No native invocation before that
protocol is fixed; no F6 retry, old unsaved-weight reconstruction or new audit/
capture/reader increment. Warm adaptation/retention/context-return still require
their own learner evidence.
