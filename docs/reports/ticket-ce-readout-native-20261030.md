# Four known tasks succeed after bounded TRAIN output refitting

R2.1-TW.CE.N / [#321](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/321).
One prospectively frozen Linux native case completed. The added CE-readout control
solved all four known ordinary tasks in three steps each. The matched original
source readouts at its final12-update checkpoint failed all four after six steps.
Independent review verified every raw action, applied transition and committed
tool, together with source/actor/budget preservation and measured costs.

This is useful **known-TRAIN task resubstitution** with extra output learning.
The original source learner remains unchanged. The result does not establish
new-task adaptation, retention during new learning, recovery after forgetting,
held-out generalization, statistical reliability, matched compute or CPC benefit.

| Fixed task | Original final source | CE-readout actions | Outcome |
|---|---|---|---|
| v1 invoice |fail/max6|open → route → commit|success/3|
| v1 password |fail/max6|open → inspect → commit|success/3|
| v2 invoice |fail/max6|open → inspect → commit|success/3|
| v2 parcel |fail/max6|open → defer → commit|success/3|

Every transition returned applied, with phases ready → staged → committed and
the correct committed tool. All12 predictions returned, validated and dispatched;
12 effects applied, no environment events. Allowed catalogs remain the unchanged
five actions per policy version, including phase-illegal actions. No phase mask,
target-fed decoder, rollout training, task selection or success-rule change.
The original24 development episodes remain preserved, all0/24 successful; the
four-task comparison uses only the prospectively fixed final12 ordinary subgroup.

Final TRAIN classification is12/12 globally and among allowed actions, including
all4/4 ready/routing rows. First paid closure CE is1.646444320678711; final native
float32 CE0.0027961048763245344. The existing scalar metric summary from final
logits reports float64 CE0.0027961079047948145. Precision differs; neither was
chosen as a more favorable metric. Ready allowed raw margins are4.056640625,
4.556640625,5.12109375 and42.068359375 in the table's task order. Large observed
logits/coefficients are retained without rescaling or tuning.

## Procedure, data reuse and isolation

Physical checkout HEAD `d9119e80926a139a2f38460564d7eab1b6d2a2b0` plus accepted320
dirty source, unchanged during this case. Existing Linux Python3.12.3,
NumPy1.26.4, Torch2.14.0+cpu, one CPU thread requested. The unchanged chronological
runner produced source12 SGD updates/72 row exposures, seed17/D9/H3/K6/lr.03/
momentum.9/inference2/A4B4A4. Its24 original readouts and source report were paid
and retained. Source reconstruction was required for a live owned hidden state;
historical scalar checkpoint observations never became model restore authority.

After successful source/checkpoint/12-receipt qualification, existing
`prepare_train_fit` supplied the exact12 old TRAIN cases. Its temporary original
TRAIN teachers are selected by the established old task order; unselected rows
do not reach fitting or a probe. Accepted320 fits only the detached float32 output,
starting from source output weights, with fixed hidden30/output24 parameters.
One LBFGS step uses the unchanged32-iteration/48-evaluation settings and hard64
closure ceiling. No LP warmstart, alternate solver, seed, weighting or retry.
Complete finite fit and source binding admit the reader and all four tasks;
accuracy does not determine which tasks run.

**Auxiliary rehearsal is paid:** the12 already admitted TRAIN rows were reread in
34 CE forward/backward evaluations,408 fitting row exposures, plus12 final scoring
rows. This joint old-TRAIN output fit is additional rehearsal/data reuse. No
stream replay helper ran and the original source replay counter stayed zero;
that counter must not be interpreted as zero total rehearsal. The408 exposures
were within the prospectively declared768-row ceiling. No dataset, budget or
procedure changed in this clarification. There is no retention-without-rehearsal
claim or optimizer-only causal comparison.

Actual candidate, seed and actor remain separate complete owned models. Their
parameters/gradients/SGD bindings, actor version, input, RNG,12 receipts and original
budget identities/birth are pinned around fit, reader construction, every typed
read and original-clock observation. The reader checks its own complete payload,
coefficients and canonical paid work. At most four earlier task records are pinned;
there is no unbounded success-record chain. Original source artifact bytes are
checked unchanged. No deployment/promotion or restoration hides foreign changes.

Original source budget remains30s/512MiB, born4.357808707,12 updates and zero source
replay throughout. Last clock7.270726058 before control,7.959247575 after; total
elapsed3.601438868s. Separate control allowance15s/512MiB begins before TRAIN
preparation and includes fit/reader/tasks: observed0.643462023s. Fit itself took
0.414770703s. Original wall and memory checks remain active through the control;
no budget restarts or added native source updates.

## Exact execution and observed costs

Stage `artifacts/runs/ticket-ce-readout-native-20261030/` holds PROTOCOL.md,
actual.py/observe.py, full `source.json.gz`, `control.json`, `native-work.json`,
compact `summary.json`, static receipts and permanent NATIVE_SPENT.txt.
The single native allowance is **COMPLETE/SPENT**; do not retry or rename it.

From physical `D:/Dev/Repositories/Circadian`, PowerShell parent sets
TEMP=TMP=D:/Dev/_Temp. One native command:

```text
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/ticket-ce-readout-native-20261030/observe.py artifacts/runs/ticket-ce-readout-native-20261030/native-work.json
```

EXIT0, budget_verifiedTrue, **5.562200167s /338092032B**. All numerical, source,
control, external60s/768MiB and intentional-stage caps hold. Physical counters
precede operations; fit costs are distinct from original native SGD:

- 12 native SGD/backward updates/72 source rows.
- One detached CE optimizer/step,34 CE forwards and34 backwards/408 rows each,
  one final TRAIN read/12 rows. This is extra gradient work, not one native SGD update.
- 27 native component constructions,54 named seeded draws/1215 values. Two of
  those constructions create/restore the independent CE typed reader.
- 13 hidden reads/24 rows: fit12 plus12 single-row task predictions;12 fitted
  affine reads/12 rows. Original/native-head work153 forwards/213 rows.
- 153 typed calls total,141 original source reads plus12 CE reads;28 episodes,
  201 dispatches including TRAIN preparation and the12 new task transitions.
- No LS/LP/SVD/autograd.grad, PC latent, stream replay helper, probe/final updates,
  GPU or remote model execution. Auxiliary rehearsal is separately counted above.

Wall/RSS include imports, validation, copies and orchestration; all internal
LBFGS scalar/tensor/copy operations are not separately enumerated. Equal capacity
does not make this matched compute with the untouched source or other families.
summary.json is a PowerShell projection of saved JSON/gzip metrics and transitions,
with no new model forward/fit. An initial manual lookup used returned metadata
for status; corrected inspection used the actual transition.status/state fields.
No task/model rerun or reinterpretation of null metadata as a failure.

Scoped static prefixes target the stage's actual.py and observe.py:

```text
.venv/Scripts/python.exe -B -m ruff check
.venv/Scripts/python.exe -B -m ruff format --check
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent --platform linux
```

Receipts preserve exact argv/output. First batch Ruff/format passed; both mypy
targets failed two errors on the copied observer's direct Tensor.backward method
assignment. Replaced it with setattr, same observation semantics, before native
execution. First24.481763s retained; final four checks EXIT0/24.2303698s, total
48.7121328s under120s. No static or scientific limit amendment/overrun/retry here.

## Acceptance and next substantial outcome

Lead self-reviewed source, procedure, all task transitions, counters and limits.
Independent reviewer accepted prelaunch procedure and the actual source/control/
cost results, verifying all12 raw steps and committed tools. Four read-only
commands0.5906513s, all EXIT0; no project/numerical-library imports, models, fit,
tests, writes, network or children. First verbose review display was truncated;
the final compact assertion pass checked every task/step/receipt/budget field.
Exception ended. Self-review is distinct from independent review.

Skipped unchanged320 model tests, broad repository suites, clean clone, wheel,
other families, new-task probes, permission/stale-task stress variations, statistical
replications and release qualification. The accepted software/math is unchanged;
this is the new real known-task control evidence. No install/download/delete,
commit/push or global settings/cache changes. All earlier failures/SPENT evidence,
#318's unverified temporary proof and #319's supporting static overrun remain intact.

The critical path now advances to **learn a fresh task, retain the old tasks and
measure return/recovery with the existing FreshTaskProbe**. Concrete dependency:
the CE reader is deliberately readonly; FreshTaskProbe requires fork/train_batch.
Next implement and qualify an explicit detached native initialization from the
actual source plus actual CE fit, copying fitted output coefficients only into
the private learner. Preserve complete original source/SGD/gradients/RNG; bind
source/fit/init provenance and auxiliary costs separately. Do not fabricate native
SGD credits, silently reset momentum, rescale logits or tune from probe outcomes.
Use independent private callbacks: the source's counted fork retains source-bound
callbacks and must not charge F work to the original source. Reuse the existing
private factory/probe rather than add another runner or model-state loader.
Then freeze a small F16/return16 case. This handoff/adaptation work is unfinished;
no new probe or initialization implementation is claimed by321. Fullgoal/#21/#18/
R2.1 acquisition-retention-recovery/scientific comparisons/release stay open.


Closeout: stage12files158,283B, below4MiB. Scoped tracked/new-document
whitespace checks emitted no whitespace errors (LF/CRLF advisories only).
All intentional project outputs remain on D; no new independently attributed
library-temporary-byte claim or closure of318's historical gap.

GitHub closeout receipts: canonical issue321 comment6527522290; issue321
closed completed with four scoped acceptance criteria checked. Parent tracker18
comment6527589514 and R2.1 parent21 comment6527590008 preserve the open full
goal and exact next implementation action. Neither parent was closed.
