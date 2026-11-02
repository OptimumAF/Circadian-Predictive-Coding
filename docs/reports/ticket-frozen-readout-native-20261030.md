# BP frozen-readout diagnosis: partial TRAIN repair, two routing errors remain

R2.1-TW.FR.N / [#318](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/318).
One prospectively frozen Linux CPU case through the unchanged chronological
ticket runner. Software #317 is reused. PhysicalD checkout HEAD
`d9119e80926a139a2f38460564d7eab1b6d2a2b0` plus accepted dirty work; no production
code or defaults changed. All earlier negative/SPENT results remain intact.

The trained BP source gets8/12 known TRAIN decisions correct and none of the four
ready/routing decisions. One disposable least-squares readout on its unchanged
hidden features gets10/12 and two of the four routing decisions. This identifies
room for improvement in the original readout, but does not establish complete
TRAIN feasibility, a unique cause, native task usefulness or a CPC advantage.

| Fixed TRAIN metric | Original BP output | Frozen-feature LS output |
|---|---:|---:|
| Full-class and allowed correct rows |8/12|10/12|
| Ready/routing correct rows |0/4|2/4|
| Open/commit correct rows |8/8|8/8|
| Mean full-class cross-entropy |1.646444208|1.237843546|

| Ready TRAIN row | Target | BP choice | LS choice | LS allowed raw margin |
|---|---|---|---|---:|
| v1 invoice |route|open|inspect|−0.069153|
| v1 password |inspect|commit|inspect|+0.351977|
| v2 invoice |inspect|open|open|−0.256039|
| v2 parcel |defer|open|defer|+0.548801|

The H3-plus-intercept design has rank4, singular values
`[3.473369085, 0.460978481, 0.115530838, 0.040184831]`, and one-hot squared error
`2.6422530812470075`. The solve retains24 output coefficients; the30 hidden
parameters stay fixed. Float64 equal-row one-hot LS differs from native float32
CE optimization. Rank and positive one-hot SSE do not prove classification
infeasibility, hidden-feature collapse, inadequate maximum capacity or convergence.
Partial success supplies a known-TRAIN witness only for the correctly classified
rows; it does not prove the remaining rows cannot be separated by another output.

The original source's unchanged development readouts remain0/24 task successes.
**No fitted-head task rollout was executed.** Do not turn10/12 imitation decisions
into task completion, retention, recovery, unseen adaptation or release evidence.
The four fixed TRAIN tasks and all failed rows remain included; no task selection,
new seed, dose, class weights, alternative solver, tuning or favorable tie change.

## Procedure and isolation

Existing Linux Python3.12.3 / NumPy1.26.4 / Torch2.14.0+cpu, one CPU process with
thread environment1. Fixed ordinary `run_example(context_return=True, method="bp")`,
A4/B4/A4, seed17/D9/H3/K6/lr.03/momentum.9/inference2,12SGD/72 row exposures.
All24 existing development episodes ran unchanged. Reconstructing this source was
necessary because saved scalar checkpoint observations are not owned hidden-state
restore authority. This is newly paid reconstruction cost, not a repeated baseline
comparison or renewed #278 class-balanced experiment.

A temporary constructor observer retained the actual stream and was restored
after the runner returned. Only after source completion,12 receipts and final
checkpoint equality passed did the diagnostic read the candidate. Its seed,
candidate and stable actor were distinct actual owners with complete state,
optimizer binding, policy/payload and RNG pins. Only original12 admitted old TRAIN
cases from existing `prepare_train_fit` were supplied to accepted #317. Source,
seed, actor/version, input, receipts, RNG and original budget stayed unchanged.
No parameter loader, new runner, deployment update, promotion, probe or final data.

The original30s/512MiB budget birth remained `5.200862928`, with12 updates and zero
replay throughout. Last clock before diagnosis `8.579998662`, after `8.634531476`:
elapsed3.433668548s, within the original allowance. No reset or renewal. Separate
10s diagnostic timer began before TRAIN preparation and observed0.025038335s.
Original wall and memory checks also surrounded diagnostic ports. Stream remains
finished; no new native training was admitted after it.

## Commands, results and costs

Local stage `artifacts/runs/ticket-frozen-readout-native-20261030/` contains frozen
`PROTOCOL.md`, local producer/observer/preflight/extraction, raw `source.json.gz`,
`diagnosis.json`, fixed `summary.json`, native work receipt, static receipts and
permanent `NATIVE_SPENT.txt`. Native budget is **COMPLETE/SPENT**; do not retry it.

All commands start in physical `D:/Dev/Repositories/Circadian`; PowerShell parent
sets `TEMP=TMP=D:/Dev/_Temp`. Linux prefix:

```text
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B
```

Using that prefix, exact suffixes (one invocation each):

```text
artifacts/runs/ticket-frozen-readout-native-20261030/preflight.py
artifacts/runs/ticket-frozen-readout-native-20261030/observe.py artifacts/runs/ticket-frozen-readout-native-20261030/native-work.json
artifacts/runs/ticket-frozen-readout-native-20261030/extract.py
```

Preflight: EXIT0,0.023224439s/13643776B, two source files compiled and fixed
entrypoint checked with zero project/NumPy/Torch/model imports or fitting. Native:
EXIT0, **5.872483927s /335781888B**, under60s/768MiB externally and original source
limits internally. Work counts before operations:

- 12 native BP backward/SGD updates,72 source training-row exposures.
- 25 native component constructions,50 named seeded draws/1125 values.
- 142 public logit calls/153 rows,154 native head forwards/225 rows including
  training and readouts. The additional TRAIN read is one call/12 rows.
- One hidden forward/12 rows, one LS solve/12 fit rows, one affine readout/12 rows.
- 24 unchanged development episodes and189 total dispatches including preparation.
- Zero autograd.grad, SVD, PC latent work, replay, probe updates, final-score
  forwards, GPU or remote execution. These counts and wall/RSS do not establish
  matched compute; complete internal tensor/BLAS operation counts are not claimed.

Extraction: EXIT0,0.029586969s/20090880B, zero project/NumPy/Torch/model imports,
parameter arithmetic, new forward or fit. Recomputed the frozen CE/argmax/ready
metrics from saved logits and checked budget fields; no new model scoring or
metric selection. Prospectively used the unused second stdlib engineering gate
for this post-result projection (issue comment6524455733), same30s/64MiB bound.
Distribution metadata reports installed SciPy1.15.3; no SciPy import or solve.

Scoped static commands targeted the stage's `actual.py`, `observe.py`,
`preflight.py`, with these prefixes:

```text
.venv/Scripts/python.exe -B -m ruff check
.venv/Scripts/python.exe -B -m ruff format --check
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent --platform linux
```

`statics-first.json` retains exact argv/output: Ruff/format passed; mypy failed
dynamic record/retained annotations, temporary constructor assignment and Windows
resource stubs (14 default /12 Linux errors). Fixed with explicit Any annotations,
`setattr` for observation and `getattr` on Linux resource API; no scientific/math
change. Four-command first batch42.5286091s. Final four commands EXIT0,
38.1862606s. `extract.py` was a later stdlib metadata command, executed successfully;
the three-file static qualification does not cover that later file. No pytest or
unchanged baseline-model rerun. Initial guessed-path and wildcard read failures
were resolved through actual imports/file discovery, with no model execution.

## Review and critical path

Lead reviewed actual source, procedure and costs. Finite independent reviewer
accepted prelaunch protocol/source and then saved source completion, receipts,
ownership, original clock and recomputed metrics. Exactly four read-only commands,
0.7979286s aggregate, all EXIT0; no project/NumPy/Torch imports, models, fitting,
writes, network or children. Exception ended. Self-review is distinct. There was
no native failure/retry or scientific budget amendment; initial static failures
remain preserved.

Skipped broad repository tests, clean clone, installed wheel, other families,
new source/probe cohorts, fitted task rollout and release checks. Production
source is unchanged and accepted #317 behavioral/math/isolation evidence applies.
Scientific benefit and repository readiness require their separate evidence.

Reassessment: direct output refitting improves two routing decisions, but the
failed LS objective cannot distinguish readout optimization from representation
infeasibility. Avoid another source/dose/capture cohort. **Next: freeze and run
one linear-separability feasibility witness on all12 saved TRAIN hidden rows,**
using the existing installed SciPy solver without new models, label selection or
production dependencies. Declare full-class versus allowed-action inequalities
explicitly; a failed full-class witness alone does not refute allowed-action
feasibility. Independently check any positive witness margins. This is a bounded
prerequisite for choosing a learning repair, not deployment or another optimizer
search. Later useful-control qualification must retain all four tasks together.
No such feasibility solve is admitted or executed by318.

Research diagnosis318 accepted; its temporary-byte ceiling remains unverified,
so the ticket retains that requirement unchecked. Do not rerun the spent case to
manufacture missing retrospective storage proof. The next learning witness is
unblocked by that cost qualification. #21/#18/R2.1 acquisition-retention-recovery, useful adaptation,
comparison and release requirements remain open. Both plans/log retain archival
evidence pointers; GitHub remains the live tracker. All old failures preserved.


Closeout: local stage14files100,762B, below4MiB. Scoped tracked/new-file
whitespace checks emitted no whitespace errors (LF/CRLF advisories only).
An unnecessary recursive read of shared D temporary storage was interrupted
with Ctrl-C; no deletion or mutation. Temporary-library bytes were not
independently attributed, so the proposed4MiB temporary ceiling is not verified
by this result. Do not convert that uncertainty into a storage-cap acceptance
claim. All source/diagnostic numerical, wall/RSS and owned stage limits above
have direct receipts. No install/download/delete/commit/push/global-cache change.
