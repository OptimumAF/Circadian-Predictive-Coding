# R7-CTX.LS.S — owned catalog study chooses waiting

[Issue295](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/295)
owns live status. Parents18/21/26/33 and the full goal remain open. Previous294
fixed-action evidence and292 calibration result were reused, not rerun. The
failed/SPENT287 comparison remains UNKNOWN; this study does not replace it.

## Working behavior and observed result

The ordinary CLI now trains the existing finite catalog chooser from its current
owned LS source, discards six TRAIN candidates, freezes once, and runs one
development application against that original source. It completed seven child
delegations/returns. Source, actor, both RNGs and original source verification
passed; candidates were never promoted. The source budget was born once before
training:12updates/240seconds/512MiB. Legacy30second modes remain unchanged.

The frozen policy chose wait for mistakes and clean. Application used zero
updates/replay exposures, kept old-task4/4 and new-task1/2 unchanged. One missing
new task remains unlearned. This is functioning empirical selection, not useful
new-task adaptation, generalization or circadian advantage.

All before vectors are invoice-v1/password-v1/invoice-v2/parcel-v2/legacy-v1/
legacy-v2 = PASS/PASS/PASS/PASS/FAIL/PASS. Each private readout contains six
complete three-step TRAIN episodes. Native observations:

| Context/action | New count before→after | Old before→after | Private fits/rows | Utility |
|---|---:|---:|---:|---:|
| mistakes/wait |1→1|4→4|0/0|-0.15601414047630208|
| mistakes/wake1 |1→2|4→2|1/3|-0.18118671062916666|
| mistakes/replay6-wake1 |1→2|4→2|2/9|-0.20551113365416662|
| clean/wait |1→1|4→4|0/0|-0.2257328030515625|
| clean/wake1 |1→1|4→4|1/6|-0.24973221255078126|
| clean/replay6-wake1 |1→1|4→4|2/12|-0.2751748206322917|

Both mistakes updates learned legacy-v1 but lost invoice-v1/password-v1. Clean
updates did not change task counts. Both replay continuations armed/completed
once, deferred twice, and charged exactly six examples without duplicate replay.
All seven cases completed fourteen controller opportunities with the original
actor. The application task vectors equal the original source vectors.

The original utility gives equal weights to normalized new gain, old drop,
execution time/120seconds and process RSS/512MiB (plus violations). Here gain1/2
and drop2/4 cancel. Costs then favor waiting. No metric/weight/order/seed/action
was changed after observing results. All utilities were negative. Fixed order,
one row per context/action and accumulating evidence-validation/process memory
costs prevent a causal efficiency or population-optimal policy claim. RSS peaks
are process observations, not isolated per-arm allocations.

## Source and isolation

Changed scripts:context_catalog_learning,context_head_live,context_head_rehearsal,
recurring_ticket_workflow_example. New test:test_context_linear_catalog_study.
No core controller/learner/replay law or dependencies changed. The study accepts
an explicitly owned LS source/guard/sampler; the saved Perceptron decoder/default
path is preserved. Complete original12/72 source/history/digest stays binding.
Prior paid outcomes are sealed/checked before further delegation; corruption,
replacement or lost returns close future work. Missing child work is UNKNOWN,
not zero. Six complete known rows precede scalar fitting and original-source
application. Existing selected-wake continuation and task readouts are reused.
Final/probe labels are unavailable to fit, promotion and selection. Ordinary
known TRAIN templates/public context cues do not establish held-out learning.

## Frozen execution, costs and artifacts

Stage:`artifacts/runs/context-linear-catalog-study-20261029/` retains PROTOCOL.md,
native.py/native.attempted,actual.json,native-work.json and all pure receipts.
Native protocol was frozen and reviewed before model work. ONE Linux invocation,
EXIT0, no retry. Native observer includes main/new Python threads with shared
locked counters. Attempt marker precedes model import/work. Recorded costs:

| Quantity | Observed |
|---|---:|
| LS fits/optimized row exposures/lstsq |18/102/18|
| Source fits/rows |12/72|
| Six discarded candidates fits/rows |6/30|
| Frozen application fits/rows |0/0|
| Replay example exposures |12|
| Teacher traces/derived teacher steps |86/258|
| Task episodes (includes source, private readouts and workers) |206|
| Sandbox dispatches |930|
| Outer typed calls/delegates |672/1344|
| Typed affine calls/rows |636/636|
| Full probability calls/rows |10/45|
| Context route calls/rows |690/774|
| Unfitted reads |36|
| Recorder body seconds |90.11192544|
| Sampled recorder peak RSS bytes |106061824|
| Raw stage bytes at execution closeout |5537526|

All frozen22fits123rows/300seconds512MiB/stage32MiB/output24MiB and other native
caps passed. Source lifetime stayed240seconds; study body83.085197998seconds.
Native peak includes output serialization; source sampler peak81756160bytes
is not the outer process peak. Scalar chooser fit once on six completed real
rows is separate from the18native LS solves. Counts are API quantities, not
total FLOPs/SVD/copies/validation work.0Torch/GPU/modelrandomdraw/final/probe.
Boundary/post checks do not claim hard preemption of an in-flight native call.

Run command (physical D checkout; TEMP/TMP point to D:\Dev\_Temp):

```text
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/context-linear-catalog-study-20261029/native.py
```

This was the normal CLI with --method linear_fit --context-return
--separate-context-heads --catalog-study --output stage/actual.json. Do not rerun
the spent recorder or overwrite its marker/output. No further numerical grant.

## Focused checks, failures and review

Final pure cohort:22new+39live-catalog+24incident cases =85PASS/0FAIL/0SKIP on
Windows and Linux. Actual existing scalar trainer is exercised with scripted
ports: six rows/freeze/application, lost child return, corrupted prior evidence,
saved Perceptron dispatch/defaults, strict options and source-budget witness.
Literal LS state is a fixture, not observed source qualification. Native fit/
prediction/teacher/task counters were zero and Torch absent in these runs.
The observer's scripted counters do not cover the new fixture functions; they
must not be interpreted as zero scalar work. Source-derived completed scalar
rows14/two fits and literal monitoring callbacks6/24rows per final cohort.

Windows win-2 body3.4168312000037986seconds/80068608B; Linux linux-1
25.370151454seconds/81514496B. Preserved win-1:18PASS/1fixtureFAIL,4.4287801seconds/
79818752B. Mock budget omitted the required RSS limit; sampler correctly refused
it. Fixture declared512MiB; no guard weakening. Initial typing/unused import
failures were corrected, not ignored. One failed patch anchor was reread/fixed.

Exact pure commands (all output/basetemp paths on D):

```text
.venv/Scripts/python.exe -B artifacts/runs/context-linear-catalog-study-20261029/check.py artifacts/runs/context-linear-catalog-study-20261029/win-2-work.json tests/test_context_linear_catalog_study.py tests/test_context_linear_live_catalog.py tests/test_context_incident_serving.py --basetemp=D:/Dev/_Temp/circadian-lscatstudy-win2
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/context-linear-catalog-study-20261029/check.py artifacts/runs/context-linear-catalog-study-20261029/linux-1-work.json --capture=sys tests/test_context_linear_catalog_study.py tests/test_context_linear_live_catalog.py tests/test_context_incident_serving.py --basetemp=/mnt/d/Dev/_Temp/circadian-lscatstudy-linux1
```

Five-file Ruff check/format and mypy --explicit-package-bases
--follow-imports=silent (default and --platform linux) passed for the four changed
scripts and new test. Final repeated static body8.021894800003793seconds;
static RSS was not instrumented, so no hard peak claim. Exact commands:

```text
.venv/Scripts/python.exe -B -m ruff check scripts/context_catalog_learning.py scripts/context_head_live.py scripts/context_head_rehearsal.py scripts/recurring_ticket_workflow_example.py tests/test_context_linear_catalog_study.py
.venv/Scripts/python.exe -B -m ruff format --check scripts/context_catalog_learning.py scripts/context_head_live.py scripts/context_head_rehearsal.py scripts/recurring_ticket_workflow_example.py tests/test_context_linear_catalog_study.py
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent scripts/context_catalog_learning.py scripts/context_head_live.py scripts/context_head_rehearsal.py scripts/recurring_ticket_workflow_example.py tests/test_context_linear_catalog_study.py
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent --platform linux scripts/context_catalog_learning.py scripts/context_head_live.py scripts/context_head_rehearsal.py scripts/recurring_ticket_workflow_example.py tests/test_context_linear_catalog_study.py
```

Scoped git diff --check passed with inherited CRLF advisories; both new guides'
relative links resolved. No source edits followed the final pure cohort.
Self-review reconciled all six scalar task vectors, seven original actors,
continuations, source/RNG flags, aggregate child+source work and native caps.
Finite independent readonly source/protocol/visible-result review accepted;
4calls included two partially truncated outputs. Reviewer read aggregate and
selected trial results, not every middle field; source RNG enforcement was
reviewed, but its queried nonexistent rng_unchanged key was not result evidence.
Lead checked actual numpy_rng_unchanged/python_rng_unchanged=True. Review exception
ended, no reviewer imports/tests/models/writes. This is bounded review, not an
independent scientific replication or repository-wide acceptance.

Skipped: broader unchanged math/raw-backed/baseline suites, current wheel/clean
clone, Windows native duplicate, remote CI, GPU/physics/release and comparative
BP/PC/CPC reruns.279installed wheel predates current changes. No install/download/
global change/delete/commit/push. Inherited dirty work and failed evidence retained.

## Critical path and precise next action

R2.1 diagnostics/private fresh-task probes already exist (284/285). Repeated-update
and checkpoint infrastructure are not the next dependency. This study's chooser
can decline harmful work but cannot create a retention-preserving training arm.
The present replay-before-wake arm refits on the last narrow packet, overwriting
same-context old knowledge; the executed negative result makes that limitation
concrete. Existing TRAIN-only joint-packet rehearsal (286) is the relevant learner
capability to reuse. Adaptive serving also already exists; do not rebuild it.

Exact next implementation action: inspect the final wake-packet construction in
context_catalog_replay/context_head_live against existing prepare_joint_rehearsal,
then scope a separately tracked ordinary-TRAIN joint-packet action that keeps old
and newly arrived rows in the SAME final LS fit. Preserve existing arms/defaults
and this frozen study; do not revive287 or rerun295. Declare its distinct behavior,
costs and finite protocol before any numerical work. Subsequent acquisition,
retention and return-recovery claims still require their own credible evidence;
no CPC advantage, comparative efficiency or release claim is supported here.
