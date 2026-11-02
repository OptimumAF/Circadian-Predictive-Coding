# Optional once-per-packet old return — #359 software evidence

R5.3-LIN.JOINT.RETURN implements a separately explicit `--probe-return-once` option
for the existing ordinary shared-head joint/current_only isolated fresh/return
workflow. It removes only the duplicate old TRAIN packet on return; it does not
establish better retention. [Canonical scope](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/359).
Full goal and parents18/21/23/24 remain open. Physical checkout HEAD d9119e8,
inherited dirty work preserved; no Git publication/dependency/global/delete work.

## Why this change and what works

Accepted358 showed old-task current3→2→3 versus joint3→4→3, fresh already2/2.
Both finally failed v1invoice with the wrong inspect→commit workflow. Original
data hashes prove each old return equals one retained TRAIN packet, so the default
18-row joint return gives the current context twice and the other once. This
declared exposure remains the original default/protocol/result, not an accounting
bug to rewrite. The new15-row F/12-row return recipe is a distinct hypothesis.

`src/core/shared_linear_replay.py` accepts exact-bool current_once. It first
validates ALL original data, labels, contradictions and128-row bound. It requires
exactly one full feature/target match among exactly two retained packets, then
removes only that duplicate. Other packet, current data and owned outputs remain;
current_only validates the same information and fits equal rows. No kernel/score/
tie/tolerance changes or task-score selection, no new runner/framework.

`scripts/ticket_linear_rehearsal.py` opts in with fresh separate32/288authority:
16F×12extra +16return×6extra. Original source and F/return clocks/consent/receipts
and budget limits stay; no old384session is renewed. Admissions/expected extra
rows are paid before composition; a nonmatching return charges1/6 but does not
fit or continue. Unknown post-state remains unknown, failed work permanent. Public
counter changes require observed owned deltas; changed recipe/data/evidence refuse.
Receipt definition and optional metadata distinguish the new route, bounded32events.
SourceTrainPackets admission/ownership/arrived TRAIN provenance is unchanged.

`scripts/ticket_native_probe.py` and `scripts/recurring_ticket_workflow_example.py`
bind the explicit flag through the existing callback. Original intent presence/
exact bool are guarded before further work; unsupported API/CLI modes refuse
before data/output creation. No default keyword/metadata is emitted. The source
grants original replay consent at its birth. Only detached private owners fit;
probe_train stays transient/private, final/probe_test/diagnostic-monitor labels
never authorize fitting/replay/source-bank storage. Existing summary projects
actual288extra rows and32applications without new training or fake SGD credit.
[Usage](../shared-linear-probe-rehearsal.md).

## Executed evidence

Stage `artifacts/runs/shared-linear-once-return-20261031/` contains owned preimages/
diffs, exclusive software attempt markers, logs, receipts, totals and statics.

| Cohort | Result | Body seconds | Peak sampled RSS bytes | Real LS / rows | Scripted / rows |
|---|---|---:|---:|---:|---:|
| windows | 69 pass,2 fail | 6.788729900 | 83423232 | 8 /120 |337 /2844 |
| windows-affected |2 pass |1.047746800 |72871936 |0 /0 |24 /144 |
| linux |71 pass |17.194943539 |76685312 |8 /120 |337 /2844 |

71distinct cases qualify per platform:40new and31changed-default/legacy checks.
Windows uses unaffected passing evidence plus the two affected reruns; Linux
executes the final combined cohort. Whole ordinary fitting/task readouts are
SCRIPTED: real owners/stream/consent/receipts/controls/budgets, not learned outcomes.
Only tiny two-step callbacks use real LS: new15/12 and original15/18 controls.
All16actual fits/240rows and48scheduledproducts/720row visits are paid; the16-fit/
48-product correctness caps are consumed. Remaining row capacity grants no fits.
Generic prediction192calls/864rows,0typed logits; scripted698calls/5832rows.
Actual deterministic teacher data generation36builders/144traces/432sandbox
dispatches is counted, not free; no autonomous learned/simulator/physics episodes.
Measured costs include metadata/validation/copies; counters are not complete heap/
FLOPs. No scientific model case, random draw, neural/GPU/final evaluation.

Two initial failures incorrectly expected unknown probe cost after the original
intent guard had refused before ANY private owner/fit. Kept production unchanged;
strengthened tests require known0/noauxiliary and exact12source scripted calls/
72rows. Actual unreadable private application already passes with unknown1 and
permanent6-row charge. These distinctions preserve uncertainty without pretending
that an observed pre-admission refusal performed hidden fitting. All failures
and charges remain; no criteria or desired outcome weakened.

Scoped six-file Ruff/format/Linux-target mypy pass. Initial missing row annotation
failed types and was fixed before tests; initial receipt retained. Static RSS not
sampled. Test limits declared prospectively: <=3calls/platform/90s256MiB, shared
16LS256rows/48products768rows,2048scripted16384rows,512prediction4096rows;
64teacherbuilders256traces768dispatches. No cap renewal. Each invocation also
had a60s watchdog, all three processes terminal. Total software-body25.031420239s.
Dedicated cache initially0 under64MiB, stage under4MiB excluding cache.

Commands from physical D:cwd with TEMP/TMP=`D:\Dev\_Temp`:

```text
.venv/Scripts/python.exe -B artifacts/runs/shared-linear-once-return-20261031/check.py windows D:\Dev\_Temp\circadian-once-return-win-20261031
.venv/Scripts/python.exe -B artifacts/runs/shared-linear-once-return-20261031/check.py windows-affected D:\Dev\_Temp\circadian-once-return-win-affected-20261031 tests/test_shared_linear_once_return.py::test_original_once_intent_corruption_refuses_before_private_fitting
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/shared-linear-once-return-20261031/check.py linux /mnt/d/Dev/_Temp/circadian-once-return-linux-20261031
```

Exact expanded nodes in each *-checks.json; full roster in stage/check.py.
`.venv/Scripts/ruff.exe format`, `ruff.exe check`,
`mypy.exe --platform linux --follow-imports skip` on six exact paths in
statics-final.json; dedicated stage/cache RUFF_CACHE_DIR/MYPY_CACHE_DIR.
Observer charges real and scripted/helper admissions before delegation. It adapts
the existing bounded observer, not another experiment orchestration framework.

Skipped: full suite/clean clone/current wheel/build/install, unchanged mathematical/
matched BP/PC/neutral/active CPC/native suites, full genuine ordinary comparison,
GPU/physics/browser/remote CI/release, independent numerical replication. Reuse
accepted kernels/runner/357/358 evidence within unchanged assumptions. Nothing
establishes usefulness of this new recipe, or full repository readiness.

## Review and next gate

Lead self-review inspected actual source/owned diffs, original defaults, authority,
row/exposure/unknown accounting and all executed results. Independent scoped actual
source/evidence review ACCEPTED:4readonly commands, all exit0,0.7451185s. All four
owned diffs matched current source, full new tests/observer/receipts and usage/report
inspected;71distinct/platform verified. No imports/tests/models/network/children
or numerical replication/full native/usefulness acceptance. Finite exception ended.
Its nonblocking doc note was fixed: subsequent384/600authority/work paragraphs
explicitly describe the original default, avoiding confusion with the new288/504.
No source/test change or numeric repeat for that wording correction.
Any later native comparison needs a separate small frozen reviewed protocol and
new originally consented source owners: expected44fits504rows/arm, private32/432,
not the old600-row matched control. No scientific allowance under359. Preserve
358's transient benefit/final failure and every previous FAILED/SPENT case; no
seed/dose/score/tie tuning or large sweep to manufacture a circadian win.
