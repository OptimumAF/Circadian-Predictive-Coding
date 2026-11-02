# Ordinary TRAIN CE task command — 2026-10-30

R2.1-TW.CE.TASK / [issue330](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/330)
implements `--context-return --method bp --train-readout ce` through the existing
runner. Previous #329 was PROGRESS. The accepted #321 known-task CE observation,
#320 fitter/predictor and prior R2.1/repeated-update/matched controls are reused.
No historical baseline, spent #323 adaptation experiment or native source12
case was rerun. Full repository/science/release requirements remain OPEN.

New files `scripts/ticket_ce_tasks.py`, `tests/test_ticket_ce_tasks.py`; narrow
changes to `scripts/recurring_ticket_workflow_example.py`; [usage](../ordinary-ce-tasks.md),
README/architecture and archival plan/log pointers. Physical D checkout,
HEAD d9119e80926a139a2f38460564d7eab1b6d2a2b0 with inherited dirty work preserved.
No dependency, global settings, install, download, commit, push or delete.

Delivered: strict opt-in admission before models/data/output effects; original
source12/72 stream/checkpoints/24 development episodes and defaults; one
separate fixed TRAIN CE fit and frozen reader; all four existing known tasks,
full allowed actions/six-step bound, no accuracy-based selection or early stop.
Source/seed/actor/optimizer/RNG authority remains with the original runner.
The helper pins actual source and fit, original inbox receipt identities plus
copied public view semantics, source and auxiliary budget identities/births,
canonical paid row/container identities and final auxiliary budget record.
It returns a retained guard checked after `stream.finish()` and after final
report/sampler callbacks. Late faults preserve canonical paid records and refuse
root completion; foreign mutations are not restored. The CLI fallback retains
separate original and auxiliary work/budget records.

No source/actor weights are installed or promoted. No probe, final, replay,
source SGD or PC/LS/LP work occurs in these software fixtures. Runtime auxiliary
15s512MiB and original source30s512MiB clocks overlap and never renew.

## Executed checks and failures

Stage `artifacts/runs/ordinary-ce-tasks-20261030/` retains every receipt,
`check.py`, `statics.py`, `summary.json` and final Windows log. All project and
TEMP/TMP outputs are D. Linux TEMP/TMP/TMPDIR is `/mnt/d/Dev/_Temp`.
Existing Windows Python3.14.7/NumPy2.4.6/Torch2.14CPU and Linux Python3.12.3/
NumPy1.26.4/Torch2.14CPU runtimes used; CPU threads1, pytest plugin autoload off.

The actual composition fixture uses a native D9/H3/K6 learner with **synthetic
valid12/72 counters, zero gradients/momentum and scripted receipts**. Those
numbers are test inputs, not executed source training or scientific evidence.
It performs the real existing CE fitter, frozen typed predictor and existing
four-task loop, and verifies source/NumPy/Torch RNG preservation and separate
work. Other guard faults use a scripted fitter/readout and are not counted as
physical gradients merely because their test reports contain simulated charges.

| Receipt | Outcome | Seconds | Peak RSS bytes |
|---|---|---:|---:|
| windows-first-work | 34 pass, 1 setup error | 5.173775900 | 330,256,384 |
| windows-second-work | 34 pass, 1 failed assertion | 4.867625500 | 334,012,416 |
| windows-third-work | 53 pass | 4.812435100 | 333,053,952 |
| windows-combined-work | 83 pass | 5.688681800 | 331,550,720 |
| windows-guard-work | 17 pass | 5.204862500 | 330,780,672 |
| windows-final-work | 66 pass | 4.381683600 | 327,929,856 |
| linux-combined-work | collection/capture failure, no models | 7.599284316 | 266,915,840 |
| linux-final-work | 89 pass | 10.722510446 | 349,749,248 |

**89 distinct cases qualify per platform**: 59 new +30 affected existing CLI/
mode/default/refusal cases. Windows final guard17 and metadata/default/negative/
fit-failure66 are on the final tree; six unchanged source-admission cases reuse
the prior combined passing evidence. No repeated full numeric cohort solely to
make one larger green count. All final checks have zero skips/failures.

Initial fixture error applied `zeros_like` to snapshot metadata; fixed to use
parameter keys. Second assertion used a tensor-only equality function on NumPy
RNG arrays; fixed to ordinary tuple/array equality. Its already-completed real
fit and task reads remain paid. Linux fd capture failed on the D mount before
model construction; disabled capture with `-s`, preserving the failure and its
time. Expected parser refusals appear in passing test output. Two patch attempts
missed formatted anchors and made no edits; reapplied against actual source.

Initial Ruff/format passed but both type checks found seven typing errors.
Corrected reader typing, snapshot optional-state assertions and NumPy tuple
narrowing. Final three-file Ruff/format/default and Linux-target mypy all PASS;
two static batches66.292117s/120s, initial failure retained in statics-first.json.

Commands (exclusive receipts are already consumed; do not automatically rerun):

```powershell
.venv/Scripts/python.exe -B artifacts/runs/ordinary-ce-tasks-20261030/check.py artifacts/runs/ordinary-ce-tasks-20261030/windows-third-work.json tests/test_ticket_ce_tasks.py
.venv/Scripts/python.exe -B artifacts/runs/ordinary-ce-tasks-20261030/statics.py statics-final.json
```

Each `*-work.json` includes the **exact complete command argument list** for its
selectors; `windows-final.log` preserves the final Windows output. Final Linux
used `wsl.exe bash -lc` with physical `/mnt/d/Dev/Repositories/Circadian`, D TEMP
settings and `/home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B`,
then the same checker and selectors recorded in linux-final-work.json. The final
cohort is the new test file plus mode/CLI-only selections from
test_ticket_ce_probe.py and test_ordinary_native_reserved_probe.py. No existing
native numerical cohort or source-training fixture was repeated.

## Costs, review and remaining scope

All eight software invocations48.450859162s/shared300s, each120s768MiB.
Windows cumulative128component constructions/256 named seeded draws5760values,
4CE optimizer steps/140 closure forwards+backwards1680 TRAIN gradient-row
exposures,4final TRAIN reads48rows,52hidden reads96rows/48fitted affine reads48rows/
48typed predictions. Linux53components106draws2385values,1CE optimizer step/
35closures420gradientrows,1finalTRAIN read12rows,13hidden reads24rows/
12affine reads12rows/12typed predictions. All limits held; Windows128/128
component and256/256 draw allowances are spent, not renewed. Counts include
failed attempts; see summary.json and original per-invocation limits/receipts.

Zero source SGD/backward, PC latent, LS/LP/SVD/autograd.grad, random mutation,
probe/replay/final, physics, GPU or remote experiment work. Teacher preparation
and task dispatch use the original pure sandbox: code-scheduled accounting is
seven24-dispatch teacher preparations +48 typed steps on Windows (216), and
two24 +12 on Linux (60), below384/platform. These dispatch totals are derived
from the fixed paths and recorded test execution, not separately instrumented
process-wide counters. Actual typed calls and model primitives are instrumented.

Independent review initially REQUESTED CHANGES: the helper discarded its guard
before source finish, allowing a late clock callback to corrupt successful
auxiliary evidence. Original review/failure preserved. Fixed with retained
guard/canonical budget record; six no-fit late-clock cases cover row values/
owner, work owner, episode owner, budget value/owner. One prospectively bounded
read-only addendum accepted the final source/recorded results. Four commands,
all EXIT0, .6085621s, no edits/imports/models/tests/network/children, exception
ended. Lead source/procedure/results self-review is separate; cumulative fixture
aggregation is lead-verified, not an independent whole-repository audit.

Skipped: genuine native source12 first-party invocation, new adaptation case,
full suite, clean clone, current wheel/build/install, browser/device, simulator,
GPU, statistical/comparative/matched-compute/generalization and release checks.
Source implementation correctness and tiny composed execution do not establish
native task success or adaptation/recovery. #321's actual four-known-task success
remains accepted under its original assumptions. #323 remains FAILED/SPENT;
new mode is old-TRAIN task execution without F/return, not a retry of that case.

Next concrete action: freeze **one first-party known-task command qualification**
through this new option, using unchanged original source12/72, existing fixed CE
fitter and all four known tasks, original source/aux clocks and paid work. Inspect
the concrete producer/protocol before execution; no source-loader or monkeypatch
handoff is needed by users. This would qualify the new product path, not supply
new comparative science or renew #323. Reuse existing baseline results and admit
no seed/dose/objective/metric selection. Broader R2.1/R4/R5/R7/science/release/full
goal and parent issues18/21/23 remain OPEN. No further inspector/capture sequel.

Closeout receipts:330 canonical6532522774, closed/completed scoped software;
tracker18 6532530679, parent21 6532531123, parent23 6532531477 remain OPEN.
Final retained stage54,699B before closeout JSON, below8MiB. Scoped tracked and
new-file whitespace checks PASS, LF/CRLF advisories only.
