# R2.1-CTX.LS.M.P: isolated retained-task minimum-change probe

[#309](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/309) scoped
software implementation/evidence. Full goal/#21/#18/scientific/release requirements
remain open; #270 closed dependency. Accepted #307/#308 math/owner behavior reused.
Actual checkout HEADd9119e8 includes separately authorized publication bfab471/
d9119e8 (tracker6512786967); preserve it and unrelated local/ignored evidence.

## Delivered behavior

Existing CLI/API `probe_update="minimum_change"` restricted to ordinary owned LS
separate-context-head `retained_joint_probe`, without joint rehearsal. Exact option/
mode validation precedes source/model work at each public path. Omission retains
the old cold behavior/report shape. `ticket_linear_probe.MinimumChangeProbeLearner`
composes #308's named operation and preserves the choice through disposable forks.
`run_arm` binds the original source and stages a detached probe owner, reusing
existing FreshTaskProbe direct owned-learner F/return sequence, role guards and
clocks. AttributionStream remains the separate rehearsal path; this sequential
variant does not execute that path.
No numerical kernel, core, model schema, adapter or new orchestration change in
this increment. Source/retained/default fitting remains cold.

Affected script modules: `ticket_linear_probe.py`, `probe_ticket_policy_learning.py`,
`context_retained_probe.py`, `context_head_live.py`,
`recurring_ticket_workflow_example.py`. Paid outer/arm `linear_probe_update` intent
is pinned; successful F/return each require16 typed diagnostics with minimum-change
definition. Existing original consent/receipts/budgets/source/actor/donor/RNG guards
and prior-success admission remain. No source promotion/probe-to-TRAIN replay/final
data authority. Source lifetime120s at birth, retained child60s, both original
16-update30s512MiB F/return sessions stay, no renewal.

## Executed fixtures and limits

New `tests/test_context_minimum_change_probe.py`:22 cases including exact option/
mixed-mode/CLI forwarding and55/288 work metadata, paid-intent tamper/fork-isolation,
actual existing retained11fit72row fixture then32privatefits144rows, lost fit stop,
8 missing/empty/truncated/wrong-definition intent cases. The warm source is literal
metadata; not a trained/qualified native donor. No favorable endpoint required.
Both platform numerical fixtures observed old4→4→4 and selected1→2→2, all six
probe_test tasks per phase complete, no task-read training, counters23/144→39/192→
55/288, source/actor/donor/RNG unchanged, sessions original. This positive fixture
does NOT establish native retention/generalization, CPC benefit or release readiness.
Saved #303/#305/#306 native results/negative losses remain unchanged.

All commands physical `D:\Dev\Repositories\Circadian`, process TEMP/TMP
`D:\Dev\_Temp`. Existing WindowsPython3.14.7 and Linux3.12.3/NumPy1.26.4. No installs.

```powershell
$stage = 'artifacts/runs/context-minimum-change-probe-20261029'
$checks = @('tests/test_context_minimum_change_probe.py','tests/test_context_retained_probe.py','tests/test_context_retained_joint_probe.py')
.venv/Scripts/python.exe -B "$stage/check.py" "$stage/windows-pure-work.json" tests/test_context_minimum_change_probe.py -k 'not execute_retained and not lost_fit'
.venv/Scripts/python.exe -B "$stage/check.py" "$stage/windows-final-work.json" @checks -k 'not finish_retained_serving and not finish_actual_retained_stream'
.venv/Scripts/python.exe -B "$stage/check.py" "$stage/windows-final2-work.json" @checks -k 'not execute_retained and not finish_retained_serving and not finish_actual_retained_stream'
.venv/Scripts/python.exe -B "$stage/check.py" "$stage/windows-intent-work.json" @checks -k 'not execute_retained and not lost_fit and not finish_retained_serving and not finish_actual_retained_stream'
```

Linux prefix: `wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env
TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp
/home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B`.
`linux-final-work.json` uses the windows-final check list/filter; `linux-intent-work.json`
uses windows-intent check list/filter. Both add `--capture=sys`.
Observer adds `-q -ra --tb=short --maxfail=1`; exact argv saved in receipts.

| Receipt | Outcome | Core/real NumPy fits | Solver rows | Seconds | Peak RSS B |
|---|---|---:|---:|---:|---:|
| windows-pure |11PASS/1FAIL|0|0|1.808264300|73,134,080|
| windows-final |13PASS/1FAIL|44|219|8.870341200|82,460,672|
| windows-final2 |52PASS|1|3|2.964966900|85,594,112|
| linux-final |53PASS|44|219|29.426312559|83,263,488|
| windows-intent |59PASS|0|0|2.400849600|81,932,288|
| linux-intent |59PASS|0|0|22.954027776|81,752,064|

Initial failure: wrong new test writer module; correct existing module is
`context_catalog_learning`. Initial mypy caught it too. No production change.
Next failure: full retained/private fixture passed, lost-fit test expected wrong
error text. Runtime correctly refused the failed session via original paid-session
gate; corrected match,1fit3rows known paid and stopped, no next update. Completed
fixture never rerun on Windows. Both failed receipts retained, no model work erased.

Full Linux53 and Windows executed full fixture+corrected remaining52 preceded the
last intent-validator strengthening. Independent source review found `any(empty
diagnostics)` could pass vacuously; now exact16-list/count/dict/definition validation.
Final-tree59pure checks each platform cover that guard + old default/boundary/CLI
checks; numerical/private operation unchanged, no paid cohort repeated. One extra
Windows pure call (original3→4) prospectively justified6513040226, Linux2 within3;
original shared128fit768row caps unchanged. Actual89fits441rows including failed
and repeated lost fixtures,68.424762335s observed checks, max85,594,112B. No scientific
allowance or #287FAILED/SPENT retry. Each call120s512MiB; port solver/readout caps
fail closed before work, wall/RSS sampled/asserted at exit (not asynchronous kill).
Readout counts: full44fit cases42probability189rows/186affine186rows, lost-only
4probability18rows/36affine36rows; pure0. No Torch/modelrandom/neural/GPU/optimizer/
latent/final calls. Stage11files23,398B before final review, below4MiB.

```powershell
$files = @('scripts/ticket_linear_probe.py','scripts/probe_ticket_policy_learning.py','scripts/context_retained_probe.py','scripts/context_head_live.py','scripts/recurring_ticket_workflow_example.py','tests/test_context_minimum_change_probe.py')
.venv/Scripts/python.exe -B -m ruff check @files
.venv/Scripts/python.exe -B -m ruff format --check @files
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent @files
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent --platform linux @files
```

`statics-final3.json` all4exit0/8.757741800s. statics.json wrong test import failure,
statics-final.json passing before added intent cases, statics-final2.json inferred
heterogeneous invalid test-dict typing failure all retained. Only deliberately
heterogeneous test metadata annotated Any; no production type/validation weakening.
Final test annotation only after final pure run; runtime logic unchanged. Static RSS
not instrumented. Lead reviewed source/defaults/isolation/receipts. Independent
source/isolation/actual receipt/static/report review accepted the scoped evidence:
3 read-only commands0.7383843s, no project imports/tests/models/network/writes.
Diagnostic gap fixed, runtime wording clarified; finite exception ended. Not
independent experimental replication or native benefit/release acceptance.

Skipped broad unrelated families/CLM/Torch/simulator/GPU/remoteCI/clean-clone/wheel/
release/native-cohort checks. Two old numeric retained/joint fixtures excluded to
avoid repeated baseline work; old pure controls/default/failed-boundary/CLI retained.
Existing #279 wheel predates source. No download/install/delete/global edits/
commit/push in this work. Unrelated dirty/ignored/frozen/negative data preserved.
Scientific parents remain open; scoped source/procedure review is not replication.

## Precise next action

Freeze and independently review ONE Linux native sequential minimum-change variant
of the existing #303 protocol, reusing the saved cold control where exact source,
cohort, clocks/limits and task selection still match. Expected source12/72+
retained11/72+private32/144 =55fits288rows. Fresh conventional update rule is a
substantive changed intervention, not a retry of #287's spent joint study. Keep
all original outcomes; old/new/return endpoints may lose, tie or improve. Review
actual producer/limits and source/control applicability before any native launch,
no sweep/repeated control/favorable seed/metric selection. This software issue
grants no native allowance.
