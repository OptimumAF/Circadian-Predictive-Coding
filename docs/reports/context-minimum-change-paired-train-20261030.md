# Ordinary paired-joint minimum-change TRAIN

R2.1-CTX.LS.M.PAIR / [#313](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/313).
Scoped software qualification, not a new comparative experiment. HEAD d9119e8
plus accepted dirty308/309/311 and this change; unrelated work preserved.

The public existing ticket workflow now accepts `--train-update minimum_change`
with `--paired-serving joint`. `scripts/recurring_ticket_workflow_example.py`
admits that exact combination; `scripts/context_head_live.py` forwards it to the
existing counted LS helper and rejects other child compositions before source
access. No new fitter, recipe, orchestration or dependency. Defaults remain cold.
Shared paid-intent guards, state/fork validation, source/actor locks, sharing,
receipts, permanent attempted work and original clocks/budgets remain in force.
Paired roles/data validation and publication/rollback mechanisms are unchanged.

Why this: #312's failed v2 legacy task was untrained, with the v2 head unchanged
throughout the nonpaired child. The existing paired recipe already teaches both
v1 parcel/v2 password and uses matching `PAIRED_TASKS`, alongside four old tasks.
Accepted #270's paired Perceptron correctness evidence is reused, not rerun;
it is not a matched LS control. #287 FAILED/SPENT and every negative capture remain.

The actual ordinary software fixture on each platform paid source12fits72rows
and child6fits48rows, using the existing six packets9,9,9,9,6,6 with delayed labels.
Existing guard accepted at first learning boundary tick10: candidate new6/6
inner rows, old12/12; source new inner rows Windows4/6, Linux5/6, old12/12.
Published closed-loop tasks were6/6 on both platforms; rollback restored the
source at generation2, Windows6/6 tasks and Linux5/6 (v2 password failed).
The guard row metric and closed-loop success are different measurements.
Original source30s and child60s limits, source/actor state and both RNGs were
preserved; no final/probe/replay examples. This establishes scoped software
behavior in the declared known development tasks, not broad acquisition or
retention benefit. Windows source already solved all six closed-loop tasks;
it does not demonstrate new-task acquisition. Linux's returned source failure
stays visible. No favorable threshold, action, seed or metric was selected.

The lost-first-child fixture performed source12/72 plus one real9-row fit then
raised before returning. Child attempted1/9, returned0, budget committed0,
no applied receipt or promotion; source/actor unchanged. The attempted native
fit is counted despite the absent successful receipt. No retry or implicit drain.

## Executed checks

From physical `D:\Dev\Repositories\Circadian`, TEMP/TMP=`D:\Dev\_Temp`:

```powershell
$stage='artifacts/runs/context-minimum-change-paired-train-20261030'
$checks=@('tests/test_context_minimum_change_paired_train.py','tests/test_context_minimum_change_train.py')
$filter='not counted_fork and not stop_existing and not execute_source and not preserve_paid_source'
.venv/Scripts/python.exe -B "$stage/check.py" "$stage/windows-first-work.json" @checks -k $filter --basetemp=D:/Dev/_Temp/circadian-pair-313-windows
.venv/Scripts/python.exe -B "$stage/check.py" "$stage/windows-final-work.json" @checks -k $filter --basetemp=D:/Dev/_Temp/circadian-pair-313-windows-final
.venv/Scripts/python.exe -B "$stage/check.py" "$stage/windows-cli-work.json" tests/test_context_minimum_change_paired_train.py -k forward_paired_cli --basetemp=D:/Dev/_Temp/circadian-pair-313-windows-cli
```

Linux prefix `wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env
TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp
/home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B`.
Same final and CLI check lists/filter, `linux-final-work.json`/`linux-cli-work.json`,
`--capture=sys` and `/mnt/d/Dev/_Temp/circadian-pair-313-linux-final`/`-cli` basetemp.
Observer adds `-q -ra --tb=short --maxfail=1`; receipts record exact argv.

| Receipt | Outcome | Shared solves / actual NumPy | Rows | Seconds | Peak RSS B |
|---|---|---:|---:|---:|---:|
| windows-first |38PASS1FAIL|18/18|120|7.093323800|80,773,120|
| windows-final |64PASS|31/31|201|6.640939900|85,704,704|
| linux-final |64PASS|31/31|201|21.442625266|84,123,648|
| windows-cli |2PASS|0/0|0|1.671078400|73,072,640|
| linux-cli |2PASS|0/0|0|9.067800728|77,709,312|

64 distinct focused cases qualify per platform; two CLI cases repeated after
their mock implementation changed, without model work. Pure accepted311 cases
cover unchanged selection/paid intent; its numeric math/fork/retained and
unchanged270 packet/role suites are reused, not retrained. New tests cover early
unsupported public/owned/child combinations, default and selected CLI forwarding,
full paired integration and a lost paid update.

All five invocations total80shared solves522rows,80actual NumPy/522,
45.915768094s, max85,704,704B. Readouts:50probabilitycalls324rows,
686affinecalls686rows. Under shared prospective128/896,4096calls65536rows,
120s512MiB each and4MiB stage; Windows3calls/Linux2. Zero model random, Torch,
neural/optimizer/gradient/latent/GPU/simulator/private probes/final/remote work.
Process sampled memory/time, not a process-tree limit or exact heap accounting.
The observer enforces forward caps per invocation, unlike cumulative solve/row
caps. Observed aggregate736forwardcalls1,010rows is below the shared declared
cap; future protocols must preserve this enforcement/observed distinction.

Initial failure: source-work assertion omitted existing
`budget_committed_updates=12` and `unreturned_work_unknown=False`. Also corrected
unexecuted tick14 expectation copied from the older Perceptron case to the
actual recipe first learning boundary10. Full failed receipt/work retained;
production timing/guards unchanged. Correction prospectively recorded6521355990.
Static first run Ruff/format passed; both mypy targets failed test-only lambda
using `list.append` as a value. Replaced with explicit function, two affected CLI
cases/platform passed. `statics.json` retains both errors. `statics-final.json`:
Ruff check/format and mypy default/Linux all exit0,9.656118300s. Commands:

```powershell
$files=@('scripts/context_head_live.py','scripts/recurring_ticket_workflow_example.py','tests/test_context_minimum_change_paired_train.py')
.venv/Scripts/python.exe -B -m ruff check @files
.venv/Scripts/python.exe -B -m ruff format --check @files
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent @files
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent --platform linux @files
```

Initial statics took14.253153600s; static RSS was not measured. One lead's
read-only metadata projection failed before execution with an unterminated
Python expression; simplified projection then confirmed totals and new-file
whitespace. No model imports/work or process cost receipt for that failed helper.
Scoped `git diff --check` passed on the two scripts and usage guide.

Lead source/procedure/results self-review completed. Finite independent review
accepted actual combined source, shared helper, tests, state/publication/rollback
and recorded results without blocking findings: four read-only commands,
0.929152s aggregate command wall, no writes/imports/tests/models/network/children.
`review.txt` preserves the handoff; exception ended. Acceptance covers this
software scope, not native/scientific/release claims.
Final local stage9files30,525B, below4MiB. Final changed-file whitespace check
passed, with Git LF/CRLF conversion advisories; inherited dirty files retained.
No full suite/clean clone/installed wheel/browser/neural/robotics/comparative
study or native qualification: no corresponding allowance and no new risk
requiring unchanged broad suites. Current installed wheel predates this choice.
No commit/push/install/download/delete. Local artifacts do not imply publication.

Next freeze ONE native Linux ordinary paired-joint18fit120row development case
with before/published/return/rollback task traces, original limits and complete
costs, after this software/state review. Freeze all six task slots, failure policy
and interpretation before launch; do not retry #287 or train a new control under
this allowance. Reliable broader adaptation, CPC benefit, total-compute
comparison, research/release readiness, #21/#18 and full goal remain open.
