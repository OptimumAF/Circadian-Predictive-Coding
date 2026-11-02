# R2.1-CTX.LS.M.H: contextual explicit minimum-change owner

Local scope [#308](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/308).
Full goal/#21/#18/research/release remain open; #270 is a closed scoped dependency.
Reuse #307 shared mathematics/generic owner and frozen #305/#306 native results.
No native cohort, source training, final evaluation, baseline tuning or #287
FAILED/SPENT retry. Parent/native benefit is not accepted by these software checks.

## Delivered behavior and evidence

`src/adapters/context_linear_fit_learner.py` exposes the explicitly named nonsticky
`train_batch_minimum_change`. Existing homogeneous public routing delegates one
selected private head to #307, then atomically replaces the validated whole state.
Cold defaults, numerical kernel, policy/snapshot fields, counters and no-history
storage remain. Strong raw owner references plus identity/value/input/RNG checks
refuse equal-valued nested replacement without ID recycling; foreign changes are
not restored away. Distinct contextual diagnostic and private/whole commit counts
retain attempted/returned/lost work. Roles, exclusive lease, consent and budgets
remain caller responsibilities. No production probe/CLI update mode added.

New `tests/test_context_linear_minimum_change.py`:34 cases, both-head orthogonal
old/new/return, inactive preservation each update, mixed inference, counters/forks,
unfitted equivalence/cold nonsticky,128-row prior/current, routing/label/counter
refusal, active/inactive/raw/target/policy/state/staged-prior/RNG/lost-return faults,
equal nested replacement on both paths, existing delayed TRAIN composition with
source/actor isolation/causality/nonTRAIN refusal/permanent stop after lost work.
Literal warm metadata is not hidden training. Fault RNG changes restored. Reuse
29 applicable old contextual/reader/CLI dispatch checks and2 generic math/owner
checks,65 combined. Test-only subclass selects the operation; no new orchestration.

## Exact checks, failure and limits

All project commands ran at physical `D:\Dev\Repositories\Circadian` with process
TEMP/TMP `D:\Dev\_Temp`; Linux corresponding `/mnt/d` paths/TMPDIR. Windows
Python3.14.7; existing Linux3.12.3/NumPy1.26.4, no installation.

```powershell
$stage = 'artifacts/runs/context-linear-minimum-change-owner-20261029'
$checks = @('tests/test_context_linear_minimum_change.py','tests/test_context_linear_heads.py','tests/test_linear_minimum_change_owner.py::test_should_preserve_invisible_prior_and_match_independent_rank_one_oracle','tests/test_linear_minimum_change_owner.py::test_should_keep_unfitted_equivalence_forks_and_default_operation_nonsticky')
.venv/Scripts/python.exe -B "$stage/check.py" "$stage/windows-pure-work.json" tests/test_context_linear_minimum_change.py -k 'invalid_route or nontrain_roles'
.venv/Scripts/python.exe -B "$stage/check.py" "$stage/windows-final-work.json" @checks -k 'not execute_existing_context_stream and not two_head_failed_fit'
.venv/Scripts/python.exe -B "$stage/check.py" "$stage/windows-final2-work.json" @checks -k 'not execute_existing_context_stream and not two_head_failed_fit'
.venv/Scripts/python.exe -B "$stage/check.py" "$stage/windows-owners-work.json" tests/test_context_linear_minimum_change.py -k 'equal_valued or changed_state'
```

Linux used `wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env
TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp
/home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B`
with the same check.py/checks/filter, `linux-final-work.json`, `--capture=sys`.
Observer adds `-q -ra --tb=short --maxfail=1`; full exact argv in each receipt.

| Receipt | Outcome | Core/actual NumPy attempts | Rows | Seconds | Peak RSS B |
|---|---|---:|---:|---:|---:|
| windows-pure |11PASS|0|0|1.206272000|66,088,960|
| windows-final |29PASS/1FAIL|35|162|2.082585600|78,127,104|
| windows-final2 |65PASS|51|194|2.083908100|78,540,800|
| windows-owners |18PASS|18|18|1.210508100|69,140,480|
| linux-final |65PASS|51|194|15.879334229|75,866,112|

First combined test incorrectly asked for `candidate_snapshot()` after `finish()`;
existing runtime correctly refused. Fix reads candidate before finish. Failed
solver work retained/charged. Windows65PASS preceded the final stronger raw-reference
pin; final-tree Windows18PASS covers all changed owner/failure paths and Linux65PASS
is the full final combined tree. No source/test changes after those final checks.

Original prospective shared160solve attempts768rows held; actual155/568 including
failure/repeats,22.462608029s summed observed test time, max78,540,800B. Core and real
NumPy counts equal here. Original <=3pytest calls/platform was prospectively amended
to exactly one additional Windows focused call after reviewer found id-only pins
could recycle; original failed records and solver/wall/RSS/storage caps preserved.
Issue comment6512895589 records rationale before executing, not scientific renewal.
Per invocation120s512MiB, readout4096calls65536rows; observed full final each has
7 probability calls17rows/5affine calls6rows, failed run2/4, others0. Model-port
limits fail closed before delegation; wall/RSS sampled and asserted at exit, not
asynchronously terminated. Torch guard remained clear; no neural/GPU/latent/random
model work. Stage well below4MiB; retained failed and all successful records.

```powershell
$files = @('src/adapters/context_linear_fit_learner.py','tests/test_context_linear_minimum_change.py')
.venv/Scripts/python.exe -B -m ruff check @files
.venv/Scripts/python.exe -B -m ruff format --check @files
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent @files
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent --platform linux @files
```

Final `statics-final.json`: all4exit0,8.515835900s (static RSS uninstrumented).
`preliminary-statics.json` preserves initial test-only typing errors: pair inferred
variadic tuple and starred untyped packet; corrected explicit pair/tuple annotation,
no weakened production type/validation. Earlier statics.json all0 retained.
Lead self-reviewed full patch, state/default compatibility, receipts and scope.
Finite independent source/procedure review found missing active fault coverage and
ID-lifetime caveat; both fixed before final checks. Final independent source/actual
receipt/static/report review accepted scoped software evidence:3 read-only commands,
0.6594572s command wall, no project imports/tests/models/network/writes. Exception
ended. This is not independent experimental replication or native efficacy.

Skipped broad unrelated families/CLM/Torch/GPU/native/simulator/remoteCI/clean-clone/
wheel/release tests; unchanged #307 mathematics/wide limits reused. Two old context
numeric CLI fixture tests excluded to avoid repeated native recipe work; cold owner,
typed readers and CLI dispatch/refusal included. #279 installed wheel predates this
code. No installs/downloads/delete/global edits/Git commit/push. Unrelated dirty/
ignored/frozen/failed captures preserved. #285/#305/#306 loss results and #287
FAILED/SPENT unchanged. This is software acceptance, not experimental replication.

## Exact next implementation action

Integrate an explicit per-call minimum-change selection into the existing isolated
contextual fresh-task probe composition, retaining cold default and independent
F/A/return forks, original counters/roles/clocks/budgets and no final/probe promotion.
Inspect existing application/CLI options first; add only a small option through
existing components, then qualify with software fixtures before freezing any new
native retention variant. Do not rerun old baselines or #287 failed study. Native
retention comparison requires a separate reviewed prospective protocol.
