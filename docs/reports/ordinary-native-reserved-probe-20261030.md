# Ordinary four-family workflow to isolated reserved probe

R2.1-TW.P / [#315](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/315).
Implemented `--reserved-probe` for the existing ordinary context-return workflow
with BP, PC, neutral CPC and active/gated CPC.50distinct focused cases qualify
per platform, final scoped statics pass; independent review pending.
This delivers the handoff and its isolation/accounting, not useful task learning:
all six typed tasks failed before F, after F and after return for every method
on both platforms. Preserve that negative result, with no tuning or advantage claim.

Actual checkout HEADd9119e8 plus inherited accepted dirty work308/309/311/313;
#314 native LS acquisition/retention/return evidence remains unchanged. Existing
R2.1 diagnostics/probe/return and four-family pilot already existed. Inspection
found ordinary `private_adaptation` restricted to Perceptron heads; the historical
BP/heads probe path did not provide this live ordinary four-family handoff.

## Behavior and boundaries

`scripts/recurring_ticket_workflow_example.py` adds one narrow CLI/API opt-in,
rejecting other control/replay/diagnostic/publication modes before models/data.
Its original source12updates72rows and all checkpoints/task readouts stay intact.
`scripts/ticket_native_probe.py` validates the actual live checkpoint12 geometry,
method/seed/policy, complete source receipt count, source sampler/budget and
original callback before building reserved data or models. It uses the existing
paired cohort builder: new v1parcel/v2password, four old v1invoice/password and
v2invoice/parcel tasks. These are known TRAIN-derived templates, not unseen data.

Why this: source counted forks inherit source callbacks. A detached native owner
restored from the complete qualified state instead has independently paid probe
callbacks. Existing `run_fresh_task_probe` performs16Fupdates48rows then16oldTRAIN
return updates96rows on disposable descendants. Source/actor never receive those
updates. Source budget stays12/30s/512MiB; F/return each16/30s/512MiB born once
at admission. Return clock includes F time. Separate counters preserve attempts
even on lost returns; root `probe_updates` is null for uncertain work, not zero.
Reusing a populated completed/failed report is refused without resetting it.

All four methods retain original D9/K6, hidden3, seed17, learning rate .03,
inference2 and their existing method configurations. No learner/optimizer/math,
capacity, label, confidence, seed, metric, tie-break or controller tuning.
Probe training sees only `probe_train`; `probe_test` labels only score. Old TRAIN
monitors do not update during F, separate old TRAIN templates train return.
Final evaluation unavailable, no promotion/replay or deployed/benchmark change.
Six typed episodes at each fixed before/after/return phase use existing frozen
native readers/evaluator; paid native readouts and source guards remain active.

Completed invocation costs source12/72 +F16/48 +return16/96 =44updates216rows.
Root costs cover source only; nested `reserved_probe` owns probe/return, input/
categorical diagnostics and18typed task readouts. All methods passed source,
actor, NumPy/Torch RNG, source budget/counters, full private state/fork and task
count28/44 checks. Typed task success remains0/6atallthreephases, distinct from
loss/accuracy diagnostics. No reliable acquisition/retention/recovery conclusion.

Deliberately lost first probe update performs one real3-row update then loses
its return: attempted1/3, returned0, F/return budget commits0/0, rootunknown/null,
no return work. Original source still12/72, source/actor verified unchanged.
Replacing the source budget with a value-equal new object after cohort building
is refused before any probe model/update, without restoring that foreign object.
Root source/actor flags stay unknown on that integrity failure.

## Executed checks and costs

Physical cwd `D:\Dev\Repositories\Circadian`; TEMP/TMP=`D:\Dev\_Temp`.

```powershell
$stage='artifacts/runs/ordinary-native-reserved-probe-20261030'
$test='tests/test_ordinary_native_reserved_probe.py'
.venv/Scripts/python.exe -B "$stage/check.py" "$stage/windows-first-work.json" $test --basetemp=D:/Dev/_Temp/circadian-native-probe-315-windows
.venv/Scripts/python.exe -B "$stage/check.py" "$stage/windows-second-work.json" $test --basetemp=D:/Dev/_Temp/circadian-native-probe-315-windows-second
.venv/Scripts/python.exe -B "$stage/check.py" "$stage/windows-final-work.json" $test --basetemp=D:/Dev/_Temp/circadian-native-probe-315-windows-final
.venv/Scripts/python.exe -B "$stage/check.py" "$stage/windows-row-work.json" $test -k paid_report_reuse --basetemp=D:/Dev/_Temp/circadian-native-probe-315-windows-row
```

Linux prefix `wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env
TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp
/home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B`;
same check.py/test, `linux-final-work.json`, `--capture=sys`,
`--basetemp=/mnt/d/Dev/_Temp/circadian-native-probe-315-linux-final`.
Observer adds `-q -ra --tb=short --maxfail=1`, fixes CPU thread env1; exactargv in
receipts. Existing Torch2.14.0+cpu both; NumPyWindows2.4.6/Linux1.26.4.

| Receipt | Outcome | Native updates | Rows | Seconds | Peak RSS B |
|---|---|---:|---:|---:|---:|
| windows-first |23PASS1FAIL|0|0|3.486997200|245,567,488|
| windows-second |34PASS1FAIL|44|216|15.462179600|335,147,008|
| windows-final |44PASS|201|1011|87.024912400|348,188,672|
| windows-row |2PASS|0|0|2.860647500|243,961,856|
| linux-final |46PASS|201|1011|66.229691073|366,579,712|
| windows-endpoint |4PASS|0|0|2.521490200|243,777,536|
| linux-endpoint |4PASS|0|0|8.742174926|263,905,280|

Windows44combined plus2pure fresh-row and4endpoint cases qualify50distinct,
Linux46combined plus4endpoint cases. Late fresh-row guard rejects populated rows before the unchanged valid
empty-row path; no paid Windows cohort repeated afterward. Final list annotation
fix changes no runtime semantics; numerical results reused.0skips.

Shared all attempts446updates2238rows,186.328092899s/max366,579,712B, under480/2560,
180s768MiB per invocation. Original embedded512MiB/30s source/F/return limits held.
728component constructions,1456named seeded draw calls32760values; caller RNG
unchanged.296predictcalls1332rows+2877logitcalls2877rows;182backward/SGD steps,
0autograd.grad,0SVD;486episodes3501dispatches. Named API counts are not complete
internal tensor/manual PC-gradient computation; PC inference2 is the existing
policy bound. All such work is included in sampled wall/RSS, not a matched-
compute or speed comparison. No LS, GPU, simulator, final, install/download or
remote experiment. Full reports were retained for the eight final fixtures.

## Preserved failures and prospective changes

First CLI fixture caught a real double-dispatch bug: new option fell through
to ordinary source training again.0modelwork occurred in that failed literal
check. Fixed dispatch and terminal return, preserving defaults and fresh output.

First complete BP integration passed isolation/accounting, all tasks failed; next
encoding test failed because the combined report exceeded legacy512KiB even
compact. Its full report was not persisted after the test process exited; only
the original cost/outcome receipt remains, a disclosed evidence gap. No claim
of complete raw trace retention for that first attempt. Before further runs,
6522608369 scoped ONLY the new option to existing exclusive
`write_catalog_report`4MiB/failure-cost fallback. Old modes512KiB unchanged and
pure-tested. Replacing the incorrect new-mode512KiB test was explicit scope
correction, not silently discarding paid traces or weakening scientific criteria.
Final pretty reports1,682,518..2,064,663B; eight gzipped full reports retained
under the original8MiB stage. D: pytest output admission<=32MiB separately
declared for8finite writer fixtures, no retention deletion or global cache change.

After Windows' three calls, final lead review added refusal of populated paid
rows. Prospective6522743676 allowed one fourth Windows call for only2pure cases,
0additionalmodel allowance, before executing it. Shared480/2560 unchanged.
Initialstatics.json Ruff/format passed; both mypy targets failed missing local
list annotation. Equivalent explicit typed list/assignment fixed it; final
statics-final.json all4exit0/7.886155200s, initial12.402476200s retained. Commands:

```powershell
$files=@('scripts/ticket_native_probe.py','scripts/recurring_ticket_workflow_example.py','tests/test_ordinary_native_reserved_probe.py')
.venv/Scripts/python.exe -B -m ruff check @files
.venv/Scripts/python.exe -B -m ruff format --check @files
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent @files
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent --platform linux @files
```

Independent review found a real endpoint isolation gap after the original
cohorts: a frozen snapshot/reader was checked, but the actual disposable learner
was not rechecked after reads/clocks before return training. Corrected observer
retains actual learner plus a separate deep pin, compares both actual and captured
state, and routes constructor/prediction through existing
`EndpointBoundary(read_ports,after_clocks=unchanged)`. Foreign changes are refused,
not restored. Four final literal cases per platform cover healthy ports, native
read mutation, last-clock mutation and equal mutation of actual/detached state.
All passed0realmodelwork; prospective extra pure calls recorded6523035748 and
6523060833, no model-cap increase. All learning/readers/cohorts are unchanged;
earlier eight full integrations and numerical negative results are reused.
Full workflow latency/extra state-validation overhead AFTER this hardening was
not remeasured;186.328092899s is observed test work, not a final pipeline estimate.
Final guarded statics all4exit0/7.950226800s, `statics-guarded.json`, after the
earlier successful7.886155200s batch and preserved initial failures.

Lead source/procedure/result self-review completed. Finite independent final
source/evidence/interpretation review accepted after the endpoint fix, with no
remaining blocker:4read-only commands1.0875232s plus transmitted exact callsite
verification. No writes/imports/tests/models/network/children; exception ended.
Review covered actual source, tests, all8saved fixture metadata/outcomes,
work/static receipts and scoped interpretation; no exhaustive individual trace
audit. `review.txt` preserves the handoff. No scientific study asserted.
Full suite/cleanclone/currentwheel/browser/neural
comparison/robotics/native study skipped: this is focused software integration,
no new experiment allowance; accepted unchanged probe/math/source suites reused.
No commit/push/install/download/delete, and inherited edits/frozen failures kept.
Final local stage20files706,914B, below8MiB, including all8compressed final
fixture reports, failed work/static receipts and review. Scoped changed-file
whitespace passed with LF/CRLF advisories; new source/test format checks pass.

The final pure endpoint invocations use the same check.py/test prefixes above,
`windows-endpoint-work.json`/`linux-endpoint-work.json`, `-k endpoint_guard`,
basetemp suffix `-windows-endpoint`/`-linux-endpoint`. Five Windows/two Linux
pytest invocations total, no skips. No whole numeric cohort repeated for the
independently reviewed guard fix; its actual private-state checks are new while
the recorded native numerical computation is unchanged.

Next inspect the saved BP probe class/loss diagnostics and earliest failed typed
query against its existing TRAIN truth/raw scores, without fits/encoders/new
forward calls. Determine a concrete optimization-versus-control mismatch before
choosing a separately frozen TRAIN-only repair; do not launch another cohort,
pick seeds or add circadian mechanisms to force a win. R2.1 broader learnability,
reliable/unseen adaptation/CPC advantage/release/#21/#18/full goal remain OPEN.
