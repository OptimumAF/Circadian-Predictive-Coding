# Opt-in BP TRAIN-update diagnostics — R2.1-BP.UPDATE / #334

Previous323 attempt2 was progress with a negative native result: old-task
success4→0→0, fresh0→0→0. This increment supplies observations needed to locate
large changes without altering optimization. Full learning/scientific/release
claims remain open. No spent source12/F16/return16 experiment was repeated.

## Preservation correction — reopened regression

The prior increment incorrectly treated an EXISTING tracked module as a new
file. It overwrote torch_bp_update_diagnostics.py, removing the legacy invariant
MiniGrid diagnose_bp_updates API/DEFINITION. Lead and scoped review missed this
regression. Prior closeout did not prove repository preservation; no broad
readiness claim is valid. Original report bytes are retained at
artifacts/runs/bp-update-diagnostics-20261030/report-before-preservation-fix.md.txt.

Restored legacy module EXACT HEAD bytes (15,210B), which were clean in the
previous initial checkout. New helper moved unchanged to torch_bp_step_norms.py;
displaced body preserved in norms-before-relocation.py.txt. Only new imports
redirected; legacy consumers/tests stay at their original module. No unrelated
change reset. Reopened334/start6534500726 prospectively allocated one bounded
software preservation cohort/platform before execution; old charges remain.

Combined45checks/platform PASS:26new-helper cases plus19existing legacy
admission/refusal cases. Legacy cases reject before Torch/model work, and
unchanged old native135-row cohort is not repeated. Windows4.0636768999975175s
336,318,464B/Linux9.310958512s351,612,928B; each22components44draw220values/
7SGD21rows/7head21rows/7backward10summaries, all per-run caps held. Linux session
37591 observed to terminalEXIT0 without restart. Exact commands:

```text
.venv/Scripts/python.exe -B artifacts/runs/bp-update-diagnostics-20261030/check_preservation.py windows-preservation
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/bp-update-diagnostics-20261030/check_preservation.py linux-preservation
```

Final scoped Ruff/format/default+Linux-target mypy PASS; exact argv in
statics-preservation*.json. Cumulative known28SGD84rows; conservative charged
36SGD108rows including original unknownLinux ceiling, below amended40/120.
Original180s testwall/90sstatic/512MiBprocess/4MiBstage/cache32MiB unchanged.
Preservation correction independently ACCEPTED:5 read-only commands,
1.4269389s, no project/numerical execution. Missing optional-log reads and final
CP1252 console-print error are retained: strict UTF8 decoding/roundtrip passed
before that console error. REGRESSION_REVIEW.md distinguishes source/file proof
from command exits. Prior closeout claims below are historical and superseded
for preservation. Finite exception ended. Full goal and broader science remain open.

## Delivered and executed behavior

New `src/adapters/torch_bp_step_norms.py::summarize_bp_update` validates
two consecutive complete BP snapshots under the same policy and row clock.
For hidden parameters and output parameters separately it reports parameter
before/after L2, gradient L2, momentum before/after L2, actual parameter-delta L2,
nominal SGD-delta L2 and maximum absolute SGD residual. It uses after-update
gradient/momentum buffers and actual parameter differences, no new update,
forward/backward, model construction, restore or RNG. Original validated SGD
has zero dampening/weight decay and no Nesterov. Residual includes float32
rounding; it is an observation, not a replacement update or convergence proof.
Snapshot validation/copies and scalar reductions are additional paid work.

Existing `scripts/ticket_native_probe.py` accepts explicit
`probe_update_diagnostics=True`; the ordinary runner and CE handoff forward it.
CLI flag is `--probe-update-diagnostics`, BP/reserved/context-return only.
Wrong types/modes refuse before model preparation. Default path omits new
report fields and extra snapshots/summaries, preserving the optimizer and
existing report/work definitions. No labels/features are stored by this adapter;
only the existing private TRAIN update port invokes it. Probe-test/final labels
never become training inputs; source/actor are not measured or updated.

Report fields are `reserved_probe.update_diagnostics.{records,work}` and
`preserved_update_diagnostics`. At most32 scalar records. Two snapshot attempts
and one summary attempt are separately charged before delegation, with returns
recorded afterward. Five scalar-vector values per parameter include explicit
zero placeholders for absent momentum. These counts are not total FLOPs or
independent tensor-copy-byte attribution. Existing paid_report_guard protects
preceding values and container owners. A changed prior measurement refuses the
next update; corruption remains visible alongside the trusted canonical prefix.
A summary failure preserves the already returned native SGD and a pending
measurement record. CLI bounded-writer fallback also retains diagnostic work.

New `tests/test_probe_update_diagnostics.py` has26 cases. Two real paired tiny
updates show complete state/SGD/gradient equality and unchanged Torch RNG with
and without measurement. A literal snapshot oracle verifies group norms; bad
row clocks/policy/gradient values refuse. Three intentionally interrupted tiny
private prefixes cover default/no diagnostic calls, corruption before another
update, and summary failure after a paid SGD return. Source12 admission,
cohort/task and memory seams are scripted in these integration fixtures;
they do not qualify a genuine source12/F/return run. CLI/CE forwarding and
admission/fallback checks are model-free. Consolidated into one test module to
share the tiny native fixture, avoiding duplicate model work.

## Commands, costs and failures

All work physical `D:/Dev/Repositories/Circadian`; PowerShell TEMP/TMP
`D:/Dev/_Temp`. Existing SDKs/dependencies only. WSL TEMP/TMP/TMPDIR
`/mnt/d/Dev/_Temp`, Python3.12.3/NumPy1.26.4/Torch2.14CPU, threads1.
Stage `artifacts/runs/bp-update-diagnostics-20261030/`.

```text
.venv/Scripts/python.exe -B artifacts/runs/bp-update-diagnostics-20261030/check.py windows
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/bp-update-diagnostics-20261030/check.py linux
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/bp-update-diagnostics-20261030/check.py linux-final
```

Windows26PASS/EXIT0,4.41773529999773s329,424,896B. First Linux failed pytest
FD-capture cleanup on D TEMP with FileNotFoundError and emitted no count
receipt. Original SPENT/temp retained; unknown work conservatively charged at
the entire original32components64draws320values/8SGD24rows/8head24rows/
8backward60summary ceiling, RSS512MiB ceiling and13s observation wall upper
bound (combined toolcall12.6s, including subsequent statics). No unobserved
zero-work claim. Prospective amendment6534277106 allowed ONE Linux software
replacement using established `--capture=sys`, separate preserved temp/output,
original per-invocation caps and original180s aggregate wall. Linux aggregate
allowance64components/16SGD48rows; this did not renew a scientific allowance.

Linux-final EXIT0/error null,9.211994734s346,075,136B, same defined26-case cohort.
Both passing runs each counted22components44seededdraws220values/7SGD21rows/
7head21rows/7backward10summary attempts. No CE fit/LBFGS/LS/LP/PC-latent/typed
tasks/simulator/final/GPU/remote work. Known completed total14SGD42rows;
including conservative failed-Linux charge22SGD66rows, within revised pooled
24SGD72rows. Known passing test process time13.62973003399773s plus failed
observation bound13s remains below original180s. Per-process90s512MiB held for
passing cohorts. Raw results/caps in windows-work.json/linux-final-work.json;
failed allowance in linux-first-FAILED.json. No actual model/test failure was
changed into a pass by weakening expectations.

Initial type check found8errors in annotations/test helpers, fixed before
native fixtures. One failed formatted patch anchor changed no files; exact
edits simplified the repair. Ruff/format sixfiles and default+Linux-target
mypy fivefiles now PASS; expanded commands/output/times in statics-*.json and
runner-statics.json. Cache D:/Dev/_Temp/circadian334-mypy, not stage. Stage4MiB,
testtemp16MiB/cache32MiB limits remain. No deletes or global cache changes.

The first Linux-final tool response yielded after ten seconds; its session ID
was not printed. It was NOT restarted. The completed work receipt plus explicit
`pgrep -af '[c]heck.py linux-final'` exit1/no live target established terminal
state. An initial WSL rg process query failed Permission denied and was replaced
by pgrep. This is an observation-tool limitation, not extra model execution.
Runner now records exceptions/counters as well as successful returns.

Broad suite/unchanged math/four-family cohorts/full native source12/F-return/
new wheel/clean clone/browser/GPU/remote CI/release/sweeps skipped for bounded
correctness scope. Existing failed/negative323/331,333 reference-capacity scope
and328 LS behavior/caveats remain unchanged. Preserve all inherited dirty work.

## Prior scoped acceptance and next action (preservation corrected above)

Lead reviewed actual formulas, before/after state ownership, default paths,
role/cost routing, pending failure records and the combined source/tests.
Independent actual-source/math/state/isolation/results ACCEPT4readonly commands
0.6777152s, no numerical execution/independent replication. Finite exception
ended. Self-review remains separately identified.
This qualifies software observations on tiny fixtures, not the cause of323's
native failure or any new acquisition/retention/recovery/advantage claim.

Next precise action: implement one bounded read-only first-step SGD reference
decomposition using323's saved initial source/momentum, CE coefficients and
fixed private probe_train rows, with its already-paid first/second native loss anchors.
Freeze arithmetic/tolerances/cost before calculation; distinguish reference
counterfactuals from runtime authority. This can test the hidden-update
instability hypothesis without reconstructing a native model or repeating the
spent F/return study. Do not tune a baseline or select a favorable variant.
Full R2/R4/R5/R7/matched science/release/repository goal remain open.

Prior closeout, superseded by preservation correction:334 scopedsoftware
closed/completed, canonical6534449167; tracker18
6534454353,parent21 6534454711,parent23 6534455086. Final retained stage22683B
<4MiB. Guide/bothplans/log updated; full goal and scientific parents remain open.

Preservation closeout:334 reclosed with correction6534651407; parent receipts
18 6534651757,21 6534652028,23 6534652279. Current stage55954B<4MiB;
cache16785637B<32MiB. Legacy source restored, unqualified former
preservation claim corrected; full repository goal remains active.
