# Runtime mutation enforcement (P6.7d2b2j6b / j6b1)


## Current status — 2026-10-06

The confirmed foreign nested-context exit invalidates the earlier claim of release
only at the exact owning `with` boundary. Current `_exit_offsets` accepts exit sites
across the owner's code without binding them to the actual guard entry. The retained
probe assigns this same guard's bound `__exit__` to a preconstructed nested context;
its exceptional exit removes monitoring before the genuine owning context ends.
Defaults change from 5 to 9 and the target body runs with 9, without poison or denial;
the genuine later cleanup raises `ValueError`. The probe performed no scientific
calls or complete V2/native observation.

P6.7d2b2j6b2's general correctness acceptance is reopened. Its historical 23 controls,
three recorded observations and whole-closing evidence remain preserved, with their
original scope and spent budgets. They do not establish this missing owning-entry
contract. P6.7d2b2j6c is unchecked and user-deferred; no repair is implemented here.
Complete runtime/source/native/private/lifetime and scientific admission remain
unavailable; `require_complete_admission()` still raises.

The recorded commands and outcomes below describe historical validation. The current
native guard test has an explicit platform skip on unsupported native configurations;
a skip is neither positive coverage nor admission. This documentation audit reruns
neither the probe nor the full native case. Resume the deferred task only on user
instruction: bind actual entry to genuine normal/exception/return exits, then satisfy
all original and new controls under its prospectively declared budget. Full composition
and every remaining source/native/continuous proof obligation stay mandatory.

Why this: a negative execution witness overrides a broader correctness claim even
when the previous finite controls passed. See [the plan](../DEVELOPMENT_PLAN.md),
[the log](development-log.md) and the original local probe receipt at
`artifacts/runs/p67-untouched-seed-usage/prior-evidence/runtime-owning-with-contract/foreign-exit-probe-validation.json`.

## Historical mechanism and validation record

Status: mechanism implemented and independently verified; complete runtime admission
and whole observer-lifetime integration are unavailable. Original120s gates remain open.

## Responsibilities and boundaries

Inputs are the complete actual prepared GC function/code catalog and exact owning
Python code. The infra module installs global monitoring over its explicit lifetime;
it emits primitive event identities, persistent denial reasons and actual cleanup.
Its callbacks do not retain operation/argument objects or execute user callbacks.
No core/app dependency changes or existing observer changes. It grants no scientific
execution, source provenance, native/private/build certificate or freshness.

Why this: the prior installed experiment showed invisible foreign callback execution
and tracing loss after a caught error. Reject those capabilities before arming and
keep monitoring plus a poisoned state after denial. Native/implicit/internal behavior
still requires a complete verified operation contract before full runtime admission.

The [monitoring reference](https://docs.python.org/3.14/library/sys.monitoring.html)
describes global events, before-call/before-instruction callbacks and independent tools.
Installed actual tests establish the reported behavior; docs and build descriptions
do not attest installed source or private ABI. The
[audit reference](https://docs.python.org/3.14/library/sys.html#sys.addaudithook)
describes observation hooks with limits. This module combines actual VM denials with
audit refusal; it is not a Python sandbox or a certificate of native coverage.

## Evidence and remaining requirements

New optional infra RuntimeExecutionGuard implements global CALL/PY_START/INSTRUCTION
mutation enforcement over declared lifetimes. It retains every actual prepared GC
function/recursive code in immutable maps, rejects foreign tools/tracing/profile/other
Python threads before arming, denies creation/binding/attribute/closure/subscript writes
and unsupported native/C-API/monitoring operations before execution, and keeps poison
after a caught denial without resetting or returning DISABLE. It releases only through
the exact owning with boundary and reads all used callbacks after actual reacquisition.

Full corrected new case passes0 failures/errors/skips with23 actual controls and3
complete actual V2/native-owner observations. Catalog includes7841 actual GC functions,
8068 complete recursive raw code/native public fields/full disassembly, not a target
subset. All84 focused raw probes, all319 failed partial raws, every delivered event,
32 persistent denials and54 actually empty callback slots pass independent readback.
Legitimate pure precompiled Python returns10; mutation denials preserve default/code/
global/closure/attribute/subscript bindings, block the later target before its body,
keep actual mask81 after caught errors and retain consumer failure/release behavior.

The full case ran before one narrow inline typing directive for the actual stdlib
Bytecode.exception_entries member omitted by typeshed. Installed dis.py assigns it and
actual normal/exceptional with controls exercised its regions. Current609-file mypy/
Ruff/three-file format/diff pass. Complete whole old/current module compilation, every
recursive native code field and exact retained raw module bytes match. Both physical
source versions and the exact comment difference remain saved. This proves executable
identity, not new live physical-file/source correspondence: the three full records
are historical and source/version/native/private/continuous authority remains false.

Failures preserved and charged: first full47.9627813s failed on native disassembly
arguments, second45.3513869s on an actual slice marshal constant, both before arming;
27+292 complete partial raw files/source/XML retained, missing headers/events/terminal
counters not invented. Add descriptive native argument identities/reprs separately
from raw/native constants, exact slice start/stop/step and remove redundant new catalog
tuples while retaining every identical function/code in immutable maps. First static18
errors corrected by exact typing and nullable assertions. A0.5359952s live boundary
probe observed actual3.14 CALL3/WITH_EXCEPT_START, then the CALL2 assumption denied
release and lost stderr. Preserve it; correct exact owning code/guard identity/exception
regions, retaining premature release denial. Corrected0.5574522s two-control probe and
4.2374754s actual22-control matrix pass, explicitly without complete V2 observations.

The original120s full correctness gates remain unaccepted; spend98.6684586s is retained.
Prospectively separate200s successor includes all original failures/probes/preparation
plus the full94.1936794s attempt and declaration: combined192.8703373/200s, not reset.
Only correctness successor j6b1 may complete after full whole2061 preservation; original
j6b stays unchecked. Existing entry/static/metadata/shared180s, child90s and owned32MB
remain fixed. Failed reader7.0778772s frozenset display order retained; restricted literal
readback preserves both complete strings and exact raw/native values. No normalization.

All77 prior source/test files and original150 sources79 tests174 inputs are unchanged.
Prior full581 at73 pins, j5 single-case75 and j6a single-case77 receipts are pinned,
not rerun or claimed as584 full. No existing observer/graph/hash/audit edit, dependency,
environment/config/algorithm/seed/metric/baseline change or scientific dispatch. Full
observer-lifetime integration, implicit/native/internal/source/build/private/opaque/
omitted/nested membership and all five j5 gaps remain mandatory in parent j. Whole
GC maps/events/diagnostic intervals/unchanged boundary bodies do not grant admission.
All original j3_400s/j3a_500s/output221533189/j_180s/scientific360s failures remain
unaccepted without rekey or original semantic reader/repeat. Actual arrival/ledger/
prior/b3 and P6.7 remain open; seeds/source/arrays unset, independence unknown and
execution/freshness/precision false. Unrelated changes/held proposals are preserved.

## Files and commands

```text
src/infra/runtime_execution_guard.py          immutable whole catalog/global denial/poison/cleanup
tests/runtime_execution_guard_fixtures.py     pytest-free actual V2/native process controls
tests/test_runtime_execution_guard.py         bounded90s child/full artifact handoff
docs/adr/ADR-0195-poison-runtime-mutation-enforcement-after-caught-denials.md
```

Prepare the actual catalog and guard/owning code before the hold, install its audit
hook once, then enter the guard from the prepared with boundary. It is single-use;
any denial remains poisoned. Arbitrary native calls are refused. A request for complete
admission raises because required source/native/private/lifetime facts are unproved.
On unsupported interpreter versions, the test requires refusal, not a skip or approval.

```powershell
.venv\Scripts\python.exe -B -m pytest -q tests/test_runtime_execution_guard.py
.venv\Scripts\python.exe -B -m ruff check --no-cache src tests scripts
.venv\Scripts\python.exe -B -m ruff format --check src/infra/runtime_execution_guard.py tests/runtime_execution_guard_fixtures.py tests/test_runtime_execution_guard.py
.venv\Scripts\python.exe -B -m mypy --no-incremental --cache-dir nul
git diff --check
```

Artifacts: `artifacts/runs/p67-untouched-seed-usage/prior-evidence/runtime-execution-guard/`.
Complete catalog and independent result are losslessly compressed; every raw code
file remains present. Before-publication documents reuse identical retained whole
snapshots, with the entire additional plan suffix saved, rather than duplicate bytes.
No original input, artifact, graph or acceptance is omitted to meet the output cap.

## Exact next action

P6.7d2b2j: implement complete observer-lifetime enforcement and trusted source/native correspondence, using all23 current actual mutation controls, all five j5 gaps, j6a callback/trace failures and all96 old controls as mandatory regressions. First compose the guard with actual observer/reader/native ownership through an explicit verified support-operation contract; reject unsupported implicit/native/internal execution before admission and require effective pre-execution denial throughout entry/use/finally/consumer failure and real release. Do not exempt broad framework modules or treat guarded diagnostic intervals, GC maps, VM events, historical equal bodies or the exact comment/code bridge as complete native/source/continuous authority. Preserve current80 code/original150 closure/prior581 plus j5/j6a/j6b receipts, original120s and successor200s ledgers, all failed j3_400s/j3a_500s/output221533189/j_180s/scientific360s gates without reset/rekey. Require genuine matched-source/loaded-mismatch Python, full native memory/file/build/private localsplus/kinds correspondence and opaque/omitted/nested-route completeness. Then finish actual sequential arrival, immutable full actual ledger, all five saved inputs4471 contents13149 aliases/current-historical Git/prior effects/all25 Unbound helpers and every b3 isolation/init/parity/resource/repeat/artifact/independent-readback gate. Seeds/source/arrays unset, independence unknown, execution/freshness/precision false; no baseline/metric/seed tuning or held CI/P6.4 integration.


## Session outcome

The mechanism's23 actual controls, current609 static and complete independent readback pass. Whole closing remains unaccepted: first verifier path failure then corrected hard shared timeout,180.01753679988906/180s total. Both j6b/j6b1 stay unchecked. Full manifests are losslessly preserved; missing timeout END/terminal counters remain unknown. See the development log for all receipts, resources and the prospective inclusive closing successor required before another attempt. Complete runtime/native/source/private/lifetime/scientific admission remains unavailable.


## Complete whole-closing successor

Only P6.7d2b2j6b2, the prospectively declared inclusive whole-closing successor,
completes. Original j6b120s and j6b1 shared180s remain unchecked with criteria unchanged.
All original180.01753679988906s shared failures/documents/preparation and every new
declaration/preparation/closing/terminal step charge the fixed360s total. Original
correctness192.87033730000257/200s and owned32MB are unchanged; no full case rerun.

Fresh complete verification observes all2061 physical files/current-historical Git,
all80 current code47 docs/original150 sources79 tests174 inputs/proofs/387 prior
criteria244 tables25 Unbound/three held whole clones/commits/patches beforeafter.
One actual original END completes and all24 actual terminal scientific guards are0.
All9 actual phase receipts are saved; no old unobserved stage or counter is reused.
Prior full581-at73/j5-at75/j6a-at77 receipts remain pinned, not reexecuted or combined
as584 full. Unrelated seed-evidence doc/test and all77 old code unchanged. HEAD
28e71ee9de46230fdd6232cca3f9f59ba8ff4fb8 remains13 commits after reviewed8793c49.

Mutation mechanism's saved current1 real-process case23 controls0 skips/failures/
errors, Ruff/three-file format/no-incremental mypy609/diff and full independent
8068 raw/native public code/disassembly7841 joins84 probes319 failed partials/every
literal event32 denials54 actually empty callbacks/3 complete historical V2-native
records pass at unchanged current80 code pins. Legitimate pure Python returns10,
denials preserve bindings/poison/mask81 and block the repeated target before its body,
actual consumer exception/release/native ownership work and complete admission refuses.
Both complete physical source versions and the exact comment-only whole compiled/raw/
recursive native-field bridge remain retained. Historical records are not new live
physical source/native/build/private/continuous correspondence.

The failed basename closer and corrected original shared180s timeout stay unaccepted,
with complete producers/operations/whole lossless manifests retained. Old timeout
END count/terminal scientific guards remain unknown. Complete containers and full
contents hashes validate; no artifact, reference/history field or criterion omitted.
No scientific/algorithm/baseline/seed/metric/model/dependency/environment change,
semantic original reader/repeat, sweep, branch/commit/PR or held CI/P6.4 integration.
All five j5 gaps, j6a callback/trace failures and96 old controls remain mandatory for
complete source/native/private/support-operation/observer-lifetime/opaque/omitted/
nested/arrival/ledger/prior/b3/P6.7. All original j3_400s/j3a_500s/output221533189/
j_180s/scientific360s failures stay unaccepted without reset/rekey. Seeds/source/
arrays unset, independence unknown, execution/freshness/precision false.

Exact next action:P6.7d2b2j: implement complete observer-lifetime enforcement and trusted source/native correspondence, using all23 current actual mutation controls, all five j5 gaps, j6a callback/trace failures and all96 old controls as mandatory regressions. First compose the guard with actual observer/reader/native ownership through an explicit verified support-operation contract; reject unsupported implicit/native/internal execution before admission and require effective pre-execution denial throughout entry/use/finally/consumer failure and real release. Do not exempt broad framework modules or treat guarded diagnostic intervals, GC maps, VM events, historical equal bodies or the exact comment/code bridge as complete native/source/continuous authority. Preserve current80 code/original150 closure/prior581 plus j5/j6a/j6b receipts, original120s and successor200s ledgers, all failed j3_400s/j3a_500s/output221533189/j_180s/scientific360s gates without reset/rekey. Require genuine matched-source/loaded-mismatch Python, full native memory/file/build/private localsplus/kinds correspondence and opaque/omitted/nested-route completeness. Then finish actual sequential arrival, immutable full actual ledger, all five saved inputs4471 contents13149 aliases/current-historical Git/prior effects/all25 Unbound helpers and every b3 isolation/init/parity/resource/repeat/artifact/independent-readback gate. Seeds/source/arrays unset, independence unknown, execution/freshness/precision false; no baseline/metric/seed tuning or held CI/P6.4 integration.
