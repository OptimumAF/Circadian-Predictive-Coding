# Live process runtime code observations (acceptance open)

## Files and boundaries

```text
src/core/prospective_runtime_closure.py immutable observations and live proof ports
src/app/prospective_runtime_closure.py full V2/native owner/runtime composition
src/infra/prospective_runtime_closure.py current-process live freeze/recheck
src/infra/runtime_python_objects.py actual Python membership/code/namespaces
src/infra/runtime_native_images.py Windows full files/images/executable memory
tests/prospective_runtime_fixtures.py full synthetic independent process controls
tests/test_prospective_runtime_closure.py bounded subprocess/pure invalid controls
```

Caller -> app -> core protocols; injected infra supplies process/file/native IO.
Core/app import no infra/adapters. Inputs: full unchanged fixed design/V2 spec,
reader/native owner/trusted observer and unique ordered `module:attribute` names.
Outputs: immutable point-in-time process observations with whole canonical JSON
identity, actual scope/PID/nonce/sequence/UTC and every original proof obligation.
No new dependency/environment variable/scientific default. All53 earlier lead
files/original150 source closure unchanged. This is an unfinished observation
component; execution/freshness/precision/source-version attestation remain false.

## Why inspect the actual runtime

Matching caller file declarations do not prove loaded modules or executing code.
The outer observer reads every sys.modules entry (including aliases/None), module
object/native dictionary/loader/file origin, actual GC-tracked functions including
detached ones, whole recursive marshaled code, defaults/closures/globals and
code-bearing namespaces/entrypoints/import hooks/path/tracing. Native dictionary
slots avoid user properties; weak proxies are not forwarded as real types. Baseline
function/type objects are retained to prevent garbage collection/id reuse while
held. Paths/files are fully read and physically bound; native system hardlink
counts are recorded, not falsely assumed absent. Closed physical bundle link rules
remain unchanged. No selected module/file subset substitutes for process membership.

64-bit Windows enumerates all actual process images via K32EnumProcessModulesEx,
pins whole native file bytes and checks image membership twice, then VirtualQueryEx
walks the complete address space and ReadProcessMemory hashes every executable
committed region, including private executable allocations. Fail truncation,
unsupported platforms, unreadable/guarded executable pages and nonadvancing walks.
This records actual memory bytes; it does not prove approved binary/source
correspondence. Python/runtime addresses/object IDs are process-local, not expected
equal across independent processes. A recorded observation is historical after exit.

Audit hooks deny the explicit import/exec/compile/direct code/function construction,
ctypes load/symbol and tracing events while active; Python cannot remove a hook,
so it becomes inactive after every exit. Full late checks retain earlier failures
through exception context. Existing compiled MAKE_FUNCTION/class operations and
transient changes are not comprehensively audited; observed GC/code drift is only
point-in-time evidence. These and generated/opaque source-version correspondence,
actual requested-manifest-to-loaded-code correspondence and native callable/source
attestation remain open. No full runtime closure/immutable execution capability is
claimed. Trusted composition must close these gaps before constructing sources.

## API

```python
from src.app.prospective_runtime_closure import observe_prospective_generation_runtime
from src.infra.prospective_runtime_closure import ProcessGenerationRuntimeObserver

# All dependencies/entrypoints are loaded before the live freeze. The unchanged
# full fixed design/spec and reader/native owner have already been configured.
runtime = ProcessGenerationRuntimeObserver()
with observe_prospective_generation_runtime(
    design, spec, reader, owner, runtime,
    ("src.infra.datasets:generate_two_cluster_dataset",),
) as observed:
    assert observed.runtime_process_observed
    assert not observed.runtime_source_version_attested
    assert not observed.execution_authorized
# Full observation/code/version/source admission remains a separate gate.
```

`runtime.phase_timings` retains complete measured capture costs, not scientific
resource acceptance. Public core ports return only immutable observations. Full
request files/owner are checked before entry and on success/failure exit; runtime
scope/raw graph/file/memory/entrypoint membership and monotone sequence/UTC are
checked inside the native lease. Actual runtime source-version identity and every
remaining prior/arrival/resource/repeat/b3 proof remain required.

## Commands and evidence

PowerShell from repository root, using existing .venv (Python3.14.7):

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q tests/test_prospective_runtime_closure.py
.\.venv\Scripts\python.exe -B -m ruff check --no-cache src tests scripts
.\.venv\Scripts\python.exe -B -m ruff format --check src/core/prospective_runtime_closure.py src/app/prospective_runtime_closure.py src/infra/prospective_runtime_closure.py src/infra/runtime_python_objects.py src/infra/runtime_native_images.py tests/test_prospective_runtime_closure.py tests/prospective_runtime_fixtures.py
.\.venv\Scripts\python.exe -B -m mypy --no-incremental --cache-dir nul
git diff --check
```

Do not launch the missing full correctness gate under an exhausted receipt or
reset the168.7993043/180spent shared family. Commands are documented workflows,
not a claim that the current full20 or all502 related rerun passed. Current bounded
16pass, native/failure complete diagnostic pass, prior positive16pass at its pinned
version, full Ruff/format/mypy589/diff pass; all failures/exact versions retained.
Full current positive/python_a/whole20/502 related and source-version/continuity/
all late variants remain unverified. Original scientific350.7925872/360spent and
9.2074128remaining unchanged. No scientific source/array/RNG/model/final/scoring
dispatch; test data is only full invented V2, real process code observation and an
unexecuted4096-byte temporary native fixture allocation.

Measured full captures:341modules/7663functions/51images/74-or75 executable regions,
11.18MB canonical graph. Python1.05–1.07s/native0.50–0.51s/serialization0.09–0.10s
means; exact per-capture rows in owned `runtime-code-closure` diagnostic receipt.
The 40-second eight-control Python group timed out; split into two complete four
control groups with unchanged40s limits. Preserve the whole180 shared gate failure/
remaining checks; no assertion/semantic condition removed or source-version claim.

Primary API references: [Python runtime types](https://docs.python.org/3/library/types.html),
[module membership](https://docs.python.org/3/library/sys.html#sys.modules),
[Windows image enumeration](https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-enumprocessmodulesex).
Implementation acceptance comes from local actual process controls, not these links.

## Next step

P6.7d2b2j remains unchecked. First inspect the complete capture-diagnostic-validation.json and every preserved failure/version, then declare a bounded runtime capture cost/parity diagnosis using its actual full341-module/7663-function/51-image/74-or75-executable-region/11.18MB observations. Measure duplicate Python namespace/graph traversal and full physical-path/native reads before optimizing; retain every actual module/function/code/closure/default/global/namespace/entrypoint/import-state/native file/executable-memory binding and prove complete equality or explicit lossless graph equivalence, with no subset/caching-away late drift. Current shared tests168.79930429987144/180spent (11.20069570012856remaining) must not be reset or represented as passed. Finish current positive and python_a controls, malformed/continuity/owner/request/native/late-code variants and all502 related cases under a prospectively justified bounded gate protocol preserving the failed180-second requirement and receipts. Close generated/opaque source-version correspondence and MAKE_FUNCTION/other non-audited transient callable gaps before claiming full runtime closure. Then implement sequential arrival/immutable complete actual ledger, all five saved inputs/full4471 contents/13149 aliases/history/prior effects/25 Unbound helper cases and every b3 gate. Preserve original350.7925872/360spent/9.2074128remaining and held CI/P6.4; actual scientific seeds/source/arrays unset and execution false.


### 2026-10-06T05:50:41.703922+00:00 — Terminal runtime observation progress; j unchecked

Completed IDs:[]; classification:verified_progress. The new actual process observer
and bounded positive/drift/release/cost evidence are implemented; P6.7d2b2j remains
unchecked for full current20/all502 correctness/resource/source-version/transient/
continuity requirements. Earlier pending preservation note is superseded only by
accepted whole preservation, not by an invented runtime/scientific admission pass.

Single-use closing exit0/115.2227169000s, whole2023 files
and current/historical Git before/after; all53 prior code files unchanged, all60
current lead pins, original150 sources79 tests174 inputs/proofs,29 current docs,
375 prior task criteria/244 raw240 actual tables,25 original Unbound helper cases,
all three whole held clones/commits/patches. Exactly one original END rebuild;
all24 scientific guards0. No original reader/semantic audit/scientific dispatch,
source closure rekey/helper integration/favorable seeds/baselines/metrics/cap changes.

Closing source:2499268bytes/SHA256
ee543679024fa25d206c0e9929e100db43684415682f75ae4a248430eb95604c; validation:10257bytes/SHA256
b3e8e3963ff406a230670693b30a8ad4b2f7ce81a42a86a7818beb3ffb015266; complete source/references retain every
earlier accepted bridge/whole saved input/history witness and failed receipt.
Current bounded16pass, zero skips/full589 staticpass. Real native/failure diagnostic
passed at the retained annotation-equivalent version; current positive/python_a/
full20/all502 rerun remain unverified. Shared tests168.79930429987144/180spent,
11.20069570012856remaining; static93.08297840005253/180spent; no budget reset.
Documents/preparation/closing spent116.6358261001/180 before this final append;
final-append-operation records its own charged elapsed and whole output <16MB.
Original resource failure350.7925872/360spent/9.2074128remaining unchanged.

Actual scientific seeds/source/arrays remain unset; runtime closure/source-version,
arrival/chronology/freshness/execution/precision authority false, independence
unknown; all original b2/b3/d2b/d2/P6.7/d3 and held CI/P6.4 requirements remain open.
No external blocker. Internal next gates: lossless complete capture/resource
diagnosis, remaining full tests and actual source/version/transient correspondence.

Exact next action:P6.7d2b2j remains unchecked. First inspect the complete capture-diagnostic-validation.json and every preserved failure/version, then declare a bounded runtime capture cost/parity diagnosis using its actual full341-module/7663-function/51-image/74-or75-executable-region/11.18MB observations. Measure duplicate Python namespace/graph traversal and full physical-path/native reads before optimizing; retain every actual module/function/code/closure/default/global/namespace/entrypoint/import-state/native file/executable-memory binding and prove complete equality or explicit lossless graph equivalence, with no subset/caching-away late drift. Current shared tests168.79930429987144/180spent (11.20069570012856remaining) must not be reset or represented as passed. Finish current positive and python_a controls, malformed/continuity/owner/request/native/late-code variants and all502 related cases under a prospectively justified bounded gate protocol preserving the failed180-second requirement and receipts. Close generated/opaque source-version correspondence and MAKE_FUNCTION/other non-audited transient callable gaps before claiming full runtime closure. Then implement sequential arrival/immutable complete actual ledger, all five saved inputs/full4471 contents/13149 aliases/history/prior effects/25 Unbound helper cases and every b3 gate. Preserve original350.7925872/360spent/9.2074128remaining and held CI/P6.4; actual scientific seeds/source/arrays unset and execution false.
