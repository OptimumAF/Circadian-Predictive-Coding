# Live prospective generation ownership

## Modules and boundaries

```text
src/core/prospective_generation_ownership.py  immutable scope/observation, live ports
src/app/prospective_generation_ownership.py   full V2 owner context composition
src/infra/prospective_generation_ownership.py native local lease and physical checks
tests/test_prospective_generation_ownership.py full V2/native no-science regressions
docs/adr/ADR-0185-observe-live-generation-ownership-through-native-lease-port.md
```

Inputs are the full fixed design/spec, trusted V2 reader and owner ports; the outer
owner receives existing bundle/canonical shared local registry roots and an aware
UTC clock. Outputs are a live context and immutable observations describing the
recorded time held. Caller -> app -> core protocols -> injected infra owner/reader.
App uses the unchanged full H preflight; it imports no infra/adapters. Core does
no IO. Native IO remains in its focused outer module. All49 previous lead files
and original150 source closure are unchanged, including the older V14 owner.
No new dependency, environment variable, scientific default or algorithm change.

Responsibilities: validate complete metadata before claim; bind the whole request
scope, acquire a native handle, verify regular single-link registry file identity
and complete V2 files, observe monotone scope/handle/time/sequence, release on all
exits. Non-responsibilities: source/labels/arrays/RNG/models/final/scoring, complete
runtime code closure, actual arrival/assignment/chronology, remote/global ownership,
source independence/prior usage/resource fit/repeat/b3 or scientific admission.

## Why native ownership is a separate gate

Existing V14 uses `msvcrt.locking` on Windows and `fcntl.flock` on POSIX. Preserve
that original source and use the same standard native mechanism behind a new inner
port. Why this: an owner declaration and matching request files cannot observe a
live lease. Native owner lifetime is independently testable before implementing
actual code/import closure and sequential source-arrival evidence. No owner proof
can replace those remaining full acceptance requirements.

Registry key `.p67-generation-<whole-inner-generation-request-SHA256>.owner.lock`
uses g's complete request identity, not a mutable outer owner/file identity. Full
different-owner/file copies sharing that generation request must contend in the
same configured registry. Use one existing canonical absolute registry directory
for every cooperative local executor. This is scoped local cooperative ownership;
different registries/hosts or unrelated requests are not globally fenced. Prior
freshness/collision/release uncertainty remains separately required.

Permanent lock bytes are exactly `b"0"`. Create once with exclusive creation; open
the existing file without rewriting and never unlink/reclaim its path. Native
Windows locks its first byte; POSIX uses a nonblocking exclusive flock. A fresh
inode at a deleted/replaced name could permit two holders, so the trusted registry
must not be reclaimed. Reject symlink/junction paths, nonregular/missing/drifting
registry or lock identities, hardlink aliases and wrong/extra bytes before an
observation. Recheck path/handle device+inode/single-link identity around full V2
physical rechecks, even if the physical check fails. Native denial of namespace
mutation is accepted prevention; otherwise observations reject that drift. A
hostile actor who changes a shared registry cannot obtain scientific authority
from this library. No cross-host/distributed lock or global source freshness claim.

## Public API

```python
from pathlib import Path
from src.app.prospective_generation_ownership import claim_prospective_generation_bundle
from src.infra.prospective_generation_bundles import FileProspectiveGenerationBundleReader
from src.infra.prospective_generation_ownership import FileGenerationRequestOwner

# Complete design/spec are caller declarations, not actual seed selection here.
registry = Path(shared_registry).resolve(strict=True)  # existing trusted directory
reader = FileProspectiveGenerationBundleReader(Path(bundle_root))
owner = FileGenerationRequestOwner(Path(bundle_root), registry)
with claim_prospective_generation_bundle(design, spec, reader, owner) as observation:
    assert observation.ownership_at_entry.native_lock_observed
    assert len(observation.bundle.snapshot.generation_request.role_recipes) == 480
    assert len(observation.required_actual_bindings) == 12
    assert not observation.runtime_code_closure_verified
    assert not observation.execution_authorized
# ownership_at_entry is historical now, not a live execution capability.
```

`GenerationRequestOwner.claim(snapshot)` yields `LiveGenerationRequestLease` whose
`observe(snapshot)` binds exact immutable `GenerationOwnershipScope` and a native
handle's point-in-time evidence. Scope includes whole physical expected-file spec,
inner generation request/design/source-map/code-manifest identities and owner.
Both prospective/observed snapshot UTC declarations are checked before native file
access. Outer native checks bind all whole physical metadata and code files; app
owns full semantic g inspection before claim. A direct native lease does not
replace that app gate. Direct released handles raise inactive errors; a nonce or
immutable observation does not prove a currently held handle.

App first inspects every full V2 recipe/source/stream/binding/resource/repeat/owner
declaration, then claims and validates observation0, rereads under the lease,
validates observation1 and late bytes before yielding. Success/failure exits still
owe complete file and owner checks, including observation2 continuity. Nested
finally blocks retain ownership checks on file failure and native release on any
exception. Strict record/identity/integer/boolean/path/nonce/time checks deny
foreign/aliased/unheld/detached/late changes; counts are never coerced. Original
consumer exceptions remain chained if a final check also fails.

The default UTC clock is a live system observation; injected clocks are useful
for deterministic invalid/rollback controls. It is not actual source-generation
chronology. Non-scientific UUIDs identify handles without touching any declared
source/model/selector/local-noise RNG stream. Bound observations are not persisted
as a complete execution/repeat ledger by this module; that future ledger remains
required. Core/app/infra results keep all12 parent obligations and no actual
fresh-role/execution/precision/source-independence admission.

## Verification and evidence

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q tests/test_prospective_generation_ownership.py
.\.venv\Scripts\python.exe -B -m ruff check --no-cache src tests scripts
.\.venv\Scripts\python.exe -B -m ruff format --check src/core/prospective_generation_ownership.py src/app/prospective_generation_ownership.py src/infra/prospective_generation_ownership.py tests/test_prospective_generation_ownership.py
.\.venv\Scripts\python.exe -B -m mypy --no-incremental --cache-dir nul
git diff --check
```

Complete V2 file preflight now composes a live owner port with a real native local
lease. Contention keys use the entire inner generation request, so different
outer owner declarations/file copies of that request contend in one configured
canonical registry. All480 recipes/100 sources/400 streams/60 bindings, unchanged
560 cells/116 contrasts/settings/analysis/stopping/count/caps and all12 admission
obligations remain. Reread after acquiring and recheck whole physical files/owner
before yield and on success/failure exit. Immutable observations bind scope,
native single-link file identity, nonce and monotone sequence/UTC at the recorded
time held. Native handles release on exception and process death; permanent lock
paths stay in place and are never reclaimed/unlinked by the adapter. Observations
remain historical after exit and cannot authorize execution. Runtime code closure,
actual arrival/assignment/chronology, cross-host ownership/source independence,
prior effects/resource/repeat/b3 remain unverified; all fresh/execution/precision
flags false and independence unknown.360 development IDs remain unset and120
final-ID tuples declare geometry only. Existing49 lead files/APIs unchanged.

Full502 cases pass (46 owner/456 related),0errors/failures/skips in
131.0861267s. Real native Windows controls
cover same-process and subprocess contention, different outer owners for the same
whole inner request, normal release and os._exit73 process-death release, failures,
clock rollback/type rejection, physical/registry drift and early/late invalid
observations. All new cases guard scientific source/final/model/scoring/RNG/array
work; IO/native non-scientific UUID/time are intentional. Ruff/all4-file format/
full no-incremental mypy582/diff pass in
25.8889968s. Preserve initial missing
API and static fixture mapping-type failure, all exact versions/receipts. No
optional repeated tests; shared tests132.2325310/180spent, static50.7586573/180spent.
Whole preservation pending.


All valid controls use real temporary full V2 metadata/five invented code files,
all480/100/400/60 rows and native holders; invented code is never executed.
Subprocesses run the actual app/outer native owner with independent raising science
guards and complete V2 files. Child os._exit73 bypasses Python cleanup, proving OS
release on process death; stdin EOF/10s bounded cleanup prevents abandoned holders.
Negative port wrappers alter only observations from a real held native port for
late drift, or supply invalid initial metadata that never yields. They cannot
serve as actual scientific proofs. Native Windows platform executed; actual POSIX
native/symlink/junction privileges, backend/CUDA/training/resume/clean-clone CI,
full repository pytest outside502, original semantic repeat/audit/science and
optional repetitions are not run. All executed502 cases have zero skips.

Evidence: `artifacts/runs/p67-untouched-seed-usage/prior-evidence/generation-request-ownership/`
entry/declaration and exact before copies; red/JUnit; initial static failure/four
preserved versions; format/v2/full green/static; document-change/operation; pending
whole handoff/terminal receipts. Static failure was one fixture dictionary inferred
as object at `**values[kind]`; add its actual mapping type, no ignore/exclusion.
Review strengthens snapshot UTC, clock monotonicity and nested failure-exit owner
checks and adds complete controls before the single full502 passing invocation.
Entry/tests/static180s each including failures; documents/preparation/whole closing/
terminal share180s,16MB output. No experiment artifact or source construction,
favorable seed/baseline/metric selection, cap increase or original reader restart.

All25 original helper requirements remain individually Unbound in the F guide;
CI/seed-role/P6.4 proposals held. Preserve original4471 contents/13149 aliases,
whole historical uncertainty/effects, negative precision and unaccepted original
350.7925872/360spent/9.2074128remaining. Whole original/current/history/held-clone
preservation and one original END rebuild precede checking i only. All b2/b3/d2b/
d2/P6.7/d3 parent criteria remain required and open.

## Next safe extension

P6.7d2b2/b3: implement complete trusted runtime code closure through an inner proof port, composing full V2 physical files and the live owner context. Bind actual loaded Python/native dependencies and actual callable/code objects, closed complete membership and late loaded/monkeypatched/detached code drift before source construction; a caller-declared code manifest or native owner observation cannot substitute. Then implement the live sequential phase-arrival observer against frozen split/exposure recipes with an immutable complete actual proof ledger, without constructing future B/source/labels early. Compose all five saved pending inputs/full4471 contents/13149 aliases/whole historical witnesses and all prior effects; bind or explicitly revise all25 original helper cases. Resolve original failed resource acceptance without resetting350.7925872/360spent or9.2074128remaining. Preserve b3 isolation/init/parity/resource/artifact/repeat/readback and held CI/P6.4 proposals. Actual scientific seeds/source/arrays stay unset and execution false until complete admission passes.

Keep full runtime evidence separate from declared file membership and local native
owner observations. Bound code-object/dependency drift before any actual source
work, then bind each allowed arrival sequentially with actual class-label split/
exposure evidence and full immutable ledger. Broad experiments remain deferred.


## Terminal scoped acceptance

P6.7d2b2i alone is complete for full V2 live local native owner component.
Core immutable complete scope/point-in-time observations/live protocols, app full
H/g owner context and outer standard native lease delivered with46-case full
controls, guide and ADR-0185. Full480 recipes/100 sources/400 streams/60 bindings,
unchanged560 cells/116 contrasts/settings/count/caps/analysis/stopping/all12 proof
obligations remain. Registry key is the whole inner generation request; different
outer owners/file copies cannot evade local contention. Permanent single-link
regular lock bytes/handle identity, aware monotone UTC/sequence/scope/nonce and
whole files rechecked before/while/after, including failure exits. Native handles
release after exception and actual os._exit73 process death; never unlink/reclaim
registry files. Observations are historical after exit; runtime closure/arrival/
assignment/chronology/global source independence/prior/resource/repeat/b3 remain
required, independence unknown and fresh/execution/precision authority false.

One full502 invocation passes46 new native/456 related0errors/failures/skips.
Full Ruff/4-file format/no-incremental mypy582/diff pass without ignores/exclusions.
Initial missing API and fixture mapping-type failure preserved with exact four
failed versions and all receipts; explicit mapping annotation and stronger UTC/
rollback/failure-exit checks applied before full pass. All49 prior lead files and
original150 source closure unchanged and unrekeyed. Native Windows executed; actual
POSIX/symlink/junction privileges/backend/CUDA/training/resume/clean-clone CI/full
repo pytest outside502/original semantic audit/repeat/science/optional repetition
not run. Parent and child scientific sentinels remain raised throughout controls.

Full preservation passes116.1562081s including
4.9806298s preparation:all2014 physical files,
3793 current Git objects/34 commits,
all current/historical aliases, original150 sources/79 tests/174 inputs/proofs,
all53 lead files and three held clone/commit/patch pins before/after. Exactly one
original END rebuild,24 science guards0; no original reader/semantic audit or
scientific dispatch. All374 previous task criteria and244 raw240 nonseparator
tables unchanged. Original4471 contents/13149 aliases/full uncertainty/effects,
negative precision and failed350.7925872/360spent/9.2074128remaining persist. All
b2/b3/d2b/d2/P6.7/d3 tasks open;25 helper requirements Unbound/three proposals held.

Artifacts:generation-request-ownership entry/declaration/red/format/static failure/
complete failed versions/full green/JUnit/static-v2/document-change/operation/
handoff-source.json/handoff-validation.json/handoff-operation.json and terminal
handoff-final-log-append.json/final-append-operation.json. No experiment, actual
scientific seed/source/arrays or favorable metric/baseline/algorithm selection.
Commands: start-generation-request-ownership.py; run-gates.py red; run-format.py;
run-gates.py static; prepare-static-repair.py; run-format-v2.py; run-gates-v2.py
green/static-v2; update-documents.py; prepare-preservation.py; close-session.py;
record-terminal.py. Every owned producer immutable/single-use, every launched
process terminal. Overall goal active; external blockers:none. Full caps and
output remain unchanged; budgets before terminal:{"closing_and_documents_including_preparation": 117.05128449999029, "entry_including_failures": 1.277946999995038, "static_including_failures": 50.758657299971674, "tests_including_failures": 132.23253100004513}.
Terminal reuses accepted whole before/after gate, checks all27 docs/53 code before
publication and exact4 document edits/374 old criteria/tables/log prefix/unchanged
23 other docs/53 code after. Exact before-terminal texts reconstruct from pinned
whole before-publication bases plus complete recorded append text/new-guide text;
no duplicate whole snapshots or optional whole physical/clone/END repeat. Final
receipt binds terminal elapsed/shared180 total/remainder/16MB output/diff.

Exact next action:P6.7d2b2/b3: implement complete trusted runtime code closure through an inner proof port, composing full V2 physical files and the live owner context. Bind actual loaded Python/native dependencies and actual callable/code objects, closed complete membership and late loaded/monkeypatched/detached code drift before source construction; a caller-declared code manifest or native owner observation cannot substitute. Then implement the live sequential phase-arrival observer against frozen split/exposure recipes with an immutable complete actual proof ledger, without constructing future B/source/labels early. Compose all five saved pending inputs/full4471 contents/13149 aliases/whole historical witnesses and all prior effects; bind or explicitly revise all25 original helper cases. Resolve original failed resource acceptance without resetting350.7925872/360spent or9.2074128remaining. Preserve b3 isolation/init/parity/resource/artifact/repeat/readback and held CI/P6.4 proposals. Actual scientific seeds/source/arrays stay unset and execution false until complete admission passes.
