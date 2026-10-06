# Prospective V2 generation bundle files

## Responsibilities and module boundaries

```text
src/core/prospective_generation_bundles.py     immutable snapshot/result, reader port
src/app/prospective_generation_bundles.py      exact V2 decoder and full recipe preflight
src/infra/prospective_generation_bundles.py    V2 physical reader adapter
src/infra/prospective_bundle_files.py          shared whole-file physical policy
src/infra/prospective_request_bundles.py       preserved V1 adapter (one prior edit)
tests/prospective_generation_bundle_fixtures.py full invented temporary files
tests/test_prospective_generation_bundles.py   no-science physical regressions
docs/adr/ADR-0184-share-whole-file-proofs-for-versioned-generation-bundles.md
```

Inputs are the unchanged complete fixed design, caller's immutable expected-file
spec, verifying reader/root and injectable aware UTC clock. Outputs are immutable
physical snapshots and a preflight retaining every actual admission obligation.
Composition: caller -> app preflight -> core reader protocol -> injected infra
adapter -> shared outer IO; app also composes the existing pure g recipe gate.
Core imports core only; app imports no infra/adapters. Versioned decoders/typed
snapshots remain separate. No new dependency, environment variable or scientific
default. This module does not create sources/labels/arrays/RNG/models/final values,
execute pinned code, acquire a live lease, observe real arrival, certify runtime
closure/source independence/prior-effect resolution/resource fit/repeat/b3 or run
an experiment. Physical metadata matching retains execution and release denied.

## Why share the physical policy

V1 concrete-role file metadata and V2 generation recipe files need identical
path/alias/publication/canonical/whole-byte/code-membership/recheck guarantees.
Extract only those outer IO checks, preserving V1's public constructor/read/recheck
API, errors and operation sequence. Its exact earlier6599-byte file is retained
under `generation-request-bundle/before-code/src/infra/prospective_request_bundles.py`.
This file is absent from the original150-source closure. The prospectively declared
single prior edit preserves that closure without rekeying; all42 other prior lead
files are unchanged. Why this: one real IO policy prevents version drift while
full schema and scientific recipe checks remain in their focused app modules.

Full V1 specialization proves all original six constructor/private helper bodies
and whole read/recheck bodies match exact saved AST after substituting the fixed
callbacks/parameters, with unchanged public signatures. Direct recheck must compare
the physical source map with the **supplied snapshot**. Rebuilt dataclass equality
can equate a float byte count with an int; strict JSON comparison of the supplied
snapshot rejects that alias. Two tests initially failed, then passed after the
original comparison was restored. All failed bytes/receipts remain evidence.

## File and declaration contract

Reuse `ProspectiveBundleSpec` from the existing core V1 module. It pins distinct
root-relative request/source-map/code-manifest paths and the closed ordered tuple
of expected code paths, with whole byte count/SHA256 identities. Spec validation
precedes file access. Shared IO denies traversal/absolute/noncanonical paths,
symlinks/junctions, nonregular/outside-root files, hardlink aliases and pending
`.claim`/`.failure.json` publication markers. Every metadata file uses full strict
duplicate/nonfinite rejecting JSON and exact canonical bytes (sorted compact JSON
plus newline). Every code file is read whole against its closed manifest identity
before/after the app's inspection. Manifest omission/reorder/extra/schema/type or
late byte drift fails; manifest membership is caller-declared, not runtime closure.

The outer schema is `p67_complete_prospective_generation_bundle_v2` with exactly
`schema_id`, `generation_request`, `prospective_utc`, `owner_id`,
`resource_envelope` and `independent_repeat_envelope`. The typed inner request
retains `p67_complete_prospective_generation_request_v2`. Complete JSON arrays
decode to immutable tuples without numeric/bool coercion. Full g validation
rederives all480/100/400/60 rows and all ordered proof obligations; request/source
map/manifest identities remain whole. V1 expects its own schema and rejects V2;
V2 rejects V1. Fabricated concrete development IDs cannot become V2 recipes.

Prospective and observed UTC must be canonical aware UTC ISO strings, and the
prospective declaration cannot exceed the observed clock. Owner is exactly32
lowercase hexadecimal characters. The full resource envelope must equal the
unchanged fixed design. Repeat requires whole design/source-map/V2 generation
request identities, identical caps, charging success/failure/every attempt and
`actual_independent_repeat_verified=False`. A declaration is not live ownership,
actual before-source chronology or successful independent repeat. All360
development IDs remain unset;120 final-ID tuple declarations contain positions
0..39 from geometry, with final values unavailable until complete global freeze.
Actual arrival-dependent stratified splitting and B class-balanced exposure
retain original positions and frozen rules/seeds; no early source work fills IDs.

## Public API and usage

```python
from pathlib import Path
from src.app.prospective_generation_bundles import preflight_prospective_generation_bundle
from src.infra.prospective_generation_bundles import FileProspectiveGenerationBundleReader

# design/spec are complete caller declarations, pinned before source construction.
# This example does not choose actual scientific seeds or publish those files.
reader = FileProspectiveGenerationBundleReader(Path(bundle_root))
result = preflight_prospective_generation_bundle(design, spec, reader)
assert result.physical_files_verified
assert len(result.snapshot.generation_request.role_recipes) == 480
assert len(result.required_actual_bindings) == 12
assert not result.runtime_code_closure_verified
assert not result.exclusive_owner_verified
assert not result.execution_authorized
```

The app's pure `decode_prospective_generation_bundle(body, spec, code_files,
observed_utc)` decodes metadata only; callers requiring physical evidence use the
preflight with a trusted verifying reader. Implementations of the core protocol
must bind all physical inputs and recheck late changes. The app validates exact
snapshot/spec/code-member types, canonical envelope strings, full g recipes and
same-request repeat before its final `recheck_bundle`. Matching files do not
resolve any of the twelve independent actual-proof obligations.

## Verification and evidence

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q tests/test_prospective_generation_bundles.py
.\.venv\Scripts\python.exe -B -m ruff check --no-cache src tests scripts
.\.venv\Scripts\python.exe -B -m ruff format --check src/app/prospective_generation_bundles.py src/core/prospective_generation_bundles.py src/infra/prospective_generation_bundles.py src/infra/prospective_bundle_files.py tests/prospective_generation_bundle_fixtures.py tests/test_prospective_generation_bundles.py src/infra/prospective_request_bundles.py
.\.venv\Scripts\python.exe -B -m mypy --no-incremental --cache-dir nul
git diff --check
```

The V2 physical boundary binds the full request/source map/code manifest and every
closed expected code file to all480 recipes/100 source declarations/400 streams/
60 bindings, unchanged560 cells/116 contrasts/settings/analysis/stopping/count/caps
and all12 actual-proof obligations. Immutable snapshots and inner read/recheck port
keep app/core free of IO. Both versions share whole bytes, canonical strict JSON,
root-contained regular paths, physical alias/publication-marker checks and late
rechecks. V1 public constructor/read/recheck behavior is preserved. All360
development realizations stay unset;120 final-ID tuples declare geometry only.
UTC/owner/resource/same-generation-request repeat are checked as declarations.
Matching physical metadata proves neither runtime code closure, exclusive owner,
actual sources/roles/arrival/assignment/chronology, unrecycled independence, prior
effect resolution, resource fit, independent repeat nor b3 admission. These stay
required; independence unknown and fresh/execution/precision flags false.

Initial full454 cases pass (66 new/388 related),0skips in
138.3763548s. Two added V1/V2 direct source
numeric-alias recheck cases expose comparing the rebuilt snapshot with the source
map instead of the supplied snapshot. Restore the original supplied-snapshot
comparison; all68 current new cases pass in
30.2081090s,0skips. Current unique
coverage456 is bound by both full receipts and exact V1 specialization: all six
original private IO/constructor bodies, whole read/recheck bodies after controlled
callback substitution and three public signatures match the saved V1 AST; all42
other lead files unchanged, all66 prior fixture ASTs unchanged and rerun. This is
coverage across receipts, not a single456-case pytest invocation. Ruff/all7-file
format/full no-incremental mypy578/diff pass in
45.6565722s. Whole preservation pending.


All new cases use full invented temporary-file controls and raising source/final/
array/RNG/model/scoring guards; IO is intentionally permitted. Cover full positives,
last-row resealed scope/rule/seed/availability/type/array/release/execution/UTC/
owner/resource/repeat corruption, equal-size changes of all four physical kinds,
closed manifest membership, unsafe spec before IO, actual hardlink/pending/failure
namespace conditions, late after-app recheck drift, foreign port snapshots,
independent V1/V2 rejection and direct V1/V2 numeric aliases. Invented five code
files are hashed text, never executed. No actual scientific seed selection,
source generation, independent replication, metric/baseline tuning or experiment.

Whole evidence is under
`artifacts/runs/p67-untouched-seed-usage/prior-evidence/generation-request-bundle/`:
entry/declaration; original V1 bytes; exact red/format/green/correction versions;
full JUnit/gates; two failure details; `v1-specialization-and-suite-reuse.json`;
document-change/operation; whole handoff source/validation/operation; terminal
handoff receipt/final operation. Initial missing API is red1 collection error,
not acceptance. A failed patch anchor mutated no files; its diagnostic is charged
conservatively1s through recheck preparation. All producer executions and failures
spend unchanged entry/tests/static180s each and shared documents/preparation/whole
closing/terminal180s; owned output16MB. The original failed semantic/repeat budget
stays350.7925872/360spent and9.2074128remaining; no original reader/audit restart.

All25 original helper requirements remain individually Unbound in
`docs/p67-prospective-request-bundles.md`; three CI/seed-role/P6.4 proposals remain
held. Full4471 prior contents/13149 aliases/historical uncertainty/effects and
negative precision remain. The scoped file implementation does not complete any
b2/b3/d2b/d2/P6.7/d3 parent acceptance. Whole original150/current/history/held-clone
preservation with exactly one original END rebuild is still required before h
completion. No optional whole related repeat after the restoring fix; complete
case/AST/unchanged-code evidence preserves scope without reducing acceptance.

## Safe next extension

P6.7d2b2/b3: add the trusted live exclusive request-owner port and complete runtime code closure around the V2 full physical generation request, then a live phase-arrival observer which binds each permitted sequential arrival to the frozen split/exposure recipes and a complete immutable proof ledger. Do not construct future B/source/labels early to fill complete claims. Compose all five saved pending inputs/full4471 contents/13149 aliases/whole historical witnesses and all prior effects; bind or explicitly revise all25 original helper requirements without promoting toy fixtures to actual proof. Resolve original failed resource acceptance without resetting350.7925872/360spent or9.2074128remaining. Preserve all b3 isolation/init/parity/resource/artifact/repeat/readback gates and held CI/P6.4 proposals. Keep actual scientific seeds/source/arrays unset and execution false until full admission passes.

Add owner/runtime/arrival proofs through focused inner ports and outer adapters.
Validate schema/caps/closed provenance before lease or source use. Observe each
allowed arrival sequentially and append immutable evidence of labels/content and
actual seeded assignment; do not move later source construction early to produce
all480 concrete claims. Keep recipe/file inspectors usable and all actual gates
denied until the full parent proof chain passes. Broad sweeps remain deferred.


## Terminal scoped acceptance

P6.7d2b2h alone is complete for its full V2 physical generation-file metadata scope.
Deliver core immutable snapshot/result/reader protocol, pure app full decoder/g
preflight, V2 outer adapter, shared whole-file IO, V1 delegation, full fixtures/
68-case suite, guide and ADR-0184. Bind all480 recipes/100 source declarations/
400 streams/60 bindings and unchanged whole560 cells/116 contrasts/settings/count/
analysis/stopping/caps/all12 obligations.360 development realizations remain unset;
120 final-ID tuples declare geometry without final values. Whole request/source
map/code manifest/every expected code file, canonical UTC/owner/full resource/
same-generation-request repeat declarations and late rechecks are verified as
physical metadata. Actual runtime closure/live owner/arrival/seeded assignment/
chronology/unrecycled independence/prior effects/resource fit/repeat/b3 gates stay
required; independence unknown and fresh/execution/precision authority false.

Initial full454 cases pass66 new/388 related. Review's two direct V1/V2 source
numeric-alias regressions actually fail, preserve all seven failed versions and
restore original supplied-snapshot comparison. Entire current68 module passes,
0errors/failures/skips. Exact six old private/constructor bodies and entire
read/recheck bodies specialize to the saved V1 AST; public signatures unchanged.
All388 related case IDs match earlier complete coverage, all66 prior V2 fixtures
AST-identical and rerun,42 other lead files unchanged.456 current unique cases
across full receipts; no claim of one456-case invocation or criterion reduction.
Full Ruff/7-file format/no-incremental mypy578/diff pass without ignores/exclusions/
dependencies. Missing API, failed patch anchor before mutation/diagnostic1s, both
alias failures and all exact versions/format/preparation receipts retained/charged.
One permitted V1 infra edit retains full original6599 bytes; original150 closure
unrekeyed,42 other prior lead files unchanged,6 new code/test files added (49 total).

Full preservation passes79.0719877s including
0.0078148s preparation:all2008 physical files,
3771 current Git objects/34 commits,
all current/historical aliases, original150 sources/79 tests/174 inputs/proofs,
all49 current lead code pins, old V1 bytes and held clone/patch pins before/after.
One original END rebuild,24 science guards0; no original reader/semantic audit or
scientific dispatch. All373 prior task criteria/244 raw240 nonseparator tables
unchanged. Original failed350.7925872/360spent/9.2074128remaining,4471 prior contents/
13149 aliases/full effects and negative precision remain unresolved. All b2/b3/
d2b/d2/P6.7/d3 parent tasks open;25 helper cases Unbound/three proposals held.

Evidence:generation-request-bundle/handoff-source-v2.json, handoff-validation-v3.json,
handoff-v3-operation.json, handoff-final-log-append.json/final-append-operation.json;
earlier entry/declaration/all gates/JUnit/exact versions/V1 specialization/reuse/
document receipts. No experiment, actual scientific seed/source/array or tuning.
Exact next action:P6.7d2b2/b3: add the trusted live exclusive request-owner port and complete runtime code closure around the V2 full physical generation request, then a live phase-arrival observer which binds each permitted sequential arrival to the frozen split/exposure recipes and a complete immutable proof ledger. Do not construct future B/source/labels early to fill complete claims. Compose all five saved pending inputs/full4471 contents/13149 aliases/whole historical witnesses and all prior effects; bind or explicitly revise all25 original helper requirements without promoting toy fixtures to actual proof. Resolve original failed resource acceptance without resetting350.7925872/360spent or9.2074128remaining. Preserve all b3 isolation/init/parity/resource/artifact/repeat/readback gates and held CI/P6.4 proposals. Keep actual scientific seeds/source/arrays unset and execution false until full admission passes.

Closing correction: initial close-session.py failed after42.7518605s (including5.0792412s preparation) because its red fixture loop looked for the preserved old V1 reader under red-fixtures; its exact whole original is in before-code. No original END rebuild had run. Retain failed source/operation/producer unchanged; accepted=False. prepare-closing-recovery.py verifies the pinned exact unhandled failure line and entire unconditional completed prefix AST: all retained/2008 physical/full-history/original whole before pins, three clone checks and initial24-guard zero observation completed. close-session-v2.py corrects only that lookup, reuses the full before evidence, installs/checks all original guards/bindings again while skipping only the redundant initial original whole-pin read, executes every remaining full gate, all retained/original/2008 physical/current/history/clone after checks and exactly one original END rebuild. No criteria/count/cap reduction, failed attempt spends the same180s family. v2 accepted whole receipt is handoff-source-v2.json/handoff-validation-v3.json/handoff-v3-operation.json; original generic handoff operation stays failed. record-terminal-v3.py is the actual terminal producer; preserve unused record-terminal.py. Conservative0.5s diagnostic charge is included in recovery preparation. Initial prepare-closing-recovery.py asserted one occurrence of the output-name tuple, which also appears in previous-stage refs; it stopped before either new supervisor existed. Retain that producer/prefix JSON, correct only the first occurrence in prepare-closing-recovery-v2.py, charge1s for failed attempt/copy preparation and keep earlier-stage refs unchanged. Future sessions must use the accepted v2 source/operation from this terminal receipt, never the failed generic source as current acceptance.

The v2 recovery next failed during preparation at its occupied handoff-launch-contract.json name, before any worker/guard/END launch. Preserve its entire prepared handoff-source-v2.json and producer unchanged, retain launch-name-failure-v2.json and charge5s for the4.8307972s tool wall. prepare-closing-recovery-v3.py/close-session-v3.py use a distinct launch receipt and a small operative handoff-recovery-source-v3.json bridge that pins the full unchanged v2 source, current operative producer and every newer artifact. Accepted handoff-validation-v3.json/handoff-v3-operation.json bind both source identities and all complete acceptance gates. No second full source copy, cap increase or retained artifact deletion. record-terminal-v3.py reconstructs exact before-terminal documents from already preserved whole before-publication bases and their complete document-change append text (or full new-guide text), verifying actual whole byte identities before edits. This retains all old bytes without duplicating2.5MB and stays within16MB. Future sessions must use this terminal's accepted v3 operation/validation, full v2 base source plus v3 operative bridge; generic failed gate and unused v2 prepared source alone are not acceptance. Commands/outcomes: initial close-session.py failed; first recovery preparation asserted repeated anchor before new supervisors; corrected prepare-closing-recovery-v2.py passed with debt1s; close-session-v2.py failed before worker with debt5s; prepare-closing-recovery-v3.py passed; close-session-v3.py completed all remaining gates; record-terminal-v3.py publishes only after acceptance.

Terminal scope: reuse accepted whole2008/current/history/original/held-clone before-and-after gate; check all25 documents/49 current code pins before publication, exact four document edits/373 previous criteria/plan tables/log prefix and unchanged21 other documents/49 code pins after. No optional repeated whole physical/held-clone/END/scientific run after terminal publication. Every operation uses the budgeted filename glob; preparation includes0.5s conservatively charged prior-binding shape inspection. Terminal producer prepared/pinned before whole closing. All families and output remain within their original declared caps.
