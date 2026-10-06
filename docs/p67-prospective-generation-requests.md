# Prospective generation requests

## Responsibilities and boundaries

`src/core/prospective_generation_requests.py` contains immutable recipes, a V2
generation request and inspection results. `src/app/prospective_generation_requests.py`
derives/checks the complete fixed request and inspects separate concrete role ID
claims through the existing e gate. Inputs are declarations only; outputs bind
metadata and retain every actual-proof obligation. Neither module performs IO,
array/source/RNG/model/final/scoring work or grants execution/release authority.

```text
src/core/prospective_generation_requests.py   immutable records
src/app/prospective_generation_requests.py    full recipe/claim validation
tests/prospective_generation_fixtures.py      invented full declarations
tests/test_prospective_generation_requests.py no-science behavior regressions
docs/adr/ADR-0183-freeze-role-assignments-before-source-arrival.md
```

Dependency flow: caller -> app generation gate -> app fixed-design/stream/source/
concrete-role gates and core records. App/core import no infra/adapters. No new
dependencies, environment variables or scientific defaults.

## Why a versioned request

The real generator (`src/infra/datasets.py`) draws class normals before permuting
source rows. The splitter (`src/infra/continual_roles.py::_partition_training_rows`)
permutes arrived class rows to choose train/inner/outer positions, sorts them and
keeps original source positions in IDs. Phase B exposure
(`src/app/continual_arrived_benchmark.py::_reduce_phase_b_source`) also reads arrived
labels and chooses class-balanced original positions. Knowing concrete development
IDs before data would require source-related RNG work at the wrong boundary.

Why this: freeze the existing assignment rules before data and bind their realized
development declarations afterward. Do not substitute contiguous IDs, construct
labels early or alter stratification/exposure. Existing e/f are completed concrete
candidate/file metadata inspectors with actual authority denied; retain their APIs.
V2 supplies the proper pre-source representation, not actual before-source proof.

## Frozen recipe semantics

All480 rows follow the unchanged fixed layout. A development source has120 original
positions and retains120; B has120 original positions and retains60. Counts remain
A72/24/24 and B36/12/12 for train/inner/outer. Development IDs are `None` in the
generation request; their namespace is `phase_<a/b>/seed_<base>/development` and
their availability is the respective permitted phase arrival. Assignment policy
`arrived_class_stratified_roles_v1` refers to the existing seeded per-class
permutations, inner reservation then outer reservation then train remainder, with
per-class reservations max(1, round(class_count * fraction)) for fractions0.2/0.2
and sorted original positions. Split seeds
are base+17 for A and base+138 for B. A exposure is identity; B exposure uses the
existing class-balanced selection at base+118 and preserves original positions.
These policy identifiers are declarations, not proof that the pinned code ran.

All120 final-ID tuple declarations contain canonical positions0..39 from fixed
geometry, with namespace `phase_<a/b>/seed_<base>/final`. Their ID declarations
are available before the first source; final source/labels/values remain unavailable
until `complete_global_freeze`. Final assignment has no exposure or split RNG.
Array identities stay unset and final release false. Shared views preserve the
same source/configuration/recipe declarations; copies do not add replications.

Canonical JSON binds the full design,100-source map and whole V2 request excluding
only its self identity. Factory/inspector validate all60 bindings/400 externally
claimed streams and every ordered row with exact types and values, including
immutable tuples and all12 ordered proof obligations. Resealing malformed scope,
unknown dataclass subclasses, bool/int/float aliases, wrong rules/seeds/counts/
availability/IDs/arrays/execution cannot validate a request. All original matched
settings/metrics/contrasts/analysis/stopping/count/caps and negative precision stay.

## Public API and usage

```python
from src.app.prospective_generation_requests import (
    freeze_prospective_generation_request,
    inspect_prospective_generation_request,
    inspect_arrived_role_declarations,
)

# Whole fixed declarations are supplied by the caller; no actual seed choice here.
request = freeze_prospective_generation_request(
    design, bindings, declared_code_identity, claimed_streams
)
preflight = inspect_prospective_generation_request(design, declared_code_identity, request)
assert preflight.final_id_declarations_complete
assert not preflight.development_sample_ids_bound
assert not preflight.execution_authorized

# Separate full concrete claims from a future arrival observer. This API alone
# cannot certify that arrival or the frozen assignment actually occurred.
claims = inspect_arrived_role_declarations(
    design, declared_code_identity, request, concrete_role_id_declarations
)
assert not claims.actual_arrival_verified
assert not claims.assignment_execution_verified
assert not claims.concrete_role_inspection.execution_authorized
```

The concrete bridge requires all480 rows; its early exact-schema check prevents
foreign payload copying before canonical encoding. Existing e then checks every
ID namespace/range/count/order, disjoint complete development partitions and
shared-view identity. The bridge never fills or mutates the frozen generation
request, and it does not verify labels, class membership or seeded realization.
Full contiguous fixture positions are invented claims, not actual source outputs.
Final ID geometry can match while actual freshness or before-source chronology
is still unproved. No positive fixture can authorize a source or final value.

## Verification and extension

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q tests/test_prospective_generation_requests.py
.\.venv\Scripts\python.exe -B -m ruff check --no-cache src tests scripts
.\.venv\Scripts\python.exe -B -m ruff format --check src/app/prospective_generation_requests.py src/core/prospective_generation_requests.py tests/prospective_generation_fixtures.py tests/test_prospective_generation_requests.py
.\.venv\Scripts\python.exe -B -m mypy --no-incremental --cache-dir nul
git diff --check
```

The V2 generation request freezes480 ordered role recipes/100 source declarations/
400 streams/60 bindings against the unchanged complete fixed design (560 cells/
116 contrasts). Assignment/exposure rules and seeds, original/retained geometry,
counts, allowed uses, availability, all120 canonical final-ID tuples and all12
actual-proof obligations are frozen; all360 development role ID realizations are
unset. The separate full concrete-role declaration bridge composes e's partition/
shared-view checks after strict type rejection and binds its result to the V2
request. It cannot prove actual arrival, seeded assignment, runtime code, owner,
chronology, source independence/freshness or execution. All actual/fresh/execution/
precision flags remain false; independence unknown. Existing e/f remain intact.

Initial full382 cases pass (78 new/304 related) in
99.9559949s. Six added sentinel cases then
expose deepcopy before rejection; preserve every failed version/receipt and reject
full concrete schema before encoding. All84 current new cases pass in
35.8429476s. Current unique coverage388:
all304 related cases match earlier complete receipts; all776 older physical Python
files unchanged, no existing source/test/script references new modules or symbols,
and all78 earlier generation fixtures have identical runtime AST and rerun. No
optional full repeat: related-suite-reuse.json binds full reuse evidence without
scope/test removal. Ruff/all4-file format/full no-incremental mypy572/diff pass in
63.6039481s. Whole preservation pending.


Evidence is under `artifacts/runs/p67-untouched-seed-usage/prior-evidence/generation-role-recipes/`:
entry/declaration, exact red/pass/failure/format versions, full gates and JUnit,
schema correction, related-suite-reuse, document-change and whole handoff/terminal
receipts. The failed preparation's added-newline assumption and subsequent missing
producer call are retained and conservatively charged3s with their diagnostic.
Entry/tests/static180s each; shared docs/preparation/closing/terminal180s,16MB output.
No old cap reset, new scientific source/RNG/array, experiment, confidence claim or
favorable seed/baseline/metric selection. Whole closure requires one original END
rebuild, never the exhausted original reader/semantic audit. All25 original helper
requirements remain individually Unbound in `docs/p67-prospective-request-bundles.md`;
three CI/seed-role/P6.4 proposals remain held. This module does not complete them.

Next: P6.7d2b2/b3: implement the complete V2 physical generation-request reader and inner read/recheck port for all480 recipes/100 sources/400 streams/60 bindings before any source. Preserve e/f APIs; do not feed fabricated concrete development IDs to the V1 bundle reader as a pre-source request. Add full late physical request/source-map/code/UTC/owner/resource/repeat corruption fixtures. Then bind a live phase-arrival observer to each permitted sequential arrival and original split/exposure recipes, retaining a complete immutable proof ledger rather than constructing future sources early to fill the full declaration bridge. Compose trusted runtime closure/live exclusive lease, all five saved pending inputs/full4471 contents/13149 aliases/whole historical witnesses and all prior effects; bind or explicitly revise all25 original helper cases. Resolve original failed resource acceptance without resetting 350.7925872/360spent or9.2074128remaining. Preserve all b3 isolation/init/parity/resource/artifact/repeat/readback gates and held CI/P6.4 proposals. Keep actual scientific seeds/source/arrays unset and execution false until full admission passes.

The future arrival observer must bind each sequential permitted arrival as it
happens, including actual source/label/content/seeded assignment and chronology,
then produce a complete immutable ledger. This full declaration bridge is not a
partial runtime arrival gate, and future B/source construction must not be moved
early just to obtain full metadata. All b2/b3/d2b/d2/P6.7/d3 parent criteria stay open.


## Terminal scoped acceptance

P6.7d2b2g alone is complete for its full generation-recipe/claim metadata scope.
Core immutable V2 request/recipe/results, pure app factory/inspector/concrete-role
bridge,84-case suite/full invented fixture helper, guide and ADR-0183 delivered.
Freeze all480 recipes/100 source declarations/400 streams/60 bindings and whole
560 cells/116 contrasts/settings/count/analysis/stopping/caps/all12 obligations.
All360 development realizations remain unset until permitted arrival;120 final-ID
tuples are declared from geometry without final values. Existing label-dependent
stratified split/class-balanced B exposure/original positions remain. Separate
full concrete claims compose e partition/shared-view checks and retain actual
arrival/seeded assignment unverified. Existing e/f unchanged; actual V2 physical
reader/live sequential arrival/runtime closure/lease/prior/resource/repeat/b3
proofs remain required. Independence unknown, fresh/execution/precision false.

Initial full382 cases pass78 new/304 related; all84 current new cases pass after
six actual copying-risk failures and strict early schema repair. Verified complete
reuse of unchanged304 case IDs/dependencies (all776 prior physical Python files
unchanged, no forward imports/symbol references), all78 original generation
fixtures AST-identical and rerun:388 current unique coverage,0errors/failures/skips.
Ruff/four-file format/full mypy572/diff pass without ignores/exclusions/dependencies.
Retain initial missing API, failed newline preservation/missing producer attempts,
diagnostic, six copy sentinels, every exact version/receipt and corrections; all
failures spend original declared caps. Guide reservation formula clarified with
exact earlier guide retained. All39 older lead code/test files unchanged,4 added.

Full preservation passes173.1951707s including
6.6674974s preparation:all2000 physical files,
3753 current Git objects/34 commits,
all current/historical aliases, original150 sources/79 tests/174 inputs/proofs,
all43 lead code pins and held clone/patch pins before/after. One original END
rebuild,24 science guards0; no original reader/semantic audit or new science.
All372 older task criteria and244 raw240 nonseparator tables unchanged. Original
failed350.7925872/360spent/9.2074128remaining and negative precision unresolved;
b2/b3/d2b/d2/P6.7/d3 remain open,25 helper cases unbound/three proposals held.

Evidence:generation-role-recipes/handoff-source.json, handoff-validation.json,
handoff-operation.json and handoff-final-log-append.json/final-append-operation.json.
Exact next action:P6.7d2b2/b3: implement the complete V2 physical generation-request reader and inner read/recheck port for all480 recipes/100 sources/400 streams/60 bindings before any source. Preserve e/f APIs; do not feed fabricated concrete development IDs to the V1 bundle reader as a pre-source request. Add full late physical request/source-map/code/UTC/owner/resource/repeat corruption fixtures. Then bind a live phase-arrival observer to each permitted sequential arrival and original split/exposure recipes, retaining a complete immutable proof ledger rather than constructing future sources early to fill the full declaration bridge. Compose trusted runtime closure/live exclusive lease, all five saved pending inputs/full4471 contents/13149 aliases/whole historical witnesses and all prior effects; bind or explicitly revise all25 original helper cases. Resolve original failed resource acceptance without resetting 350.7925872/360spent or9.2074128remaining. Preserve all b3 isolation/init/parity/resource/artifact/repeat/readback gates and held CI/P6.4 proposals. Keep actual scientific seeds/source/arrays unset and execution false until full admission passes.

Terminal verification reuses the accepted173.1951707s whole2000/current/history/original/held-clone before-and-after gate, then checks all23 documents/43 lead code pins before publication, exact four document mutations/372 older task criteria/plan tables/log prefix and unchanged19 other documents/43 code pins after. No optional repeated whole physical/held-clone read after publication, no new whole-history/END/scientific run or cap increase. Exact previously prepared terminal producer retained unexecuted. Commands: prepare-budgeted-terminal.py; record-terminal-v2.py. document-operation-v2.json's0.1687795s was outside the operation filename glob; its full elapsed is now explicitly charged once through terminal-budget-accounting-operation.json. Whole closing actual spent including that debt remains below180; final receipt binds all charged families.
