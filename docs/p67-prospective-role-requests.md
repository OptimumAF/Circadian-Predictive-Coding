# Complete prospective role/source/request metadata

## Scope and modules

`src/core/prospective_role_requests.py` defines frozen source, role, request and
inspection records. `src/app/prospective_role_requests.py` derives source
configuration declarations, encodes whole metadata identities and inspects the
complete fixed request. Inputs: the exact fixed design, expected code declaration,
60 bindings,400 streams,100 phase sources,480 roles and all12 required actual proofs.
Output: immutable inspected source/role records and full metadata identities.
All actual source/role/code/chronology/fresh/execution/precision flags remain false;
independent source replications remain unknown. No IO, arrays, RNG, models,
scoring, seed selection or real request admission is performed.

Why this: validate complete declarations before constructing data, while keeping
physical proof and execution at their proper outer boundaries. A checksummed
request can still lie about actual sources or UTC. Resealing cannot bypass its
fixed design, source configurations, types, role partitions or availability rules.

## Preserved geometry and policies

A/B each originate from160 samples with.25 final fraction:120 development and40
final positions. A retains all120; B exposure retains60 original rows in0..119.
Development roles are72/24/24 for A and36/12/12 for B; IDs are increasing within
each role, disjoint within each phase and equal across planned shared views.
Final IDs are exactly0..39 in a separate namespace. B IDs may include60..119;
inspection does not prove the seeded/class-stratified exposure or splitting.

Canonical IDs: `phase_a/seed_<base>/development/<original_row>` (likewise B) and
`phase_a/seed_<base>/final/<0..39>`. All numeric positions use canonical decimals.
Development availability is phase arrival; final availability is the declared
`complete_global_freeze`. This is a policy declaration, not release chronology.
Outer selection remains unscored/unavailable for confirmation setting selection.
No role has an actual array identity, and `final_released` must be exactFalse.

Configuration identities cover sample counts, noise, transforms, final ratio,
exposure and its seed, source/split seeds, original/retained/final geometry and
inner/outer fractions. Gating/replay share data arguments; method/sleep settings
remain bound by the complete design identity. Code identity is an expected whole
declaration only.100 phase source rows deduplicate only planned shared groups.
The complete560 cells/116 contrasts/settings/analysis/stopping/caps are unchanged.

## API use

For a caller that already holds the complete ordered `bindings`, `streams` and
`roles` metadata and an expected `code_identity` declaration:

```python
from dataclasses import replace
from src.app.prospective_confirmation_design import (
    fixed_prospective_design, prospective_design_identity,
)
from src.app.prospective_role_requests import (
    declare_prospective_sources, inspect_prospective_role_request,
    prospective_source_map_identity, prospective_role_request_identity,
)
from src.core.prospective_role_requests import ProspectiveRoleRequest
from src.core.seed_stream_screening import EvidenceIdentity

design = fixed_prospective_design()
sources = declare_prospective_sources(design, bindings, code_identity)
request = ProspectiveRoleRequest(
    EvidenceIdentity(**prospective_design_identity(design)),
    bindings, streams, sources, roles,
    prospective_source_map_identity(sources), EvidenceIdentity(0, "0" * 64),
    tuple(design["required_actual_request_bindings"]),
)
request = replace(request, request_identity=prospective_role_request_identity(request))
result = inspect_prospective_role_request(design, code_identity, request)
assert not result.execution_authorized
assert not result.actual_code_verified
assert result.independent_source_replications is None
```

The encoders hash canonical sorted compact UTF8 JSON plus newline. Request identity
excludes only its own identity field; source map identity covers every source row.
Encoding an invalid declaration is allowed; only the inspector validates it.
Records and every nested primitive/tuple must have exact schemas/types. `ValueError`
rejects omissions, duplicates, reorder, foreign/unknown/mutable/type-coerced values,
late corruption, overlap, changed shared views, wrong policies and execution claims.

## Commands and evidence

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -ra -p no:cacheprovider tests/test_prospective_role_requests.py tests/test_prospective_stream_declarations.py tests/test_prospective_replications.py tests/test_prospective_confirmation_design.py tests/test_seed_stream_screening.py
.\.venv\Scripts\python.exe -B -m ruff check --no-cache src tests scripts
.\.venv\Scripts\python.exe -B -m ruff format --check src/core/prospective_role_requests.py src/app/prospective_role_requests.py tests/test_prospective_role_requests.py
.\.venv\Scripts\python.exe -B -m mypy --no-incremental --cache-dir nul
git diff --check
```

Budgeted full scoped gates pass174 tests (54 new/120 existing),0skips, Ruff,
format and full mypy563. All source/RNG/model/IO guards raise. Full stdout/stderr,
JUnit cases, pins and failed attempts are retained under
`artifacts/runs/p67-untouched-seed-usage/prior-evidence/role-source-request-contract/`.
Owned producers are immutable and single-use; do not relaunch occupied acceptance
paths. Whole preservation and terminal acceptance are recorded separately.
See development log and ADR-0181. No dependencies/environment variables added.

## Safe extension and unfinished work

Implement actual IO/provenance through an inner proof port and outer adapter.
Bind complete code/source-map/request bytes, UTC before any source, exclusive
ownership, full prior uncertainty, joint resource/repeat envelopes and b3 proofs.
Keep all12 obligations and original25 unbound helper cases explicit. Matching
metadata or distinct numeric seeds never proves independent/unrecycled sources.
Do not grant authority from this inspection result or choose seeds/settings from
outcomes. Actual request/source/seed/array bindings remain unset. Original failed
resource350.7925872/360spent/9.2074128remaining and negative precision are unresolved;
parent milestones stay unfinished. No baseline, metric, stop or cap changes.


## Terminal scoped acceptance

P6.7d2b2e alone is complete for the full declared metadata envelope:100 ordered
phase sources,480 roles,400 streams,60 bindings/50 groups, all560 cells/116
contrasts/settings/counts/analysis/caps/stopping unchanged. Exact schemas/types,
whole identities, generator declarations, disjoint original development positions,
final IDs, shared views and availability/use policies checked. Actual source,
role, code and chronology unverified; independence unknown; fresh/execution/
precision authority false. All12 actual-proof obligations and25 unbound helper
cases remain required. All370 older task criteria and244 raw240 nonseparator
table rows preserved. Original b2/b3/d2b/d2/P6.7/d3 remain unfinished.

Full174-case scope passes (54 new/120 related),0errors/0failures/0skips. Ruff,
3-file format/full mypy563/diff pass. Preserve missing API, reserved pytest fixture
and local tuple/set typing failures, exact versions, name-only corrections and
formatting. All31 previous source/test files remain byte-identical,3 new files
added; no dependency, scientific setting/seed/baseline/metric/cap/stop changes.

Whole preservation passes in115.2141911s including
4.5584410s preparation:1987 physical files,
3716 current Git objects/34 commits,
all current/historical aliases, original150 sources/79 tests/174 inputs/proofs,
all34 lead code pins and all held clone/patch pins match before/after. One complete
original END rebuild,24 science guards0. No original reader/semantic audit/science.
Original failed resource350.7925872/360spent/9.2074128remaining, all earlier prior
unknowns and negative precision remain unresolved. Development goal remains active.

Evidence: role-source-request-contract/handoff-source.json, handoff-validation.json,
handoff-operation.json and handoff-final-log-append.json/final-append-operation.json.
Exact next action: P6.7d2b2/b3: implement actual prospective request admission before first source. Read src/infra/prospective_confirmation_inputs.py, src/app/prospective_confirmation_evidence.py and the new role request modules, then define the inner proof port and outer full-envelope IO adapter for whole code/source-map/request bytes, prospective UTC and exclusive ownership, all prior uncertainty effects, joint resource and independent-repeat envelopes. Start with full100-source/480-role fixtures and late physical-pin/UTC/owner/resource corruption tests; bind or explicitly revise each of the25 unbound helper cases against this real full request, never a dummy adapter. Preserve actual seeds/source/arrays unset, unknown independence, fresh/execution authority false until full admission and b3 isolation/parity/resource/reproducibility/artifact/readback proofs pass. Explicitly resolve original failed resource acceptance without resetting350.7925872/360spent or9.2074128remaining. Keep held CI/P6.4 proposals unintegrated until source/version and boundary corrections pass.
