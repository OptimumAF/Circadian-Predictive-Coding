# Complete prospective stream declarations

Task: P6.7d2b2d. The public metadata boundary validates a caller's complete
derived-stream declaration against the existing fixed study.

## Modules and responsibilities

```text
src/app/prospective_stream_declarations.py   whole identity and external stream validation
tests/test_prospective_stream_declarations.py full layout and adversarial callers
```

`validate_prospective_stream_declarations` accepts the unchanged full design,
60 ordered `ReplicaSeedBinding` records, its exact `EvidenceIdentity`, and an
immutable tuple of 400 exact `DerivedSeedStream` records. The existing public
design and replica validators enforce all 50 planned source groups, shared
gating/replay bindings, all eight offsets and within/between group collisions.
The new boundary checks every caller-supplied record in order, including the last.

Missing, extra, reordered, duplicate, mutable, symbolic, detached and inexact
records fail with `ValueError`. Boolean/integer and float/integer equality does
not satisfy exact types. Unknown record classes or caller-proposed RNG-domain
schema extensions are rejected. A design checksum with matching syntax must also
equal the whole canonical fixed design identity; resealed scope drift still fails.

The function returns the existing `DeclaredReplicaStreams` value, with actual
independent replications unknown and `fresh_roles_authorized=False`. It performs
no IO, random sampling, source construction, training, scoring or release.
Actual history, RNG-domain provenance, source/request/role identity, chronological
availability, resource/repeat acceptance and scientific execution remain the
responsibility of the unfinished full prospective gates.

## Caller example

The caller supplies the complete ordered bindings and claimed stream tuple from
its prospective declaration. The identity is the expected whole design identity,
not a mutable file locator or a checksum used as an execution permission.

```python
from src.app.prospective_confirmation_design import fixed_prospective_design
from src.app.prospective_stream_declarations import validate_prospective_stream_declarations

design = fixed_prospective_design()
declaration = validate_prospective_stream_declarations(
    design, ordered_bindings, expected_design_identity, claimed_streams
)
assert len(declaration.streams) == 400
assert declaration.independent_source_replications is None
assert declaration.fresh_roles_authorized is False
```

Use `EvidenceIdentity`, `ReplicaSeedBinding` and `DerivedSeedStream` from the
existing core modules. The tests contain an executable complete literal fixture;
its numeric values are software inputs and do not select scientific seeds.

## Verification and extension

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_prospective_stream_declarations.py tests/test_prospective_replications.py tests/test_prospective_confirmation_design.py tests/test_seed_stream_screening.py
.\.venv\Scripts\python.exe -B -m ruff check --no-cache src tests scripts
.\.venv\Scripts\python.exe -B -m ruff format --check src/app/prospective_stream_declarations.py tests/test_prospective_stream_declarations.py
.\.venv\Scripts\python.exe -B -m mypy --no-incremental --cache-dir nul
git diff --check
```

The cache device `nul` is for Windows; use the platform's null device elsewhere.
Supervised results and exact source/test versions live under
`artifacts/runs/p67-untouched-seed-usage/prior-evidence/stream-declaration-contract/`.
The plan/log retain the initial missing-module failure and every gate outcome.

Extend through a separate pure complete-envelope inspector and an outer IO
adapter. Validate the full historical and request inputs before composing this
stream check. Preserve all 560 cells, 480 roles, 116 contrasts and all original
uncertainty/resource/precision criteria. The 25 original helper cases remain
unbound; this API supplies no dummy adapter or historical-proof shortcut.


## Terminal scoped acceptance

P6.7d2b2d is complete for its full declared external-stream scope. The function
checks the complete pinned design/60 ordered bindings/50 planned groups/all400
typed claims through existing public validators.120 current metadata tests pass
(42 new/78 related),0errors/0failures/0skips; Ruff/two-file format/full mypy560/diff
pass. Initial missing-module and mypy failures, exact fixture versions and local
annotation-only correction remain. No test/import exclusion or old cap reset.

Full preservation passes116.0157001s including
4.7566800s preparation. All1982 physical files,
3701 current Git objects/34 commits
and all historical aliases, original150 sources/79 tests/174 inputs/proofs, all29
previous code files plus2 new and all held clone/patch pins match before/after.
One complete original END rebuild/24 science guards0. No new original reader,
semantic corpus audit or scientific dispatch. Whole d acceptance is recorded;
all369 prior task lines/criteria and244 raw240 nonseparator tables remain.

Actual future source/seed/role/request provenance, original failed resource
350.7925872/360spent/9.2074128remaining, all prior unknowns, negative precision and
full b2/b3/d2b/d2/P6.7/d3 criteria stay required. Independent replications remain
unknown, fresh authority false;25 helper cases remain unbound and all helper
integrations held. No scientific seed/config/baseline/metric/cap/stop change.

Evidence: stream-declaration-contract/handoff-source.json, handoff-validation.json,
handoff-operation.json and handoff-final-log-append.json/final-append-operation.json.
Exact next action: P6.7d2b2/b3: define the complete pure prospective source/request envelope and compose the new400-stream check. Start with all480 ordered role declarations and their expected counts/availability: implement typed role/source/request identity preflight and late omission/reordering/overlap/mutation tests on the full560-cell/116-contrast layout, with actual final values unavailable. Keep actual source/request/seeds unset, all prior unknowns and fresh/execution authority false. Bind or explicitly revise each remaining helper case only against the real full envelope; no dummy adapter or skipped coverage. Resolve the original failed resource acceptance explicitly without resetting 350.7925872/360spent or9.2074128remaining. Complete source/version and b3 isolation/parity/resource/reproducibility/artifact proof before science or integrating the held CI/P6.4 proposals.
