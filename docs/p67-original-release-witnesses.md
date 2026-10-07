# P6.7d2b2b2: original observed execution and release witnesses

## Responsibilities and boundaries

```text
src/core/seed_release_chronology.py      within-run observer trace positions
src/app/scored_release_witnesses.py     complete decoded scoring/audit links
src/infra/original_release_witnesses.py fixed unchanged complete reader boundary
tests/test_seed_release_chronology.py   ordering and exact count validation
tests/test_scored_release_witnesses.py  full fixed scope, corruption and unknowns
tests/test_original_release_witnesses.py whole IO, late drift and fixture ports
```

Dependency direction: infra -> app -> core. Core describes only the order
enforced by the original observer. The app reuses the unchanged complete scoring
validators and retains exact pointers and canonical record digests. The boundary
verifies whole scoring files and current request sources before/after readback,
then calls the unchanged scoring reader and both unchanged full training readers.
Neither core nor app imports infra. No publisher, worker launch, source/RNG/model,
training, label release, prediction, candidate selection or fresh admission occurs.

The entire historical saved evidence and prior complete chronology remain bound
separate inputs. No alias, raw witness or uncertainty is erased by these records.
No new dependency/configuration/environment variable; existing .env.example and
CONTRIBUTING.md remain applicable. ADR-0178 explains the decision and limits.

## Meaning of the records

Each complete original scoring bundle covers 60 family/seed rows, 560 cells,
120 final role views, 240 source input/target reads and 1,680 endpoint calls.
The derived trace has 2,043 nodes: three whole training-state barriers, every
source read and release, then every prediction. All original numerical failures,
null results, resource observations, executed-update facts and audit links remain.
Each node points into the full original result or observed audit and binds that
entire record. The original full artifact identity protects omitted or changed
trailing records as well as headers.

Within-run causal order is supported by the source-bound observer and complete
readback. Exact actual release UTC and cross-run release order are unrecorded.
Request UTC is documentary; elapsed duration is an observed duration, without a
recorded matching absolute release anchor. Neither is an exact release timestamp.
Different family views may share numeric seeds; deterministic repetitions and
copied bodies do not add independent source replications. That count stays unknown.

The pure dictionary consumer always reports complete_original_reader_verified
false. The private whole-IO reader ports also report false. The fixed public
adapter can report full original-reader readback after it actually calls those
readers; the supervised metadata receipt must additionally pin the full original,
current and historical source/input/proof/document boundary. No interface grants
fresh roles or repairs the original failed current-semantic resource acceptance.

## Usage

The actual readback requires the existing ignored local original bundles. Pure
and boundary unit fixtures use complete fabricated data and temporary files,
without requiring those local research artifacts or proving historical execution.

```python
from pathlib import Path
from src.infra.original_release_witnesses import read_original_scoring_release

# Actual repository root; names are fixed protocol bundles, not selectable seeds.
readback = read_original_scoring_release(Path.cwd(), 'canonical')
assert readback.complete_original_reader_verified
assert readback.chronology['coverage']['prediction_events'] == 1680
assert readback.chronology['exact_actual_release_utc'] is None
assert not readback.chronology['fresh_roles_authorized']
```

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_seed_release_chronology.py tests/test_scored_release_witnesses.py tests/test_original_release_witnesses.py
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src/core/seed_release_chronology.py src/app/scored_release_witnesses.py src/infra/original_release_witnesses.py tests/test_seed_release_chronology.py tests/test_scored_release_witnesses.py tests/test_original_release_witnesses.py
.\.venv\Scripts\python.exe -m mypy
git diff --check
```

## Prospective gates and next action

Gates are pending. The first pure increment passed 37 tests with zero skips.
The complete raw inspection checked 198 pinned files/169 distinct contents,
all 150 original source files/79 original test files and every full original
training/scoring triplet. Its failed parser import and corrected version remain
recorded, sharing the declared inspection budget. No complete original reader
has yet been dispatched for this component.

Freeze the complete source/test/prospective scope, then pass the full meaningful
fixture/static gates. Read canonical and repeated original scoring bundles
independently through their unchanged readers under 120 seconds each/shared240
including failures. Preserve all original/current/history source/test/input/proof/
docs and one original END rebuild at closing before task completion. Inspect
terminal receipts instead of restarting occupied producer names.

Keep original b2a/b2/b3/d2b/d2/P6.7/d3 unchecked until their full unchanged criteria
pass. Original failed semantic repeat remains 350.7925872/360 seconds spent and
9.2074128 remaining. Scientific 600 seconds/16,000 updates/512MiB is unchanged.
Next, bind remaining unknown execution/release effects and the complete future
independent role/config/count/caps/analysis/stopping/source/request contract.


## Completed implementation gates; closing pending

Canonical/repeated full-reader parents pass in62.0712639s/59.9263664s, each<120s; shared121.9976303/240s including failures.

Each original bundle retains all60 family/seed rows,560 cells,120 release views,240 source reads,1680 endpoint calls/67200 examples and2043 causal nodes. One unchanged scoring reader reconstructs both unchanged complete training references per stage. All source/request/audit/state/failure/resource links and full original parts are bound to the prior4471-content/13149-alias ledger; every old uncertainty is preserved. Request UTC is documentary; exact actual release UTC/cross-run order/independent replication count stay unknown. The two original request/audit/resource records remain distinct even though their full causal traces agree. All24 science guards are0 in each accepted stage. Fresh admission and original resource acceptance remain false.

The complete659-case correctness gate passes by603 unchanged existing cases from the preserved full run plus all56 new cases retested with0failures/0skips. The initial full run's single guarded-IO fixture failure is retained. Only that new fixture function changed to closed binary streams; all production/old tests/guards are unchanged. Ruff, six-file formatting, mypy550 and diff pass.

Artifacts are under artifacts/runs/p67-untouched-seed-usage/prior-evidence/release-witnesses/: source-v3.json, tests-v3-validation.json and both JUnit runs, static-v3-validation.json, canonical-witness.json/repeat-witness.json and their validation/operation receipts. Complete physical/Git preservation and one original END rebuild must pass before completing onlyb2b2. Earlier prospective pending sections are retained as historical declarations.

Exact next action after preservation: P6.7d2b2: conservatively bind remaining unknown historical execution/release effects and complete the prospective ordered independent source/seed/final-role/config/count/caps/analysis/stopping/source/request contract, retaining all original informative/matched cells and116 contrasts. Add the complete b3 isolation/reproducibility/resource/artifact fixtures before any d3 source release. Resolve original b2a repeat/resource acceptance explicitly without resetting350.7925872/360spent or9.2074128remaining. Keep fresh admission false; no seed selection, confirmation or sweep.


## Terminal component acceptance

P6.7d2b2b2 is complete after the full145.3345128s preservation gate,
one original END binding rebuild and24 science guards0. All1968 physical files,
whole current/historical Git objects/aliases and original/new source/test/input/
proof/docs are preserved. Complete659-case aggregate correctness (603 unchanged
existing+56 corrected new),0skips, Ruff/format/mypy550/diff and both actual full
original-reader stages pass. Complete causal trace SHA256
318266f467f8d23a3d13c9673c68194bf0aa4b84f91a104a581164127b2dd0e0 agrees in both bundles; whole witness outputs
{"canonical": {"byte_count": 435509, "sha256": "6d757b6ce07b25bf28addbc58ea53b217af075f116c1b9ea7d99912db20e4cce"}, "repeat": {"byte_count": 435540, "sha256": "1aba932a6ecee03f43bdb05db18f66229103a79660c77e9572cee37ef7e0f7f4"}} retain the
separate original request/audit/resource identities. All prior witnesses/aliases/
unknowns stay bound. Original failed resource acceptance/fresh authority remain
false; no actual independent replication count or exact release UTC is inferred.
See handoff-validation.json, handoff-operation.json, final live receipt and
docs/development-log.md for exact commands/budgets/skips/failures/next action.

Safe extension and exact next action: P6.7d2b2: conservatively bind remaining unknown historical execution/release effects and complete the prospective ordered independent source/seed/final-role/config/count/caps/analysis/stopping/source/request contract, retaining all original informative/matched cells and116 contrasts. Add the complete b3 isolation/reproducibility/resource/artifact fixtures before any d3 source release. Resolve original b2a repeat/resource acceptance explicitly without resetting350.7925872/360spent or9.2074128remaining. Keep fresh admission false; no seed selection, confirmation or sweep.
