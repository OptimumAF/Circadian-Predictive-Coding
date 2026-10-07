# P6.7c1c1: Complete scored JSON and training-reference readback

Declared and implemented: 2026-10-01. No actual reserved final value opened.
See ADR-0159 for the prospective split and rationale. C1c2 still owns the
full scored request, worker, observed execution/resources and artifact lifecycle.

## Boundaries and usage

```text
src/app/continual_confirmation_scoring_validation.py   pure complete scored JSON
src/infra/continual_confirmation_training_references.py sequential verified references
scripts/inspect_p67_scoring_training_references.py     read-only inspection adapter
tests/test_continual_confirmation_scoring_validation.py
tests/test_continual_confirmation_training_references.py
tests/test_p67_scoring_training_reference_inspection.py
```

The app validator requires the unchanged complete scoring manifest. It reads
all 60 family/seed rows, 120 final views, 1,680 endpoint records and 560 cells
in fixed order. Require all three exact training result SHA/length/inventory
proofs and unchanged app authority/seal fields. Final IDs/counts follow the
original training validator; shared phase/seed signatures agree. Independently
derive correct/count accuracies, explicit numerical failure strings, null
cells and every total from the endpoint records. Require exact saved links
and recursive JSON field/type/value identity. No rounding or metric change.
Do not reuse the producer's cell helper. Whole typed outcomes remain compatible
with all 580 paired seed observations and 116 predeclared statements.

This establishes declared JSON links, not actual arrays/live model state,
observed calls, file bytes or resources. The returned app authority remains
false. The future process boundary must establish those facts separately.

```python
from src.app.continual_confirmation_scoring_manifest import fixed_scoring_manifest
from src.app.continual_confirmation_scoring_validation import verify_scored_payload

checked = verify_scored_payload(already_decoded_scored_json, fixed_scoring_manifest())
assert not checked.external_execution_verified
```

The infra reader accepts a complete-reader port; the inspection adapter
supplies the actual unchanged pinned public training reader. Verify both
bundles' six exact request/result/audit hashes and result lengths before the
first read. Read each complete body independently, compare decoded canonical
JSON SHA/length incrementally with its original bytes, bind the original
manifest/scope/source-map/adapter and check bytes/markers again. Only small
reference/cost/historical resource metadata survives a read. Check all six
files/markers again after the last reader, including earlier-bundle drift.
Require identical complete derived work and observed update counts across
the two successful bundles. There is no second retained decoded result.

Why this: the old public reader already validates complete raw training,
checkpoints, costs, request, observed work/resources and audit. Reuse that
code and discard its 134-MB result graph before reading the next bundle.
One-MiB file chunks and incremental canonical JSON avoid another full encoded
string. This is a memory-conscious design, not a newly measured RSS claim.

Requires both original ignored completed training bundles and their existing
scope/development evidence. Choose a fresh inspection destination; outputs
are exclusive. This command performs existing unscored readback only:

```powershell
.\.venv\Scripts\python.exe -m scripts.inspect_p67_scoring_training_references --result-file artifacts/runs/p67-reference-inspection.json
```

Occupied outputs fail before either reader; a later writer collision preserves
its bytes. Compute intended metadata SHA before publication and verify actual
bytes afterward. Inspection metadata is not a scientific scoring bundle.
`verify_training_reference_bytes` can later recheck the six fixed files without
decoding their bodies beside held models. Complete readback before the child
and all late scoring/serialization checks remain mandatory.

No new dependency, environment variable, baseline, seed, metric or cap.

## Source freeze before fabricated scored fixtures

The conservative static local closure is now **90 files**: authoritative
c1b V2's 87 unchanged files plus these three production modules. Conditional
unused Torch/package imports remain included. No scientific algorithm pin
is replaced. The freeze extends V2 record SHA `00226f6b...5c246` and is metadata
only; `full_scored_worker_bound` remains false.

| Artifact/declaration | SHA-256 | Bytes |
|---|---|---:|
| `artifacts/runs/p67-confirmation-scoring-readback-source.json` | `7f44987225e0e0be4e9dbae861b670d65aeddb0e4c249cabe33030fb494768de` | 11,396 |
| Source map, compact canonical JSON | `e1db9b94e7c0db5695fc391156e6ec48f264e98af93d2f9e83ffb18b8dd11b68` | — |
| Unchanged scoring manifest | `76cf873e5942a661bdb76e6fd7f28fc490fc6b8001e2ae0ccc4afe87063a4223` | — |
| Unchanged analysis declaration | `5e33ef28862bcdf9d92fe14dd6cf6b71672a2336ffd760a1214ef04666b594b1` | — |
| `artifacts/runs/p67-confirmation-scoring-train-reference-readback.json` | `cc1c1deb4c721af5d8250f17783c9daada2501c108be91f825fa37324626b001` | 72,050 |

Freeze UTC **2026-10-01T08:39:58.629325+00:00**, before fabricated scored JSON
and reader/publication fixtures. No production source changed after freeze;
all 90 current pins revalidate. No repair or superseded c1c1 freeze.

## Correctness and actual readback evidence

The first new fixture runs fail at collection because each new implementation
module is absent (one error each, 0.42/0.41 s). Add the production modules and
format/type check before freezing. All **197 new tests pass in 14.89 s**, zero
skipped, on the frozen sources. They cover the entire fabricated matrix,
all numerical failure policies (first/last/all calls), every late proof/role/
endpoint/cell/count/order/type/unknown-field/authority error, independently
derived totals, and resealed shared-source drift. Source/model/train/final/
RNG/file sentinels hold during pure readback. The full-scope fixture uses an
explicit global-state spy and metadata model tokens, never reserved sources.

IO-only tests use deliberately incomplete scientific bodies through private
fixture seams. Public gates reject those nonfixed manifests before IO.
They cover missing/changed bytes/lengths, failure/claim markers, detached
decoded bodies, resealed original declaration drift, reader failures and
first/last/earlier-bundle drift. Adapter tests cover exclusive output, foreign
writer collisions, changed publication bytes, nonfinite metadata, real CLI
help and exact conservative dependency closure. These spies are not training
or resource evidence.

The actual public inspection adapter's CLI `main()` independently reads both
original complete unscored bundles through the pinned reader, under a
**180-second validation timeout** with source/model/train/prediction/final/
outer-role sentinels. Exit **0**, **28.3516246 s** including the guard child's
startup, **zero forbidden accesses**. It exclusively publishes the 72,050-byte
inspection metadata above. Both original 134,554,378-byte results remain SHA
`3d85c606...89e547`. Exact request/audit identities and all 60 seed costs are
retained; the saved source/adapter/scientific/analysis declarations revalidate.

Each old bundle independently retains 560 cells, 13,440 wakes, 1,724 applied
and 46 rejected executed replay updates = **15,210 actual optimizer calls**;
BP/PC/circadian partition 3,708/3,948/7,554. Guards 770/evaluations 1,540/examples
28,320; retained labeled arrays 46,080 bytes before copies, width fourteen.
Historical training RSS peaks 462,479,360/462,348,288 and worker times
32.5046013/32.6507508 s are carried from the original verified audits. They
are not new scoring measurements or an inspection memory benchmark.

The complete combined regression gate passes **596 tests in 114.01 s**, zero
skipped, including all 197 new cases and 399 prior related cases. Commands
are recorded in the current development log. Ruff/six-file format/mypy
(408 files) pass.
Full CPU/CUDA/clean-clone/actual CI, new reserved training/scoring, sweeps and
actual intervals are skipped; no selected test skip or dependency change.

## Handoff history and current next action

Subsequent c1c2 progress: the [final execution observer](p67-confirmation-final-observation.md)
and pure whole observation links now pass their development/fabricated checks,
on a linked 92-source available component V2 freeze. This does not bind the
full request/worker or complete c1c2 by itself. The subsequent
[full worker](p67-confirmation-scored-worker.md) now completes all original
remaining criteria below on a prospective 97-source V2 freeze.

The required **P6.7c1c2** extension of this unchanged 90-file composition was:

1. Extend/freeze the entire new scored worker/adapter closure and strict
   request/command/environment, binding both verified training bundles and
   the unchanged analysis declaration. Do this before fabricated fixtures.
2. Compose unchanged training/c1a/b in a bounded child; avoid another full
   decoded reference result beside held models. Verify current source/request/
   reference bytes before data/release and after scoring/serialization.
3. Externally observe optimizer calls and final release/input-label/prediction/
   example calls, including rejected replay. Preserve 16,000 updates/600 s/
   5-ms observed 512-MiB limits. Keep final views for post-serialization checks.
4. Test every late state/content/endpoint/source/request/reference/resource/
   serialization failure and exclusive request/result/audit/failure/readback
   using development/fabricated fixtures. No reserved final until all c1 gates.

C1c1/c1c2/c1c/c1 correctness is complete with 984 related passing tests,
guarded actual complete reference preflight and bounded development execution.
Both actual full 560-cell scored processes, complete independent readbacks
and exact deterministic result bytes now pass under unchanged caps (see the
[complete results](p67-confirmation-scored-results.md)). **Next: P6.11b**,
exhaustive seed/interval/raw-cost reporting. P6.11b and original scientific/
resource/reporting parents stay open.
