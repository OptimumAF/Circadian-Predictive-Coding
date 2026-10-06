# Saved historical environment evidence

This is a scoped evidence inventory for P9.3b2a, captured on 2026-10-06 at
checkout `28e71ee9de46230fdd6232cca3f9f59ba8ff4fb8` with its existing local edits.
Original P9.3b2 remains unfinished. The inventory reads saved metadata;
it does not rerun experiments or establish original execution or source admission.

## Files and scope

- [Machine-readable ledger](../constraints/published-run-environments-2026-10-06.json).
- [Current dependency snapshot and fresh-install validation](dependency-reproducibility.md).
- Local full inputs, producers and verification receipts:
  `artifacts/runs/p93-dependencies/historical-environments-20261006/`.

The ledger contains 134 saved request/audit/manifest or legacy benchmark files,
plus all 10 nonignored JSON files from the frozen checkout. Those extra inputs
include three root benchmarks, document metadata, three configuration files,
two test fixtures and today's constraint snapshot. Their origin labels keep
fixtures, configurations and current versions separate from historical records.
Every selected JSON body and all 318 frozen Markdown bodies are retained byte
exact locally. File hashes identify the original input of each saved field.

There are 893 saved environment, version, hardware, precision, determinism and
device fields, with their exact JSON pointers and values, and 1,320 provenance
locations. There are 813 literal reference tokens at 1,339 document locations.
These are counts of records and occurrences, not independent runs. Nested
upstream declarations keep their original location; they do not describe the
enclosing report's execution environment.

## Examples of recorded versions

| Original saved record | Recorded fields | Limit |
|---|---|---|
| P5.7 `p57-repro-a/manifest.json` and `p57-repro-b/manifest.json` | `/dependency_versions/python`: `3.14.7`; `/dependency_versions/numpy`: `2.4.6`; saved CPU/precision facts | Two saved manifest declarations; no complete transitive/tool/wheel inventory |
| P6.2/P6.3 request files | Explicit `python_version`: `3.14.7`, `numpy_version`: `2.4.6` | Saved requests; this inventory does not newly verify execution |
| P6.7 confirmation train/scored requests | `/environment/python_version`: `3.14.7`, `/environment/numpy_version`: `2.4.6` | Original environment declarations; original seed/source/admission limits remain |
| Legacy `benchmark_cifar_pretrained_cuda_v1_request_smoke.json` | `/torch`: `2.14.0+cu130`; `/torchvision`: `0.29.0+cu130` | Saved CUDA request; do not replace with today's CPU versions |
| Today's dated Windows CPU constraint JSON | Current snapshot in its own labeled record | Describes the separately tested current environment, not missing historical facts |

The complete ledger retains every selected record, including ones with no
version fields. Exact values are not coerced or flattened. Missing original
transitive packages, tooling, installation/index details and wheel/build
identities remain unresolved. Current metadata in older snapshot directories
does not establish their historical execution environment.

## Coverage that remains open

Literal references are classified as selected inputs, existing directories,
unread files, missing paths, embedded paths, symbolic/compound references or
excluded operational/current validation. The scanner does not interpret usage
examples as actual publications or silently expand templates. It captures
artifact/data paths, bare benchmark JSON names and document JSON paths in frozen
Markdown; other reference formats may require manual reconciliation.

Outside the 134 artifact candidates, 420 artifact JSON files total
3,411,076,715 bytes. Their names and observed sizes are inventoried; their bodies
were not read or hashed by this stage. Documented ignored `data/` results also
remain unreviewed. An inventory of literal references does not prove complete
publication coverage. The ledger sets `complete_original_publication_coverage`
to `false`; P9.3b2/P9.3b/P9.3 stay unchecked.

Why this split: inspection found nonstandard legacy/result schemas and large
saved derivative bodies. Recording the inspected subset with its gaps preserves
the original acceptance criterion and prevents falsely attributing a complete
environment to a historical run.

## Validation and safe extension

Two complete derivations have identical bytes: 1,324,990 bytes, SHA-256
`0107b295f1b014abc0505b8715eecf677947030e4d518e2afa6bf8e8a436ff7c`.
Independent readback checks every selected field, provenance location and
reference occurrence against the complete frozen input bodies. All 15 behavior
checks pass: nested differing environments, null/type retention, escaped
pointers, invalid/duplicate/nonfinite JSON, changed membership/bytes, omissions
and boolean/number conflation. All 969 repository files outside the two
preflight document updates, current package versions and HEAD were unchanged.

Exact commands and failures are in the development log and local receipts.
Validation is local, with a declared 180-second aggregate, 45-second operation
limits and 64 MB of owned receipts. No scientific execution, dataset/weight
download, installed-package or production/test change was needed.

Next: P9.3b2b, classify original publication references and freeze the complete
saved primary records lacking selected metadata. Start with the existing
`data/cifar-representative-confirmation-v1-result.json` and its saved request,
manifest, feasibility and gate records. Add their original fields in a new
versioned ledger under a declared metadata budget. Preserve missing references,
this immutable ledger, all original bodies and earlier failed scientific budgets;
use no scientific reader or new experiment. Resolve large unreviewed bodies
before claiming full original publication coverage.


## Complete available saved-record review (P9.3b2b)

The immutable v1 inventory above remains as its original scoped evidence.
Its unread-structured-body gap is now resolved by the separately versioned
[v2 ledger](../constraints/published-run-environments-2026-10-06.v2.json) and
[lossless field sidecar](../constraints/published-run-environments-2026-10-06.v2.saved-fields.json.gz).
These are evidence data, not pip installation constraint files.

The successor covers all 640 available JSON files in the declared original
artifact/data/nonignored-record scope: 3,430,491,542 complete bytes, including
the previously unread 420 artifact files and all 76 data JSON files. It also
covers all 68 saved journals, 9,166,614 complete bytes and 13,558 lines. Complete
SHA-256 and membership checks precede/follow validation; originals remain at their
original paths. Smaller bodies and all 319 Markdown documents are additionally
archived in full. No input was sampled, rewritten or deleted.

Full byte-identical input groups share one analysis per independent pass, with
each original file path/origin retained: 515 unique JSON bodies and 19 unique
journal bodies. This storage rule does not count duplicate files as independent
runs. Two full passes have identical metadata values. Independent forward
readback verifies all 40,190 JSON field locations and 279,384 provenance locations,
and every journal line with 1,053 field locations. The sidecar preserves every
complete field value, pointer and provenance identity; the readable ledger gives
input identities, context labels and exact known dependency-version assertions.

Why compression: the largest full profile yielded 33,984,063 bytes of metadata.
A deterministic gzip representation retains every value and fits the original
128 MB output cap. Full decompression/parity was checked. Schema `format_version`
identifiers are retained in the sidecar but are not software package versions.
Nested metadata remains at its original pointer and is not attributed to the
enclosing report run.

### CIFAR result-family finding

The original `data/cifar-representative-feasibility-v1-request.json` records
Torch `2.14.0+cu130` and torchvision `0.29.0+cu130`. The saved study request,
selection manifest and later confirmation aggregate/per-scope result files
do not record dependency versions. Their original environment stays unknown;
neither that earlier feasibility request nor today's retained environment can
supply the later confirmation's execution-time versions. No scientific run,
final-source reader or metric recomputation was invoked.

### Validation and remaining publication scope

All 26 local behavior controls pass, including exact nested types/signed zero,
invalid/duplicate/nonfinite JSON, full-byte drift and positive/negative actual
worker memory controls. A first memory control exposed the venv launcher's PID
being sampled; that failed control and original profile remain retained, and
the profile's original memory claim is reopened. The corrected profile samples
the actual Python worker, with observed peak about 1.07 GB. Full passes/readback
remain below 4 GiB and their 45-second per-operation limit. A first independent
readback timed out at 45 seconds; it remains unaccepted and charged. Exact typed
value comparison avoids repeated serialization; the same full 515-body readback
then passed in 43.173 seconds. No input, criterion or time/memory/output cap changed.

The v2 ledger inventories 828 literal reference groups at 1,371 locations in all
319 frozen documents. It distinguishes commands/examples, narrative candidates,
symbolic references and operational/current diagnostics. These contextual labels
are rules, not exhaustive publication authority. Missing original files, embedded
paths, symbolic/compound references and non-JSON formats remain explicit. The
ledger retains `complete_original_publication_coverage: false` and
`complete_original_execution_environment_facts: false`.

P9.3b2b completes the available structured-record reconciliation and publishes
that explicit incomplete-coverage ledger. P9.3b2/P9.3b/P9.3 stay unchecked.
Next is P9.3b2c: build a manual publication register from README's actual results,
the fixed-v14 reproducibility scope, and the P6.2/P6.3/P6.7/P6.12 result guides.
Bind each named publication to original record identities or explicit missing
originals; distinguish commands, diagnostics, current constraints and prose-only
claims. Record missing versions without retrospective assignment. No original
scientific replay or deferred guard repair is needed for that register.

Complete producers, raw failure outputs, profiles, repeated fields and readbacks
are retained locally at `artifacts/runs/p93-dependencies/published-coverage-20261006/`.
The declared aggregate remains 300 seconds, 45 seconds per operation, 4 GiB per
worker and 128 MB owned outputs. Earlier failed scientific/runtime budgets are
unchanged. Source/tests/framework versions, seeds, baselines and metrics are
unchanged; the 331 prior backend tests are not rerun for saved-data/doc changes.

To read exact values without an application dependency:

```python
import gzip
import json

with gzip.open("constraints/published-run-environments-2026-10-06.v2.saved-fields.json.gz", "rt", encoding="utf-8") as stream:
    fields = json.load(stream)
# A ledger record's analysis_key selects its complete original saved fields.
```

Preserve v1/v2 and original bodies. Add a new immutable evidence version when
the publication register or new original metadata is available; do not assign
current installed package versions to the missing historical facts.


## Explicit original publication register (P9.3b2c)

[Published experiment register](published-experiment-register.md) and the separate
[immutable publication ledger](../constraints/published-run-environments-2026-10-06.publications.json)
bind62 named result/evidence families to their original saved records or explicit
missing originals. Their scope is all319 frozen repository Markdown documents,
2306 complete context sections,1380 literal reference occurrences and all708
original JSON/journal inputs plus16 original text/figure/dashboard records.
Each original identity, source paragraph/section and exact saved dependency
pointer is retained. All848 saved version assertions across100 structured files
are independently joined to the unchanged lossless v2 sidecar.

Requests, corrected development results, confirmation results, derived reports,
illustrations, example commands, current constraints and operational diagnostics
retain separate reviewed roles. Symbolic related file sets are matched only to
the frozen concrete inventory. Explicitly missing benchmark originals have their
own unresolved-original labels; the initial generic labels and both initial
outputs remain retained. Both corrected full derivations are byte identical;
all19 behavior controls and complete original/source readback pass.

This completes the publication-register scope for the frozen documentation and
available original inventory. It does not recover the missing historical chart
source/per-seed records, the raw48-epoch record or full original execution
environments. Earlier feasibility CUDA assertions still belong only to their
actual original request; later confirmation versions stay unknown. V1/v2 ledger
coverage flags remain immutable historical statements of their own scope.

Local evidence is retained at
`artifacts/runs/p93-dependencies/publication-register-20261006/` under the original
180s aggregate/45s operation/32MB output budget. No source/test/framework,
seed/baseline/metric, historical result or earlier failed budget changes occurred.
The owning-with guard repair remains deferred. Checklist completion awaits the
final static, whole-preservation and original-acceptance receipts for this session.
