# P6.7d2b1: retained JSON seed declarations

## Scope

This is the first boundary of the complete prior-usage audit. Read every retained
JSON metadata file in the checkout, including failed operations and preserved
copies. Freeze whole physical bytes and ordered corpus membership before checks;
check them again after parsing. Runtime/cache directories and this new exclusively
owned output transaction are the explicit exclusions. A virtual environment is
recognized by its `pyvenv.cfg` marker, including the existing CPU/CUDA snapshots;
library example seeds are not repository experiment metadata. Refuse an unowned directory
or an external/symbolic path rather than hiding it from the inventory.

Keep regular seed fields, canonical dictionary/tuple forms, whole embedded JSON
configuration strings and recorded CLI seed arguments. Preserve every declaration's
file identity, declaration location, field, value and representation. Keep unresolved/null/
invalid declarations and unparsed files visible. Two known `distinct_*_seeds`
coverage quantities are excluded as counts. Other recognized seed-named fields
remain conservative declarations, including quantities that need schema context;
they do not prove that a source was actually constructed.

Locations use escaped JSON path segments for ordinary fields and canonical
mapping rows. The `embedded_json` segment denotes a decoded string; a decimal
text token index denotes its normalized comma-separated declaration. These
virtual segments describe the declaration view and cannot always be directly
dereferenced in the raw JSON. Whole file bytes, the field and representation
remain bound so the original scalar or embedded configuration can be recovered.

This JSON inventory is **not the complete prior-usage acceptance**, execution
proof or fresh-role authorization. Text/CSV provenance, source/factory defaults,
symbolic seed expressions, derived RNG streams, previous role release and every
remaining ambiguous declaration still need their own evidence in d2b2. Unknown
historical use stays unknown. No seed, winner or outcome is selected here.

## Why this increment

The old runner/validators bind already scored historical confirmations. Code
inspection finds source B uses `base + 101`, A/B role splits use `base + 17`/
`base + 138`, B exposure uses `base + 118`, models use `base + 1001` and parent
selection uses `base + 5001`. Checking only base-seed membership misses possible
stream overlap. D2b2 must verify these bindings and the complete prior usage
before declaring any fresh source/final roles. No derived RNG is sampled here.

Split d2b into metadata inventory, complete usage/untouched-role contract and full
execution fixtures. Preserve the original d2b/d2/P6.7 acceptance and all unfinished
work. The original five-point precision target remains uncertified at ten seeds;
future design must keep an honest exploratory precision status and fixed caps.

## Modules

```text
src/core/seed_usage.py              # pure full decoded metadata traversal
src/infra/seed_usage_inventory.py   # whole physical corpus discovery/read/pins
tests/test_seed_usage.py            # all representations and unresolved cases
tests/test_seed_usage_inventory.py  # membership/corruption/ownership/no science
```

The infrastructure reader depends on the pure core. Core has no filesystem,
model, dataset, RNG, training or scoring import. No new dependency is added.
The API accepts the entire prospectively frozen file tuple; it refuses subsets,
reordering, added files, changed identities and late physical corruption.

```python
from pathlib import Path
from src.infra.seed_usage_inventory import (
    freeze_seed_metadata_files,
    read_seed_usage_inventory,
)

root = Path(".")
files = freeze_seed_metadata_files(root)
report = read_seed_usage_inventory(root, files)
assert not report["fresh_roles_authorized"]
```

The fixed output directory is `artifacts/runs/p67-untouched-seed-usage` and requires
the exact ownership marker. The actual producer must establish that it did not
exist at entry, claim it exclusively, and bind its owner before excluding its
new publications. All pre-existing history remains in the input corpus.

## Prospective local gates

Before any actual full parsing, freeze source/tests/method/file membership and
whole identities. Each fixture/static/catalog/metadata operation has a hard
180-second cap. Declare two complete actual inventory reads, each hard180, with
a shared360-second envelope including any failures. Do not increase an exhausted
budget or replace complete membership with representative files. These are
metadata operations; the scientific600-second/16,000-update/512-MiB envelopes
remain unchanged. There is no new source, training, scoring or sweep permission.

## Accepted metadata evidence

P6.7d2b1 passes its metadata gates. The complete frozen corpus contains **593
files / 2,690,579,418 physical bytes**. All four complete outputs agree byte for
byte: **123,220,118 bytes**, SHA-256
`d5be21d468cb225f41504ab086f05babbc74bd30a657698e8c256c491970525f`.
They retain **494,389 declaration occurrences**, **97 conservative numeric
values**, **30,318 unresolved fields** and **zero unparsed files**. Copies and
repeated declarations are retained; these counts are not independent trials.
No claim that all 97 values are source seeds or actually executed is made.
For example, `bounded_mean_required_seeds`, `distinct_source_seeds` and
`persistent_labeled_array_bytes_per_seed` still require semantic classification.

| Unresolved field | Value type | Occurrences |
| --- | --- | ---: |
| `by_seed` | `dict` | 5,586 |
| `parent_method_facts_by_seed` | `dict` | 10 |
| `projected_feature_seconds_per_seed` | `float` | 2 |
| `python_hash_seed` | `NoneType` | 12 |
| `scored_seeds` | `dict` | 192 |
| `seed` | `NoneType` | 24,460 |
| `seeds` | `dict` | 56 |

Null policy/hash seeds, per-seed record containers and duration/quantity fields
remain explicit. D2b2 must resolve their complete schema/source context or bind
the uncertainty conservatively; null does not establish historical non-use.

The corrected gates pass **130 tests: 30 new + 100 related; zero skips**.
Ruff over `src tests scripts`, formatting of the four new code/test files,
configured mypy over 529 sources and `git diff --check` all exit 0. The initial
passing version is preserved completely. A subsequent review added symbolic/
external containment checks for the owned directory and its marker, plus two
deterministic fixtures. Parsing and the input corpus did not change.

Every operation obeys its prospective hard cap. All four actual full-read
parents, including both passing versions, total **301.2022952 seconds < 360**;
each is under 180. Catalog, fixture and static version totals are respectively
15.1238912, 18.6775069 and 5.1516408 seconds < 180 each. The independent full
readback validates every row, ordered file identity, count, numeric set and all
four complete bytes. Original source/test/input/proof pins are checked before
and after; all 24 scientific guards remain zero. No new source/model/RNG,
training, scoring, final-role or scientific reader is dispatched.

Artifacts are in `artifacts/runs/p67-untouched-seed-usage/`: `source-v2.json`,
`tests-validation-v2.json`, `static-validation-v2.json`, complete output/repeat,
`complete-readback.json` and the full v1 copies/repair receipt. The source is
149,330 bytes/SHA `a75b3666dea85d600a0830e636f93a81341ad0443466be6ec685bde2780c9c68`;
readback is 5,166 bytes/SHA `3fbbd8bf2ff35a234a15703cff28056764cbdd9dc5097a6c920b3637a19fb8f1`.
The session log and terminal handoff record the preservation closing outcome.
All original P6.7/d2/d2b criteria and later work stay unchecked and unchanged.
`fresh_roles_authorized`, `complete_prior_usage_acceptance` and original P6.7
acceptance stay false. Scientific caps/metrics/baselines/precision target remain.

### Commands and extension

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_seed_usage.py tests/test_seed_usage_inventory.py tests/test_seed_precision_budget.py tests/test_continual_precision_feasibility.py tests/test_pilot_precision.py tests/test_continual_pilot_variability.py
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src/core/seed_usage.py src/infra/seed_usage_inventory.py tests/test_seed_usage.py tests/test_seed_usage_inventory.py
.\.venv\Scripts\python.exe -m mypy
git diff --check
```

The actual exclusive producers were `artifacts/runs/p67-seed-usage-gates.py`
and `artifacts/runs/p67-seed-usage-gates-v2.py`, each with `prepare`, `tests`,
`static`, `inventory` and `repeat`. Their occupied publications must not be
rerun. Use a prospectively bound new transaction for later audits. Add semantic
schema/source classification in a separate d2b2 module, preserving every b1 raw
declaration and unresolved location. Do not infer execution from field names.

## Exact next action

P6.7d2b2: audit complete remaining text/CSV/historical provenance, schema/source/factory/default/symbolic expressions, all derived source/split/exposure/model/selector RNG streams and release chronology; resolve or conservatively bind every declaration ambiguity before freezing fresh independent ordered roles, unchanged informative/matched configurations, fixed count with honest exploratory precision status, caps/analysis/stopping/source/request identities. P6.7d2b3 supplies full execution fixtures before d3. No new confirmation, favorable selection, narrowed scope or cap increase.
