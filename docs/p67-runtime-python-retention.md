# Retain observed Python code values (P6.7d2b2j4b)

## Responsibilities and boundaries

`src/infra/runtime_python_retention.py` gathers every GC/loaded-module native
namespace root and native function binding, then strongly retains reached code,
recursive constants and all public native code fields for the real freeze lease.
Input: the existing native namespace reader. Output: a strong-reference tuple.
The existing runtime observer hashes unchanged full marshal bytes and releases
the tuple in finally. This module performs no IO, source attestation, hash
normalization, graph filtering, scientific work or private ABI inspection.

Why this: unchanged values acquiring a second reference can change marshal's
sharing flags. Constants-only retention misses metadata aliases such as line tables.
The diagnostic mechanism controls in ADR-0191 precede this production increment.

## Private storage inspection and limits

CPython's [marshal code writer](https://github.com/python/cpython/blob/3.14/Python/marshal.c)
uses reference flags and serializes private localsplus names/kinds as well as
code, constants, names and line/exception metadata. Its
[code-object implementation](https://github.com/python/cpython/blob/3.14/Objects/codeobject.c)
constructs new filtered public locals tuples and caches projections; the native
constructor builds private localsplus storage, and replacement routes through
that constructor. Inference: holding public projections alone is not holding the
private tuple/bytes objects themselves. The observed full-byte alias/lifetime
controls prove the declared paths; they do not prove every private sharing route.
These moving upstream3.14 sources explain a mechanism and do not attest the
installed Python3.14.7 binary or a source/build correspondence. No guessed ctypes
layout/private pointer access is used. The parent trusted-runtime gate stays open.

## Results, failures and preservation

Current581 tests pass0 failures/errors/skips, preserving exact original548 plus
marshal18 plus retention15 case IDs. Full Ruff/six-file format/no-incremental
mypy602/diff pass. All96 original payload/continuity/lifecycle control names and
passed outcomes remain exact; denial reasons/routes match, with literal diagnostic
indices checked against their own complete process graphs. New actual retention
contributes11 named controls.

Production retains native graph code objects, recursive constants and every public
native code field before first observation, with unchanged raw marshal identity,
full graph/native membership, audit, scope/sequence/UTC and finally/release logic.
Detached function globals, defaults/keyword defaults, closures, attributes and
native namespace containers are traversed explicitly. No arbitrary descriptor or
foreign metaclass equality/hashing runs. Original function/class lifetime remains;
release reclaims the tuple. Local native getter warning suppression restores caller
filters and omits no field. The old marshal controls keep their original warnings.

The real V2/native-owner fixture joins every observed function and namespace code
to the actual adapter's retained tuple, then proves full entry/use/finally bytes
equal across six actual returns and a real controlled-code exception/traceback.
Code/default/namespace/native changes continue to fail in the full old controls;
the real native owner reacquires after finally. Independently read29 complete
runtime bodies+snapshots+headers/scopes,63 whole raw code/native-value artifacts
from both focused successors and full retention, and42 partial failed raw code
files. Raw code hashes also join to corresponding observed function hashes.
No unsaved failed headers or historical executable contents are reconstructed.

Failures remain preserved: red controls2 pass11 fail (actual-process case excluded);
first focused14 pass1 fail from opaque Pygments imported by a pytest helper;
first static unused binding; next focused and mistakenly routed static-v2 each
14 pass1 fail from fixture validator arity; static-v3 eleven type errors fixed by
native generator/traceback assertions and typed collections, without new broad
suppressions. The misrouted static-v2 is charged to correctness, never claimed as
static. Missing first failure temporary directory/manifest is explicitly unknown;
its failure preceded raw artifact writing. Two later partial directories preserve
every21 raw code file each. All original budgets include failed attempts.
The first independent reader failed an incorrect demand for identical sorted
process-array indices in old/current denial strings. Its producer/traceback and
5.5583523s metadata failure are preserved. Every old/current literal string and
actual graph index is read; matching full reason/route and unchanged96 names/passed
outcomes are required. No observed graph/code bytes are normalized.

Before correction, the plan added detached-globals coverage (581 rather than580)
and a pytest-free shared helper (602 static/73 code/41 docs/2048 physical rather
than601/72/41/2047). Original18 marshal IDs/function ASTs are unchanged. Every67
other current code file is unchanged; both permitted older files are fully saved.
All382 prior criteria,244 table lines,25 Unbound helper rows and held proposals
stay required. Current HEAD28e71ee has13 commits since reviewed8793c49; no new
commit/branch/remote/CI integration in this increment and unrelated changes preserved.

Private interpreter ABI/build/source-version coverage remains unproved. Public
projection retention is not direct ownership of private localsplus/kinds storage.
Full trusted disk-to-loaded Python/native/source-version, opaque/omitted/nested/
transient gaps and all actual arrival/ledger/prior/b3 gates remain unchecked.
Original failed400s/500s/output221533189/j_180s/scientific360s gates remain failed,
without reset/rekey or semantic-corpus reader/repeat. Seeds/source/arrays unset,
independence unknown, execution/freshness/precision false. No baseline/metric/seed/
algorithm/dependency/config/environment change or new scientific dispatch.

## Files and safe extension

```text
src/infra/runtime_python_retention.py    native traversal and lifetime ownership
src/infra/runtime_python_objects.py      existing observer retention wrapper
tests/runtime_marshal_value_fixtures.py  pytest-free native diagnostic helpers
tests/test_runtime_marshal_controls.py  same original18 cases
tests/test_runtime_python_retention.py  fifteen retention behavior cases
tests/runtime_retention_fixtures.py      complete guarded real-process proof
docs/adr/ADR-0192-retain-observed-code-values-before-runtime-freeze.md
```

Add a newly observed binding route through this focused collector and require a
real-code behavior control plus the full existing denial/whole-runtime gates.
Keep descriptor reads native and avoid creating callables after freeze activation.
Source/build/private/transient authority requires its own declared acceptance gate.

## Run and evidence

```powershell
.venv\Scripts\python.exe -B -m pytest -q tests/test_runtime_python_retention.py tests/test_runtime_marshal_controls.py
.venv\Scripts\python.exe -B -m ruff check --no-cache src tests scripts
.venv\Scripts\python.exe -B -m ruff format --check src/infra/runtime_python_objects.py src/infra/runtime_python_retention.py tests/test_runtime_marshal_controls.py tests/runtime_marshal_value_fixtures.py tests/test_runtime_python_retention.py tests/runtime_retention_fixtures.py
.venv\Scripts\python.exe -B -m mypy --no-incremental --cache-dir nul
git diff --check
```

Exact full seventeen-module581-case command, all outcomes/source versions/complete
artifacts, single-use fixed-budget supervisors/readback/preservation and terminal
receipts: `artifacts/runs/p67-untouched-seed-usage/prior-evidence/runtime-python-retention/`.
The normal pytest command writes no owned repository output. Do not reexecute an
occupied supervisor or treat saved observations as a current live lease.

## Next action

P6.7d2b2j: prospectively declare a bounded trusted-runtime/source-correspondence increment. Inspect the actual installed Python/native build and complete disk-to-loaded Python/native/source-version relationships, including private localsplus/kinds/public-projection aliases; produce positive and genuine mismatch controls without source manifests granting live authority. Resolve or explicitly demonstrate remaining opaque type, omitted unreferenced membership, nested endpoint route and ordinary MAKE_FUNCTION/non-audited transient limits before full runtime admission. Preserve current73 code and original150-source79-test174-input closure, every failed j3_400s/j3a_500s/output221533189/j_180s/scientific360s gate and full581 current regression IDs. Then continue actual sequential source arrival, immutable full actual ledger, full five saved inputs4471 contents13149 aliases/current-historical Git/prior effects/all25 Unbound cases and every b3 isolation/init/parity/resource/repeat/artifact/independent-readback requirement. Scientific seeds/source/arrays unset, independence unknown, execution/freshness/precision false; no baseline/metric/seed tuning or held CI/P6.4 integration.


## Terminal acceptance

P6.7d2b2j4b complete after581 full cases0 skips/static602/29 whole records63 rawcode
42 failed partial bytes independent readback/whole2048 preservation/one original
END/24 zero guards. Only the declared public-value retention mechanism is verified.
Private/build/source/transient/full scientific admission remains open. Exact gates,
failed versions and next action are recorded in the development log.
