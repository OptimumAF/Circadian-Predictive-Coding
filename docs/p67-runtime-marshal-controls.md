# Whole marshal reference controls (P6.7d2b2j4a)

## Responsibilities and evidence

This diagnostic increment compares full marshal bytes with every native public
code field and recursive constant contents. It does not modify production code
or prove loaded code came from trusted disk source.

CPython's [`w_ref` writer](https://github.com/python/cpython/blob/3.14/Python/marshal.c)
checks unique-reference status before writing object reference flags. Its code
writer also serializes metadata such as the line table. The local Python3.14.7
controls causally reproduce these reference effects; the moving upstream branch
is an explanatory primary source, not a build/version attestation.

18 current diagnostic tests pass0 failures/errors/skips; full Ruff/new-file format/
no-incremental mypy598/diff pass. All68 earlier source/test files unchanged, with
one new test module. Prior full548 runtime regression and49 complete+11 partial
readback receipts stay pinned at identical68 bytes; those expensive groups are
not reexecuted or claimed as a new566-case full gate.

Six fresh non-interned str/bytes/float/tuple/frozenset/nested-code controls keep the
same actual code/constant identity and every public native field/recursive value.
Adding one reference changes marshal bytes (a FLAG_REF128 bit is observed); native
decode yields the same full public contents. Reference observations2->3->2 and
byte-for-byte reversal after release isolate the reference-lifetime mechanism.
Six full public-field/recursive-constant retention controls preserve raw bytes on
actual first return and release. An actual exception raised by controlled code
retains the same message and its real traceback; raw bytes remain unchanged.
Genuine constant/bytecode/nested-code replacements alter contents and raw identity.
Two metadata controls show an unchanged original code's shared co_linetable alias
changes serialization without native-field retention and stays stable with it.
Every54 current raw serialization/complete native field snapshot/reference outcome
is saved and independently decoded/read back; no code/graph filtering or byte
normalization. All public native fields, including deprecated co_lnotab, inspected;
73 deprecation warnings in the final pytest run are retained, not skipped fields.

All three historical full drift groups read and independently checked: first-use
three changed code-hash rows/no added function; malformed-first one changed row;
late-request one changed row plus one actual lambda. Every saved related compressed
body/difference/malformed artifact retained. Three earlier tuple/list comparison
entries had equal saved JSON values, while three hash differences remain actual.
Historical raw marshal/code contents were never saved: they are not reconstructed,
and these controls do not retrospectively certify that all old drift was harmless.

Initial16 gate7 pass9 fail0 skip: six fixture-return failures from an assumed None
slot that Python3.14 omits; three shared line-table reference failures. Full v1
test/producer/XML/outcomes and pre-correction18 raw field-probe records retained;
the probe's None return is explicitly excluded from successful first-return proof.
Before correction, the plan recorded actual None assignment and strengthened
retention to every public native field/recursive constant plus two metadata cases.
All original criteria/caps preserved. Later18 gates pass at saved source versions;
final adds a genuine controlled-code exception/traceback and explicit type narrowing.
First static gate fails two mypy errors (object Iterable and optional traceback),
fixed by explicit assertions without suppressions. First independent readback fails
an anticipated source_version_attested key; actual V1 source_attestation field is
checked exactly in the correction. Whole failed producer and1.1282154s tool parent
failure retained/charged. All failures remain in their original budget families.

Only diagnostic j4a may complete after whole2042/current-historical Git/original150
sources79 tests174 inputs/69 code39 docs/381 prior criteria244 tables25 Unbound/
three held proposal preservation, one original END and24 guards0. Production
observer retention, private localsplus/kinds/native/source-version correspondence,
opaque/omitted/nested-route/transient gaps and all arrival/prior/ledger/b3 gates
remain mandatory and unfinished. No production observer/audit/semantic reader,
scientific source/RNG/arrays/model/train/final/scoring/repeat, original closure rekey,
dependency/config/environment, held proposal integration or baseline/metric/seed
change. Scientific execution/freshness/precision false and independence unknown.
Original failed400s/500s/output221533189/j_180s/scientific360s gates unaccepted.

## Run the focused controls

```powershell
.venv\Scripts\python.exe -B -m pytest -q tests/test_runtime_marshal_controls.py
.venv\Scripts\python.exe -B -m ruff check --no-cache src tests scripts
.venv\Scripts\python.exe -B -m ruff format --check tests/test_runtime_marshal_controls.py
.venv\Scripts\python.exe -B -m mypy --no-incremental --cache-dir nul
git diff --check
```

Owned single-use diagnostics/gates/readback/whole closing receipts and complete raw
artifacts: `artifacts/runs/p67-untouched-seed-usage/prior-evidence/runtime-marshal-controls/`.
Do not rerun occupied supervisors; ordinary focused pytest uses no repository output.

## Next extension

P6.7d2b2j: prospectively declare a production code/value-retention increment. Use j4a's full raw/public-field/reference controls to retain every actual observed code object, all recursive constants and native serialized metadata before the first runtime observation; inspect CPython private localsplus/kinds/alias coverage rather than assuming public fields suffice. Preserve unchanged68/current69 code, original150 source closure and all failed j3_400s/j3a_500s/output221533189/j_180s/scientific360s gates. Require complete actual whole-graph before/use/finally controls, genuine code/default/namespace/native mutation denials, lease release/reacquisition, full prior548+new18 cases (566) with0 skips at current bytes, whole preservation/one original END/all guards0 and separately justified budgets before completing stabilization. Then close trusted full disk-to-loaded-Python/native/source-version, opaque type/omitted unreferenced membership/nested endpoint route/ordinary MAKE_FUNCTION/non-audited transient gaps before runtime admission. Continue actual sequential arrival/immutable ledger/full five saved inputs4471 contents13149 aliases/current-historical Git/prior effects/all25 Unbound and every b3 isolation/init/parity/resource/repeat/artifact/independent-readback requirement. Scientific seeds/source/arrays unset, independence unknown, execution/freshness/precision false; no baseline/metric/seed tuning or held CI/P6.4 integration.


## Terminal acceptance

P6.7d2b2j4a complete after18 controls/static598/complete raw+historic readback/
whole2042 preservation/one original END/24 zero guards. Production observer/source
admission remains open. Receipts and exact next action are in the development log.
