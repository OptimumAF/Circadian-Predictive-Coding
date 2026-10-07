# Exact-type runtime namespace optimization

## Structure and boundaries

```text
src/infra/runtime_python_objects.py       existing native observer; _namespace optimized
tests/test_runtime_python_namespaces.py   new22 native namespace behavior cases
docs/p67-runtime-namespace-optimization.md this guide
docs/adr/ADR-0188-preserve-full-runtime-parity-for-exact-type-namespace-lookup.md
artifacts/runs/p67-untouched-seed-usage/prior-evidence/runtime-namespace-optimization/
  before/                                complete original source/document bytes
  runtime_python_objects_candidate.py     fixed full candidate before adoption
  complete-original-runtime.json          full actual original canonical body
  complete-candidate-runtime.json         byte-identical full candidate body
  complete-namespace-ledger.json           every actual45793 GC object/native result
  parity-worker.py / verify-full-readback.py
  *-validation.json / *-operation.json     all current/failed gates/pins/budgets
```

Input is the same actual value formerly read by the outer native observer. Exact
dict/list/tuple/set/frozenset type identities return None; all other branches still
read native class/module/instance descriptors. Identity checks cannot invoke a
foreign metaclass's equality/hash. No persistent cache, IO/core/app interface,
configuration, dependencies or scientific behavior change. The native file/GC/
function/closure/default/global/import/entrypoint/executable-memory paths remain.

Why this: j1 measured608760 namespace lookups/0.743523s cumulative. Optimize this
measured boundary only after preserving the complete old implementation and full
actual original/candidate parity. Builtin values may be mutable; their exact native
types have no instance dictionary. Subclasses are always inspected normally.

## Evidence and limits

Whole unfiltered45793-object native lookup parity and complete original/candidate
10,334,507bytes/SHA25693d1e7dabbe6791b3cb94d5711c55b6a2e245ca2f42cea1ff9653c52f6b459db
match exactly without normalization. Actual342 modules/7700 functions/1627 executable
namespaces/51 native images/74 executable regions, full files/import/defaults/closures/
globals/entrypoints retained. Both original/candidate/driver modules/functions stay
in actual membership. The slotted full GC adapter supplies the identical actual
unfiltered inventory only in this parity fixture; production still calls gc.get_objects.
Independent whole-byte/GC-ledger readback and17 pinned-content corruption denials
pass; these attest content fidelity, never semantic source/version/admission.

Fixed original-first all-object lookup0.0182809000s
vs candidate0.0131953000s (27.819% less),
whole Python/native/entrypoint pair1.1505740000s vs
1.1002188000s (4.377% less). One pair;
OS/page/cache/order effects and variance are unmeasured. No seed/workload/repeat
selection or future speed claim. j1 slower lossless codec is retained and unused.

Current single full544 test cases pass0 failed0 errors0 skipped: original502 exact
case IDs plus unchanged20 full runtime cases plus22 new namespace behaviors. Real
runtime groups include positive/reacquisition, module/callable/detached/default/
closure/instance/file/import/code/native executable memory/native loading/consumer/
inactive/request/finally rechecks; each original40-second child limit unchanged.
New controls retain builtin subclasses, descriptor/property/metaclass equality/
hash denial, class/module/wrapper/native inherited dictionaries, live/dead weak
proxies, actual compatible type/MRO drift and late callable members. No production
cache, GC subset, file/native-read bypass, new dependency/environment/config.
Full Ruff,2-file format, no-incremental mypy590 and diff pass. New fixture's11 static
errors fixed using explicit dynamic setattr/delattr and three narrowly documented
readonly-property override ignores; production has no new suppression.

## Commands (PowerShell, existing environment)

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q tests/test_runtime_python_namespaces.py tests/test_prospective_runtime_closure.py
.\.venv\Scripts\python.exe -B -m ruff check --no-cache src tests scripts
.\.venv\Scripts\python.exe -B -m ruff format --check src/infra/runtime_python_objects.py tests/test_runtime_python_namespaces.py
.\.venv\Scripts\python.exe -B -m mypy --no-incremental --cache-dir nul
git diff --check
```

The saved full544 command/case results are in full-gate-validation.json; they add
every original502 module to the two focused modules above. These example commands
describe future local checks, not new executions this session. All actual single-
use supervisors and exact commands/outcomes/failed versions are in the log and
owned receipts; do not rerun occupied saved destinations.

Successor tests hard300s including preparation/failures/unchanged40s child limits;
full parity/static/shared docs-preparation-whole-closing-terminal180 each. Old
j168.7993043/180spent/11.2006957remaining and original scientific350.7925872/360spent/
9.2074128remaining unchanged/unaccepted. Separate64MB saves both full bodies and
whole before/reference proofs; old j16MB/j1_32MB unchanged. No scientific experiment.

## Extend safely

Keep the exact identity shortcut small. Further types require native descriptor/
subclass/proxy controls and full original-input parity before edits; caches require
explicit invalidation for MRO/descriptor/type/member drift. Do not sample GC or
native/file membership to meet a budget. Admission must still prove nested shape/
continuity/trusted source correspondence/transient executable lifetime, full prior/
arrival/ledger/resource/repeat/isolation/parity/artifacts/independent readback.

Exact next action:P6.7d2b2j: prospectively declare the next nested-runtime-schema/continuity increment. The app currently checks outer payload keys but accepts nonempty foreign nested records; implement a small pure closed-schema validator for every actual module/function/code/default/closure/namespace/import/entrypoint/native-file/executable-memory record, exact types/IDs/complete membership and joins, with useful early errors and no IO. Preserve the complete prior app bytes before any permitted edit and all current61 code/original150 closure. Positive/continuity fixtures must originate from full real guarded V2/native-owner/runtime observations; mutate every malformed/foreign/omitted/boolean/float/duplicate/reordered/conflicting identity/lifetime/time/PID/nonce/request/late record and verify before-use/finally rechecks/releases without science. Prospectively justify bounded current full-regression controls, retaining original j168.7993043/180spent/11.2006957remaining and current j2 successor outcomes without declaring old resource acceptance repaired. Then close trusted whole Python/native disk-to-loaded-code/source-version correspondence and ordinary MAKE_FUNCTION/non-audited transient callable gaps before claiming full runtime closure; no recorded hash/caller manifest/shape validation substitutes. Continue sequential arrival/immutable full actual ledger, all five saved inputs/full4471 contents13149 aliases/history/prior effects/all25 Unbound cases and b3 isolation/init/parity/resource/repeat/artifact/readback admission. Preserve scientific350.7925872/360spent/9.2074128remaining, held CI/P6.4, fixed negative results and seeds/source/arrays unset/execution false until full admission passes.
