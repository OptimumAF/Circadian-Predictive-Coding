# P6.7c1: Independent confirmation scoring correctness

Declared: 2026-09-30. Increments a/b/c1c1 validated: 2026-10-01.
No confirmation final value has been opened.

## Increment a: scientific manifest and global training-state proof

Before implementation, bind the complete original manifest and analysis
declaration to both saved train request/result/audit identities recorded in
`p67-confirmation-training-results.md`. Both results have identical SHA
`3d85c60627de63769d0f0fc0bf5ec781c77d50468673dab466d8bbe28089e547`.
Retain all six families/ten seeds per family/560 cells/580 original pairs,
three final endpoints per cell and forty examples per role. Settings,
baseline behavior, metrics and the 16,000-update/600-second/observed 512-MiB
limits stay fixed. The full P6.11a declaration stays unchanged.

The implemented app boundary accepts reproduced live training and the fixed
scoring manifest. It incrementally hashes every fact in the existing saved
JSON encoding, checks exact complete family/seed/held/fact inventories, and
reuses the original full model/role checker across the entire matrix. It
returns a state proof only after the last check. Reuse the check after
future scoring; keep original sealed roles and released final views separate.

Why this: a digest of the entire reproduced artifact binds costs, raw decisions,
roles and every checkpoint, while incremental encoding avoids another whole
decoded result alongside the held models. Digest equality relies on SHA-256;
actual file/source/resource provenance remains a separate boundary. No
resource measurement or final authorization follows from this app proof.

Fixtures use all six first existing development seeds, raising original
final and outer access sentinels. The public API never accepts that partial
scope. Synthetic sixty-row delegation tests may demonstrate global dispatch
and late failure only; they are not a reserved-source/model experiment.
No actual or fabricated final accuracy is used in this increment.

## Verified evidence for a

`continual_confirmation_scoring_manifest.py` fixes both complete bundle
request/result/audit identities, scope, original source map/adapter and full
analysis contract. Strict nested dataclass/tuple/scalar types and exact JSON
values reject partial, reordered, retuned or ambiguous declarations.
`continual_confirmation_scoring_state.py` checks exact held/fact attachments,
the complete bound training artifact digest **and byte count**, and all
original full live model/role checks. It rechecks inventory and fact bytes
after the last live check. The public proof records the validated scoring
manifest digest; the private development proof has no production digest.

All **69 new tests** pass in **18.16 s**, zero skipped, after the final
receipt metadata/test-type edits. The earlier related gate passes **296
tests in 50.58 s**, zero skipped. It precedes those final metadata edits;
all scientific/checkpoint/encoding behavior is unchanged and the final
new-module gate reruns every affected case. Ruff, four-file format and
mypy (**396 files**) pass on final sources.

The original writer and incremental encoder match for the entire six-row,
56-cell development training artifact: **13,523,045 bytes**, SHA
`f28a439142171688363ce11892913535b52ef463a896ae8c6d60981e9b58edee`.
Late A/B parameter/width/traffic/chemistry/model-RNG/selector-RNG changes,
original role phase/seed/final seal/content/hash/count drift, whole-artifact
cost/type/checkpoint/arrival/seal/forbidden metric-key changes, held inventory
and detached facts all fail. Mutating facts or inventory in the last check
also fails. Original final/outer access is sealed during development training;
verification forbids source/model construction, training, prediction/scoring,
final release, RNG construction and file reads. A sixty-row delegation spy
checks full public dispatch and last-row failure only. Unbound fabricated
whole metadata and real partial development training fail the public gate.

Read-only, one-MiB chunk hashes confirm all six actual saved request/result/
audit byte identities and result lengths, with no failure/claim present.
Current old execution bindings revalidate twelve historical bundles/twenty
usage records and all 78 original source-map entries plus adapter, unchanged.
These are byte/source checks; the new complete scored bundle reader and
bounded process remain c1c. No new scientific artifact/resource measurement.

Frozen identities before any future scored fixture:

| Declaration or source | SHA-256 |
|---|---|
| Scoring scientific manifest (compact canonical JSON) | `76cf873e5942a661bdb76e6fd7f28fc490fc6b8001e2ae0ccc4afe87063a4223` |
| Full unchanged analysis declaration | `5e33ef28862bcdf9d92fe14dd6cf6b71672a2336ffd760a1214ef04666b594b1` |
| `src/app/continual_confirmation_scoring_manifest.py` | `21c86b9f78e6fef0c475400b87f026daa6bca11e20f64f7fa369d853207efca8` |
| `src/app/continual_confirmation_scoring_state.py` | `cedd370866fdd7223c83ba4398cc104156d2690b98f77f6547d78f1a6aacdad3` |

The later scored closure must include these and the unchanged three analysis
sources and original training closure; preserve every earlier pin. No claim
that this two-source record is the full scored closure or a memory benchmark.

## Usage and validation

Within the future fully bound worker, validate already reproduced training:

```python
from src.app.continual_confirmation_scoring_manifest import fixed_scoring_manifest
from src.app.continual_confirmation_scoring_state import verify_scoring_training_state

proof = verify_scoring_training_state(already_reproduced_training, fixed_scoring_manifest())
assert not proof.final_release_authorized
```

There is no new scientific run command. Development-only validation:

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts= -q -ra --maxfail=1 tests/test_continual_confirmation_scoring_manifest.py tests/test_continual_confirmation_scoring_state.py
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src/app/continual_confirmation_scoring_manifest.py src/app/continual_confirmation_scoring_state.py tests/test_continual_confirmation_scoring_manifest.py tests/test_continual_confirmation_scoring_state.py
.\.venv\Scripts\python.exe -m mypy
```

No dependency or environment-variable change. Full CPU/CUDA suites, new
reserved training/scoring, sweeps and actual confirmation intervals were
skipped for this state-only increment; no selected test skip.

## Remaining required increments

### Prospective c1b declaration (2026-10-01, before code/fixtures)

ADR-0158 declares pure final-array/ID/hash and endpoint count/failure contracts,
app orchestration, and a new infra adapter composing the pinned release and
model prediction. Verify all training before any release; bind all final
views, shared phase/seed signatures and original IDs/counts before any score;
globally recheck training after release and both training/released views after
all fixed evaluations. Keep original roles sealed and separate.

Retain exact correct/count endpoints at the original `>= 0.5` threshold.
Only nonfinite predictions or `FloatingPointError` become typed numerical
endpoint failures. Continue every original three-call cell and retain all
raw endpoint failures/null aggregate cells; other errors fail the gate.
No retry/substitution/drop/partial scope or fabricated original-final value.
App-local access/call/example totals are not external process observations.
Freeze every new source identity and the available composition dependency
closure before scored fixtures; c1c must extend that record to its complete
worker/adapter/request/resource/artifact boundary. All old pins stay fixed.

Before the first fabricated final fixture, the read-only AST traversal of
training/app-scoring/infra-final entry points resolves exactly **87 local
source files**. Every original 79-file training pin and all five analysis/
c1a pins revalidate unchanged. Publish the exclusive metadata-only record
`artifacts/runs/p67-confirmation-scoring-composition-source.json`, SHA
`c096145a770e10e3fc1b13114d5147f9d1d888fcf982b8f48902b427848c1c8c`;
source map SHA `40e3309c6e061453ac155039b32b871746ce96a1d659adf80d7714372fd74775`.
It binds the available composition closure, not the unimplemented full scored
worker/request/resource boundary. Its explicit `full_scored_worker_bound`
is false. Scientific manifest remains `76cf873e...3a4223`.

| New source, frozen before fabricated fixtures | SHA-256 |
|---|---|
| `src/core/confirmation_final_roles.py` | `acfd451e68280469731c94dcf943b25bb9db4f179ffcd1ab3f4c854321aaf276` |
| `src/app/continual_confirmation_scoring.py` | `00495c6b0ad300b53ff9eefdd9a958a77c990675cda6156fd9e49d6d83174a7a` |
| `src/infra/continual_confirmation_final.py` | `2270fb0c2f2ddb36fa8ae1434324643e417e4a9b54b25589d29c2c18bd479b96` |

The first orchestration fixture subsequently fails with `NameError` before
returning any result: refactoring placed the cell-return body in the endpoint
identity helper. Restore that body to its own function; this changes no
scientific contract. Preserve the first freeze as failed-code history. Before
the next fabricated fixture, publish exclusive
`artifacts/runs/p67-confirmation-scoring-composition-source-v2.json`, SHA
`00226f6bf37bf66c35991694e8c86f5dde308f8a9abd507deb2780bc2975c246`.
Its 87-file map SHA is
`21eba485124316799bcc6364e9502ed5e9a86e2eefa06ccbf9d09fa052fe1ae4`;
the only changed source is app scoring, now SHA
`2458044d2118203a9b94fe728a67b3cbebcf6ce417495a1abe04e08bf82c14ee`.
The record links the original freeze and repair rationale; every other source,
scientific manifest, metric/threshold/failure rule, seed, baseline and cap is
unchanged. This supersedes the first app pin for c1b fixture correctness only;
c1c's full worker/request/resource/artifact gate remains required.

### Verified c1b evidence (2026-10-01)

Three new production modules and three tests preserve the original source
and scope contracts:

```text
src/core/confirmation_final_roles.py          released data/hash/count contracts
src/app/continual_confirmation_scoring.py     global ports/barriers/endpoint rows
src/infra/continual_confirmation_final.py     pinned release/prediction adapter
tests/test_confirmation_final_roles.py
tests/test_continual_confirmation_final_adapter.py
tests/test_continual_confirmation_scoring.py
```

All **103 new/399 related tests** pass on final sources in **99.82 s**, zero
skipped; Ruff/six-file format/mypy **402 files** pass. Core cases independently
match the original hash helper on fabricated arrays, reject nonfinite/wrong
dtype/shape/IDs/hash/count/failure policies, and preserve exact correct/count.
Adapter cases observe each fabricated source's input/label field once,
preserve original seals/objects, reject altered released metadata and preserve
the threshold boundary. Numerical prediction failures are explicit nulls;
other exceptions and invalid probability contracts propagate.

Original all-six first-development-seed training supplies all **56 held cells**;
its complete unscored facts retain SHA `f28a4391...8edee`. Replace only unopened
backing sources with fabricated final fields; original final/outer/reserved
access remains sealed. Every **12 role releases/24 field reads/168 endpoint
calls/6,720 examples** is observed locally, at most once per declared source
field. Two independent fixture clones repeat every metric/count/role/proof.
All three global training proofs equal and all original roles remain sealed.
Late checkpoint/selector/fact/original-role/final-content mutations, including
resealed content, fail; retained endpoint-count drift fails its cell link.
First/all numerical failures retain all endpoint calls and 56 cells/nulls.
Late final role/IDs/hash/dtype/count/shared-source errors prevent first score.
Boundary failures propagate and the last callback precedes final state checks.
Partial real development training fails the public gate before final access.

A separate **explicit global-state spy** uses only fabricated role arrays and
metadata model tokens for all 60 reserved-seed rows: dispatch/order, 120 views,
1,680 calls/67,200 examples and all **560 cells/580 paired seed observations**
plug into the unchanged analysis contract. It tests one numerical failure and
complete cells, keeps all 116 primary statements, constructs no reserved
dataset/model and makes no actual confirmation or resource claim. Without
that spy, unbound whole metadata fails. Source/model construction, training,
RNG construction and file reads are forbidden during evaluation fixtures.

The first corrected-app fixture additionally exposed a test assumption:
model dictionary insertion order differs from frozen sleep arm order. The
implementation follows the authoritative manifest; fix the expected order
to that declaration without changing production or acceptance. All red/static
commands and source-freeze repair history are retained in the log. V2 is the
authoritative available composition source map; every 87 current byte pin
revalidates unchanged. Metadata records are 10,769/11,041 bytes, frozen at
`2026-10-01T07:33:21.080407+00:00` and `07:44:51.451282+00:00` respectively.
There is no new scientific scored artifact, reserved final value, measured
resource result, dependency or environment/configuration change.

Validation commands:

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts= -q -ra --maxfail=1 tests/test_confirmation_final_roles.py tests/test_continual_confirmation_final_adapter.py tests/test_continual_confirmation_scoring.py
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src/core/confirmation_final_roles.py src/app/continual_confirmation_scoring.py src/infra/continual_confirmation_final.py tests/test_confirmation_final_roles.py tests/test_continual_confirmation_scoring.py tests/test_continual_confirmation_final_adapter.py
.\.venv\Scripts\python.exe -m mypy
```

The public `evaluate_confirmation` takes complete reproduced training, the
fixed scoring manifest and injected final-release/evaluation/boundary-check
ports. The future c1c worker supplies actual source-bound, independently
observed ports and verified requests/resources/artifacts. Keep the three
declared global barriers and original scientific helpers fixed; add outer
boundary modules rather than changing the scoring/analysis rules. No new
scientific CLI until complete c1c correctness. Full CPU/CUDA/clean-clone/CI,
new reserved training/scoring, sweeps and actual intervals were skipped;
no selected test was skipped. C1b alone is complete, parents remain open.

- **c1b (fixture correctness complete):** final-role IDs/counts/content/hash checks, all three endpoint calls,
  exact final access/call/example accounting, explicit failures/nulls, complete
  post-score state and released-role verification. Freeze source identities
  before fabricated final fixtures; keep original final/outer/reserved access
  sealed.
- **c1c1 (readback correctness complete):** the [complete JSON/reference gate](p67-confirmation-scoring-readback.md)
  passes 197 new/596 related tests, zero skipped, and actual complete readback
  of both bound unscored bundles with source/model/train/final/outer sealed.
  All 90 available source pins are frozen before fabricated fixtures; every
  endpoint/cell/failure/total and global proof link is independently checked.
- **c1c2 (correctness complete):** extend/freeze full scored worker/request/command/environment;
  before/after source/request/reference and post-serialization checks; bounded
  child, external optimizer/final work/evaluations/RSS/wall observations;
  exclusive request/result/audit/failure/readback and all late failures.
  No cap change, actual reserved final or partial production run in correctness.
  The [final observer increment](p67-confirmation-final-observation.md) now
  supplies direct source/model-call observations and whole pure event links;
  its 92-source component freeze is not a full scored request/worker freeze.
  The [full scored worker](p67-confirmation-scored-worker.md) now extends it
  to 97 sources and passes every original lifecycle/resource/readback gate:
  203 new/984 related tests, guarded two-reader preflight and a bounded genuine
  development child with fabricated final fields. C1c2/c1c/c1 correctness
  is complete; those fixtures make no reserved reproduction claim.
- **c2:** only after all c1 gates, two complete real scored processes and
  independent readbacks with every metric/cost/source/artifact/resource identity.
  The [actual confirmation and repeat](p67-confirmation-scored-results.md) now
  pass these criteria, including exact scientific result-byte equality.
- **P6.11b:** all individual independent seeds and predeclared uncertainty/
  contrasts under the original missing/constant/negative/null rules.

P6.7c1 correctness and c2/P6.7c independent final scoring are complete;
P6.11b and original matrix/resource/hypothesis/reporting parents remain unchecked.
See ADR-0157 and `development-log.md` for prospective decisions and evidence.
