# Independent confirmation envelope, checkpoint and parameter-link validation

2026-09-30. B2a and b2b schema/parameter/work/state scopes have implemented
independent checks. Bounded source/artifact/resource execution (b2c), actual
confirmation, uncertainty and final scoring remain unfinished.

## Boundaries and structure

```text
src/app/continual_confirmation_json.py         finite/type/canonical schema primitives
src/app/continual_confirmation_checkpoints.py declared model/configuration/state/view checks
src/app/continual_confirmation_parameter_links.py original parameter contracts and raw links
src/app/continual_confirmation_validation.py  exact joint envelope, roles and witness schemas
src/app/continual_confirmation_fact_schema.py closed dataclass/TypedDict raw fact schemas
src/app/continual_confirmation_simple_work.py independent gating/replay/sleep rules
src/app/continual_confirmation_work_validation.py complete scope, derived work and held state links
tests/test_continual_confirmation_validation.py
tests/test_continual_confirmation_work_validation.py
```

Inputs are decoded finite JSON and the exact frozen P6.7a manifest; outputs
are success or explicit failures. No source/model, training/scoring or IO.
Only initial tensor fingerprints and RNG identities are derived locally from
declared seeds. Existing scientific helpers/sources are unchanged. The public
envelope-only API remains `verify_confirmation_envelope`. The full pure train
fact gate is `verify_confirmation_payload`; source-file/artifact/live/resource
enforcement must still pass b2c before reserved scientific execution.

The whole envelope must match all six families, exact sixty family/seed rows
in reservation order, all named cells, configurations/source references and
literal global-arrival/evaluation seals. Numeric type matters: an integer
budget/count cannot become a float or bool. Unknown/duplicate canonical tags,
mapping keys, fields, container ordering, dtypes and nonfinite/scored values
are refused. Model class, exact state-field inventory, configuration, initial
shape, width bounds and parameter shapes/counts are checked independently.

Every state digest is recomputed from canonical JSON. Initial parameters,
noise RNG, optional selector RNG, zero state/clocks/lineage and traffic are
seeded/checked independently of captured hashes. Duplicate clock/retention/
lineage/selector views must agree with state, including lineage array hashes,
selector decision/cursor/RNG and fixed control state. Retained rows preserve
declared FIFO/8-example/192-byte shapes/storage. All guard witnesses have the
exact declared inventory, wake/row clocks and full equality on rollback/skip.

Role ID order/count/coverage/disjointness, source/phase/seed prefixes and all
forty sealed final IDs are checked. Role hashes/counts link to raw family role
records; no final hash, target or array is accepted. A's full 120 development
positions are covered, while B keeps sixty original positions within the same
120-row source after exposure reduction. All six hashes must be distinct.

## Evidence and commands

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts= -q -ra --maxfail=1 tests/test_continual_confirmation_validation.py tests/test_continual_confirmation_state.py tests/test_continual_confirmation_training.py tests/test_continual_confirmation_manifest.py tests/test_p67_confirmation_scope.py
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src/app/continual_confirmation_json.py src/app/continual_confirmation_checkpoints.py src/app/continual_confirmation_validation.py tests/test_continual_confirmation_validation.py
.\.venv\Scripts\python.exe -m mypy
git diff --check
```

**125 passed in 21.30 s, 51 new, zero skipped**; mypy 375 files and other
static gates pass. All 56 actual development fixture model cells and initial/
A/B/guard records pass with constructors, source builders, training and score
functions forbidden during validation. Original first development seeds only.
Separate metadata-only delegation tests prove complete sixty-row/fifty-source
scope ordering and rejection of partial/reordered/extra/scored requests before
seed-body checks. Those spy fixtures contain no trained checkpoint bodies;
they are not scientific confirmation or complete 560-cell evidence.

Digest-resealed clock/configuration/selector/lineage/parameter/RNG, role/guard,
ambiguous canonical, nonfinite/type/nesting and schema forgery regressions pass.
Twelve complete historical bundles and twenty P6.3 usage files revalidate; the
saved scope JSON stays identical. No new experiment file or reserved model/
dataset was produced. Tests need no ignored artifacts; real historical
inspection requires the saved scope/development files. Full CPU suite/CUDA/
large sweeps/runtime-resource/final scoring were skipped, no selected skips.

## P6.7b2b1 original parameter contracts and available raw links

The original helpers hash the same normalized tensor names, shapes and
float64 bytes with three different domain prefixes. A trained digest cannot
be converted to another prefix without those bytes. Before implementation,
the plan split b2b1 from remaining b2b2 work and prospectively amended the
still-unrun checkpoint schema. No published record or pinned helper changes.

Every initial/A/B/guard checkpoint now contains an exact
`parameter_sha256_by_contract` map:

| Original family | Map key/domain prefix |
|---|---|
| Gating | `p63-shallow-parameter-tensors-v1` |
| Replay and uniform confirmation | `p63-replay-shallow-parameter-tensors-v1` |
| Sleep | `p63-sleep-factor-shallow-parameters-v1` |

The existing `parameter_sha256` stays the uniform/replay entry. Capture all
three from actual live tensors; independently regenerate seeded initial
parameters and domain hashes in JSON validation. Require exact keys/finite
SHA strings and the uniform field link. All available raw initial/final
endpoints link for all six families. Raw A links are checked where recorded;
sleep pre/post parameters link to complete supplemental before/after and held
A, schedule guards link to complete witnesses and held boundaries, and
combined/parent before/after hashes link to wake/after-epoch records. Epochs
12/24 link to held A/B checkpoints. Rejected/skipped chains preserve hashes;
accepted parent proposals agree with the committed hash. No app imports CLI.

**Fifty added/223 related tests passed in 43.39 s, zero skipped**, with Ruff,
mypy (376 files), format/diff checks. The final command is in the b2b1 log;
the smaller state/validation command passes 119 in 7.18 s:

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts= -q -ra --maxfail=1 tests/test_continual_confirmation_state.py tests/test_continual_confirmation_validation.py
.\.venv\Scripts\python.exe -m ruff format --check src/app/continual_confirmation_state.py src/app/continual_confirmation_parameter_links.py src/app/continual_confirmation_checkpoints.py src/app/continual_confirmation_validation.py tests/test_continual_confirmation_state.py tests/test_continual_confirmation_validation.py
```

Live backprop/PC/circadian models match every original encoder before/after a
parameter change. All six normal development bodies (56 cells) preserve exact
unscored trajectories; four additional fixtures force all attempted guards
to reject in sleep/schedule/combined/parent and validate with source/model/
train/score functions forbidden. Missing/extra/swapped/malformed contracts,
uniform alias mismatch, resealed seeded initial plus matching raw forgeries,
raw initial/final/A/guard/wake/epoch links and missing-method cases fail.
Historical twelve bundles/twenty usage files and saved scope stay exact.
No new experiment record, reserved model/source or final/outer score.

## P6.7b2b2 all-family raw work/controller and full state links

The full public gate verifies exact whole scope before any raw work body.
It accepts every required family/seed/cell, never partial scientific scope.
Closed dataclass annotations and explicit TypedDict schemas enforce exact
raw keys and JSON scalar/container types, including all nested telemetry,
lineage/clock/selector/guard objects. Bool/float counters and unsupported/open
annotations fail. The only omitted old fields are forbidden development
metrics and nondeterministic event durations; those old scientific schemas
and sources remain unchanged.

Gating/replay/sleep rules are independently implemented in app using the
inspected original rules. Schedule/combined/parent reuse unchanged pure
per-seed seams while their old strict three-seed whole-result gates remain
intact. Wake/latent costs, matched FIFO/application IDs, actual and rejected
replay, guard semantics/exposures, component budgets, transient split/prune
capacity, adaptive/periodic decisions and selector order/RNG are checked.
Derived typed `SeedWork` records expose actual executed updates, guards,
examples, peak width and declared retained array storage. Complete family
wake totals, maximum updates/guards and the joint 16,000 update cap bind.

New links bind raw full held A/B state digests and clock/lineage/selector/
retention/capacity views. Sleep and schedule supplemental states agree with
raw outcomes and complete held boundaries. Schedule trigger windows bind to
the configured tail of full captured energy history. Committed exposure IDs
agree with captured sets; combined lineage continues across every decision.
Each baseline traffic count equals applied wake and replay updates at A/B;
guard predictions never record baseline traffic. Every nested sleep final
seal is checked independently of the sealed envelope.

Development fixtures use all six original first seeds (56 normal cells), plus
four families with every attempted guard forced to reject. Validation runs
with model/source/train/score functions forbidden. Forced rejected replay is
12 schedule/72 combined updates, and zero sleep/parent; rejected work remains
counted. Resealed traffic, forged raw/nested costs/types/keys, guard role,
trigger history/clock, structural/capacity/chemistry, retained/exposed supply,
full held state and selector/RNG/metric forgeries fail. Whole-scope delegation
and family-bound checks use metadata-only spies, never trained reserved
bodies; they are not 560-cell scientific evidence. This increment adds 110
tests; the final related gate passes 333 tests in 56.53 s with zero skips.
Exact commands and the reproduced final-seal gap/repair are in the latest
development log. B2b2 and b2b meet their pure validation criteria; b2c remains.

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts= -q -ra --maxfail=1 tests/test_continual_confirmation_work_validation.py
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src/app/continual_confirmation_fact_schema.py src/app/continual_confirmation_simple_work.py src/app/continual_confirmation_work_validation.py tests/test_continual_confirmation_work_validation.py
.\.venv\Scripts\python.exe -m mypy
git diff --check
```

Twelve historical development bundles/twenty usage files and the saved scope
still validate exactly. No old pin, baseline, setting, seed, threshold or
metric changed. No new experiment artifact, reserved source/model or outer/
final score. Tests need no ignored artifacts; real historical verification
uses existing evidence.

## Remaining work and safe extension

Independent schema checks bind hashed state and declared links. They cannot
reconstruct after-trained tensors from byte fingerprints, prove physical
alias relationships, or establish actual role content from hash strings.
Future source-bound production, live capture/copy checks and exact all-seed
train reproduction remain required before evaluation. In particular, pure
JSON cannot reconstruct all three after-trained domain hashes from per-array
fingerprints; live hashing and source-bound reproduction remain mandatory.
Captured chemical variance/means also cannot be recomputed from array hashes.
Decision arithmetic, complete declared links and finite schemas are checked;
live comparison and exact train reproduction must establish actual values.

The later b2c boundary now binds exact scope/source/request identities and
passes exclusive artifact/update/wall/RSS failure gates before data. Both
full source-bound reserved runs and independent readbacks verify all 560
cells and exact result bytes under unchanged caps; see
`docs/p67-confirmation-training-results.md`. Live hashing/copy/state checks
establish the actual values described above; pure JSON retains its stated
limitations. P6.11a's predeclared analysis/correctness gate now passes;
next bind/test the separate P6.7c scored checkpoint/final-release boundary.
Original P6.7/P6.3 criteria and separate P6.7c/P6.11 scoring gates remain open.
