# P6.9a: Explicit stored confirmation stage/task matrices

Status: **P6.9a is complete** after correctness, both actual publications and
both independent complete readbacks pass. **P6.9 is complete within the
prospectively declared current two-task scope** after its separate original
criterion audit. Every future-task B-after-A slot remains unmeasured.

## Module boundaries and fixed protocol

```text
src/app/continual_confirmation_matrix_inputs.py
  Rebuild every original declared endpoint/cell/summary/cost link.
src/app/continual_confirmation_matrix.py
  Every individual 2x2 matrix, null/status and descriptive transfer.
src/app/continual_confirmation_matrix_rendering.py
  Exhaustive deterministic matrix/count/role/pointer Markdown.
src/infra/continual_confirmation_matrix_bindings.py
  Current complete source/request/input identities and report reader port.
src/infra/continual_confirmation_matrix_artifacts.py
  Exclusive request/result/Markdown/audit/failure/claim and full readback.
scripts/run_p69_confirmation_matrix.py
  Fixed CLI supplying the unchanged complete official report reader.
```

Dependency direction is CLI → infra → app/core; infra never imports the CLI.
The public report reader independently reads both original scored bundles and
fresh complete cost evidence, itself reading both training bundles for each
scored reader. Every operation therefore includes one complete report, two
scored and four original training readbacks. Both original report bundles
and all original inputs are byte-bound and rechecked around the consumer.

The pure input bridge constructs the *expected training declaration* only to
reuse the public scored/report validators on stored endpoint/cell/summary/
cost fields. This is not source/execution proof. Complete actual authority is
owned by the unchanged infrastructure reader and current byte/source bindings.
All original metrics, failures, contrasts, intervals and costs are rederived
before any matrix is exposed. No new dependency/environment variable, primary
endpoint/interval family, model, dataset, tuning rule or scientific override.

## Matrix and missing-value interpretation

The global stage axis is `[after_a, after_b]`; task axis is `[a, b]`:

| Stage / task | A | B |
| --- | --- | --- |
| After A | Measured A-after-A | Unmeasured before B arrival |
| After B | Measured A-after-B | Measured B-after-B |

Each of the 560 family/seed/arm rows preserves all four slots and three exact
original endpoint records. Measured values are derived from correct-count /
example-count with task/checkpoint/role SHA and exact original JSON pointer.
Failed endpoints keep their reason; successful raw endpoints in the same
failed cell remain visible. Original whole-cell derived metric failures stay
unchanged rather than inventing a partial successful cell.

Signed A forgetting is A-after-A minus A-after-B. Descriptive A backward
transfer is its sign reversal; positive/negative/zero values remain. It adds
no new primary metric or CI. B forward transfer is unavailable: neither
B-after-A nor an untrained B reference was declared. No missing value is
inferred and no new final view is opened retrospectively. Retention remains
undefined at zero A-after-A and may exceed one. Weak initial A remains visible
beside forgetting. Families are not pooled; repeats add no seed replications.

The [original reporting audit](phase6-reporting-acceptance-audit.md) closes
P6.11 on its four actual criteria. The subsequent P6.9 scope audit reconciles
the matrix with the already accepted P6.8 contract: that contract explicitly
defines the full current two-task matrix as A-after-A/A-after-B/B-after-B.
The original P6.9 wording does not require evaluating a future task before
arrival. Requiring a fourth measured cell would add an undeclared requirement.
P6.9 therefore passes without inferring that cell, changing the protocol or
opening final data. The immutable `original_fully_measured_matrix_acceptance_complete`
flags remain false: no fully measured four-slot matrix is claimed. They do
not redefine the original accepted three-endpoint task scope.

The read-only scope audit command
`python artifacts/runs/inspect-p69-original-acceptance.py` exits 0 in 6.63 s;
its exclusive `artifacts/runs/p69-original-acceptance-audit.json` is **6,466
bytes**, SHA `0435a1d72c3b1ed16d2d08b5fcc5b15c7edcf28813fb603ef6a8cfbb3e96cc37`.
It verifies the original criterion, committed prospective metric contract,
unchanged scored manifest/analysis/source/test/input/output identities,
whole pure matrix/Markdown reconstruction and the recorded actual readbacks.
All 24 scientific guards remain zero. This establishes current preservation
and criterion consistency, not another complete scientific readback.
P6.10/P6.12 remain open for scoped resource presentation and explicit
H1–H4 plus development/confirmation conclusions; later stream criteria remain.

## Prospective source and correctness evidence

The source/request V1 record was written before fabricated matrix fixtures:
`artifacts/runs/p69-confirmation-matrix-source.json`, **32,923 bytes**, SHA
`9d4ff61cb1c167640589793a5db1fbe04e6707d88fa7045a5f1805e5b1ad5165`.
All **112** local runtime dependencies were checked by AST import closure;
the map is `4885420a055a3d23f852baddcd32fcc23416c614d33cd30091ac5f543cca7f92`.
All original 106 report source pins remain unchanged. Production bytes have
not changed after the freeze. Historical prospective correctness/publication
flags remain false; later terminal evidence establishes completion separately.

All **110 new tests pass in 40.61 s**; **556 related tests pass in 78.87 s**,
zero skipped. Ruff, eleven-file format, mypy (450 files) and diff gates pass.
Tests cover the complete fabricated matrix, original endpoint/count/role/
checkpoint/arithmetic links, negative/zero-denominator/above-one retention,
mixed/all endpoint failures, whole original summary/cost corruption, strict
request/source/reader/input bytes, late changes, occupied/partial outputs,
failure-marker/audit ownership, budgets and fixed CLI dispatch.

Correctness receipt V1 (5,776 bytes, SHA ef47e2f7...471ff) retains the initial
metadata record. Linked V2 corrects shell quoting in recorded argv and binds
fresh static checks after a test-only filesystem fixture repair:
`artifacts/runs/p69-confirmation-matrix-correctness-validation-v2.json`,
**7,552 bytes**, SHA
`94ee5a17cee2ee7f4412a7981d00ea5794dc408b8850039f1441a109c10ade81`.
No production, test outcome, source or scientific setting changed for that
receipt correction. Tests use fabricated scores and the existing unscored-cost
metadata fixture; they do not establish actual scientific readback authority.

## Actual publication/readback protocol

```powershell
.\.venv\Scripts\python.exe -m scripts.run_p69_confirmation_matrix --publish --output-dir artifacts/runs/p69-confirmation-matrix
.\.venv\Scripts\python.exe -m scripts.run_p69_confirmation_matrix --read-only --output-dir artifacts/runs/p69-confirmation-matrix
```

Existing publications are preserved. Publication refuses every occupied
request/result/Markdown/audit/failure/claim path before its reader. Failed
failure-marker IO revokes only our unchanged owned completion audit and
preserves foreign/partial bytes. Readback independently reconstructs the
entire body/Markdown/audit from the complete original reader, then checks
current request/source/input/output/marker identities.

The prospective **180-second derivative budget** uses the prior four complete
report operations' measured 160–163-second cost; it does not change original
scientific work/wall/RSS caps. The separate bounded producer runs two complete
publications and two independent complete readbacks sequentially, each with a
hard 180-second timeout and all 24 scientific boundary guards:

```powershell
.\.venv\Scripts\python.exe artifacts/runs/validate-p69-confirmation-matrix.py
```

All four operations and the parent producer have terminal **exit 0**. Every
operation stays within its original prospective 180-second hard budget:

| Operation | Child pipeline seconds | Parent seconds | Exit |
| --- | ---: | ---: | ---: |
| Canonical publication | 174.4392318 | 178.3326424 | 0 |
| Repeat publication | 174.0555048 | 177.9656485 | 0 |
| Canonical complete readback | 172.4643405 | 176.4029250 | 0 |
| Repeat complete readback | 172.4175417 | 176.3804770 | 0 |

All 24 scientific guards record zero calls in every operation. The four
operations invoke four complete original report readers, eight complete
scored readers and sixteen complete original training readers. No scientific
training, prediction or new final-source access occurs. Every required
artifact part exists, with no claim/failure marker. Preserve these occupied
outputs; the producer must not be restarted over them.

The final exclusive record `artifacts/runs/p69-confirmation-matrix-validation.json`
is **18,108 bytes**, SHA
`112813b519843c08cfbec06bb6e1e3160d32f4320cf42d4314b16ba304bcf481`.
It binds the prospective source/correctness records, exact test/fixture and
producer bytes, all commands, terminal outcomes, times, guards and artifact
identities. The two bundles are `artifacts/runs/p69-confirmation-matrix` and
`artifacts/runs/p69-confirmation-matrix-repeat`:

| Part | Bytes | SHA-256 |
| --- | ---: | --- |
| Both complete result JSON files | 1,973,394 | a1e968677a7ff98ee0c8b43b3ed3945aef34547d3f7c835edccc9f564d06a086 |
| Both complete Markdown files | 749,183 | 8acd86ca2566b3b40e55fa5aa9bd30944162a720f6950e6a56472e03c3f742a0 |
| Canonical request | 18,704 | e6d42168f7e3229d4ec32ac02443778dd1ffe2318d10a266cd7a8c7e1915f9ba |
| Repeat request | 18,711 | f3a455eb13c3c198f6f9057a2869d6f8097f43cbf3fac17c37216590425ff881 |
| Canonical completion audit | 5,985 | 4b3143a6a154c0237ead843769d694c4e24cb55e85a1d5dcf645505db1511a76 |
| Repeat completion audit | 5,987 | 23e512ddc340c3c66d453d315c7b2a84995f4c2537922b87326c4d73feb86e3a |

The entire scientific result JSON and Markdown repeat byte for byte; request
paths/times and observed derivative audit durations retain their actual values.
Coverage is **560 rows / 2,240 slots / 1,680 successful original endpoints /
560 unmeasured slots**, zero failed endpoints/cells. There are two undefined
zero-base retention observations: schedule seed 233 / pc_12_no_sleep and
parent seed 373 / pc_13_off. Raw A endpoints remain beside their null ratios.
Descriptive A backward transfer is positive in 241 cells, negative in 87 and
zero in 232. These heterogeneous cell counts are not pooled inference or a
model ranking. Every B forward-transfer value remains unavailable.

Closing read-only preservation also exits 0 in 6.52 s:
`python artifacts/runs/validate-p69-confirmation-matrix-handoff.py` publishes
`artifacts/runs/p69-confirmation-matrix-handoff-validation.json`, **26,130
bytes**, SHA `ce733130f93cacfa7346192605c21ff781a7b9485f514428a56d032421f83f25`.
It rechecks all 112 frozen sources, original inputs, exact tests/fixtures and
producers, both complete matrix bundles, original scope/metric contract,
whole pure reconstruction/rendering, absent markers and unchanged HEAD.
All 24 scientific guards stay zero; completed and unchecked task IDs match
the handoff. This is preservation with recorded full readback authority,
not another full scientific readback. Fresh static gates subsequently pass.

## Verification commands and extension

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q --tb=short tests/test_continual_confirmation_matrix.py tests/test_continual_confirmation_matrix_rendering.py tests/test_continual_confirmation_matrix_bindings.py tests/test_continual_confirmation_matrix_artifacts.py tests/test_p69_confirmation_matrix_cli.py tests/test_continual_confirmation_report.py tests/test_continual_confirmation_report_rendering.py tests/test_continual_confirmation_report_bindings.py tests/test_continual_confirmation_report_artifacts.py tests/test_p611_confirmation_report_cli.py tests/test_continual_confirmation_analysis.py tests/test_seed_statistics.py tests/test_continual_confirmation_scoring_validation.py tests/test_continual_confirmation_report_costs.py tests/test_continual_confirmation_report_cost_references.py tests/test_p611_confirmation_cost_inspection.py tests/test_continual_confirmation_training_references.py
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src/app/continual_confirmation_matrix_inputs.py src/app/continual_confirmation_matrix.py src/app/continual_confirmation_matrix_rendering.py src/infra/continual_confirmation_matrix_bindings.py src/infra/continual_confirmation_matrix_artifacts.py scripts/run_p69_confirmation_matrix.py tests/test_continual_confirmation_matrix.py tests/test_continual_confirmation_matrix_rendering.py tests/test_continual_confirmation_matrix_bindings.py tests/test_continual_confirmation_matrix_artifacts.py tests/test_p69_confirmation_matrix_cli.py
.\.venv\Scripts\python.exe -m mypy
git diff --check
```

Skipped: full repository/fresh-clone/actual CI, Torch/CUDA, new training,
new source/final views, metric/seed/baseline/cap changes and sweeps. The changed
consumer/analysis/binding/artifact boundaries are covered by the related gate.
Next: P6.10a must inventory every original resource field, unit and measurement
scope over all 560 raw costs, checkpoint capacities and sixty shared contexts.
P6.10b then owns the complete accuracy/forgetting versus compute/memory ledger.
New resource or hypothesis presenters should consume the same complete verified
report through separate modules, preserve unknown fields/measurement scopes,
and define their own source/input/output gates. Do not edit pinned scientific
sources to fill missing measurements or manufacture a favorable ranking.
