# P6.12a complete primary confirmation findings

Status: **P6.12a complete** after its original declared acceptance audit.
**P6.12b and P6.12 remain unchecked.** ADR-0168 records the split and rationale.

## What changed

The pure consumer requires the whole accepted P6.11 report and rebuilds its
complete existing declarations before interpreting any primary statement.
The output preserves the entire original report, all 560 cells, 626 metric
vectors, 6,260 observations, 58 ordered contrasts and 116 prospective primary
statements. Each statement retains its exact summary, ten ordered seed values,
original eligibility, preferred direction, units, three secondary endpoint
summaries and source pointer. Every marginal/secondary result, cost reference,
endpoint/role/failure and undefined retention reason remains in the original
report appendix. Negative values are not automatically labeled regressions:
lower signed forgetting and higher accuracy have different preferred directions.

Current confirmation evidence: **105 available primary simultaneous intervals
all include zero; eleven are ineligible for zero observed variance**. The full
original report retains **two undefined retention observations** and **935 raw
negative observations across all metric vectors**. H1–H4 are **unresolved
within this primary evidence**. Crossing zero is not equivalence or broad
rejection; constant observed differences do not justify fabricated intervals.
The original model-based df9 Bonferroni family stays at 116, conditional on its
original independence/normality assumptions. Marginal intervals never replace
primary simultaneous intervals. No hypothesis vote or composite winner.

Combined contrasts contextualize all hypotheses without claiming isolated
mechanism effects or extra independent replications. H1 retains the weaker
A-after-A caveat; H2 requires actual capacity/compute context; H3 requires
matched exposure and homeostasis activity; H4 requires actual attempts,
inactivity and sleep work. Their complete synthesis remains P6.12b.

## Module boundaries and usage

```text
src/app/continual_confirmation_findings.py            whole input/pure ledger
src/app/continual_confirmation_findings_rendering.py  exhaustive deterministic Markdown
tests/test_continual_confirmation_findings.py         fabricated arithmetic/scope/failure/sentinels
tests/test_continual_confirmation_findings_rendering.py  completeness/corruption/repetition
docs/adr/ADR-0168-preserve-complete-primary-evidence-before-hypothesis-synthesis.md
```

```python
from src.app.continual_confirmation_findings import build_confirmation_findings
from src.app.continual_confirmation_findings_rendering import render_confirmation_findings

# Supply the complete accepted original report at an IO boundary.
body = build_confirmation_findings(complete_report)
markdown = render_confirmation_findings(body)
```

The public API rejects fabricated, partial, changed or resealed whole inputs.
The private development seam validates fabricated complete declarations for
tests; it grants no scientific/artifact/current-source/official-reader authority.
Rendering rebuilds and compares the entire body before displaying any result.
Application modules perform no file IO, source/model construction, training,
scoring, final-role release, tuning or publication. No new dependencies,
configuration settings or environment variables. New evidence belongs in
separate modules with prospective source/input/budget gates; preserve these
frozen sources and the complete original analysis.

## Prospective and correctness evidence

At unchanged HEAD `57b6fd014190d5039a285db26ecc43ccc946aa89`, all earlier
127 accepted source and 42 test/helper/fixture byte pins match the current
P6.10 closing handoff. The new prospective freeze covers **129 sources**, all
**75 local runtime imports** for these pure modules and **44 test dependencies**,
the complete eight original report parts, the previous handoff and environment.
Map SHA: `768eeba133a0dfc0b010caa756292a391d7b5955288d47f1680c564e0a5ebdfb`.
Freeze precedes every new fixture and actual derivation. It declares **120
seconds per complete pure operation**; prior scientific/reader caps are unchanged.

| Evidence | Bytes | SHA-256 |
| --- | ---: | --- |
| p612-confirmation-findings-source.json | 36,756 | 56c62bc7bd279b2f357f733ec7e134f002a12d2a04cec0d660faef3d5412e50b |
| p612-confirmation-findings-new-checks.json | 1,382 | 4decfd709fd833b26db25185e89f4893728dd43a16842d150a635d1799f039f3 |
| p612-confirmation-findings-related-checks.json | 2,059 | d0b905b6d33f687a74aeab9ce1721b400ec2e61d3ddfdb7194c99d363baeb840 |
| p612-confirmation-findings-static-checks.json | 3,141 | 57c14f13939b3ebe677e6b446b7ddf6eab31c988c86a1eaeed93b78a5d35ff1d |
| p612-confirmation-findings-validation.json | 11,566 | 2f670a7a60493f5740330743666ca1f0522481820e1105fe02dc69928c3b32c1 |
| p612-confirmation-findings-acceptance-audit.json | 42,744 | d04a0bf37e885ea80fbc37fcb08d9316898965992d8bd5aa1c0b0a5b136fb940 |
| p612-confirmation-findings-handoff-validation.json | 31,530 | 77262966c5206199f6b52af09f2ffe68bdc6e992059cb24cee7eda48f673a876 |

All evidence is local in ignored `artifacts/runs/`. Exclusive producers are
occupied: **do not rerun their mains or overwrite their output**. Receipts bind
their complete producer, command, transcript, source/test/input/output and
environment identities. The before-completion plan snapshot preserves every
criterion. The audit checks every a requirement and preserves original P6.12
wording unchanged; it does not close b or the parent.

Closing current preservation exits 0 in **2.0258474000002025 seconds**, under
the same prospective saved-metadata envelope. It rechecks all source/test/
input/environment and saved output/operation/producer bytes, verifies that a's
status metadata/evidence is the only task-line change from the complete declared
plan snapshot, and preserves every other original task criterion/status.
No new declaration reconstruction, official reader or scientific call. This
primary-only closing scope retains the previous P6.10 handoff reference;
complete development/activity/cost/operational revalidation remains P6.12b.

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q --tb=short tests/test_continual_confirmation_findings.py tests/test_continual_confirmation_findings_rendering.py
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q --tb=short tests/test_continual_confirmation_findings.py tests/test_continual_confirmation_findings_rendering.py tests/test_continual_confirmation_analysis.py tests/test_seed_statistics.py tests/test_continual_confirmation_report.py tests/test_continual_confirmation_scoring_validation.py tests/test_continual_confirmation_matrix.py tests/test_continual_confirmation_matrix_rendering.py
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check src/app/continual_confirmation_findings.py src/app/continual_confirmation_findings_rendering.py tests/test_continual_confirmation_findings.py tests/test_continual_confirmation_findings_rendering.py
.\.venv\Scripts\python.exe -m mypy
git diff --check
```

Observed terminal exit 0: **36 new tests in 22.71 s**, **331 related tests in
57.60 s**, **zero skips**, Ruff, four-file formatting, mypy **490 files** and
diff check (configured LF/CRLF notices only). Fixtures cover sign/direction,
zero boundaries, marginal fallback refusal, missing/ineligible/failed rows,
late/resealed corruption, complete scope/order, detachment, full repetition,
model/source/final-release and IO sentinels. All-failed fabricated input retains
every original seed and both failed-side reasons; it is not a real divergence
or resource/execution result.

## Actual complete saved-input derivations

Both separate hard-120-second children terminate successfully; parent times
**4.844548300010501 / 4.757446499992511 seconds**, worker times
**4.727242799999658 / 4.639014900007169 seconds**. Each performs one public whole
input build, one complete renderer and two unchanged complete declaration
reconstructions. Every one of the 24 scientific boundary guards is zero.
**Zero fresh official reader calls.** Neither operation creates a new scientific
source/model, trains, scores, opens a final view, profiles or selects settings.
Full current source/test/input/environment bindings pass before and after.

Both `p612-confirmation-findings{,-repeat}-pure/` contain:

| Part | Bytes | SHA-256 |
| --- | ---: | --- |
| findings.result.json | 9,069,422 | 89de3237ac7deb6ca443d3e2f9a19038331f322ab3659763e785b4dabdecad95 |
| findings.md | 7,573,427 | db66ee6209fe11b5904f6574c8c4199006b3006e41fdbb37de1cfd77bc94c5b1 |

Whole JSON/Markdown repeat exactly. Operation records bind each result;
no claim/failure marker remains. Every statement appears exactly once in
Markdown, with round-trippable original numbers and a complete original report
appendix. The original report is 7,537,678 bytes at
`363ed97dba808281d13224526b4b36614ae452188cade256992b530b6ab03088`.
The acceptance audit terminates in **3.331197700012126 seconds** under the same
prospective saved-metadata envelope, reconstructs no new original declaration
and calls no fresh reader/scientific boundary.

Production byte pins:

| File | SHA-256 |
| --- | --- |
| src/app/continual_confirmation_findings.py | f95d20526934668ee38c3e1141fc7efa9711356ae8c80c984ab738e9713b425d |
| src/app/continual_confirmation_findings_rendering.py | 73e843e36c44c50cb8bc18fd67627b7596da13583fcd2819a979da8596c3ee2f |

## Remaining acceptance and exact next action

P6.12b must bind the complete six-family development/tuning/failure and current
confirmation/cost/matrix handoffs. Keep all rejected sleeps, inactive controls,
regressions and nulls, including the first derivative timeout/partial claim and
unobserved killed-child terminal counters. Explain H1–H4 at actual measured
scopes, separate development from independent confirmation, and validate
complete current IO publication/repetition/readbacks before the unchanged
P6.12 acceptance audit. Pure results here cannot grant that authority.

Next: inspect `continual_confirmation_manifest`, the full
`scripts/inspect_p67_confirmation_scope.py` adapter and its six development
adapter imports, then all current development result/request/audit and evidence
producer dependencies;
inspect complete saved outcome/cost activity fields and operational failure
records. Record a justified small b split and freeze its complete evidence
inputs/source/budget before new fixtures or actual operations. Do not open new
scientific inputs, rerun occupied producers or change any original settings,
seeds, metrics, baselines, confidence model or budgets.

Skipped/unverified: full repository suite/clean clone, actual CI/Torch/CUDA/
vision/stream/C9/optimization gates, new scientific experiments/profiling/sweeps
and complete new IO publication/readbacks. Original broader tasks remain open.
