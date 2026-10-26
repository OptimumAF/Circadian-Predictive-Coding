# R2.1-CLM.L — wide frozen-feature categorical control

2026-10-25, [issue #157](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/157).
Implemented and executed scoped software behavior; single-agent source/procedure/
results **self-review**, not independent acceptance. Full repository goal and
R2/R5.1/R5.3/G5/R7/release parents remain open. Previous #156 gradient-observation
software is reused within its accepted scope; no repeat pretrained cohort.

## Working behavior

One fixed equal-row one-hot least-squares fit now accepts frozen CLM features
with up to16 rows/4096 coordinates/16 classes through a distinct immutable model.
The original categorical control retains its296-feature/128-row contract/type.
Only numeric solve and softmax internals are shared: original `lstsq`, rcond1e-12,
appended1 intercept, residual recomputation and stable softmax. No Gram normal
equations, optimizer, random initialization, tuning or new orchestration.

`fit_clm_linear_control` admits exact original TRAIN/probe_train catalog records,
matching ordered labels/encoder geometry and bounded detached CPUfloat32 features.
It owns float64 conversion and immutable output. `predict_clm_linear_control`
accepts matching role-bound choice records, including one-query readout; no final
role or held-out labels. Input-owner/value/identity/storage/version/label changes,
altered fitted model and incorrect returned fit-row count refuse publication.
Paid failed numeric returns remain distinct from completed control publication.
No source learner/head/optimizer/RNG mutation or installation authority.

Action embeddings are checked but not used in direct class-score computation.
This is a fixed-class diagnostic control, not native contrastive learning, a
candidate-description provenance certificate or a deployed policy. The existing
workflow CLI and typed decision predictor remain unchanged.

Files and usage: [guide](../clm-linear-fit-control.md), README,
`src/core/wide_categorical_linear_fit.py`, minimal numeric extraction in
`src/core/categorical_linear_fit.py`, `src/adapters/clm_linear_fit_control.py`,
and the two new focused test modules. No dependencies/configuration/CLI flags.
Core imports NumPy/core only; the adapter owns Torch conversion.

## Qualification and retained failure

Stage `artifacts/runs/clm-linear-fit-control-20261025/` contains the prospective
protocol and explicit amendments, original solver source, old/new profiling
harness, commands, XML/stdout/work/parent receipts, static failures/final passes,
current source pins and cost rollup. No old scientific actual/execution/ledger,
detached reviews, unsaved weights or pretrained assets were read/reconstructed.

| Run | Observed result | Body seconds | Parent seconds | Peak RSS bytes |
| --- | --- | ---: | ---: | ---: |
| Windows1 | 46 pass | 19.523912900 | 20.100137800 | 242253824 |
| Linux1 | 46 pass | 31.688088795 | 37.605949700 | 264429568 |
| Windows2 | 19 pass, 1 fail, remainder unexecuted | 12.675352900 | 13.060242000 | 244269056 |
| Windows dense repair | 1 pass | 7.972173400 | 8.394436100 | 237686784 |
| Linux dense | 1 pass | 14.141855705 | 18.470144100 | 262610944 |
| Windows row guard | 3 pass | 6.536371700 | 6.898376600 | 235843584 |
| Linux row guard | 3 pass | 12.565448539 | 17.160918700 | 259497984 |

Zero errors/skips/timeouts. Windows2 EXIT1 is preserved; all other rows EXIT0.
First cohort covers scalar coefficients/intercept/residual/spectrum/probabilities,
rank-deficient contradictory-label negative, original upper geometry and saved-
before exact legacy math, detached state/RNG, role/geometry refusal and paid
fit/readout failures. Legacy model and admission AST remain exact.

Self-review strengthened the initial one-informative-dimension4096 fixture to
16 dense orthonormal rows, repeated Hadamard signs ±1/64. For X X^T=I,
[X,1][X,1]^T=I+11^T and its inverse is I-11^T/17. That gives an independent full
coefficient/intercept/spectrum/residual reference without another solver oracle.

The first dense Windows attempt passed rank16/residual<1e-25 but failed the new
pointwise coefficient atol1e-14:10/8192 near-zero entries differed by up to
2.68794731e-14. This original precision check remains **FAILED**, not retrospectively
green. Before further execution, the engineering criterion was explicitly revised
to a dimension/analytic-conditioning normwise float64 comparison:
gamma_(4097)*sqrt(17)*max(1,||W_reference||F), gamma_n=n*eps/(1-n*eps).
Scalar/intercept/spectrum/residual checks, production solve, data and scientific
metrics stayed unchanged. This is a finite numerical fixture tolerance, not a
LAPACK error certificate or evidence of better task learning. Corrected dense
cases passed with Windows NumPy2.4.6 and Linux NumPy1.26.4; no cross-platform
bitwise equality claim. Torch2.14.0+cpu on both.

Final self-review added one row-count publication check, tested with a valid
returned model carrying the wrong fit-row count and two ordinary TRAIN/probe
paths. The final accepted scope is **47 distinct cases per platform**: reuse45
unchanged first-cohort cases, replace the initial sparse wide case with the
accepted dense case, add the row-count case. Final normal role paths were also
rerun. This is combined scoped evidence, **not a single final47-case full run**.
No third full Windows run; its two-full-run allowance was honored. Change history
and source review identified the affected fixture and later narrow row guard;
unchanged math/roles were reused rather than rebuilding the archive. Final pins
bind the current source; initial cohort had no per-function hash manifest.

## Costs and engineering amendments

All counts below are cumulative, including failed/repeated work:

| Per-platform total | Windows | Linux |
| --- | ---: | ---: |
| Fit API entries, including rejected admissions | 55 | 37 |
| Conservatively counted fit rows | 325 | 253 |
| Actual LAPACK wrapper attempts | 35 | 27 |
| Prediction API entries / rows | 28 / 216 | 20 / 198 |
| Charged design cells | 405350 | 339098 |
| Profiled detached feature cells | 776249 | 635049 |
| Feature cells including reference reserve | 1416249 | 1065049 |
| Coefficient/probability logical output cells | 32928 | 24066 |
| Body seconds | 46.707810900 | 58.395393039 |
| Parent seconds | 48.453192500 | 73.237012500 |

Design charges add6 cells per executed saved-before scalar fit outside the
current helper profiler. Oracle feature-copy reserve is10000 for each initial
cohort and210000 for each dense/subsequent targeted run, deliberately conservative
for the tiny row-guard cases. Analytic dense reference matrix work is paid, not
free. An estimate in the prospective comment omitted10000 initial reserve cells;
the rollup uses the complete charges above. No extra LAPACK oracle or source fit.

Before targeted repair, the future engineering detached-feature/copy reservation
increased **prospectively from1M to1.5M cells/platform**, using measured Windows
497629 cells plus the fixed remaining dense/reference work. Existing work was
still below1M. Original protocol/harness/failed results remain. No refund, hidden
retry, changed scientific allowance or retrospective pass. Other original limits
held:128 fits/LAPACK/predictions,2048 fit rows/4096 prediction rows,1M design cells,
2M coefficient/probability logical cells,120sbody/150sparent,1GiB sampledRSS,
CPU1 sequential/96cases/run. Logical accounting is not all allocations, Python
tuple overhead, LAPACK workspace, IO or FLOPs; the reference reserve does not
measure exact such costs. RSS starts after imports.

Native model objects/backwards are runtime-metered0; encoder/native head/optimizer/
token/backbone/assets/physics/episodes/final/replay work is absent by the executed
source/procedure. No native learner/encoder or Transformers imported. All features
are handcrafted deterministic literals, no stochastic parameter draws. Linear
fits and matrix readouts are paid work, not zero-model-computation evidence.

Six scoped statics attempts total165.903598000s. Initial four mypy errors were
intentional negative-test typing issues (immutable assignment, wrong model type,
variadic calls); corrected annotations/setattr/explicit arguments. statics2..6
Ruff/format/default+Linux-target mypy pass, final five product/test files unchanged
after statics6 and targeted row checks. Reversed target order reused the current
cache for two later runs; no checking mode weakened. Conservative statics+format
bound177.012851100s includes measured formatter parents and<=10.0016557s for the
unmetered formatter bundled before statics4. Cheap rollup/whitespace/pin checks
remain within180s. Actual Linux Ruff/mypy unavailable/uninstalled; Linux-target
typing runs on Windows.

Task-touched cache files109706680B (whole modified files, not cumulative IO),
below160MiB; global inherited+task cache596755824B is separately reported and
**not** within that task reservation. Stage before rollup85105B/stdout14001B,
well below8MiB/512KiB. No cache deletion/refund or unrelated file mutation. Two
formatted-context patches failed atomically with no writes and were retried
against explicit current lines; large display-truncated searches were reduced
to compact issue titles and narrow reads. These are distinct from the retained
numeric test failure.

## Commands and remaining scope

Run from repository root; launcher receipts contain exact cached Linux commands:

```powershell
.venv/Scripts/python.exe -m ruff format src/core/categorical_linear_fit.py src/core/wide_categorical_linear_fit.py src/adapters/clm_linear_fit_control.py tests/test_wide_categorical_linear_fit.py tests/test_clm_linear_fit_control.py
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-fit-control-20261025/statics.py statics-1
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-fit-control-20261025/statics.py statics-2
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-fit-control-20261025/statics.py statics-3
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-fit-control-20261025/launch.py windows-1
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-fit-control-20261025/launch.py linux-1
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-fit-control-20261025/statics.py statics-4
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-fit-control-20261025/launch.py windows-2
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-fit-control-20261025/statics.py statics-5
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-fit-control-20261025/launch.py windows-dense-repair tests/test_clm_linear_fit_control.py::test_should_fit_full_4096_width_and16_rows_with_literal_scalar_solution
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-fit-control-20261025/launch.py linux-dense tests/test_clm_linear_fit_control.py::test_should_fit_full_4096_width_and16_rows_with_literal_scalar_solution
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-fit-control-20261025/statics.py statics-6
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-fit-control-20261025/launch.py windows-row-guard tests/test_clm_linear_fit_control.py::test_should_refuse_different_fitted_row_count_after_paid_return tests/test_clm_linear_fit_control.py::test_should_fit_and_read_existing_roles_with_detached_source_rng_and_costs
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-fit-control-20261025/launch.py linux-row-guard tests/test_clm_linear_fit_control.py::test_should_refuse_different_fitted_row_count_after_paid_return tests/test_clm_linear_fit_control.py::test_should_fit_and_read_existing_roles_with_detached_source_rng_and_costs
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-fit-control-20261025/rollup.py '2026-10-25T16:23:49Z'
git diff --check -- src/core/categorical_linear_fit.py src/core/wide_categorical_linear_fit.py src/adapters/clm_linear_fit_control.py tests/test_wide_categorical_linear_fit.py tests/test_clm_linear_fit_control.py README.md docs/clm-linear-fit-control.md
```

Additional formatting of the two edited files and the dense fixture is described
above; exact Ruff/mypy invocations and outputs are in statics receipts. No full
unchanged suite, whole game/runtime regression, clean clone/install/package,
browser, GPU, remote CI, pretrained comparison, native task qualification or
independent review. No installs/downloads/commits/pushes/deletions. Original dirty
work and historical plans/log remain; GitHub is live status under AGENTS.md.

This delivers a conventional control, not fresh-task acquisition, deployed
retention/recovery, calibration or CPC advantage. #151 partial F acquisition and
old interference, #153 OPEN FAILED-SPENT before model load, #154 manual UI gate,
all earlier negatives/unknowns/permanent charges remain. Earlier #155 wheel
predates these changes. Full R2.1/R5.1/R5.3/G5/R7/release requirements stay open.

**One precise next implementation action:** bind this fitted control to the
existing typed decision/workflow path with an explicit opt-in provider using
`ProbabilityDecisionResult`, fixed ordered class IDs/descriptions and original
external permission checks. Reuse frozen encoder and raw-action feature ports;
preserve ordinary contrastive defaults and source/probe separation. Literal
complete ordinary/stale/denied workflow checks first. Then freeze a separate small
native development protocol that charges original feature production plus the
direct fit/readouts/episodes before any asset load. No #153 retry, old-weight
reconstruction, final/probe selection, favorable seed choice or new orchestration.
