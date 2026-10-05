# P6.7d2a: prospective precision within the complete fixed budget

## Scope and status

This increment asks whether ten replications per family can support a declared
precision objective. It consumes the complete original development boundary,
including all 116 primary paired vectors and all 348 three-seed observations.
It preserves the original six families, matched controls, metrics, signed
forgetting, seed order, work ceilings and historical intervals.

The method and objective were frozen before this calculation in the ignored
`artifacts/runs/p67-precision-prospective-contract.json`. P6.7d2a is complete after
all runtime, static, full-input and independent readback gates below. Original P6.7, d2, d2b and d3
remain unfinished. This calculation grants no confirmation execution authority.

## Objective and rationale

The objective is a mean half-width of **0.05 accuracy fractions (five percentage
points)** for each of the 116 existing primary statements, with mean family
alpha 0.05. The same absolute target applies to mean accuracy differences and
signed forgetting differences. It was selected independently of already
observed confirmation outcomes.

Why this: one decision in twenty, or two correct-label count changes in an
original 40-example task, is a modest declared comparison resolution. This is
not a clinical or biological relevance threshold, a new metric, or a change in
the original metrics' measurement resolution. The target is not relaxed after
seeing feasibility, and no contrast is chosen for having a favorable forecast.

Ten is the candidate count allowed by the original complete work budget. It is
not yet a variance-justified final execution count or an untouched-role binding.

## Conditional normal sensitivity

For three independent, exactly normal paired pilot differences with sample SD
`s`, the variance statistic has two degrees of freedom. The one-sided SD upper
bound uses the lower chi-square quantile, as obtained from the variance
statistic in the [NIST variance reference](https://www.itl.nist.gov/div898/handbook/eda/section3/eda358.htm).

Set `a = 0.05 / 116`. Our integration of the df-two density in the
[NIST distribution reference](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm)
gives `F(q) = 1 - exp(-q/2)` and its stable inverse
`q = -2 * log1p(-a)`. These steps are mathematical inferences from that density.

```text
SD_upper = s * sqrt(2 / q)
conditional_half_width(N) = 5.403490569214909 * SD_upper / sqrt(N)
minimum_N_at_fixed_critical = ceil((5.403490569214909 * SD_upper / 0.05)^2)
```

The critical value is the unchanged original df-nine simultaneous scaling
constant. The last quantity holds that constant fixed; it does not recompute
Student critical values or define intervals at a different sample count.
No historical interval or original confirmation analysis changes.

This is explicitly a sensitivity calculation. Discrete bounded accuracy
differences are not exactly normal. Three pilot seeds give weak dispersion
evidence, and development outer-selection dispersion need not transfer to
fresh final roles. Pilot variance and future mean alpha levels are separate;
this combination is not a joint 95% planning guarantee. Constant or numerically
negligible pilot variance, and incomplete pilots, retain null normal forecasts
with their original status and raw observations. They never establish zero
population uncertainty.

## Separate bounded-mean check

The original accuracy difference lies in `[-1, 1]` (width 2). A signed forgetting
value lies in `[-1, 1]`, so its paired difference lies in `[-2, 2]` (width 4).
This check needs independent source-seed replications within each vector;
shared contrasts need not be independent for the union bound. Repeated runs
of the same source seed do not add independent observations.

The research context is bounded independent tail inequalities and the
exponential moment method; see [Bentkus, On Hoeffding's inequalities](https://arxiv.org/pdf/math/0410159).
The following derivation is our explicit inference, rather than a quotation
or a claim to use that paper's sharper bound.

For a centered variable of range width `R`, exponential tilting keeps its
range. Its tilted variance is at most `R^2/4`, since variance is no larger
than mean squared distance from the range midpoint. Thus the log moment
generating function has second derivative at most `R^2/4`; its value and
first derivative at zero are zero. Integrating gives `K(lambda) <= lambda^2
R^2/8`. Independence, Chernoff's inequality with `lambda = 4h/R^2`, both signs,
and the union over `m = 116` yield:

```text
P(any absolute mean error >= h) <= 2m * exp(-2N h^2 / R^2)
bounded_half_width(N) = R * sqrt(log(2m / 0.05) / (2N))
sufficient_N = ceil(R^2 * log(2m / 0.05) / (2 * 0.05^2))
```

This conservative sufficient count does not use pilot dispersion, so constant
pilots retain a nonzero bound. It is not a necessary or optimal sample count.
A bound above the target means these declared checks do not certify that
precision within the fixed budget. It does not prove all valid methods or all
populations require that many seeds. Do not increase the scientific cap to
execute the calculated sufficient counts.

## Complete work and resource envelope

The original additive maximum update ceiling per complete family replication is:

| Family | Cells | Maximum optimizer updates |
| --- | ---: | ---: |
| Gating | 3 | 72 |
| Replay | 8 | 228 |
| Sleep | 9 | 216 |
| Schedule | 11 | 338 |
| Combined | 17 | 516 |
| Parent | 8 | 192 |
| All six families | 56 | 1,562 |

Ten complete replications cost at most 15,620 updates; eleven exceed the original
16,000-update cap. Retain the original 600-second wall limit and observed
512-MiB process RSS limit with 0.005-second sampling. These are prospective
ceilings; this pure consumer does not measure a new fit's time or memory.

Local metadata, fixtures and complete saved-input derivation each have a hard
180-second limit. The unchanged full development reader retains its own
120-second limit. Failed attempts must be preserved and share their gate's
original budget; no narrowed reader, favorable stopping or cap increase.

## Modules and usage

```text
src/core/seed_precision_budget.py         # validated pure numeric calculations
src/app/continual_precision_contract.py   # fixed target, ranges and assumptions
src/app/continual_precision_feasibility.py # whole original boundary and all vectors
tests/test_seed_precision_budget.py       # arithmetic and input edge cases
tests/test_continual_precision_feasibility.py # complete scope and no science/IO
```

Supply the entire original development declaration, not a selected vector or
a previously projected report:

```python
from src.app.continual_precision_contract import fixed_precision_contract
from src.app.continual_precision_feasibility import build_precision_feasibility

report = build_precision_feasibility(complete_inputs, fixed_precision_contract())
```

The public consumer validates the fixed objective before the original input
boundary, rebuilds the whole original development ledger and pilot projection,
and returns an independent result object. Private fabricated seams supply no
original-input authority. No filesystem, scientific source, model, training,
scoring, final release, seed selection or winner selection belongs here.

## Validation and evidence

All 57 new / 155 selected tests pass, zero skipped: 34 core precision, 23 app
feasibility, 27 prior pilot core, 16 prior pilot app, 36 seed statistics and
19 development-boundary cases. They cover known arithmetic and count thresholds,
changed numeric types/ranges/objectives, missing/ordered/duplicate/nonfinite
observations, constants, sign/shift invariance, budget changes, late scope
forgeries, original-authority refusal, deterministic detached results and
raising source/model/train/predict/score/final/IO sentinels. Ruff across
src/tests/scripts, five-file formatting, mypy525 and diff checks pass.

The first Ruff gate rejected one unused import; the other checks and all155
tests passed. Exact v1 source/producer/operations/XML/docs and complete copies
remain in `p67-precision-first-attempt-preservation.json`. Remove only that
unused import, freeze v2 and rerun the same155 tests and static criteria.
The original shared180-second gate is retained for every attempt: tests
14.3112655s, static3.3953418s and freeze9.1585093s total. No scope or cap changes.

The unchanged complete development reader dispatches once, reading twelve
development bundles, eight direct preflight bundles and eight forwarded
canonical references. It rebuilds the entire43,831,409-byte original ledger
in3.8968591s under original120. The public consumer derives and repeats every
vector with whole-byte equality in a69.0640203s worker /69.2071151s parent
under hard180. All150 original source/79 prior test/174 input/fourteen whole
file pins and prior current-reader/failure proofs remain exact before/after;
the entire original current binding is rebuilt once at the end. New source,
test, fixed-method and historical document pins are checked separately.
All24 scientific guards remain0. No source, model, training, scoring, final
release, new confirmation reader or experiment is dispatched.

An independent stdlib-only readback verifies all116 projections/provenance and
348 raw observations, whole repeated bytes, complete additive work and every
new formula/count/null status using70-digit Decimal arithmetic. Its parent
0.1246414s fits its separate180-second gate. The whole original pilot identity
remains317,173 bytes / SHA `c223c8f245804d71f15f708bca0a138c217ef11047dc314ad4a7584007d1ba03`.

## Result: objective not certified within the fixed budget

| Family | Vectors | Conditional normal forecasts | Unresolved constants | Normal forecasts meeting target |
| --- | ---: | ---: | ---: | ---: |
| Gating | 2 | 1 | 1 | 0 |
| Replay | 6 | 5 | 1 | 0 |
| Sleep | 6 | 2 | 4 | 0 |
| Schedule | 18 | 8 | 10 | 0 |
| Combined | 44 | 38 | 6 | 0 |
| Parent | 40 | 36 | 4 | 0 |
| Complete scope | 116 | 90 | 26 | 0 |

| Metric (58 statements each) | Bounded half-width at ten | Conservative sufficient count |
| --- | ---: | ---: |
| Mean accuracy difference (width2) | 1.2994206128286792 | 6,754 |
| Signed forgetting difference (width4) | 2.5988412256573583 | 27,016 |

None of the conditional normal forecasts meets0.05; constants remain null.
The separate bounded check does not certify the target at ten for any vector.
Bounds can exceed the possible metric range because they are conservative;
they are not measured intervals. These sufficient counts are **not necessary
counts**, fit recommendations or permission to exceed the16,000-update cap.
The conditional normal model and development-to-final transfer remain unproved.
The result is an honest planning limitation: precision is uncertified by the
declared checks within the complete fixed budget. It does not establish that
all methods or populations require these counts, and supplies no winner.
Keep the objective, all contrasts, baselines, seeds, metrics and caps unchanged.
Any future ten-seed design must retain an honest exploratory precision status
and the untouched-role/execution gates; original P6.7/d2 are still unchecked.

## Complete local artifacts

Artifacts are ignored, local and exclusive; do not rerun occupied producer
mains. The frozen prospective guide and first failed gate remain historical.
The live guide includes results; the source gate pins its exact prospective copy.

| File under artifacts/runs | Bytes | SHA-256 |
| --- | ---: | --- |
| `p67-precision-tests-validation-v2.json` | 1525 | `1c6bd3db91bc23211d41f08d2a4947c3728b71119eb5238120f3362101003458` |
| `p67-precision-static-validation-v2.json` | 2759 | `c6b6c9a7fabfbfa03d7a3ceb3c23e76007321848883d9a319adc67575e2947fd` |
| `p67-precision-audit-validation-v2.json` | 4767 | `ae197454328399ab628e3cc7448dc2bd0f3c64f87093d33e30e420db30f48b6c` |
| `p67-precision-readback-validation-v2.json` | 1006 | `7b9d61f278a6d0e9a7a7a12d79d81ff455db4b3db045010365cb8a3d4b64e699` |
| `p67-precision-source-v2.json` | 25942 | `b495ba82ec098dca06769df0e5592562824559ce4ecd8c3d7ecc0d9d35491562` |
| `p67-precision-feasibility.result.json` | 319140 | `b10c760c54fb2d333715dc505ccfe4f44501766ede6c6ab4f1e2b6c73c2642fe` |
| `p67-precision-feasibility-repeat.result.json` | 319140 | `b10c760c54fb2d333715dc505ccfe4f44501766ede6c6ab4f1e2b6c73c2642fe` |

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_seed_precision_budget.py tests/test_continual_precision_feasibility.py tests/test_pilot_precision.py tests/test_continual_pilot_variability.py tests/test_seed_statistics.py tests/test_continual_findings_development.py
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src/core/seed_precision_budget.py src/app/continual_precision_contract.py src/app/continual_precision_feasibility.py tests/test_seed_precision_budget.py tests/test_continual_precision_feasibility.py
.\.venv\Scripts\python.exe -m mypy
git diff --check
```

These clean fabricated tests require no ignored scientific artifact. The local
gates used `run-p67-precision-gates.py freeze/tests/static`, then distinct
`run-p67-precision-gates-v2.py freeze/tests/static/audit/readback`; each terminal
operation records its exact command and outcome. The full unrelated/Torch/CI
suite and fresh scientific fits/sweeps were not rerun for these pure additions;
unchanged accepted source/input/test proofs remain. No selected test was skipped.

## Exact next action

After d2a passes, P6.7d2b must audit complete prior source/seed usage and bind
fresh ordered independent source/final roles, a fixed count and honest precision
or exploratory status, unchanged informative factors/matched configurations,
all analysis/stopping/caps/source/request identities and complete future
isolation/reproducibility/resource/artifact fixtures. D3 owns any subsequent
bounded execution and complete repeated readbacks. Original P6.7 stays open
until its own acceptance is met. No large sweep is authorized by feasibility.
