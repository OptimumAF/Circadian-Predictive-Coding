# P6.11a prospective confirmation analysis contract

Date: 2026-09-30. Status: P6.11a implemented and verified before final scoring;
the subsequent complete P6.11b report now passes. Original parent audits remain.
Contract ID: `continual_confirmation_seed_analysis_v1`.
At declaration no independent final score had been read. These rules preceded
synthetic analysis fixtures and the separately frozen P6.7c scoring boundary.
The subsequent complete actual scoring/repetition and
[exhaustive seed/interval/cost publication](p611-confirmation-report.md) pass.
All 626 vectors and all 116 primary statements remain. The subsequent
[complete raw cost join](p611-confirmation-cost-join.md) passes b1, with all
original cost distinctions and exact repeated metadata. This changes no
prospective analysis rule. Actual statistical-report authority is established
by the separate complete b2 publication/repetition/readback evidence.

## Complete scope and endpoints

Bind the unchanged full P6.7a manifest
`8d1ed66b33bbc1bf298cc60604c3741afa22bb4b7e0f6636efe166a52672951b`
and both complete P6.7b results, each SHA
`3d85c60627de63769d0f0fc0bf5ec781c77d50468673dab466d8bbe28089e547`.
Use every original arm, seed and ordered pair, with the original
`phase6_two_task_accuracy_v1` arithmetic. The source of eventual observations
must be `independent_confirmation_final_test`, with A/B forty final examples
and identical final-role fingerprints for arms paired within each seed.
The later scored reader must establish actual role/state/source provenance;
analysis alone cannot prove that supplied numbers came from those roles.

All 56 family-arm groups/560 individual cells and 58 ordered contrasts/580
individual pairs are required, including duplicate PC/neutral, inactive,
negative and null rows. Counts per family are gating 1, replay 3, sleep 3,
schedule 9, combined 22 and parent 20. Preserve exact original order from
`fixed_confirmation_manifest()`; never select a smaller primary subset.

For **every contrast**, the two primary endpoints are final mean task
accuracy and signed A forgetting. All differences are **left minus right**.
Higher mean accuracy is favorable; lower signed forgetting describes less
deterioration. Publish all three A/B endpoints beside these metrics. Lower
forgetting caused by weaker A-after-A is not evidence of better retention.
Units are accuracy fractions, not silently converted percentages.
Raw endpoints and zero-safe retention are secondary diagnostics. Balanced
score duplicates final mean and is not another endpoint. Retention ratios
remain per-arm descriptive values and do not introduce a new paired endpoint.
No intermediate checkpoints, cost-adjusted score or composite winner.

## Replication and uncertainty

Within each family, use the ten original seed IDs as the replication units.
Compute each seed's paired difference before its mean and sample SD. Never
treat forty examples, minibatches, repeated deterministic runs or shared
gating/replay IDs as additional independent replications. Do not pool family
scores; the ten shared gating/replay source IDs remain ten distinct sources.
Publish all individual values, n planned/observed, mean, sample SD (n−1),
standard error and observed range.

For complete ten-seed, nonconstant vectors, predeclare two-sided Student-t
intervals with nine degrees of freedom: mean ± critical × sample SD/√10.
Each raw endpoint and both primary metrics has a marginal 95% interval.
This follows the paired-difference arithmetic and unknown-variance mean
interval described by [NIST paired observations](https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm)
and [NIST mean intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc221.htm).

The primary simultaneous family is **all 58 pairs × two primary metrics =
116 statements**, including duplicate/inactive rows. At family alpha .05,
each two-sided interval uses .05/116 total tail probability; each tail is
.05/232. The [Bonferroni inequality](https://www.itl.nist.gov/div898/handbook/prc/section4/prc463.htm)
bounds simultaneous error without requiring contrasts to be independent,
provided each underlying interval has its stated coverage. Dependence from
shared seeds/arms does not authorize decreasing 116 after seeing outcomes.
Marginal intervals and all secondary intervals are descriptive and cannot
substitute for primary simultaneous intervals in claims.

These are **model-based** intervals: independent source-seed draws and
approximately normal seed outcomes/differences are assumptions, not facts
proved by ten runs. With discrete small final roles, coverage can be poor;
no method switch or favorable subset is selected by a normality test after
scoring. Interpret any directional evidence only under this fixed model,
source geometry, settings and measured costs. Crossing zero is unresolved
within this budget, not equivalence. No general superiority, hypothesis
winner or biological claim follows from this contract.

The fixed critical values are:

| Scope | df | Upper tail | Critical |
|---|---:|---|---:|
| Marginal 95% | 9 | .025 | 2.2621571627982053 |
| Primary simultaneous 95%, 116 statements | 9 | .05/232 | 5.403490569214909 |

Why fixed constants: the complete design always requires ten observations
per vector. Precomputed, independently checked values avoid introducing a
distribution library or an unneeded general quantile implementation. A
changed seed count/alpha/statement count needs a new reviewed contract.
An existing local mpmath 1.3.0 calculation at 60 decimal digits independently
inverted the regularized-beta tail and integrated the
[NIST t density](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3664.htm)
above each critical; both tail errors are below 1e−50 before float rounding.
No runtime/test dependency on mpmath or SciPy is added. Exact command and
digits belong in the development log; checked reference fixtures cover the
float interval arithmetic. Return raw interval endpoints without clipping
them to the metric range.

## Failure, missing and degenerate cases

- Missing, duplicate or unknown family/seed/arm rows, changed contracts,
  reordered scope and conflicting paired final-role identities are errors.
  Input order can be arbitrary; output order must be canonical and complete.
- A failed scheduled cell remains an explicit row with its failure reason;
  it is never silently dropped or given zero accuracy. Preserve both sides'
  failures in every affected pair. Failure reporting cannot satisfy P6.7c's
  successful complete scoring gate by itself.
- Any missing/undefined value in a scheduled ten-seed vector suppresses its
  full planned mean, standard error and all intervals. Report observed-only
  mean/sample SD/range and counts with explicit conditional scope, plus every
  null seed/reason. Do not shrink df, alpha or 116 or rerun a favorable seed.
- An all-failed vector has no numeric observed summary. One observed value
  has no sample SD. Zero A-after-A produces null retention with the existing
  zero-denominator reason; ratios above one remain valid descriptive values.
- A complete constant vector retains its mean/observed SD but gets no
  t interval, status `zero_observed_variance`. Ten identical discrete values
  do not estimate a nonzero population spread or establish equivalence.
  Before app fixtures, declare an absolute SD tolerance of **1e−12 accuracy
  fractions** for this status: subtraction of exact forty-example count
  differences can otherwise create binary roundoff near 1e−17. Preserve the
  computed SD and raw differences; do not round metrics or promote roundoff
  to a narrow interval. The minimum nonzero endpoint resolution is 1/40
  (mean-task resolution 1/80), far above this numeric tolerance. This rule is
  fixed before final scoring and applies to every method, not just ties.
- Retention ratios have descriptive observed summaries only, even when all
  are defined; no normal-model interval or new multiplicity family is added.
- Reject nonfinite/Boolean/out-of-range inputs and arithmetic overflow;
  recompute derived metrics from the validated three accuracies.

Costs remain the complete raw P6.7b vector joined by family/seed/arm and
its exact result digest. No per-arm wall/RSS/FLOPs are invented. P6.10 still
requires any additional resource evidence and accuracy/cost presentation.

## Implemented boundary and verified acceptance

The pure core seed-summary and separate app contract/analysis modules
construct no model/source, read no file/final role, train,
score, choose settings or publish artifacts. Fixtures use fabricated numbers
on declared IDs with source/final/scorer sentinels; they are not experimental
outcomes. Test known arithmetic, pairing covariance, all scope/pair/metric
counts, reversed input, missing/duplicate/mismatched rows, strict types,
nonfinite/overflow, explicit failures/null/constant and exact repeat.

All **102 new tests** and the related **222-test gate in 14.20 s** pass,
zero skipped. The full fabricated matrix retains 560 cells/580 pairs/116
primary statements and exact repeated output even when input is reversed.
Late unknown/duplicate/missing cells, mismatched role/count/seed, altered
contract/container and tampered accuracy fail before the first summary.
Known paired count numerators `[2,3,4,5,-4,7,8,-1,0,1]/80` yield mean 1/32
and SE 7/480, with both independently specified critical values. The NIST
ten-observation example, covariance/constant/roundoff, null/overflow and
all-failed cases pass. Retention remains descriptive; both failed sides and
all planned seeds remain explicit. Model/source/RNG/file/final-release
sentinels pass. After strengthening two existing sentinel/metadata tests,
their direct gate passes 2/64 deselected in .39 s; the intervening full run's
final stream was unavailable and is not used as completion evidence.
Ruff, five-file format and mypy on **392 source files** pass.

```text
src/core/seed_statistics.py                         pure seed summaries/interval arithmetic
src/app/continual_confirmation_analysis_contract.py strict frozen scope/rules
src/app/continual_confirmation_analysis.py          complete pairing and all outcome vectors
tests/test_seed_statistics.py                       36 arithmetic/policy cases
tests/test_continual_confirmation_analysis.py        66 complete-scope/sentinel cases
```

The API accepts already verified observations; the eventual scored reader
must supply provenance before calling it:

```python
from src.app.continual_confirmation_analysis_contract import fixed_analysis_contract
from src.app.continual_confirmation_analysis import analyze_confirmation

contract = fixed_analysis_contract()
report = analyze_confirmation(tuple(verified_outcome_cells), contract)
# report retains raw cells, every arm/pair vector and the exact cost reference.
```

Contract SHA-256:
`5e33ef28862bcdf9d92fe14dd6cf6b71672a2336ffd760a1214ef04666b594b1`.

| New scientific component | Byte SHA-256 |
|---|---|
| Core seed statistics | `d848ea585046c0c83ed73e22953f0c5361012b73129e12c464dc5290720195cd` |
| App frozen contract | `0e3989ecfb8191ba4aea1ebb989da23581e48d7a8a919c1a21156b0505323436` |
| App complete analysis | `3fed747ac5bdd61569c33a48f768b5f54c3159991de9d2e694b5c827a9f25de0` |

These additional identities supplement the unchanged 79-source P6.7b
closure; the final scorer must bind its full new closure separately. No
original scientific source, seed, baseline, metric or setting changed.
No new experiment/scored artifact, actual final value or dependency.

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts= -q -ra --maxfail=1 tests/test_seed_statistics.py tests/test_continual_confirmation_analysis.py tests/test_continual_metrics.py tests/test_continual_confirmation_manifest.py tests/test_p67_confirmation_scope.py tests/test_continual_confirmation_execution.py tests/test_p67_confirmation_training_cli.py
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src/core/seed_statistics.py src/app/continual_confirmation_analysis_contract.py src/app/continual_confirmation_analysis.py tests/test_seed_statistics.py tests/test_continual_confirmation_analysis.py
.\.venv\Scripts\python.exe -m mypy
git diff --check
```

Exact commands/intermediate repairs/numerical-oracle results are in the
development log. Clean fixtures need no ignored artifacts; the complete old
source/scope/reference binding revalidates read-only. Full CPU/CUDA/sweeps/
scored confirmation and actual seed intervals were skipped.

P6.11a meets only predeclaration/correctness. The subsequent P6.7c and P6.11b
evidence satisfies their separate full scoring and report criteria with this
contract unchanged. Original P6.11/matrix/failure/metric/cost and broader
confirmation parents retain their own criterion-level acceptance audits.
