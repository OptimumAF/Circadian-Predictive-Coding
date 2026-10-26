# Four-phase train-only key/door sequence

The explicit recipe runs A64 → B64 → C_train64 → A64 through the existing
AttributionStream and recurrent mean-CE learner, with continuous SGD momentum.
It reports fixed train-resubstitution curves and signed acquisition/forgetting/
return differences. It never constructs final or probe data, runs a live policy,
promotes a learner, selects a checkpoint or changes the budget from its scores.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-train-four-phase --describe
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-train-four-phase
```

`--describe` allocates no learner, environment or native execution resources.
Execution is the separately budgeted local256-update recipe; describing it is
not evidence that native behavior passed. No new dependency or configuration.
The original64-update acquisition and192-update A/B/A routes remain unchanged.

## Boundaries and protocol

```text
src/adapters/train_acquisition_pilot.py  fixed schedule/data/runtime composition
src/core/context_return_metrics.py     pure strict signed endpoint arithmetic
scripts/run_coupled_attribution.py      explicit scenario dispatch/output cap
tests/test_four_phase_*.py             metric and real-stream fake-port behavior
```

Why: the existing long return recipe only schedules A/B/A, while R4.2 requires
a declared third changed context before return. C_train has rotation2,
branch_row3, swapped keys and appearance−.75. It differs from historical final
C (rotation2/branch_row2/unswapped/appearance−.5), whose reports stay untouched.
The endpoint metric alias C explicitly means C_train in this protocol.
No prior final packet, observation or label trains this new route.

Six actual complete eight-step teacher traces are collected once, all role
`train`; alternating copies supply256 unique episode IDs and2048 repeated
transition exposures. Numeric repetition is not independent evidence. The same
native H8/seed17/lr.03/momentum.9 owner continues across phases. Retained scoring
forks never update the candidate, actor or source. All six original train traces
are scored at0 and8/16/32/64 offsets in each phase (17points,102 forwards,
816 scored rows); future-phase scores are train diagnostics, not held-out tests.
Final evaluation stays absent. Historical C/D failures remain negative results.

All five endpoints0/64/128/192/256 are retained for A/B/C, including incomplete
initial acquisition. For CE define d=−1; for accuracy/exact sequence d=+1.
Gains d*(later−earlier) are desirable positive. Losses/gap d*(reference−later)
are worse positive. Exact formulas/names are in `--describe`; zero, negative and
over-recovery values remain. A returned fit does not establish restoration of
an acquired live skill, recovery savings or broad generalization.

## Verification and extension

Prefix Python commands with `.venv/Scripts/python.exe -X utf8 -B`:

```text
-m pytest -q tests/test_four_phase_return_metrics.py tests/test_four_phase_train_pilot.py tests/test_context_return_metrics.py tests/test_train_context_return_pilot.py
-m ruff check src/core/context_return_metrics.py src/adapters/train_acquisition_pilot.py scripts/run_coupled_attribution.py tests/test_four_phase_return_metrics.py tests/test_four_phase_train_pilot.py
-m ruff format --check <same five paths>
-m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 <same five paths>
```

Also run mypy with `--platform linux`. Actual outcomes, failures, skipped gates
and remaining claims are recorded in the stage results after execution, not
inferred from authored tests. No equal-compute or circadian-benefit comparison
is established by this single recurrent control. Reuse the existing fixed
policy-evaluation/probe boundaries for a separately declared future outcome;
never attach them by accepting old final data as training input.

## Observed frozen case

181 focused cases, five-path Ruff/format/both-platform mypy and one frozen local
native case passed. Native work256SGD/2048trainrows/102predictions/816scorerows/
2864unrolls, six teacher environments/48steps,5.2617parent seconds. Sampled recipe
RSS316477440B excludes any unobserved later serialization peak.

| Endpoint | A train accuracy | B train accuracy | C_train accuracy |
|---|---|---|---|
|0|.125|.1875|.125|
|64|.625|.4375|.125|
|128|.625|.875|.1875|
|192|.25|.5625|1|
|256|1|.5625|.25|

Both C_train sequences fit by192, then fail complete fitting after A return.
Both A sequences fit at256; initial A fit was incomplete. B never fits a complete
sequence at these endpoints. This demonstrates train-fit acquisition and
forgetting, not retained live navigation skill or recovery savings. Common A/B
scores/diagnostics through128 and first128 receipts match accepted old evidence
exactly. Native opaque state identity cannot be compared to the old JSON.
[Exact commands, all signed metrics, costs and skips](../artifacts/runs/r42-four-phase-train-20261016/RESULTS.md),
[independent review](../artifacts/runs/r42-four-phase-train-20261016/REVIEW.md).

Next add an explicit four-phase post-training cached/live route through existing
evaluation boundaries, collecting final data only after training finishes. Keep
old defaults and every failed outcome; freeze/review before a new native case.
