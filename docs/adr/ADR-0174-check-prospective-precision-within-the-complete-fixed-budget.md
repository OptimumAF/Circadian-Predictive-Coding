# ADR-0174: check prospective precision within the complete fixed budget

## Context

P6.7d1 retains all original three-seed paired variability. It cannot supply
missing prelaunch justification for the already released ten-seed confirmation.
P6.7d2 combines precision planning with untouched-role execution design. Pilot
variance is weak, some paired pilots are constant, metrics are discrete, and
development/final dispersion may differ. The original complete work ceiling
allows ten replications per family under the 16,000-update cap.

## Decision

Split d2 into d2a complete precision/budget feasibility and d2b fresh-role and
execution gates, preserving d2/P6.7 acceptance. Before calculating feasibility,
freeze a five-percentage-point simultaneous mean half-width for all 116 original
statements at family alpha 0.05. Its resolution rationale is one decision in
twenty, independent of prior confirmation outcomes and equal across both
primary metrics. This is not a biological relevance threshold.

Implement pure core arithmetic and a fixed app contract; the app consumer
reuses the entire original public development/pilot boundary. Preserve every
raw seed row, metric, pair, work ceiling, constant/null result and assumption.
Use a conditional df-two normal SD upper sensitivity with unchanged df-nine
scaling, plus a separate bounded-independent sufficient-count calculation.
Keep role-transfer, non-normality and separate alpha levels explicit. A failing
conservative check does not prove precision is impossible by all methods.

Why this: declaring a common resolution before feasibility prevents selection
of an objective or contrast that favors the circadian model. The separate
bounded check prevents zero pilot variance from implying certainty. Neither
method requires a new dependency, source construction, scoring or a sweep.

## Alternatives

- Treat ten historical seeds as prospectively variance-justified: violates
  chronology and preserves the original sample-size gap.
- Treat three-seed SDs as known population SDs: hides dispersion uncertainty.
- Tune the target, reduce contrasts or expand the exhausted budget after the
  calculation: changes the declared research question and stopping rules.
- Introduce a new robust interval or adaptive sample-size procedure here:
  changes the historical analysis and exceeds this small consumer's scope.

## Consequences

P6.7d2a can complete after its full correctness and complete-input gates, even
if the objective is not certified within the budget. The report grants no
confirmation authority. D2b must bind untouched independent roles and a fixed
honest precision/exploratory status with all future execution gates. Original
P6.7, d2 and d3 remain unchecked until their actual acceptance is satisfied.
See [method, sources, limitations and evidence](../p67-precision-feasibility.md).
