# ADR-0518: Score native probabilities with existing metrics

## Context

Released Open-Jev Choice replies expose probabilities and a margin confidence,
not raw logits. The existing DecisionResult requires raw_score and a distinct
tie rule. One real native CPU smoke is accepted; task scoring needs its observed
probabilities without changing that native objective or older interfaces.

## Decision

Add the small inward ProbabilityDecisionResult port, a native decoder projection
and an entrypoint in the existing evaluator. Share case admission and NLL/Brier/
five-bin ECE algebra. Preserve p exactly within native mass tolerance; retain
provider confidence separately from max p used for ECE. Unknown reported counts
remain unknown. Keep adverse order observations and paid failure prefixes.

## Alternatives

Inventing logits from log(p) would misrepresent observations and fail at zero.
Adding raw_score=None would weaken an existing strict contract. A separate
evaluator framework would duplicate role isolation and metric definitions.

## Consequences

Core/app remain independent of the wire adapter and model IO. The saved smoke
can be scored without new inference, with inherited tokens explicitly separated.
Generic callbacks remain unsandboxed; this port does not establish calibrated
confidence, model quality, adaptation, provenance authentication or CPC benefit.
Existing raw-score evaluator semantics remain intact. Subsequent comparisons
require a separately frozen protocol. [Evidence](../../artifacts/runs/native-probability-evaluation-20261022/RESULTS.md).
