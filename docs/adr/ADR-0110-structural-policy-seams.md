# ADR-0110: Keep structural policy seams local to each backend

## Context

P4.5 asks the existing `NeuronAdaptationPolicy` and typed proposal path to
remain usable while separating usage scoring, candidate selection, budget
calculation, and tensor mutation. NumPy already accepts external
`NeuronChangeProposal` objects and built-in scores. Torch's built-in sleep
chooses prune candidates on detached tensors after a noisy split; the
candidate may be a newly created child. A pre-sleep index proposal does not
describe that decision without changing its meaning.

## Decision

Keep the NumPy `NeuronAdaptationPolicy.propose(LayerTraffic)` contract and
typed `NeuronChangeProposal` result. Parse and validate external requests,
resolve phase and configured caps, find eligible original-width split
sources, and rank them in separate methods. The existing split/prune score
methods remain usage scoring, and `_split_neurons` plus prune methods remain
the only NumPy tensor mutators. A policy can request a split and prune in
one call; the prune owns its original neuron and the split uses a different
eligible source. Invalid phase budgets reject the whole event before split
noise or tensor mutation. The test exercises early and late zero budgets
and an accepted middle-phase call.

Torch keeps its own score, candidate, budget, and mutation methods and its
detached post-split planner. No common runtime policy class, new config
field, or changed default structural rule is introduced. The existing
cross-backend score parity fixture remains the check where their semantics
can be aligned.

## Alternatives

- Route Torch through NumPy's pre-sleep proposal indices. This would lose
  the existing ability to prune a newly split child and alter outcomes.
- Introduce a generic cross-backend planning framework. The two mutation
  orders differ, so a shared abstraction would hide a scientific choice.
- Leave external request parsing, caps, and ranking in one long method.
  That makes it harder to audit an invalid policy proposal before mutation.

## Consequences

The refactor changes no config, snapshot, checkpoint, metric, or protocol
identity. Deterministic phase-budget tests and NumPy/Torch structural
regressions guard the seam. External policy authors use the same traffic
input and typed proposals; they must respect the active sleep budget, and
the model reports an error without partial mutation when they do not.
