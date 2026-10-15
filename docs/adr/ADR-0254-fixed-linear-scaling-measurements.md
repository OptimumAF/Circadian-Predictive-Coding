# ADR-0254: fixed linear scaling measurements with deferred scoring

## Context

R1.4a implements raw/effective scaling and SGD. R1.4 still needs actual norms,
cost and heldout measurements across repeated widths/depths. Existing local
budgets and seed statistics already supply the necessary infrastructure.

## Decision

Add a small application runner that copies role-separated regression batches,
collects training diagnostics, then scores detached fixed checkpoints only after
all training completes. Keep every fixed grid row and withhold partial scores on
failure. Use existing wall/RSS budgets and full declared seed summaries. Compose
fixed data, initialization and JSON in an outer adapter/CLI.

Why this: a fixed grid exposes parameterization behavior while preventing
heldout feedback from influencing training or replacement of unfavorable runs.

## Alternatives

Embedding measurement logic in the core would mix scientific orchestration with
pure updates. A configurable sweep framework would increase scope without
helping this finite 81-case protocol. Reusing binary attribution data would add
unnecessary task and evaluation coupling.

## Consequences

The opt-in CLI becomes runnable without changing existing learners. Diagnostic
gradient work is recorded separately from learning; timings are descriptive.
Finite failures are legitimate retained outcomes. Tiny fixed regression data,
three seeds and six steps cannot establish asymptotic scaling, convergence,
nonlinear/CPC benefit or residual-depth correctness. Parent R1.4 remains open.
