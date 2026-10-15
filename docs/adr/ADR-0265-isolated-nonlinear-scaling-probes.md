# ADR-0265: Isolated nonlinear scaling probes and matched raw controls

## Context

R1.8a defines a local tanh/BCE raw mapping; R1.8b1 supplies matched raw BP. Neither
an equation example nor linear theory measures nonlinear finite-network behavior.
Existing final-scoring isolation must survive adding checkpoint measurements.

## Decision

Add an optional copied checkpoint observer to the existing delayed stream and a
bounded application runner. Initial probes are target-free; subsequent probes
use arrived training batches only. Final scores remain deferred until training
ends. Distinguish postupdate diagnostic probes, native preupdate receipts and
applied gated updates. Failures retain evidence without partial final scores.

Use existing full raw learner ports and process sampler, with a frozen small
96-case matched protocol. Keep all seeds/failures and explicit missing estimates.
One live CLI sampler spans factories through actual encoding/output, with a
separate command receipt for the later resource observation.

## Alternatives

Scoring during training would expose final outcomes to subsequent execution.
Reusing uniform-effective-rate BP would mismatch the raw per-layer/bias rates.
Treating fixed relaxation as equilibrium would overstate the objective contract.
Adding an experiment framework is unnecessary for one bounded local protocol.

## Consequences

Measurements are runnable and isolated, with clear finite-work and resource
semantics. Three updates and three fixed-data seeds are exploratory, not evidence
of convergence, asymptotic stability, equal compute or general CPC benefit. No
existing model defaults, prior negative results or failed budgets are changed.

## Legacy completion addendum — 2026-10-15

Keep the original two-scheme protocol fixed; an explicit `legacy_only` selection
adds the missing Gaussian control under a separate prospective allowance. Summary
groups are restricted to the original pair, legacy singleton and ordered full
triple. Retained-only aggregation checks exact settings/data and each origin's
declared case keys before the union, preserving source reports and failures.
Why this: complete the all-schemes criterion without repeating the accepted 96
models or choosing settings using their scores. The two invocations total 432
updates; they make no equal-compute or broad stability claim.
