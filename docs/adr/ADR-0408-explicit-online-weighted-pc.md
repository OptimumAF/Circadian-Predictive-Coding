# ADR-0408: Explicit weighted PC/neutral online composition

## Context

Weighted PC math, complete episode mutation and the existing weighted schedule
have accepted correctness evidence. The public MiniGrid weighted route previously
admitted only BP. Reserved probe/return updates cannot use the benchmark's sealed
TRAIN-only schedule, and its weighted energy differs from ordinary diagnostics.

## Decision

Add an explicit `weighted_pc`/`--weighted-pc` option for the two original S2 PC/
neutral qualification recipes in the existing online pilot. Extend the existing
weighted route with optional method parameters whose default retains BP. Bind
full fixed policies, complete TRAIN packets, exact receipt IDs and doses.

Convert only probe source templates to ordinary unweighted native learners with
identical full initial state. Existing helpers restore original benchmark states
on disposable forks for F16/old-A/A16. Keep metrics, budgets, paid-prefix handling,
source/actor/state/protocol guards and all old defaults unchanged.

## Alternatives

A new runner would duplicate the accepted stream/probe implementation. Allowing
reserved episodes into the sealed benchmark schedule would violate isolation.
Reusing the weighted objective ID for ordinary native probes would misstate their
legacy diagnostic. Applying this episode law to joint54 aggregates would change
its semantics without a protocol. All are declined.

## Consequences

The existing BP control and explicit weighted PC/neutral controls can be described
and run locally through the same online API/CLI. This adds no experiment, task
selection, advantage or release claim. A small prospective comparative protocol
with frozen seeds/metrics and separate costs remains the next scientific gate.
