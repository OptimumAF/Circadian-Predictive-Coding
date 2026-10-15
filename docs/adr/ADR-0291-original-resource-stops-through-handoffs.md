# ADR-0291: Original resource stops through coupled handoffs

Date:2026-10-15. Status: accepted design; validation tracked in R3.7u.

## Context

Repeated handoffs preserve update/preparation limits, but the existing example
uses constant wall time without an RSS sampler. The runtime already supports
cumulative wall/RSS budgets; those resource origins must survive owner changes.

## Decision

Add an opt-in fixed integration that retains the original budget clock and RSS
sampler/reader across two handoffs, then triggers wall or RSS refusal before
another builder/update. Preserve spent preparation/admission counts and serving
availability. Use deterministic injected resource values for these decisions and
report actual host RSS boundary observations separately in the saved driver.

## Alternatives

New budget/sampler objects per owner would reset start/peak and fail the intended
invariant. Real allocation pressure or timing races would make these correctness
controls flaky. Increasing endurance duration first would leave the resource
origin checks unverified.

## Consequences

The integration reuses existing controller/budget/sampler APIs and adds no core
authority or algorithm. Full state and stable resource metadata remain equal
while legitimate cumulative time/sample observations advance. A lower later RSS
reading cannot erase the original high-water refusal. Sharing admission may be
spent even when before-update admission prevents training. Separate host readings
are two boundaries, with no continuous peak/model-memory claim. Full long-run,
managed/process/lifecycle recovery and scientific comparisons stay unfinished.
