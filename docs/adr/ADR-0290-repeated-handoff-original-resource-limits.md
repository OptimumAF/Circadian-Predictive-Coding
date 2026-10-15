# ADR-0290: Repeated handoffs retain original resource limits

Date:2026-10-15. Status: accepted design; validation tracked in R3.7t.

## Context

The existing complete-policy integration covers one refused preparation and one
successful transfer. That does not demonstrate cumulative controller capacity
and original runtime quotas across several owner changes.

## Decision

Add an opt-in fixed three-cycle runnable integration using the same controller,
actor, clock, budget and sharing gate. Include an initial failed policy attempt,
causal training before each transfer, retired/token refusals, and final spent
preparation and update limits. Retain full state/cursor/resource observations and
matched uninterrupted references for PC, neutral CPC and active CPC in both
inference modes. Keep existing default behavior and production contracts intact.

## Alternatives

A new controller or budget per cycle renews capacity and cannot demonstrate the
required invariant. Large endurance sweeps would spend unnecessary resources
before this fixed repeated correctness control passes.

## Consequences

The example exercises cumulative state management with bounded telemetry and
explicit exhaustion. It adds no recovery framework or algorithm. Three cycles
do not prove full long-run memory stability, disk/process/lifecycle restoration
or comparative learning benefit; those parent requirements remain open.
