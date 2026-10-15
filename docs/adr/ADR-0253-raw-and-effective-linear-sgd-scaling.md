# ADR-0253: Separate raw and effective linear PC scaling

## Context

R1.4 requires published parameterization controls before scaling experiments.
Initialization-only scaling is insufficient: the same forward multiplier also
changes raw-coordinate derivatives and effective-coordinate SGD rates.

## Decision

Add an opt-in scalar linear utility with explicit raw state/configuration, exact
initial/effective tensor records, and a simultaneous raw-coordinate equilibrium
step. Mean-field factors follow paper Table 2 and pinned author source. Legacy
Gaussian and ordinary Gaussian fan-in controls remain separately named.

Use the existing full equilibrium gradient, including the denominator derivative.
Test the raw computation graph independently in Torch. Multiply raw weight
deltas with binary mantissas/exponents to avoid an observed premature-underflow
failure without narrowing the accepted mathematical configuration.

## Alternatives

Changing existing binary or legacy defaults would invalidate accepted behavior.
Using fan-in initialization factors as persistent forward factors would change
the control. Omitting squared effective SGD factors implements another algorithm.
Refusing additional extreme gamma/rate values would hide the representable-update
bug instead of repairing its arithmetic.

## Consequences

The reference can now express and execute the published linear mean-field SGD
mapping, with detached states and explicit numerical limits. It does not prove
width stability, residual depth scaling, Adam behavior, or applicability to the
binary/CPC objective. R1.4's empirical acceptance criteria remain open.
