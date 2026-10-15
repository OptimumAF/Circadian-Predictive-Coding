# ADR-0250: Separate exact linear PC reference

## Context

R1.1 requires a published reference behavior after inspecting full equations and
author code. The new coupled binary learner has a different objective; its
neutrality tests cannot reproduce a clamped linear squared-error result.

## Decision

Add a pure NumPy scalar-output linear equilibrium calculation and a fixed runnable
CLI. Reproduce the energy identity and held-equilibrium local derivatives from
equations 2–6 of the ICML 2026 paper. Pin the inspected author source and document
all modeling and execution differences in `docs/linear-pc-reference.md`.

Use independent hidden least-squares and Torch quotient differentiation as
oracles. Keep the derivative of the weight-dependent denominator. No author code
execution, external dataset, initialization study, or scaling sweep is required
for this small reference behavior.

## Alternatives

Installing the author's JAX stack and reproducing an entire figure adds cost
beyond the current small reference. Changing the existing binary learner to a
clamped linear objective would break its accepted contract.

## Consequences

The runnable reference supplies a checkable prerequisite for future scaling work.
It neither implements the paper's parameterization nor validates asymptotic,
comparative, or biological claims. Existing learners and runtime stay unchanged.
