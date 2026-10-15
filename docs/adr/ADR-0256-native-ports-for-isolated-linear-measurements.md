# ADR-0256: native ports for isolated linear measurements

## Context

The residual state and raw SGD are verified, but the measurement runner supports
only MLP states. Copying its loop would duplicate role, budget, timing and failure
semantics. Existing MLP callers and experiment results must retain their scope.

## Decision

Use a frozen typed native port record for copied state, complete effective maps
and advance state. Public MLP/residual wrappers select exact native families and
share the existing isolation engine. Ports are trusted internal composition,
not decoded caller configuration. Extract common outer case/grid execution;
the residual adapter supplies only its native factory, run function and fixed
protocol. Keep existing MLP entry points and protocol values compatible.

Why this: one implementation of deferred scoring and all-or-nothing failure
scores avoids divergent validation behavior. Native wrappers retain independent
math/state ownership and cannot substitute the other model family's updates.

## Alternatives

A copied residual runner would duplicate most logic. A fully extensible model
registry or untrusted callable API adds unnecessary configuration and validation
scope. Replacing residual state with effective MLP state would lose its raw SGD
parameterization and fixed identity semantics.

## Consequences

Both topologies have the same metrics and resource accounting while keeping
their native raw/effective factors. Focused MLP regression checks are required
for the shared refactor. Historical frozen outputs remain unchanged; current
implementation source identities may differ. A separate prospective residual
grid is necessary for residual empirical acceptance. Nonlinear/CPC and broader
scientific claims remain outside this linear measurement result.
