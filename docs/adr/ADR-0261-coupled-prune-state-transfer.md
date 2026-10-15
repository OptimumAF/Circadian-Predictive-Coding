# ADR-0261: complete native coupled prune transfer

## Context

Linear structural operations do not transfer nonlinear biases, layer precision or
CPC chemistry. Native coupled snapshots already define complete compatible state;
native training is synchronous failure-atomic and has no adaptive optimizer.
R1.6 needs a concrete nonlinear transfer without erasing trace association.

## Decision

Prune one explicit hidden coordinate into a fresh exact-family native model,
selecting incoming/bias/outgoing parameters. Preserve precision and policy/count,
selecting CPC chemistry/gates only for last-layer pruning. Validate full source
and target restores. Bound structural geometry before allocation and copy policy
records. Gate probability/BCE/fixed-step relaxed energy on declared train/validation
probes, requiring every inference step and retaining source on failure/rejection.

Why this: native restore already enforces gate/chemical consistency and avoids
publishing a partial new topology. A truncated inference diagnostic would compare
different completed procedures; it must not silently become an accepted score.

## Alternatives

Parameter-only transfer loses adaptive continuation. Resetting chemistry changes
the model's learned wake state. Applying linear isometries to tanh assumes a
false homogeneity relationship. Calling training to score probes would mutate
parameters/chemistry. Equating relaxed energy to a minimum or equal steps to equal
compute would overstate the evidence.

## Consequences

General function/future-wake preservation is not promised. Probe roles are caller
declarations; final-test selection is refused. The functional model replacement
does not mutate an existing runtime holder or grant promotion authority. Runtime
publication/rollback, broader structural operations and nonlinear raw scaling
remain open under R1.6/R1.8. Existing native defaults, snapshots and adapter restore
domains are unchanged; no scientific benefit claim follows from engineering tests.
