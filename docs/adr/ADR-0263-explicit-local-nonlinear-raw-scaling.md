# ADR-0263: Explicit local nonlinear raw/effective coupled scaling

## Context

Linear raw/effective scaling and native tanh/BCE PC/CPC are separately verified.
The linear identity-activation equilibrium does not establish nonlinear behavior.
Biases, precision, loss normalization, latent coordinates and plasticity must be
explicit before integrating raw coordinates into the coupled learner.

## Decision

Add a separate uniform-width raw config/state/kernel and an owned learner adapter.
Reuse existing weight initialization/forward/rate factors. Keep native latent
coordinates, tanh, beta and logit BCE energy. Choose bias scale1/sqrt(stored raw
rate multiplier) with zero initialization, reporting actual finite-precision
effective rates. This is a local extension and makes no published nonlinear
stability claim.

Return true ungated raw partials from a separate evaluator. Steps combine
effective partials, all scalar factors and prospective array gates before final
rounding, without requiring representable raw-gradient diagnostics. CPC gates
precondition exactly the native associations; no gradient through chemistry.
Full raw/config/wake validation and candidate energy precede replacement.

## Alternatives

An effective-only optimizer would lose raw-coordinate semantics. Scaling only
initialization omits optimizer factors. Reusing unscaled biases with a growing
raw rate would introduce width-growing effective bias updates. Applying linear
equilibrium formulas or reinterpreting energy normalization without a derivation
would silently change the objective. Changing native defaults would confound
accepted baselines and historical negative comparisons.

## Consequences

The opt-in algorithm is completely specified and testable. Full raw snapshots
are required; existing native/topology/checkpoint routes remain separate.
Independent derivatives, native fan-in agreement, neutral parity, active gate
association and continuation precede measurements. This implementation alone
does not prove convergence, width invariance, savings or circadian benefit.
Future structural operations must account for the explicit config and raw state.
