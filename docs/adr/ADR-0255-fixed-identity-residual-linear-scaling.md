# ADR-0255: fixed identity residual linear scaling

## Context

The accepted linear MLP utilities implement width scaling and raw SGD. R1.4 also
needs a depth parameterization. Paper §4 Eq. 19 changes middle forward maps and
their optimizer chain rule; an initialization-only change is insufficient.

## Decision

Add a separate pure core module composing unchanged MLP configuration,
initialization, stable delta multiplication and scalar equilibrium. Define
`L=H+1` total learned matrices. Add fixed identities only to middle maps;
mean-field branch factor is `1/sqrt(N*L)` and raw SGD remains `gamma0²*N`.
Expose copied raw state and detached complete effective maps. Validate a full
candidate equilibrium before returning a replacement state. Keep the explicit
Gaussian controls at their existing scales/rates with the same skip topology.

Why this: the identity is not a trainable parameter and must not be scaled with
the branch. Complete effective matrices belong in prediction and suffix products,
while the raw derivative contains only the branch factor.

## Alternatives

Changing the existing MLP configuration would alter accepted experiment source
and blur topology. Treating the entire effective map as raw weights would change
the optimizer. A generalized learned-skip or configurable-depth framework adds
semantics unnecessary for the fixed published residual mapping.

## Consequences

The runnable equation example and independent raw Torch controls can verify
residual SGD without changing legacy/binary/CPC defaults or prior study results.
The one-hidden-layer case has no residual blocks. Raw storage limits do not
include identity/temporary memory. Residual measurements, nonlinear applicability,
adaptive optimizers and theoretical depth-limit validation remain separate work.
