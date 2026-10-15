# ADR-0247: Add an opt-in coupled binary PC objective

## Context

ADR-0021 preserves legacy training diagnostics; they are not the gradient source
for every multilayer update. ADR-0023 limits the existing neutral PC/CPC parity
claim to shallow networks. R1.2 requires an explicit coupled scalar objective
before deeper derivative verification and mechanism attribution.

## Decision

Add separate NumPy energy and learner modules. Use binary BCE from logits plus
per-layer precision-weighted squared prediction residuals, summed over neurons
and averaged over batch. Recompute predictions from current lower latents and
differentiate the same energy for all latents, weights and biases. Keep input and
likelihood targets fixed; introduce no output latent/clamp. Fixed simultaneous
latent steps precede one simultaneous held-latent parameter step.

The new energy identifier is `numpy_coupled_binary_energy_v1`. Existing models,
initializations, energy identifiers, adaptive triggers and study results remain
unchanged. Ordinary fan-in initialization is explicit and does not claim a
published scaling reproduction. Full formulas and checks are in
[the specification](../coupled-predictive-coding.md).

## Alternatives

- Correcting legacy models in place would change frozen experiments and sleep
  triggers and obscure old diagnostic interpretation.
- Labeling a fixed-prior diagnostic as a coherent deep objective would not
  establish its missing derivative relationships.
- A neutral alias with no integrated circadian path would not prove the required
  PC/CPC mechanism-neutrality criterion.

## Consequences

The new learner is available explicitly and can be verified with central
differences and test-only autodiff before integration. Every finite update is
validated before publishing parameters. Bounded precision and inference counts
do not guarantee arbitrary-step convergence, hard memory bounds or a learning
advantage. Published reproduction, real neutral CPC integration, resource
matching and scientific evaluation remain separate gates.
