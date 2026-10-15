# ADR-0251: Match feedforward BP by explicit parameters

## Context

R1.7 requires matched baselines before attributing differences to PC or circadian
plasticity. Existing multilayer BP has a different initialization and clipped
diagnostic contract from the new coupled binary models.

## Decision

Add an opt-in stable-logit BCE model consuming owned explicit CoupledParameters.
Share inward core validation, tanh feedforward prediction and stable sigmoid
helpers. Differentiate the complete feedforward graph and retain a distinct
preupdate loss ID. Add complete fixed-topology state and owned native ports so
the same delayed-label runtime can execute all methods from identical tensors.

Independently check gradients and repeated updates with actual Torch. Preserve
the existing BP model and its historical diagnostics. Parameter-only tensor
transfer is explicit; full snapshots remain family-specific.

## Alternatives

Matching seeds across the old/new initializers does not match tensors. Changing
legacy initialization or clipped diagnostics would alter historical behavior.
Treating PC fixed-latent partials as BP derivatives would implement the wrong
baseline.

## Consequences

Representation, initial weights and feedforward likelihood can now match.
Relaxed PC total energy and inference work still differ from BP; this integration
example does not establish comparative performance or equal computation. R1.7's
study acceptance criteria remain open.
