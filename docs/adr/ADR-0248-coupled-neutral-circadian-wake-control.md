# ADR-0248: Share coupled inference with a real neutral CPC wake control

## Context

R1.2's coupled binary objective has verified derivatives. R1.3 also requires a
neutral CPC control at depth. Legacy ADR-0023 only proves shallow parity, because
the older multilayer PC/CPC algorithms differ. An alias or no-op subclass would
not demonstrate neutrality of a functioning circadian path.

## Decision

Share PC inference and validated candidate preparation through two protected
helpers. Add a separate coupled CPC wake implementation using the existing basic
activity trace and exponential gate at the final hidden layer. A dedicated
frozen configuration lists only implemented settings. Its neutral preset forces
unit gates while real chemistry still accumulates.

Independent controls compare complete parameters, energies and predictions at
1/2/4 layers across repeated matched wakes. An active-gate control must exhibit
the specified selective update scaling. All prospective parameters/chemistry and
energy validate before publication. See [the complete specification](../coupled-neutral-control.md).

## Alternatives

- Reusing legacy CPC directly would retain incompatible deep inference behavior.
- A no-op alias could match PC without testing any circadian mechanism.
- Copying the inference loop into CPC would let normalization or derivative
  changes drift between the baseline and control.
- Accepting legacy feature flags without implementation would imply unsupported
  replay, sleep or topology behavior.

## Consequences

The new control is scoped to the coupled binary wake implementation. It does not
close structural, scaling, runtime/checkpoint or scientific benefit gates. Extra
chemistry work remains explicit; numerical parity alone is not equal compute.
Historical models and experiments retain their separate definitions.
