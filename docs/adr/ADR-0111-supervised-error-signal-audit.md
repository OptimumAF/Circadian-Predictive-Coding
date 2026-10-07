# ADR-0111: Audit supervised-error signals before difficulty heuristics

## Context

The reward-named switch in both circadian backends measures mean absolute
supervised output error against an EMA baseline. Its clipped factor scales
the current wake update. P4.6 asks whether label noise and feature outliers
can distort that factor and whether clipped error, loss improvement, or no
modulation would be better. Changing the factor before observing its
behavior would alter old protocols and make a negative result hard to
interpret.

## Decision

Split P4.6 into a read-only signal audit and a later matched learning
comparison. The fixed four-row train-only probe calibrates both existing
NumPy and CPU Torch reward-error baselines on clean output errors, then
changes exactly one label or one feature. It computes a predeclared 0.5
per-row error clip and a fixed 20% label-directed correction for a
post-update BCE-improvement diagnostic. The unmodulated factor is one.
These diagnostics are not routed into the model optimizer and no config,
snapshot, checkpoint, or protocol ID is changed.

The post-update loss is unavailable when deciding the same update. Any
future learning experiment that tests improvement as a modulation factor
must define a prior-step signal and bind its work/roles before training.
P4.6b retains the matched historical-versus-unmodulated outcome gate;
no candidate is chosen from this four-row probe.

## Alternatives

- Replace the historical factor immediately with clipped error. The
  signal probe has no held-out accuracy or forgetting evidence.
- Use same-step loss improvement to scale the update that produced it.
  That would depend on future model state and extra work.
- Treat the reward name as environmental feedback. No such signal is
  observed; the targets are supervised labels.

## Consequences

One flipped label and one feature outlier both saturate the historical
factor at 1.5. The clipped-error diagnostic gives 1.375 for each. A
constructed same-row improvement is largest for the outlier, so it does
not identify a useful training example in this fixture. NumPy and CPU
Torch show the same historical factor on identical error arrays. These
observations support a bounded matched learning comparison, not a
superiority or generalization claim. The exact values and limitations
are recorded in `docs/difficulty-modulation-audit.md`.
