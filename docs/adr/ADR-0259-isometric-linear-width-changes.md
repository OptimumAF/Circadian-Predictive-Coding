# ADR-0259: isometric linear width changes

## Context

R1.6 requires structural operations under the new scaling parameterizations.
The verified MLP/residual references currently have fixed common width. Legacy
CPC structural sleep operates on different state types and objectives. R1.5's
incomplete timing result does not block an independent structural increment.

## Decision

Add pure owned uniform-width splitting through a row isometry, rebinding raw
forward factors to the new width. Residual middle branches retain complementary
identity. Add inverse projection only with version/family/config/index lineage
and scaled-branch compatibility checks. Use axis maps and stable scaled arithmetic
instead of dense isometry matrices. Keep all existing learner/default behavior.

Why this: isometry preserves the linear feedforward function and full equilibrium
suffix norms, while ordinary copy/halve duplication changes the PC denominator.
Explicit raw conversion preserves the configured effective map.

## Alternatives

Copy/halve splitting preserves only the feedforward product, losing objective
invariance. Changing just one hidden width violates current common-width configs.
Mapping complete residual middles without complementary identity changes their
fixed skip semantics. An unrestricted merge would silently drop model components.

## Consequences

The current linear function/equilibrium is preserved mathematically by splitting.
Roundoff and compatibility tolerances are explicit. A tolerated merge residual
can be amplified downstream; no unconditional global function/damage guarantee
is made. Width-dependent SGD factors change future dynamics. No optimizer/trace
state exists in these utilities. Pruning, damage gates/recovery, nonlinear/CPC
state integration and parent R1.6 remain open. No new scientific trial is required
for this engineering equation check; the failed R1.5 comparison remains intact.
