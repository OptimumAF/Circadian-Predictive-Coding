# ADR-0021: Preserve training-energy numbers and identify their diagnostics

## Context

The NumPy PC and circadian models return `energy`; the Torch heads return a
float called `energy` and the vision report keeps its last value as
`final_energy`. These values enter learning curves and some sleep trigger
decisions. They do not all differentiate to the implemented latent and
weight updates, and their hidden penalties have width-dependent means.
Historical output and sleep behavior depend on their existing values.

## Decision

Keep every existing numeric value and legacy field name. Add stable,
machine-readable training metric identifiers to new result objects, text
reports, and JSON/CSV exports. Mark PC values as **training diagnostics** and
keep held-out cross-entropy and accuracy separate. The precise formulas and
normalization are in `docs/learning-mathematics.md`.

The NumPy multilayer PC diagnostic is binary BCE plus half the mean squared
residual over all `B * sum(hidden_widths)` latent entries. The NumPy
circadian diagnostic is binary BCE plus half the mean squared residual over
the `B * adaptive_width` final-latent entries. The Torch PC/circadian
diagnostic is half the mean squared softmax-output residual over `B * C`
entries plus half the mean squared hidden residual over `B * H` entries.
All are computed from the last relaxed state and predictions before the
weight assignment, even when returned afterward. `final_energy` in the
vision export means the last **training batch** diagnostic, not held-out
energy. A fixed nonzero residual contributes less to the reported hidden
term as width grows, while the executed latent residual update has no such
division by width. Torch's squared-output diagnostic is not the CE
derivative used by the head update. Multilayer NumPy top-down latent terms
also do not follow the gradient of its logged fixed-prior diagnostic.

The one-hidden, ungated update can be interpreted as alternating steps on
per-example cross-entropy plus a **sum** of squared hidden residuals, with
the latent step using an unaveraged per-example gradient and the parameter
step averaging over the batch. This interpretation does not extend to the
current all-latent multilayer NumPy update or to circadian gates, reward
scaling, and sleep as an ordinary gradient of one static objective.

## Alternatives considered

- Replacing the logged diagnostic with a summed-residual objective would
  change historical curves and the adaptive sleep trigger. That requires a
  separately versioned experiment and trigger audit.
- Calling the Torch squared residual "cross-entropy energy" would give it
  the wrong derivative and invite comparison with held-out CE.
- Renaming `energy`/`final_energy` in place would break existing consumers
  without supplying provenance for old files.

## Consequences and open work

The new identifiers distinguish formulas but do not make energy magnitudes
comparable across models, widths, class counts, or binary/multiclass tasks.
Validation and final-test comparisons continue to use role-appropriate
accuracy and cross-entropy. Sleep plateau and budget rules still read the
historical width-dependent diagnostics; P3.2 must audit this effect when
specifying trigger semantics. P2.3 retains numeric derivative checks, and
P2.6 retains the deeper matched-formulation decision.
