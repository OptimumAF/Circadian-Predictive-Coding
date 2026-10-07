# ADR-0023: Name a shallow no-circadian parity control

## Context

The NumPy ordinary PC network relaxes every hidden latent when it has
multiple hidden layers. NumPy circadian PC relaxes only its final adaptive
hidden state and backpropagates earlier feedforward-prior updates. A blanket
claim that neutral circadian settings reproduce ordinary multilayer PC
would therefore be false. With one hidden layer, their binary local
objective and manual wake steps match. The Torch PC and circadian heads
likewise share one multiclass hidden-state update and head architecture.

## Decision

Add `CircadianConfig.matched_pc_control()` and
`CircadianHeadConfig.matched_pc_control()` as explicit, selectable presets.
They force unit plasticity, disable reward scaling and adaptive sleep,
set split/prune budgets to zero, and neutralize homeostatic scaling. The
NumPy preset also disables replay steps and memory. Chemistry remains
observable but cannot scale weights while plasticity is clipped to one.
Defaults are unchanged.

The parity claim is scoped to equal-width **one-hidden-layer** NumPy binary
PC/circadian models or equal-width one-hidden-layer Torch multiclass
PC/circadian heads, with the same seed, feature/target batches, learning
rates, and inference counts. The deterministic CPU fixtures compare all
initial parameters, every parameter after each of four wake calls,
feedforward predictions, hidden traffic, and training diagnostics. They
also force a sleep event and verify a no-op, check that NumPy replay memory
stays empty, and observe nonzero chemistry with a unit gate. A separate
two-hidden NumPy fixture starts with equal parameters and records divergent
earlier-layer weights after training, preserving the P2.6 boundary.

## Alternatives considered

- Applying the shallow parity claim to deeper NumPy networks would mix
  all-latent and final-latent-plus-feedforward algorithms.
- Disabling chemical accumulation itself would require another training
  branch and is unnecessary for a neutral **effect** control: the unit
  gate already removes chemistry's influence on the wake update.
- Making this preset the default would change existing studies and
  historical output. It stays opt-in.

## Consequences and open work

The named presets can be used directly in future ablations and provide a
regression gate for matched shallow attribution. The NumPy one-hidden
diagnostics have different identifiers even though the fixture's numeric
formula agrees; neither diagnostic is used as a common optimized loss.
This is a local CPU equivalence test, not a new image-level result or a
claim of cross-backend numeric equivalence. P2.6 retains the deeper
formulation, P2.7 retains a genuinely matched cross-backend fixture, and
Phase 6 retains the actual ablation study with data, budgets, and seeds.
