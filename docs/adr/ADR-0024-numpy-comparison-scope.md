# ADR-0024: Label NumPy algorithm versions and restrict comparison scope

## Context

The NumPy toy, in-depth, continual, and hardest-mode dynamics routes accept
multiple hidden layers; hardest-mode dynamics defaults to three. For more
than one hidden layer, ordinary PC relaxes every latent and updates earlier
weights from local residuals with fixed feedforward priors. Circadian PC
relaxes only its final adaptive latent and propagates its prior gradient
through earlier feedforward layers. A same-initialization two-hidden test
observes different earlier-weight updates. The one-hidden neutral-control
parity in ADR-0023 does not establish parity for these deeper algorithms.

The existing `toy_validation_v1`, `continual_validation_v1`, and
`validation_dynamics_v1` IDs identify data roles and label timing. They do
not identify learning rules or make the three models matched causal
controls. Even one-hidden toy and continual comparisons use different
model seeds and controls, so their scores are descriptive.

## Decision

Preserve the existing numerical algorithms and evaluation protocol IDs.
Attach these stable learning-rule IDs to new NumPy reports:

| Model | Algorithm ID |
|---|---|
| Backprop | `numpy_backprop_tanh_mlp_v1` |
| Ordinary PC | `numpy_pc_all_latent_fixed_prior_v1` |
| Circadian PC | `numpy_circadian_final_latent_feedforward_prior_v1` |

`src/app/comparison_scope.py` classifies a known one-hidden run as
`numpy_shallow_descriptive_v1` and a deeper run as
`numpy_deeper_unmatched_descriptive_v1`. Both set
`causal_attribution_supported=False`. Standalone figure builders without an
architecture receive `numpy_architecture_unknown_descriptive_v1`, also
without causal attribution. Reports, interactive payloads, and newly
rendered figures carry the description and algorithm IDs. A GIF displays
the descriptive status and embeds the IDs and scope in its comment field.
The figure's normalized training-series label now says metric because the
three training diagnostics are not one optimized objective; its existing
`objective_series` payload key remains for compatibility.

The verified one-hidden neutral control and separately matched Torch
fixed-feature head track provide foundations for future mechanism
attribution, subject to their own data and fairness gates. A deeper NumPy
score remains a valid descriptive result but cannot isolate a circadian
mechanism.

## Alternatives considered

- Replacing either deeper update rule now would change historical numerical
  behavior before a common objective, gradient gate, and fair comparison
  protocol have been specified.
- Suppressing deeper runs would discard descriptive evidence. Explicit
  provenance and scope preserve their usefulness.
- Treating a one-hidden toy leaderboard as a causal ablation would ignore
  different seeds and controls despite the core-level parity fixture.

## Consequences and open work

The scope labels do not change training, splitting, selection, or scoring.
Tiny deep toy, continual, and dynamics runs retained their pre-change
metrics and split hashes. Existing checked-in historical figures remain
untouched. P2.6a preserves the optional shared deeper formulation with
objective, gradient, neutral-control, and fairness acceptance criteria;
P2.7 retains cross-backend parity. The P1.7/P1.8 environment and larger-data
fairness gates also remain open.
