"""Versioned NumPy learning-rule identities and descriptive comparison scope.

Why this: equal hidden widths do not make the deeper all-latent PC update
equivalent to the circadian final-latent/feedforward update. These labels
travel with results so a visual or aggregate cannot imply causal parity.
"""

from __future__ import annotations

from dataclasses import dataclass

NUMPY_BACKPROP_ALGORITHM_ID = "numpy_backprop_tanh_mlp_v1"
NUMPY_PC_ALGORITHM_ID = "numpy_pc_all_latent_fixed_prior_v1"
NUMPY_CIRCADIAN_ALGORITHM_ID = "numpy_circadian_final_latent_feedforward_prior_v1"


@dataclass(frozen=True)
class NumpyComparisonScope:
    """Learning-rule provenance and limits for a NumPy three-model report."""

    scope_id: str
    hidden_depth: int | None
    causal_attribution_supported: bool
    description: str
    backprop_algorithm_id: str = NUMPY_BACKPROP_ALGORITHM_ID
    predictive_algorithm_id: str = NUMPY_PC_ALGORITHM_ID
    circadian_algorithm_id: str = NUMPY_CIRCADIAN_ALGORITHM_ID


def scope_for_hidden_dims(hidden_dims: tuple[int, ...] | None) -> NumpyComparisonScope:
    """Classify an existing run without changing its model or data protocol."""
    if hidden_dims is None:
        return NumpyComparisonScope(
            scope_id="numpy_architecture_unknown_descriptive_v1",
            hidden_depth=None,
            causal_attribution_supported=False,
            description="Architecture unspecified; descriptive only, no circadian attribution.",
        )
    if not hidden_dims or any(width <= 0 for width in hidden_dims):
        raise ValueError("hidden_dims must contain positive widths")
    if len(hidden_dims) == 1:
        return NumpyComparisonScope(
            scope_id="numpy_shallow_descriptive_v1",
            hidden_depth=1,
            causal_attribution_supported=False,
            description=(
                "Shared shallow architecture, but seeds and controls differ; "
                "descriptive only, no circadian attribution."
            ),
        )
    return NumpyComparisonScope(
        scope_id="numpy_deeper_unmatched_descriptive_v1",
        hidden_depth=len(hidden_dims),
        causal_attribution_supported=False,
        description="Unmatched multilayer update rules; descriptive only, no circadian attribution.",
    )
