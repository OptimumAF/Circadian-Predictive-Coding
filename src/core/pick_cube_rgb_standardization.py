"""Fixed sensor-range RGB scaling with fitted public vector-prefix statistics.

Pure owned array/metadata transforms for the exact frozen PickCube RGB recipe.
Raw callers own release authority; the BC adapter admits TRAIN episodes. No
clipping, labels, model updates, geometry, simulator, RNG, selection or IO.
"""

import numpy as np

from src.core.observation_standardization import (
    ObservationStandardization,
    fit_observation_standardization,
)
from src.core.pick_cube_rgb_features import PickCubeRGBRecipe, validate_pick_cube_rgb_recipe

DEFINITION = "pick_cube_train_prefix21_fixed_rgb_range_v1"
PREFIX_DIMENSIONS = 21
RGB_DIMENSIONS = 768


def validate_rgb_range_features(features: np.ndarray, recipe: PickCubeRGBRecipe) -> np.ndarray:
    """Own finite789D inputs, refusing RGB outside declared[0,1] without clipping."""
    owned_recipe = validate_pick_cube_rgb_recipe(recipe)
    if (
        type(features) is not np.ndarray
        or features.dtype != np.dtype("float64")
        or features.ndim != 2
        or not 1 <= features.shape[0] <= 256
        or features.shape[1] != owned_recipe.feature_dim
        or not np.isfinite(features).all()
    ):
        raise ValueError(
            "RGB-range normalization requires exact finite float64 features (1..256,789)"
        )
    values = features.copy()
    if np.any(values[:, PREFIX_DIMENSIONS:] < 0.0) or np.any(values[:, PREFIX_DIMENSIONS:] > 1.0):
        raise ValueError("RGB-range normalization requires public RGB features in[0,1]")
    return values


def fit_rgb_range_standardization(
    features: np.ndarray, recipe: PickCubeRGBRecipe
) -> ObservationStandardization:
    """Fit only21 public vector coordinates; RGB mean/scale follow sensor range."""
    values = validate_rgb_range_features(features, recipe)
    prefix = fit_observation_standardization(values[:, :PREFIX_DIMENSIONS])
    # Why: near-constant empirical pixels can amplify a context change by orders
    # of magnitude. Known[0,1] sensor bounds define this untuned RGB transform;
    # the fitted vector prefix is intentionally unchanged and remains unbounded.
    return ObservationStandardization(
        prefix.means + (0.5,) * RGB_DIMENSIONS,
        prefix.scales + (0.5,) * RGB_DIMENSIONS,
        prefix.fitted_rows,
    )
