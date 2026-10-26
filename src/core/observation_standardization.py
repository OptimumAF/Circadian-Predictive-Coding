"""Frozen population statistics and bounded continuous-observation transforms.

Pure float64 array math and immutable owned metadata. No roles, learner updates,
evaluation selection, refitting during application, simulator, RNG or IO. Raw
array callers own release authority; the BC adapter provides TRAIN-only fitting.
"""

from dataclasses import dataclass, field
import math

import numpy as np

from src.core import continuous_bc as bc
from src.core.continuous_demonstration import ContinuousActionSpec


def _coordinates(values: tuple[float, ...], name: str) -> tuple[float, ...]:
    if type(values) is not tuple or not 1 <= len(values) <= 1024:
        raise ValueError(f"{name} must be a native tuple with 1..1024 coordinates")
    owned = []
    for value in values:
        if type(value) not in (int, float):
            raise ValueError(f"{name} requires finite native numeric coordinates")
        try:
            number = float(value)
        except OverflowError as error:
            raise ValueError(f"{name} contains a nonrepresentable coordinate") from error
        if not math.isfinite(number):
            raise ValueError(f"{name} requires finite coordinates")
        owned.append(number)
    return tuple(owned)


@dataclass(frozen=True)
class ObservationStandardization:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    fitted_rows: int

    def __post_init__(self) -> None:
        means, scales = _coordinates(self.means, "means"), _coordinates(self.scales, "scales")
        if len(means) != len(scales) or any(value <= 0 for value in scales):
            raise ValueError("standardization scales must be positive and match the means")
        if type(self.fitted_rows) is not int or not 1 <= self.fitted_rows <= 256:
            raise ValueError("standardization fitted_rows must be a native integer in 1..256")
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "scales", scales)


def validate_scale_floors(value: tuple[float, ...], dimensions: int) -> tuple[float, ...]:
    """Own explicit positive floors; no inference or selection from features."""
    if type(dimensions) is not int or not 1 <= dimensions <= 1024:
        raise ValueError("scale floor dimensions must be a native integer in1..1024")
    floors = _coordinates(value, "scale_floors")
    if len(floors) != dimensions or any(floor <= 0 for floor in floors):
        raise ValueError("scale_floors must be positive and match observation dimensions")
    return floors


@dataclass(frozen=True)
class ScaleFlooredObservationStandardization(ObservationStandardization):
    """Keep empirical scales intact and declare separate frozen divisors."""

    scale_floors: tuple[float, ...]
    effective_scales: tuple[float, ...] = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        floors = validate_scale_floors(self.scale_floors, len(self.means))
        object.__setattr__(self, "scale_floors", floors)
        object.__setattr__(
            self,
            "effective_scales",
            tuple(max(scale, floor) for scale, floor in zip(self.scales, floors, strict=True)),
        )


def _standardization(
    value: ObservationStandardization | None, dimensions: int
) -> ObservationStandardization | None:
    if value is None:
        return None
    if type(value) is ScaleFlooredObservationStandardization:
        selected = ScaleFlooredObservationStandardization(
            value.means, value.scales, value.fitted_rows, value.scale_floors
        )
        if (
            type(value.effective_scales) is not tuple
            or any(type(scale) is not float for scale in value.effective_scales)
            or value.effective_scales != selected.effective_scales
        ):
            raise ValueError("scale-floored standardization effective_scales changed")
        owned: ObservationStandardization = selected
    elif type(value) is ObservationStandardization:
        owned = ObservationStandardization(value.means, value.scales, value.fitted_rows)
    else:
        raise ValueError("expected an exact ObservationStandardization or None")
    if len(owned.means) != dimensions:
        raise ValueError("standardization dimensions differ from observation spec")
    return owned


def standardization_numeric_bytes(transform: ObservationStandardization | None) -> int:
    """Count frozen numeric metadata payload, excluding Python/copy overhead."""
    if transform is None:
        return 0
    if type(transform) not in (ObservationStandardization, ScaleFlooredObservationStandardization):
        raise ValueError("numeric payload requires an exact supported standardization")
    owned = _standardization(transform, len(transform.means))
    assert owned is not None
    vectors = 4 if type(owned) is ScaleFlooredObservationStandardization else 2
    return vectors * len(owned.means) * 8


def fit_observation_standardization(
    features: bc.Array, *, scale_floors: tuple[float, ...] | None = None
) -> ObservationStandardization:
    """Fit population statistics once; this pure function grants no role authority."""
    if (
        type(features) is not np.ndarray
        or features.dtype != np.float64
        or features.ndim != 2
        or not 1 <= features.shape[0] <= 256
        or not 1 <= features.shape[1] <= 1024
    ):
        raise ValueError("fit requires finite float64 features with shape (1..256, 1..1024)")
    floors = (
        validate_scale_floors(scale_floors, features.shape[1]) if scale_floors is not None else None
    )
    if not np.isfinite(features).all():
        raise ValueError("fit requires finite float64 features with shape (1..256, 1..1024)")
    values = features.copy()
    constant = np.all(values == values[0], axis=0)
    # Why: summing large physical values or squaring physical deviations can
    # overflow even when their population mean and standard deviation exist.
    magnitude = np.max(np.abs(values), axis=0)
    magnitude[magnitude == 0] = 1.0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        scaled = values / magnitude
        center = np.mean(scaled, axis=0)
        means = center * magnitude
        scales = np.sqrt(np.mean((scaled - center) ** 2, axis=0)) * magnitude
    means[constant] = values[0, constant]
    scales[constant] = 1.0
    # Nonconstant subnormal spreads may have no representable positive scale.
    # Refuse them instead of silently treating a varying dimension as constant.
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("nonconstant population scales must be finite and representable")
    means_tuple, scales_tuple = (
        tuple(float(value) for value in means),
        tuple(float(value) for value in scales),
    )
    if floors is not None:
        return ScaleFlooredObservationStandardization(
            means_tuple, scales_tuple, len(values), floors
        )
    return ObservationStandardization(means_tuple, scales_tuple, len(values))


def standardize_observations(
    features: bc.Array,
    transform: ObservationStandardization | None,
    spec: ContinuousActionSpec,
) -> bc.Array:
    """Apply frozen (raw-mean)/scale; constants retain new-input deviations.

    None returns an owned raw copy. Unrepresentable subtraction/division is
    refused without clipping or changing fitted statistics.
    """
    owned_spec = bc._spec(spec)
    if transform is not None and type(transform) is not ObservationStandardization:
        # Selected metadata must be validated before any array math; foreign
        # subclasses cannot opt into different normalization implicitly.
        owned = _standardization(transform, owned_spec.observation_dim)
        values = bc._features(features, owned_spec)
    else:
        values = bc._features(features, owned_spec)
        owned = _standardization(transform, owned_spec.observation_dim)
    if owned is None:
        return values
    divisors = (
        owned.effective_scales
        if type(owned) is ScaleFlooredObservationStandardization
        else owned.scales
    )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = (values - np.asarray(owned.means)) / np.asarray(divisors)
    if not np.isfinite(result).all():
        raise FloatingPointError("standardized observations are not finite")
    return result
