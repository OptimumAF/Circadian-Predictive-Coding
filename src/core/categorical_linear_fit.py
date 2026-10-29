"""Bounded disposable one-hot least-squares categorical fit.

Inputs are exact float64 features/hard labels; outputs are immutable coefficients
and owned softmax probabilities. The fixed intercept and rank cutoff are not
tunable here. This is a TRAIN-fit control, not CE optimization or a deployed
learner, role gate, selector, clock, RNG, storage or orchestration layer.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from src.core.categorical_probability_scores import validate_class_count, validate_class_targets

RCOND = 1e-12
DEFINITION = "categorical_one_hot_least_squares_v1"


@dataclass(frozen=True, slots=True)
class CategoricalLinearFit:
    feature_dim: int
    num_classes: int
    fit_rows: int
    coefficients: tuple[tuple[float, ...], ...]
    rank: int
    singular_values: tuple[float, ...]
    squared_error: float

    def __post_init__(self) -> None:
        if type(self) is not CategoricalLinearFit:
            raise ValueError("an exact CategoricalLinearFit is required")
        validate_class_count(self.num_classes)
        if (
            type(self.feature_dim) is not int
            or not 1 <= self.feature_dim <= 296
            or type(self.fit_rows) is not int
            or not 1 <= self.fit_rows <= 128
        ):
            raise ValueError("linear fit requires1..296 features and1..128 fit rows")
        size = min(self.fit_rows, self.feature_dim + 1)
        if (
            type(self.rank) is not int
            or not 0 <= self.rank <= size
            or type(self.singular_values) is not tuple
            or len(self.singular_values) != size
            or any(
                type(value) is not float or not isfinite(value) or value < 0
                for value in self.singular_values
            )
            or type(self.squared_error) is not float
            or not isfinite(self.squared_error)
            or self.squared_error < 0
        ):
            raise ValueError("linear fit requires finite complete rank/residual evidence")
        if type(self.coefficients) is not tuple or len(self.coefficients) != self.feature_dim + 1:
            raise ValueError("linear fit requires complete immutable finite float coefficients")
        # Full validation stays per call; explicit loops avoid a generator for
        # every short row on the frequently copied296/K7 prediction boundary.
        for row in self.coefficients:
            if type(row) is not tuple or len(row) != self.num_classes:
                raise ValueError("linear fit requires complete immutable finite float coefficients")
            for value in row:
                if type(value) is not float or not isfinite(value):
                    raise ValueError(
                        "linear fit requires complete immutable finite float coefficients"
                    )


def _features(features: NDArray[np.float64], width: int | None = None) -> NDArray[np.float64]:
    if (
        type(features) is not np.ndarray
        or features.dtype != np.dtype(np.float64)
        or features.ndim != 2
        or not 1 <= features.shape[0] <= 128
        or not 1 <= features.shape[1] <= 296
        or (width is not None and features.shape[1] != width)
        or not np.all(np.isfinite(features))
        or np.any(np.abs(features) > 1)
    ):
        raise ValueError("linear features must be exact finite float64(B1..128,D1..296) in [-1,1]")
    # Why: a solver sees detached data; caller arrays and readonly views stay owned.
    design = np.ones((len(features), features.shape[1] + 1), dtype=np.float64)
    design[:, :-1] = features
    return design


def fit_categorical_linear(
    features: NDArray[np.float64], targets: NDArray[np.float64], num_classes: int
) -> CategoricalLinearFit:
    """One fixed minimum-norm solve, equal row weights and an appended intercept."""
    count = validate_class_count(num_classes)
    design = _features(features)
    validate_class_targets(targets, len(design), count)
    one_hot = np.eye(count, dtype=np.float64)[targets[:, 0].astype(np.int64)]
    coefficients, rank, singular_values, squared_error = _solve_design(design, one_hot)
    return CategoricalLinearFit(
        feature_dim=features.shape[1],
        num_classes=count,
        fit_rows=len(design),
        coefficients=tuple(tuple(float(value) for value in row) for row in coefficients),
        rank=int(rank),
        singular_values=tuple(float(value) for value in singular_values),
        squared_error=squared_error,
    )


def _solve_design(design, one_hot):
    """Shared arithmetic only; each public fit owns its distinct geometry limits."""
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, one_hot, rcond=RCOND)
    if (
        coefficients.shape != (design.shape[1], one_hot.shape[1])
        or not np.isfinite(coefficients).all()
    ):
        raise ValueError("linear solver returned invalid coefficients")
    # LAPACK returns an empty residual array for underdetermined/rank-deficient
    # systems. Compute the same residual definition for every admitted design.
    error = design @ coefficients - one_hot
    return coefficients, rank, singular_values, float(np.sum(error * error))


def predict_categorical_linear(
    model: CategoricalLinearFit, features: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Stable softmax scores; successful TRAIN fit is no generalization claim."""
    if type(model) is not CategoricalLinearFit:
        raise ValueError("an exact CategoricalLinearFit is required")
    model.__post_init__()
    design = _features(features, model.feature_dim)
    return _predict_design(design, model.coefficients)


def _predict_design(design, coefficients):
    """Same stable softmax for the separately bounded public model types."""
    with np.errstate(over="raise", invalid="raise"):
        logits = design @ np.array(coefficients, dtype=np.float64)
        if not np.isfinite(logits).all():
            raise ValueError("linear prediction produced nonfinite scores")
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return probabilities
