"""Small-row, wide-feature disposable categorical fit; no roles, IO or learner.

Shares the original fixed minimum-norm solve and softmax arithmetic. A distinct
record/API keeps the existing 296-feature contract unchanged. No Gram solve,
feature selection, scaling, regularizer, RNG or tuning is introduced.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from src.core.categorical_linear_fit import (
    RCOND as RCOND,
    _solve_design,
    _predict_design,
    _solve_minimum_change,
)
from src.core.categorical_probability_scores import validate_class_count, validate_class_targets

MAX_ROWS = 16
MAX_FEATURES = 4096
DEFINITION = "wide_categorical_one_hot_least_squares_v1"


@dataclass(frozen=True, slots=True)
class WideCategoricalLinearFit:
    feature_dim: int
    num_classes: int
    fit_rows: int
    coefficients: tuple[tuple[float, ...], ...]
    rank: int
    singular_values: tuple[float, ...]
    squared_error: float

    def __post_init__(self):
        if type(self) is not WideCategoricalLinearFit:
            raise ValueError("an exact WideCategoricalLinearFit is required")
        validate_class_count(self.num_classes)
        if (
            type(self.feature_dim) is not int
            or not 1 <= self.feature_dim <= MAX_FEATURES
            or type(self.fit_rows) is not int
            or not 1 <= self.fit_rows <= MAX_ROWS
        ):
            raise ValueError("wide fit requires1..4096 features and1..16 fit rows")
        size = min(self.fit_rows, self.feature_dim + 1)
        if (
            type(self.rank) is not int
            or not 0 <= self.rank <= size
            or type(self.singular_values) is not tuple
            or len(self.singular_values) != size
            or any(type(v) is not float or not isfinite(v) or v < 0 for v in self.singular_values)
            or type(self.squared_error) is not float
            or not isfinite(self.squared_error)
            or self.squared_error < 0
        ):
            raise ValueError("wide fit requires complete finite rank/residual evidence")
        if (
            type(self.coefficients) is not tuple
            or len(self.coefficients) != self.feature_dim + 1
            or any(
                type(row) is not tuple
                or len(row) != self.num_classes
                or any(type(v) is not float or not isfinite(v) for v in row)
                for row in self.coefficients
            )
        ):
            raise ValueError("wide fit requires immutable finite complete coefficients")


def _features(features, width=None):
    if (
        type(features) is not np.ndarray
        or features.dtype != np.dtype(np.float64)
        or features.ndim != 2
        or not 1 <= features.shape[0] <= MAX_ROWS
        or not 1 <= features.shape[1] <= MAX_FEATURES
        or (width is not None and features.shape[1] != width)
        or not np.isfinite(features).all()
        or np.any(np.abs(features) > 1)
    ):
        raise ValueError("wide features require finite float64(B1..16,D1..4096) in [-1,1]")
    design = np.ones((len(features), features.shape[1] + 1), dtype=np.float64)
    design[:, :-1] = features
    return design


def fit_wide_categorical_linear(
    features: NDArray[np.float64], targets: NDArray[np.float64], num_classes: int
) -> WideCategoricalLinearFit:
    """One fixed thin least-squares solve, equal rows, appended intercept."""
    count = validate_class_count(num_classes)
    design = _features(features)
    validate_class_targets(targets, len(design), count)
    one_hot = np.eye(count, dtype=np.float64)[targets[:, 0].astype(np.int64)]
    coefficients, rank, singular, error = _solve_design(design, one_hot)
    return WideCategoricalLinearFit(
        features.shape[1],
        count,
        len(features),
        tuple(tuple(float(v) for v in row) for row in coefficients),
        int(rank),
        tuple(float(v) for v in singular),
        error,
    )


def predict_wide_categorical_linear(
    model: WideCategoricalLinearFit, features: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Detached probabilities; fit or confidence is not a task-success claim."""
    if type(model) is not WideCategoricalLinearFit:
        raise ValueError("an exact WideCategoricalLinearFit is required")
    model.__post_init__()
    return _predict_design(_features(features, model.feature_dim), model.coefficients)


def predict_wide_categorical_active(
    model: WideCategoricalLinearFit,
    features: NDArray[np.float64],
    columns: tuple[int, ...],
) -> NDArray[np.float64]:
    """Read-only ordered columns, selected before softmax; no new fit metadata.

    Why: normalizing all classes first can underflow every allowed class when
    an inactive score is extreme. The original model and its fit evidence stay
    intact; only this readout's coefficient array is projected.
    """
    if type(model) is not WideCategoricalLinearFit:
        raise ValueError("an exact WideCategoricalLinearFit is required")
    model.__post_init__()
    if (
        type(columns) is not tuple
        or not 2 <= len(columns) <= model.num_classes
        or any(type(i) is not int or not 0 <= i < model.num_classes for i in columns)
        or any(a >= b for a, b in zip(columns, columns[1:]))
    ):
        raise ValueError("active columns require an ordered distinct original class subset")
    coefficients = tuple(tuple(row[i] for i in columns) for row in model.coefficients)
    return _predict_design(_features(features, model.feature_dim), coefficients)


def update_wide_categorical_linear(
    prior: WideCategoricalLinearFit,
    features: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> WideCategoricalLinearFit:
    """Minimum-change residual solve; metadata describes this batch only.

    With augmented design Z, return W + pinv(Z) @ (Y - ZW), using the existing
    fixed rank cutoff. Why: retain prior components invisible to the new rows;
    a cold refit would discard them. Old predictions are not otherwise protected.
    No history, roles, replay, lifetime work or optimizer state is retained here.
    """
    if type(prior) is not WideCategoricalLinearFit:
        raise ValueError("an exact WideCategoricalLinearFit prior is required")
    prior.__post_init__()
    design = _features(features, prior.feature_dim)
    validate_class_targets(targets, len(design), prior.num_classes)
    one_hot = np.eye(prior.num_classes, dtype=np.float64)[targets[:, 0].astype(np.int64)]
    coefficients, rank, singular, squared_error = _solve_minimum_change(
        design, one_hot, prior.coefficients
    )
    return WideCategoricalLinearFit(
        prior.feature_dim,
        prior.num_classes,
        len(features),
        tuple(tuple(float(v) for v in row) for row in coefficients),
        int(rank),
        tuple(float(v) for v in singular),
        squared_error,
    )
