"""Fixed physical TS12 affine control; no roles, budgets, replay or deployment.

One weighted minimum-norm solve fits physical targets directly. Prediction
reports raw values and componentwise clipping to the existing normalized action
bounds. The solve minimizes UNCLIPPED squared error, not the clipped objective,
inverse-tanh error or gripper BCE. The old hidden-readout128-row API is unchanged.
"""

from dataclasses import dataclass, replace
import math

import numpy as np

from src.core.continuous_bc import Array, _array, _spec
from src.core.continuous_demonstration import ContinuousActionSpec
from src.core.continuous_readout_fit import RCOND, _solve
from src.core.cube_taskspace_state import cube_taskspace_spec

DEFINITION = "raw_ts12_weighted_physical_affine_clipped_control_v1"
MAX_ROWS = 198


def _fixed_spec(spec):
    owned = _spec(spec)
    if owned != cube_taskspace_spec():
        raise ValueError("physical affine control requires the original TS12/four-axis cube spec")
    return owned


def _features(features):
    if type(features) is not np.ndarray or features.ndim != 2 or not 1 <= len(features) <= MAX_ROWS:
        raise ValueError("physical affine features require exact float64(B1..198,12)")
    return _array(features, (len(features), 12), "physical affine features")


def _weights(weights, rows):
    owned = _array(weights, (rows,), "physical affine positive weights")
    if np.any(owned <= 0):
        raise ValueError("physical affine weights must all be strictly positive")
    # Why: scaling first avoids overflow without changing relative row mass.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        normalized = owned / float(np.max(owned))
        normalized /= float(np.sum(normalized))
    if not np.isfinite(normalized).all() or np.any(normalized <= 0):
        raise ValueError("all physical affine row masses must remain representably positive")
    return owned, normalized


def _unchanged(actual, expected):
    return (
        type(actual) is np.ndarray
        and actual.dtype == expected.dtype
        and actual.shape == expected.shape
        and np.array_equal(actual, expected)
    )


@dataclass(frozen=True, slots=True)
class ContinuousAffineFit:
    spec: ContinuousActionSpec
    coefficients: tuple[tuple[float, ...], ...]
    fit_rows: int
    rank: int
    singular_values: tuple[float, ...]
    condition_number: float | None
    effective_condition_number: float | None

    def __post_init__(self):
        if type(self) is not ContinuousAffineFit:
            raise ValueError("physical affine fit must be exact")
        object.__setattr__(self, "spec", _fixed_spec(self.spec))
        size = min(self.fit_rows, 13) if type(self.fit_rows) is int else 0
        if (
            type(self.fit_rows) is not int
            or not 1 <= self.fit_rows <= MAX_ROWS
            or type(self.rank) is not int
            or not 0 <= self.rank <= size
            or type(self.coefficients) is not tuple
            or len(self.coefficients) != 13
            or any(type(row) is not tuple or len(row) != 4 for row in self.coefficients)
            or any(
                type(v) is not float or not math.isfinite(v)
                for row in self.coefficients
                for v in row
            )
            or type(self.singular_values) is not tuple
            or len(self.singular_values) != size
            or any(
                type(v) is not float or not math.isfinite(v) or v < 0 for v in self.singular_values
            )
        ):
            raise ValueError("physical affine fit requires complete finite13x4 coefficients/rank")
        singular = self.singular_values
        if any(a < b for a, b in zip(singular, singular[1:])) or self.rank != sum(
            value > RCOND * singular[0] for value in singular
        ):
            raise ValueError("physical affine rank disagrees with fixed-cutoff singular values")
        expected_effective = singular[0] / singular[self.rank - 1] if self.rank else None
        expected_condition = expected_effective if self.rank == size else None
        for actual, expected in (
            (self.condition_number, expected_condition),
            (self.effective_condition_number, expected_effective),
        ):
            if actual is not None and (type(actual) is not float or not math.isfinite(actual)):
                raise ValueError("physical affine conditioning must be finite or None")
            if actual != expected:
                raise ValueError("physical affine conditioning differs from singular values")


@dataclass(frozen=True, slots=True)
class ContinuousAffinePrediction:
    raw: Array
    bounded: Array
    clipped_per_axis: tuple[int, ...]


def fit_continuous_affine(
    features: Array, targets: Array, weights: Array, spec: ContinuousActionSpec
) -> ContinuousAffineFit:
    """Cold physical fit; caller owns release authority and attempted work charges."""
    owned_spec = _fixed_spec(spec)
    x = _features(features)
    y = _array(targets, (len(x), 4), "physical affine targets")
    if np.any(np.abs(y) > 1):
        raise ValueError("physical affine targets exceed original normalized action bounds")
    w, normalized = _weights(weights, len(x))
    design = np.column_stack((x, np.ones(len(x), dtype=np.float64)))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        root = np.sqrt(normalized)[:, None]
        matrix, rhs = design * root, y * root
    expected_matrix, expected_rhs = matrix.copy(), rhs.copy()
    try:
        # Reuse only fixed-cutoff validated algebra. This separate physical198
        # contract never calls or widens the frozen-hidden128-row fitting API.
        coefficients, singular, rank, condition, effective = _solve(matrix, rhs)
        coefficients = _array(coefficients, (13, 4), "physical affine coefficients")
        singular = _array(singular, (min(len(x), 13),), "physical affine singular values")
        if not _unchanged(matrix, expected_matrix) or not _unchanged(rhs, expected_rhs):
            raise ValueError("physical affine solver changed its staged design/targets")
        result = ContinuousAffineFit(
            owned_spec,
            tuple(tuple(float(v) for v in row) for row in coefficients),
            len(x),
            rank,
            tuple(float(v) for v in singular),
            condition,
            effective,
        )
    finally:
        if (
            not _unchanged(features, x)
            or not _unchanged(targets, y)
            or not _unchanged(weights, w)
            or _fixed_spec(spec) != owned_spec
        ):
            raise ValueError("physical affine fit source inputs changed")
    return result


def predict_continuous_affine(
    model: ContinuousAffineFit, features: Array
) -> ContinuousAffinePrediction:
    """Return detached raw/clipped actions; no fitted rows or caller arrays retained."""
    if type(model) is not ContinuousAffineFit:
        raise ValueError("physical affine prediction requires its exact fitted model")
    pin = (model.spec, model.coefficients, model.singular_values)
    before = replace(model)
    x = _features(features)
    coefficients = np.array(before.coefficients, dtype=np.float64)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        raw = np.matmul(np.column_stack((x, np.ones(len(x)))), coefficients)
    raw = _array(raw, (len(x), 4), "physical affine raw predictions")
    bounded = np.clip(raw, -1.0, 1.0)
    clipped = tuple(int(np.count_nonzero(raw[:, axis] != bounded[:, axis])) for axis in range(4))
    if (
        not _unchanged(features, x)
        or model != before
        or model.spec is not pin[0]
        or model.coefficients is not pin[1]
        or model.singular_values is not pin[2]
    ):
        raise ValueError("physical affine prediction source/model changed or was replaced")
    return ContinuousAffinePrediction(raw, bounded, clipped)
