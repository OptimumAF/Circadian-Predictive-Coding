"""Immutable TRAIN affine inference through the existing frozen rollout port.

Accepts an already fitted physical map. No fitting, updates, optimizer, RNG,
data reads, simulator access or deployment/promotion authority is supplied.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
import math
import sys
from typing import Any

from src.core.continuous_affine_fit import (
    DEFINITION,
    ContinuousAffineFit,
    predict_continuous_affine,
)
from src.core.continuous_bc import Array
from src.core.continuous_demonstration import ContinuousActionSpec


@dataclass(frozen=True, slots=True)
class ContinuousAffinePolicy:
    spec: ContinuousActionSpec
    definition: str = DEFINITION
    training_role: str = "train"
    observation_standardization: None = None
    binary_gripper_axis: None = None


@dataclass(frozen=True, slots=True)
class ContinuousAffineSnapshot:
    policy: ContinuousAffinePolicy
    model: ContinuousAffineFit

    @property
    def completed_updates(self) -> int:
        # Why: a cold LS fit is not a neural optimizer update.
        return 0

    @property
    def training_examples(self) -> int:
        return self.model.fit_rows


@dataclass(frozen=True, slots=True, init=False)
class ReadonlyContinuousAffine:
    _state: ContinuousAffineSnapshot
    _baseline: ContinuousAffineSnapshot
    _pin: tuple[object, ...]

    def __init__(self, model: ContinuousAffineFit):
        if type(model) is not ContinuousAffineFit:
            raise ValueError("readonly affine requires an exact physical fitted map")
        owned = replace(model)
        state = ContinuousAffineSnapshot(ContinuousAffinePolicy(owned.spec), owned)
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_baseline", deepcopy(state))
        object.__setattr__(self, "_pin", self._identities())

    def _identities(self) -> tuple[object, ...]:
        state = self._state
        return (
            state,
            state.policy,
            state.policy.spec,
            state.model,
            state.model.spec,
            state.model.coefficients,
            state.model.singular_values,
        )

    def snapshot_state(self) -> ContinuousAffineSnapshot:
        if type(self) is not ReadonlyContinuousAffine:
            raise ValueError("readonly affine owner must be exact")
        if any(a is not b for a, b in zip(self._identities(), self._pin, strict=True)):
            raise ValueError("readonly affine state was replaced")
        state = self._state
        validated = replace(state.model)
        expected_policy = ContinuousAffinePolicy(validated.spec)
        if state != self._baseline or state.policy != expected_policy:
            raise ValueError("readonly affine state or TRAIN policy changed")
        return deepcopy(ContinuousAffineSnapshot(expected_policy, validated))

    @property
    def policy(self) -> ContinuousAffinePolicy:
        return self.snapshot_state().policy

    def fork(self) -> "ReadonlyContinuousAffine":
        return ReadonlyContinuousAffine(self.snapshot_state().model)

    def predict(self, features: Array) -> Array:
        before = self.snapshot_state()
        result = predict_continuous_affine(self._state.model, features)
        if self.snapshot_state() != before:
            raise ValueError("readonly affine source changed during prediction")
        return result.bounded


def readonly_affine_from_fit_record(
    record: dict[str, Any], *, role: str
) -> ReadonlyContinuousAffine:
    """Convert an already admitted TRAIN report's model section, without IO.

    This validates the fixed report shape, not provenance/release authority.
    The caller must separately admit its envelope/source/bytes and budget.
    Diagnostic coefficient_norm is checked, never passed as model state.
    """
    keys = {
        "spec",
        "coefficients",
        "fit_rows",
        "rank",
        "singular_values",
        "condition_number",
        "effective_condition_number",
        "coefficient_norm",
    }
    if type(role) is not str or role != "train":
        raise ValueError("readonly affine record requires admitted TRAIN role")
    if type(record) is not dict or len(record) != len(keys) or set(record) != keys:
        raise ValueError("readonly affine requires the complete fixed fit report model section")
    spec = record["spec"]
    spec_keys = {"observation_dim", "action_units", "lower_bounds", "upper_bounds"}
    if type(spec) is not dict or len(spec) != 4 or set(spec) != spec_keys:
        raise ValueError("readonly affine record requires the complete physical spec")
    if any(
        type(spec[key]) is not list or len(spec[key]) != 4
        for key in ("action_units", "lower_bounds", "upper_bounds")
    ):
        raise ValueError("readonly affine record spec vectors require four JSON entries")
    coefficients, singular = record["coefficients"], record["singular_values"]
    if (
        type(coefficients) is not list
        or len(coefficients) != 13
        or any(type(row) is not list or len(row) != 4 for row in coefficients)
        or type(singular) is not list
        or not 1 <= len(singular) <= 13
    ):
        raise ValueError("readonly affine record requires bounded13x4 coefficients/singular values")
    model = ContinuousAffineFit(
        ContinuousActionSpec(
            spec["observation_dim"],
            tuple(spec["action_units"]),
            tuple(spec["lower_bounds"]),
            tuple(spec["upper_bounds"]),
        ),
        tuple(tuple(row) for row in coefficients),
        record["fit_rows"],
        record["rank"],
        tuple(singular),
        record["condition_number"],
        record["effective_condition_number"],
    )
    reported = record["coefficient_norm"]
    actual = math.hypot(*(value for row in model.coefficients for value in row))
    # Why: JSON round-trips preserve coefficients exactly, while the producer's
    # scaled NumPy sum and hypot may differ by a few rounding units. This bound
    # applies only to redundant diagnostic metadata, never fit/score selection.
    if (
        type(reported) is not float
        or not math.isfinite(reported)
        or reported < 0
        or not math.isfinite(actual)
        or not math.isclose(reported, actual, rel_tol=8 * sys.float_info.epsilon, abs_tol=0.0)
    ):
        raise ValueError("readonly affine coefficient_norm diagnostic disagrees with model")
    return ReadonlyContinuousAffine(model)
