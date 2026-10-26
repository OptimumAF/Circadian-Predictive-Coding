"""Independent weighted-loss oracles and all15 parameter derivatives.

Literal D2/H2/A3 arrays use one constant physical action axis. No learner, RNG,
optimizer, prediction API or SDK occurs. Full module forecasts102 gradient
attempts/209 rows,88 returns/180 rows and88 forwards/180 rows; malformed weights
fail before forward. Central differences account for62 of the successful calls.
"""

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.core import continuous_bc as core
from src.core.continuous_demonstration import ContinuousActionSpec


MAXIMUM = float.fromhex("0x1.fffffffffffffp+1023")
MINIMUM = float.fromhex("0x0.0000000000001p-1022")


def literal():
    features = np.array([[0.2, -0.4], [0.7, 0.3]], np.float64)
    targets = np.array([[0.5, 1.5, -1.0], [4.0, 1.5, 1.0]], np.float64)
    parameters = (
        np.array([[0.2, -0.3], [0.4, 0.1]], np.float64),
        np.array([0.05, -0.02], np.float64),
        np.array([[0.3, 0.4, -0.2], [-0.1, 0.2, 0.25]], np.float64),
        np.array([0.1, -0.3, 0.2], np.float64),
    )
    spec = ContinuousActionSpec(
        2, ("m", "fixed", "normalized_gripper"), (-2.0, 1.5, -1.0), (6.0, 1.5, 1.0)
    )
    return features, targets, parameters, spec, np.array([1.0, 3.0], np.float64)


def copied(parameters):
    return parameters[0].copy(), parameters[1].copy(), parameters[2].copy(), parameters[3].copy()


def scalar_oracle(features, targets, parameters, spec, weights, mixed):
    """Explicit scalar MLP and physical row loss; no core reduction or forward."""
    first, hidden_bias, second, output_bias = parameters
    maximum = max(weights)
    scaled = [float(value / maximum) for value in weights]
    total = math.fsum(scaled)
    masses = [value / total for value in scaled]
    predictions, row_losses = [], []
    for features_row, target in zip(features, targets, strict=True):
        hidden = [
            math.tanh(
                math.fsum(float(features_row[i] * first[i, j]) for i in range(2))
                + float(hidden_bias[j])
            )
            for j in range(2)
        ]
        logits = [
            math.fsum(hidden[i] * float(second[i, j]) for i in range(2)) + float(output_bias[j])
            for j in range(3)
        ]
        values, losses = [], []
        for axis, (lower, upper) in enumerate(
            zip(spec.lower_bounds, spec.upper_bounds, strict=True)
        ):
            prediction = (
                lower
                if lower == upper
                else (lower + upper) / 2 + (upper - lower) / 2 * math.tanh(logits[axis])
            )
            values.append(prediction)
            if lower != upper:
                if mixed and axis == 2:
                    losses.append(math.log1p(math.exp(-2 * float(target[axis]) * logits[axis])))
                else:
                    losses.append(
                        0.5 * ((prediction - float(target[axis])) / ((upper - lower) / 2)) ** 2
                    )
        predictions.append(values)
        row_losses.append(math.fsum(losses) / len(losses) if losses else 0.0)
    return math.fsum(
        mass * loss for mass, loss in zip(masses, row_losses, strict=True)
    ), predictions


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts = dict(
        gradient_attempts=0,
        gradient_returns=0,
        gradient_rows_attempted=0,
        gradient_rows_returned=0,
        forward_attempts=0,
        forward_returns=0,
        forward_rows_attempted=0,
        forward_rows_returned=0,
    )
    gradient, forward = core.continuous_bc_loss_and_gradients, core._forward

    def counted(original, kind):
        def call(features, *args, **options):
            counts[kind + "_attempts"] += 1
            counts[kind + "_rows_attempted"] += len(features)
            returned = original(features, *args, **options)
            counts[kind + "_returns"] += 1
            counts[kind + "_rows_returned"] += len(features)
            return returned

        return call

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(core, "continuous_bc_loss_and_gradients", counted(gradient, "gradient"))
        patch.setattr(core, "_forward", counted(forward, "forward"))
        yield counts
    if os.environ.get("WEIGHTED_BC_CORE_FIXTURE_LEDGER"):
        with Path(os.environ["WEIGHTED_BC_CORE_FIXTURE_LEDGER"]).open(
            "x", encoding="utf8"
        ) as output:
            json.dump(counts, output, sort_keys=True)
            output.write("\n")
    assert (counts["gradient_attempts"], counts["gradient_returns"]) == (102, 88)
    assert (counts["gradient_rows_attempted"], counts["gradient_rows_returned"]) == (209, 180)
    assert counts["forward_attempts"] == counts["forward_returns"] == 88
    assert counts["forward_rows_attempted"] == counts["forward_rows_returned"] == 180


@pytest.mark.parametrize("mixed", [False, True])
def test_should_match_independent_weighted_loss_and_every_parameter_central_difference(mixed):
    features, targets, parameters, spec, weights = literal()
    axis = 2 if mixed else None
    result = core.continuous_bc_loss_and_gradients(
        features, targets, parameters, spec, row_weights=weights, binary_gripper_axis=axis
    )
    expected_loss, expected_predictions = scalar_oracle(
        features, targets, parameters, spec, weights, mixed
    )
    assert result.loss == pytest.approx(expected_loss, rel=1e-12, abs=1e-15)
    np.testing.assert_allclose(result.predictions, expected_predictions, rtol=1e-12, atol=1e-15)
    assert result.predictions[:, 1].tolist() == [1.5, 1.5]
    assert np.array_equal(result.gradients[2][:, 1], np.zeros(2))
    assert result.gradients[3][1] == 0.0
    epsilon = 1e-6
    checked = 0
    for group, parameter in enumerate(parameters):
        for index in np.ndindex(parameter.shape):
            plus, minus = copied(parameters), copied(parameters)
            plus[group][index] += epsilon
            minus[group][index] -= epsilon
            high = core.continuous_bc_loss_and_gradients(
                features, targets, plus, spec, row_weights=weights, binary_gripper_axis=axis
            ).loss
            low = core.continuous_bc_loss_and_gradients(
                features, targets, minus, spec, row_weights=weights, binary_gripper_axis=axis
            ).loss
            assert result.gradients[group][index] == pytest.approx(
                (high - low) / (2 * epsilon), rel=2e-5, abs=2e-8
            )
            checked += 1
    assert checked == 15


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("weight", [1.0, 1e-300, 1e308])
def test_should_preserve_bitwise_original_branch_for_every_uniform_positive_weight(mixed, weight):
    features, targets, parameters, spec, _ = literal()
    axis = 2 if mixed else None
    original = core.continuous_bc_loss_and_gradients(
        features, targets, parameters, spec, binary_gripper_axis=axis
    )
    weighted = core.continuous_bc_loss_and_gradients(
        features,
        targets,
        parameters,
        spec,
        row_weights=np.full(2, weight, np.float64),
        binary_gripper_axis=axis,
    )
    expected, _ = scalar_oracle(features, targets, parameters, spec, (1.0, 1.0), mixed)
    assert original.loss == pytest.approx(expected, rel=1e-12, abs=1e-15)
    assert weighted.loss == original.loss
    np.testing.assert_array_equal(weighted.predictions, original.predictions)
    for first, second in zip(weighted.gradients, original.gradients, strict=True):
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("mixed", [False, True])
def test_should_equal_integer_replication_and_preserve_positive_weight_scaling(mixed):
    features, targets, parameters, spec, weights = literal()
    axis = 2 if mixed else None
    original = core.continuous_bc_loss_and_gradients(
        features, targets, parameters, spec, row_weights=weights, binary_gripper_axis=axis
    )
    scaled = core.continuous_bc_loss_and_gradients(
        features, targets, parameters, spec, row_weights=weights * 2.0, binary_gripper_axis=axis
    )
    repeated = core.continuous_bc_loss_and_gradients(
        np.repeat(features, (1, 3), axis=0),
        np.repeat(targets, (1, 3), axis=0),
        parameters,
        spec,
        binary_gripper_axis=axis,
    )
    assert scaled.loss == original.loss
    assert repeated.loss == pytest.approx(original.loss, rel=1e-12, abs=1e-15)
    for first, second, third in zip(
        original.gradients, scaled.gradients, repeated.gradients, strict=True
    ):
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first, third, rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(original.predictions, scaled.predictions)
    np.testing.assert_allclose(
        np.repeat(original.predictions, (1, 3), axis=0),
        repeated.predictions,
        rtol=1e-12,
        atol=1e-15,
    )


@pytest.mark.parametrize("mixed", [False, True])
def test_should_normalize_large_finite_weights_without_overflowing_the_raw_sum(mixed):
    features, targets, parameters, spec, _ = literal()
    huge = np.array([MAXIMUM / 2, MAXIMUM], np.float64)
    seal = huge.copy()
    axis = 2 if mixed else None
    large = core.continuous_bc_loss_and_gradients(
        features, targets, parameters, spec, row_weights=huge, binary_gripper_axis=axis
    )
    small = core.continuous_bc_loss_and_gradients(
        features,
        targets,
        parameters,
        spec,
        row_weights=np.array([1.0, 2.0], np.float64),
        binary_gripper_axis=axis,
    )
    assert math.isfinite(large.loss) and large.loss == small.loss
    np.testing.assert_array_equal(huge, seal)
    np.testing.assert_array_equal(large.predictions, small.predictions)
    for first, second in zip(large.gradients, small.gradients, strict=True):
        np.testing.assert_array_equal(first, second)


def test_should_leave_inputs_unchanged_and_return_arrays_disjoint_from_all_inputs():
    features, targets, parameters, spec, weights = literal()
    originals = (features, targets, weights, *parameters)
    seals = tuple(value.copy() for value in originals)
    result = core.continuous_bc_loss_and_gradients(
        features, targets, parameters, spec, row_weights=weights
    )
    for actual, expected in zip(originals, seals, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert not any(
        np.shares_memory(output, value)
        for output in (result.predictions, *result.gradients)
        for value in originals
    )
    owned = core._row_weights(weights, 2)
    assert owned is not None and not np.shares_memory(owned, weights)
    weights.fill(9.0)
    np.testing.assert_array_equal(owned, [1.0, 3.0])


def test_should_use_owned_inputs_when_original_arrays_change_at_the_forward_boundary(monkeypatch):
    features, targets, parameters, spec, weights = literal()
    expected = core.continuous_bc_loss_and_gradients(
        features, targets, parameters, spec, row_weights=weights
    )
    forward = core._forward

    def changed(values, owned_parameters, owned_spec):
        features.fill(99.0)
        targets.fill(0.0)
        weights.fill(1.0)
        for value in parameters:
            value.fill(0.0)
        return forward(values, owned_parameters, owned_spec)

    monkeypatch.setattr(core, "_forward", changed)
    actual = core.continuous_bc_loss_and_gradients(
        features, targets, parameters, spec, row_weights=weights
    )
    assert actual.loss == expected.loss
    np.testing.assert_array_equal(actual.predictions, expected.predictions)
    for first, second in zip(actual.gradients, expected.gradients, strict=True):
        np.testing.assert_array_equal(first, second)


def test_should_exclude_every_constant_axis_without_dividing_by_zero():
    features, _, parameters, _, weights = literal()
    spec = ContinuousActionSpec(2, ("fixed",) * 3, (1.5,) * 3, (1.5,) * 3)
    targets = np.full((2, 3), 1.5, np.float64)
    result = core.continuous_bc_loss_and_gradients(
        features, targets, parameters, spec, row_weights=weights
    )
    assert result.loss == 0.0
    np.testing.assert_array_equal(result.predictions, targets)
    assert all(np.array_equal(value, np.zeros_like(value)) for value in result.gradients)


class ForeignWeights(np.ndarray[Any, np.dtype[np.float64]]):
    pass


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "bool",
        "float32",
        "integer",
        "subclass",
        "scalar",
        "matrix",
        "short",
        "zero",
        "negative",
        "nan",
        "inf",
        "scale_underflow",
        "mass_underflow",
    ],
)
def test_should_refuse_malformed_or_unrepresentable_positive_mass_before_forward(
    monkeypatch, fault
):
    features, targets, parameters, spec, _ = literal()
    weights: Any = np.array([1.0, 3.0], np.float64)
    if fault == "list":
        weights = [1.0, 3.0]
    elif fault == "bool":
        weights = np.array([True, True])
    elif fault == "float32":
        weights = weights.astype(np.float32)
    elif fault == "integer":
        weights = np.array([1, 3], np.int64)
    elif fault == "subclass":
        weights = weights.view(ForeignWeights)
    elif fault == "scalar":
        weights = np.array(1.0, np.float64)
    elif fault == "matrix":
        weights = weights.reshape(2, 1)
    elif fault == "short":
        weights = weights[:1]
    elif fault in ("zero", "negative", "nan", "inf"):
        weights[1] = {"zero": 0.0, "negative": -1.0, "nan": math.nan, "inf": math.inf}[fault]
    elif fault == "scale_underflow":
        weights = np.array([MINIMUM, MAXIMUM], np.float64)
    else:
        features = np.concatenate((features, features[:1]), axis=0)
        targets = np.concatenate((targets, targets[:1]), axis=0)
        weights = np.array([MINIMUM, 1.0, 1.0], np.float64)

    def forbidden(*args, **options):
        pytest.fail("malformed weights reached a paid model forward")

    monkeypatch.setattr(core, "_forward", forbidden)
    with pytest.raises(ValueError):
        core.continuous_bc_loss_and_gradients(
            features, targets, parameters, spec, row_weights=weights
        )
