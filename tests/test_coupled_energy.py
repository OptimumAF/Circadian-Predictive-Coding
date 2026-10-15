"""Independent value, derivative and validation controls for coupled binary energy.

The NumPy reference writes the declared scalar equations directly. Central
differences cover every latent and parameter coordinate; optional Torch controls
build a separate autodiff graph. Neither oracle calls a production energy helper.
Fixed tiny fixtures establish calculus correctness, not training effectiveness.
"""

from __future__ import annotations

from dataclasses import replace
from importlib.util import find_spec
from typing import Callable, cast

import numpy as np
from numpy.typing import NDArray
import pytest

from src.core.coupled_energy import (
    CoupledParameters,
    evaluate_coupled,
    feedforward_latents,
    predict_coupled,
)

Array = NDArray[np.float64]
Fixture = tuple[CoupledParameters, Array, Array, tuple[Array, ...], tuple[float, ...]]
WIDTHS = ((3, 2), (3, 2, 3), (3, 2, 3, 2, 1))
EPSILON = 1e-6
ATOL = 2e-8
RTOL = 2e-6


def fixture(widths: tuple[int, ...]) -> Fixture:
    random = np.random.default_rng(41)
    parameters = CoupledParameters(
        tuple(random.normal(0.0, 0.3, (a, b)) for a, b in zip(widths, widths[1:])),
        tuple(random.normal(0.0, 0.15, (1, b)) for b in widths[1:]),
        random.normal(0.0, 0.3, (widths[-1], 1)),
        random.normal(0.0, 0.15, (1, 1)),
    )
    features = random.normal(0.0, 0.4, (3, widths[0]))
    targets = np.array([[0.0], [0.35], [1.0]])
    # Free latents are deliberately away from the feedforward minimum.
    latents = tuple(random.normal(0.0, 0.4, (3, width)) for width in widths[1:])
    precisions = (0.4, 1.7, 0.9, 2.2)[: len(latents)]
    return parameters, features, targets, latents, precisions


def reference_value(
    parameters: CoupledParameters,
    features: Array,
    targets: Array,
    latents: tuple[Array, ...],
    precisions: tuple[float, ...],
) -> tuple[float, float, tuple[float, ...]]:
    previous = features
    penalties = []
    for weight, bias, latent, precision in zip(
        parameters.hidden_weights, parameters.hidden_biases, latents, precisions, strict=True
    ):
        residual = latent - np.tanh(previous @ weight + bias)
        penalties.append(float(precision * np.sum(residual * residual) / (2 * len(features))))
        previous = latent
    logits = latents[-1] @ parameters.output_weight + parameters.output_bias
    task = float(np.mean(np.logaddexp(0.0, logits) - targets * logits))
    return task + sum(penalties), task, tuple(penalties)


def parameter_arrays(parameters: CoupledParameters) -> tuple[Array, ...]:
    return (
        *parameters.hidden_weights,
        *parameters.hidden_biases,
        parameters.output_weight,
        parameters.output_bias,
    )


def central_difference(array: Array, scalar: Callable[[], float]) -> Array:
    gradient = np.empty_like(array)
    for index in np.ndindex(array.shape):
        original = array[index]
        try:
            array[index] = original + EPSILON
            plus = scalar()
            array[index] = original - EPSILON
            minus = scalar()
        finally:
            array[index] = original
        gradient[index] = (plus - minus) / (2 * EPSILON)
    return gradient


@pytest.mark.parametrize("widths", WIDTHS, ids=("one-layer", "two-layers", "four-layers"))
def test_every_derivative_matches_independent_central_differences(widths: tuple[int, ...]) -> None:
    parameters, features, targets, latents, precisions = fixture(widths)
    protected = (*parameter_arrays(parameters), features, targets, *latents)
    before = tuple(array.copy() for array in protected)
    result = evaluate_coupled(parameters, features, targets, latents, precisions)
    energy, task, penalties = reference_value(parameters, features, targets, latents, precisions)
    assert result.energy_definition == "numpy_coupled_binary_energy_v1"
    assert result.energy == pytest.approx(energy, abs=1e-14, rel=1e-14)
    assert result.task_loss == pytest.approx(task, abs=1e-14, rel=1e-14)
    np.testing.assert_allclose(result.hidden_penalties, penalties, atol=1e-14, rtol=1e-14)
    assert len(result.latent_gradients) == len(latents)
    scalar = lambda: reference_value(parameters, features, targets, latents, precisions)[0]
    for latent, analytic in zip(latents, result.latent_gradients, strict=True):
        np.testing.assert_allclose(
            analytic, central_difference(latent, scalar), atol=ATOL, rtol=RTOL
        )
    for parameter, analytic in zip(
        parameter_arrays(parameters), parameter_arrays(result.parameter_gradients), strict=True
    ):
        np.testing.assert_allclose(
            analytic, central_difference(parameter, scalar), atol=ATOL, rtol=RTOL
        )
    for actual, original in zip(protected, before, strict=True):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("widths", WIDTHS, ids=("one-layer", "two-layers", "four-layers"))
def test_gradients_match_independent_torch_autodiff(widths: tuple[int, ...]) -> None:
    if find_spec("torch") is None:
        pytest.skip("Torch unavailable; independent central differences still run")
    torch = pytest.importorskip("torch")
    parameters, features, targets, latents, precisions = fixture(widths)
    parameter_tensors = [
        torch.tensor(array, dtype=torch.float64, requires_grad=True)
        for array in parameter_arrays(parameters)
    ]
    latent_tensors = [torch.tensor(z, dtype=torch.float64, requires_grad=True) for z in latents]
    depth = len(latents)
    weights, biases = parameter_tensors[:depth], parameter_tensors[depth : 2 * depth]
    previous = torch.tensor(features, dtype=torch.float64)
    penalties = []
    for weight, bias, latent, precision in zip(weights, biases, latent_tensors, precisions):
        residual = latent - torch.tanh(previous @ weight + bias)
        penalties.append(precision * residual.square().sum() / (2 * len(features)))
        previous = latent
    logits = latent_tensors[-1] @ parameter_tensors[-2] + parameter_tensors[-1]
    labels = torch.tensor(targets, dtype=torch.float64)
    task = (torch.nn.functional.softplus(logits) - labels * logits).mean()
    energy = task + torch.stack(penalties).sum()
    energy.backward()
    actual = evaluate_coupled(parameters, features, targets, latents, precisions)
    assert actual.energy == pytest.approx(energy.item(), abs=1e-14, rel=1e-14)
    for analytic, tensor in zip(
        (*actual.latent_gradients, *parameter_arrays(actual.parameter_gradients)),
        (*latent_tensors, *parameter_tensors),
        strict=True,
    ):
        assert tensor.grad is not None
        np.testing.assert_allclose(analytic, tensor.grad.detach().numpy(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("widths", WIDTHS, ids=("one-layer", "two-layers", "four-layers"))
def test_repeated_batch_keeps_energy_and_parameter_gradients_but_halves_latent_partials(
    widths: tuple[int, ...],
) -> None:
    parameters, features, targets, latents, precisions = fixture(widths)
    original = evaluate_coupled(parameters, features, targets, latents, precisions)
    repeated = evaluate_coupled(
        parameters,
        np.concatenate((features, features)),
        np.concatenate((targets, targets)),
        tuple(np.concatenate((z, z)) for z in latents),
        precisions,
    )
    assert repeated.energy == pytest.approx(original.energy, abs=1e-14, rel=1e-14)
    assert repeated.task_loss == pytest.approx(original.task_loss, abs=1e-14, rel=1e-14)
    np.testing.assert_allclose(repeated.hidden_penalties, original.hidden_penalties, atol=1e-14)
    for baseline, duplicated in zip(original.latent_gradients, repeated.latent_gradients):
        np.testing.assert_allclose(duplicated[:3], baseline / 2, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(duplicated[3:], baseline / 2, atol=1e-14, rtol=1e-14)
    for baseline, duplicated in zip(
        parameter_arrays(original.parameter_gradients),
        parameter_arrays(repeated.parameter_gradients),
    ):
        np.testing.assert_allclose(duplicated, baseline, atol=1e-14, rtol=1e-14)


def test_downstream_residual_pulls_lower_latent_with_negative_sign_and_declared_precision() -> None:
    parameters = CoupledParameters(
        (np.array([[0.3]]), np.array([[2.0]])),
        (np.zeros((1, 1)), np.zeros((1, 1))),
        np.zeros((1, 1)),
        np.zeros((1, 1)),
    )
    features, targets = np.zeros((1, 1)), np.array([[0.5]])
    latents = (np.array([[0.2]]), np.array([[0.9]]))
    result = evaluate_coupled(parameters, features, targets, latents, (0.5, 1.7))
    prediction = np.tanh(0.2 * 2.0)
    downstream = 1.7 * (0.9 - prediction) * (1 - prediction**2) * 2.0
    assert result.latent_gradients[0][0, 0] == pytest.approx(0.1 - downstream, abs=1e-14)
    assert result.latent_gradients[0][0, 0] < 0  # own residual alone would pull positive
    doubled = evaluate_coupled(parameters, features, targets, latents, (0.5, 3.4))
    assert doubled.latent_gradients[0][0, 0] == pytest.approx(0.1 - 2 * downstream, abs=1e-14)
    assert doubled.hidden_penalties[1] == pytest.approx(2 * result.hidden_penalties[1])
    changed = evaluate_coupled(
        parameters, features, targets, (np.array([[0.4]]), latents[1]), (0.5, 1.7)
    )
    assert changed.hidden_penalties[1] == pytest.approx(1.7 * (0.9 - np.tanh(0.8)) ** 2 / 2)
    assert changed.hidden_penalties[1] != result.hidden_penalties[1]


def test_feedforward_and_prediction_follow_independent_composition_without_mutating_inputs() -> (
    None
):
    parameters, features, _, _, _ = fixture(WIDTHS[-1])
    protected = (*parameter_arrays(parameters), features)
    before = tuple(array.copy() for array in protected)
    expected = []
    current = features
    for weight, bias in zip(parameters.hidden_weights, parameters.hidden_biases):
        current = np.tanh(current @ weight + bias)
        expected.append(current)
    actual = feedforward_latents(parameters, features)
    assert type(actual) is tuple and len(actual) == 4
    for latent, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(latent, reference)
    logits = expected[-1] @ parameters.output_weight + parameters.output_bias
    np.testing.assert_allclose(predict_coupled(parameters, features), 1 / (1 + np.exp(-logits)))
    for actual_array, original in zip(protected, before, strict=True):
        np.testing.assert_array_equal(actual_array, original)


def test_logits_of_both_signs_at_one_thousand_stay_finite() -> None:
    parameters = CoupledParameters(
        (np.ones((1, 1)),), (np.zeros((1, 1)),), np.array([[1000.0]]), np.zeros((1, 1))
    )
    features, targets = np.array([[1000.0], [-1000.0]]), np.array([[0.0], [1.0]])
    result = evaluate_coupled(parameters, features, targets, (np.array([[1.0], [-1.0]]),), (1.0,))
    assert result.task_loss == pytest.approx(1000.0)
    assert result.energy == pytest.approx(1000.0) and result.hidden_penalties == (0.0,)
    assert all(
        np.isfinite(array).all()
        for array in (*result.latent_gradients, *parameter_arrays(result.parameter_gradients))
    )
    np.testing.assert_array_equal(predict_coupled(parameters, features), [[1.0], [0.0]])


INVALID_CASES = (
    "features-float32",
    "features-rank1",
    "features-nonfinite",
    "empty-batch",
    "targets-float32",
    "targets-shape",
    "targets-low",
    "targets-high",
    "targets-nan",
    "latent-float32",
    "latent-shape",
    "latent-nonfinite",
    "latent-count",
    "latent-list",
    "weights-float32",
    "weights-shape",
    "weights-nonfinite",
    "weights-list",
    "bias-rank1",
    "bias-width",
    "bias-nonfinite",
    "bias-count",
    "zero-layers",
    "output-weight-shape",
    "output-bias-shape",
    "output-weight-nonfinite",
    "precision-count",
    "precision-list",
    "precision-bool",
    "precision-low",
    "precision-high",
    "precision-huge-int",
    "precision-nan",
    "precision-infinite",
    "precision-numpy-float",
    "precision-numpy-int",
)


def invalid_inputs(case: str) -> tuple[object, object, object, object, object]:
    parameters, features, targets, latents, precisions = fixture(WIDTHS[1])
    if case == "features-float32":
        features = features.astype(np.float32)
    elif case == "features-rank1":
        features = features[0]
    elif case == "features-nonfinite":
        features[0, 0] = np.inf
    elif case == "empty-batch":
        features, targets, latents = features[:0], targets[:0], tuple(z[:0] for z in latents)
    elif case == "targets-float32":
        targets = targets.astype(np.float32)
    elif case == "targets-shape":
        targets = targets[:, 0]
    elif case == "targets-low":
        targets[0, 0] = -0.01
    elif case == "targets-high":
        targets[0, 0] = 1.01
    elif case == "targets-nan":
        targets[0, 0] = np.nan
    elif case == "latent-float32":
        latents = (latents[0].astype(np.float32), latents[1])
    elif case == "latent-shape":
        latents = (latents[0][:, :1], latents[1])
    elif case == "latent-nonfinite":
        latents[0][0, 0] = np.inf
    elif case == "latent-count":
        latents = latents[:1]
    elif case == "latent-list":
        return parameters, features, targets, list(latents), precisions
    elif case == "weights-float32":
        parameters = replace(
            parameters,
            hidden_weights=(
                parameters.hidden_weights[0].astype(np.float32),
                parameters.hidden_weights[1],
            ),
        )
    elif case == "weights-shape":
        parameters = replace(
            parameters, hidden_weights=(np.zeros((4, 2)), parameters.hidden_weights[1])
        )
    elif case == "weights-nonfinite":
        parameters.hidden_weights[0][0, 0] = np.inf
    elif case == "weights-list":
        return (
            replace(
                parameters, hidden_weights=cast(tuple[Array, ...], list(parameters.hidden_weights))
            ),
            features,
            targets,
            latents,
            precisions,
        )
    elif case == "bias-rank1":
        parameters = replace(
            parameters, hidden_biases=(parameters.hidden_biases[0][0], parameters.hidden_biases[1])
        )
    elif case == "bias-width":
        parameters = replace(
            parameters, hidden_biases=(np.zeros((1, 3)), parameters.hidden_biases[1])
        )
    elif case == "bias-nonfinite":
        parameters.hidden_biases[0][0, 0] = np.nan
    elif case == "bias-count":
        parameters = replace(parameters, hidden_biases=parameters.hidden_biases[:1])
    elif case == "zero-layers":
        parameters, latents, precisions = (
            replace(parameters, hidden_weights=(), hidden_biases=()),
            (),
            (),
        )
    elif case == "output-weight-shape":
        parameters = replace(parameters, output_weight=np.zeros((4, 1)))
    elif case == "output-bias-shape":
        parameters = replace(parameters, output_bias=np.zeros((1, 2)))
    elif case == "output-weight-nonfinite":
        parameters.output_weight[0, 0] = np.inf
    elif case == "precision-count":
        precisions = precisions[:1]
    elif case == "precision-list":
        return parameters, features, targets, latents, list(precisions)
    elif case == "precision-bool":
        return parameters, features, targets, latents, (True, 1.0)
    elif case == "precision-low":
        precisions = (1e-7, 1.0)
    elif case == "precision-high":
        precisions = (1e7, 1.0)
    elif case == "precision-huge-int":
        return parameters, features, targets, latents, (10**1000, 1.0)
    elif case == "precision-nan":
        precisions = (float("nan"), 1.0)
    elif case == "precision-infinite":
        precisions = (float("inf"), 1.0)
    elif case == "precision-numpy-float":
        return parameters, features, targets, latents, (np.float64(1), 1.0)
    elif case == "precision-numpy-int":
        return parameters, features, targets, latents, (np.int64(1), 1.0)
    else:
        raise AssertionError(case)
    return parameters, features, targets, latents, precisions


@pytest.mark.parametrize("case", INVALID_CASES)
def test_invalid_schema_and_numeric_inputs_fail_before_evaluation(case: str) -> None:
    with pytest.raises(ValueError):
        parameters, features, targets, latents, precisions = invalid_inputs(case)
        evaluate_coupled(
            cast(CoupledParameters, parameters),
            cast(Array, features),
            cast(Array, targets),
            cast(tuple[Array, ...], latents),
            cast(tuple[float, ...], precisions),
        )


@pytest.mark.parametrize("precision", (1e-6, 1, 1e6))
def test_declared_precision_endpoints_and_native_integer_are_accepted(precision: float) -> None:
    parameters, features, targets, latents, _ = fixture(WIDTHS[0])
    actual = evaluate_coupled(parameters, features, targets, latents, (precision,))
    expected = reference_value(parameters, features, targets, latents, (precision,))[0]
    assert actual.energy == pytest.approx(expected)


def test_finite_inputs_that_overflow_computation_raise_floating_point_error() -> None:
    parameters = CoupledParameters(
        (np.zeros((1, 1)),), (np.zeros((1, 1)),), np.array([[1e308]]), np.zeros((1, 1))
    )
    with pytest.raises(FloatingPointError):
        evaluate_coupled(
            parameters, np.zeros((1, 1)), np.array([[0.5]]), (np.array([[1e308]]),), (1.0,)
        )
