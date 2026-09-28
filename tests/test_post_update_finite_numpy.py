"""Ordinary NumPy trainers commit only finite training results."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.core.backprop_mlp import BackpropMLP
from src.core.predictive_coding import PredictiveCodingNetwork


def _model(kind: str) -> Any:
    model = (
        BackpropMLP(2, 1, seed=41) if kind == "backprop" else PredictiveCodingNetwork(2, 1, seed=41)
    )
    model.weight_input_hidden[:] = 0.0
    model.bias_hidden[:] = 0.0
    model.weight_hidden_output[:] = 1.0
    model.bias_output[:] = 0.0
    return model


def _state(model: Any) -> tuple[np.ndarray, ...]:
    return (
        model.weight_input_hidden.copy(),
        model.bias_hidden.copy(),
        model.weight_hidden_output.copy(),
        model.bias_output.copy(),
        *(traffic.copy() for traffic in model._traffic_sums),
    )


def _train(
    model: Any,
    features: np.ndarray,
    targets: np.ndarray,
    learning_rate: float,
    inference_rate: float = 0.2,
) -> Any:
    if isinstance(model, BackpropMLP):
        return model.train_epoch(features, targets, learning_rate)
    return model.train_epoch(features, targets, learning_rate, 1, inference_rate)


@pytest.mark.parametrize("kind", ["backprop", "predictive"])
def test_overflowing_candidate_rejected_before_any_update(kind: str) -> None:
    model = _model(kind)
    before = _state(model)
    features = np.array([[1e200, 0.0]])
    targets = np.array([[1.0]])

    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(FloatingPointError, match="nonfinite.*(gradient|update|parameter)"):
            _train(model, features, targets, 1e200)

    for original, current in zip(before, _state(model)):
        np.testing.assert_array_equal(current, original)
    assert model._traffic_steps == 0
    assert model.weight_input_hidden is model._hidden_weights[0]
    assert model.bias_hidden is model._hidden_biases[0]


def test_overflowing_predictive_diagnostic_rejected_before_update() -> None:
    model = _model("predictive")
    model.weight_hidden_output[:] = 3e-154
    before = _state(model)

    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(FloatingPointError, match="nonfinite.*diagnostic"):
            _train(model, np.zeros((1, 2)), np.ones((1, 1)), 0.01, inference_rate=1e308)

    for original, current in zip(before, _state(model)):
        np.testing.assert_array_equal(current, original)
    assert model._traffic_steps == 0


@pytest.mark.parametrize("kind", ["backprop", "predictive"])
def test_nonfinite_model_parameter_rejected_before_update(kind: str) -> None:
    model = _model(kind)
    model.weight_hidden_output[0, 0] = np.nan
    before = _state(model)
    with pytest.raises(FloatingPointError, match="nonfinite.*(model|parameter)"):
        _train(model, np.array([[0.5, -0.2]]), np.ones((1, 1)), 0.01)
    for original, current in zip(before, _state(model)):
        np.testing.assert_array_equal(current, original)
    assert model._traffic_steps == 0


def test_backprop_overflowed_finite_parameter_forward_rejected() -> None:
    model = _model("backprop")
    model.weight_hidden_output[:] = 1e308
    model.bias_output[:] = 1e308
    model.bias_hidden[:] = 3.0
    before = _state(model)

    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(FloatingPointError, match="nonfinite.*(intermediate|logits)"):
            _train(model, np.zeros((1, 2)), np.zeros((1, 1)), 0.01)

    for original, current in zip(before, _state(model)):
        np.testing.assert_array_equal(current, original)
    assert model._traffic_steps == 0


@pytest.mark.parametrize("kind", ["backprop", "predictive"])
def test_finite_update_preserves_aliases_and_metric(kind: str) -> None:
    model = _model(kind)
    before = _state(model)
    result = _train(model, np.array([[0.4, -0.2]]), np.ones((1, 1)), 0.03)
    metric = result.loss if kind == "backprop" else result.energy
    assert np.isfinite(metric)
    assert model.weight_input_hidden is model._hidden_weights[0]
    assert model.bias_hidden is model._hidden_biases[0]
    assert model._traffic_steps == 1
    assert not np.array_equal(before[0], model.weight_input_hidden)
