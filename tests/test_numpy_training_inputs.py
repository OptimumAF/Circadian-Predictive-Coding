"""NumPy binary trainers reject malformed batches before changing model state."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork


FEATURES = np.array([[0.3, -0.2], [-0.5, 0.4]], dtype=np.float64)
TARGETS = np.array([[1.0], [0.0]], dtype=np.float64)


def _model(kind: str) -> Any:
    if kind == "backprop":
        return BackpropMLP(2, 3, seed=23)
    if kind == "predictive":
        return PredictiveCodingNetwork(2, 3, seed=23)
    model = CircadianPredictiveCodingNetwork(2, 3, seed=23, min_hidden_dim=3)
    model._split_cooldown[:] = [2, 1, 3]
    model._prune_cooldown[:] = [1, 2, 3]
    return model


def _train(model: Any, features: np.ndarray, targets: np.ndarray, rate: float) -> Any:
    if isinstance(model, BackpropMLP):
        return model.train_epoch(features, targets, rate)
    return model.train_epoch(features, targets, rate, 2, 0.2)


def _state(model: Any) -> tuple[np.ndarray, ...]:
    arrays = (
        model.weight_input_hidden.copy(),
        model.bias_hidden.copy(),
        model.weight_hidden_output.copy(),
        model.bias_output.copy(),
    )
    if isinstance(model, CircadianPredictiveCodingNetwork):
        return (
            *arrays,
            model.get_chemical_state(),
            model._importance_ema.copy(),
            model._split_cooldown.copy(),
            model._prune_cooldown.copy(),
            model._neuron_age.copy(),
            model._traffic_sum.copy(),
        )
    return (*arrays, *(traffic.copy() for traffic in model._traffic_sums))


@pytest.mark.parametrize("kind", ["backprop", "predictive", "circadian"])
@pytest.mark.parametrize(
    "case,features,targets,rate,message",
    [
        ("empty", np.empty((0, 2)), np.empty((0, 1)), 0.03, "nonempty"),
        ("feature_rank", FEATURES[0], TARGETS, 0.03, "features must be 2D"),
        ("feature_width", np.zeros((2, 3)), TARGETS, 0.03, "feature width"),
        ("target_rank", FEATURES, TARGETS[:, 0], 0.03, "targets must be 2D"),
        ("target_width", FEATURES, np.zeros((2, 2)), 0.03, "one column"),
        ("batch_mismatch", FEATURES, TARGETS[:1], 0.03, "same batch size"),
        ("target_low", FEATURES, np.array([[-0.1], [0.0]]), 0.03, "between 0 and 1"),
        ("target_high", FEATURES, np.array([[1.1], [0.0]]), 0.03, "between 0 and 1"),
        ("feature_nan", np.array([[np.nan, 0.0], [0.0, 0.0]]), TARGETS, 0.03, "finite"),
        ("target_inf", FEATURES, np.array([[np.inf], [0.0]]), 0.03, "finite"),
        ("feature_object", FEATURES.astype(object), TARGETS, 0.03, "real numeric"),
        ("target_complex", FEATURES, TARGETS.astype(np.complex128), 0.03, "real numeric"),
        ("rate_nan", FEATURES, TARGETS, float("nan"), "learning_rate.*finite"),
        ("rate_inf", FEATURES, TARGETS, float("inf"), "learning_rate.*finite"),
        ("rate_zero", FEATURES, TARGETS, 0.0, "learning_rate.*positive"),
    ],
)
def test_invalid_numpy_training_input_fails_before_state_change(
    kind: str, case: str, features: np.ndarray, targets: np.ndarray, rate: float, message: str
) -> None:
    model = _model(kind)
    before = _state(model)
    with pytest.raises(ValueError, match=message):
        _train(model, features, targets, rate)
    for previous, current in zip(before, _state(model)):
        np.testing.assert_array_equal(current, previous)
    assert model._traffic_steps == 0
    if kind == "circadian":
        assert model._epoch_count == 0
        assert not model._replay_memory


@pytest.mark.parametrize("kind", ["backprop", "predictive", "circadian"])
def test_numpy_training_accepts_finite_soft_labels(kind: str) -> None:
    model = _model(kind)
    result = _train(model, FEATURES, np.array([[0.25], [0.75]]), 0.03)
    metric = result.loss if kind == "backprop" else result.energy
    assert np.isfinite(metric)
    assert not np.array_equal(model.weight_hidden_output, _model(kind).weight_hidden_output)
