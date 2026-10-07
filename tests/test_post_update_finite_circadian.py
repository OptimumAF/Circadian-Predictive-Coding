"""NumPy circadian wake steps restore adaptive state after numeric rejection."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
)


ARRAYS = (
    "weight_input_hidden",
    "bias_hidden",
    "weight_hidden_output",
    "bias_output",
    "_hidden_chemical",
    "_hidden_chemical_fast",
    "_hidden_chemical_slow",
    "_importance_ema",
    "_traffic_sum",
    "_neuron_age",
    "_split_cooldown",
    "_prune_cooldown",
    "_prune_ttl",
    "_prune_marked",
)


def _model(prune_ttl: int = 0) -> CircadianPredictiveCodingNetwork:
    config = CircadianConfig(
        use_dual_chemical=True,
        use_reward_modulated_learning=True,
        prune_decay_steps=2,
        replay_memory_size=2,
    )
    model = CircadianPredictiveCodingNetwork(
        2, 2, seed=47, circadian_config=config, min_hidden_dim=1
    )
    model.weight_input_hidden[:] = 0.0
    model.bias_hidden[:] = 0.0
    model.weight_hidden_output[:] = 1.0
    model.bias_output[:] = 0.0
    model._split_cooldown[:] = [3, 2]
    model._prune_cooldown[:] = [2, 3]
    if prune_ttl:
        model._prune_marked[0] = True
        model._prune_ttl[0] = prune_ttl
    return model


def _state(
    model: CircadianPredictiveCodingNetwork,
) -> tuple[dict[str, np.ndarray], tuple[object, ...]]:
    arrays = {name: getattr(model, name).copy() for name in ARRAYS}
    scalars: tuple[object, ...] = (
        model._traffic_steps,
        model._epoch_count,
        model._epochs_since_sleep,
        model._reward_error_ema,
        model._last_reward_scale,
        tuple(model._energy_history),
        tuple(model._replay_memory),
        model.hidden_dim,
    )
    return arrays, scalars


def _assert_unchanged(
    model: CircadianPredictiveCodingNetwork,
    before: tuple[dict[str, np.ndarray], tuple[object, ...]],
) -> None:
    original_arrays, original_scalars = before
    current_arrays, current_scalars = _state(model)
    assert current_scalars == original_scalars
    for name, original in original_arrays.items():
        np.testing.assert_array_equal(current_arrays[name], original)


@pytest.mark.parametrize("prune_ttl", [0, 2, 1])
def test_candidate_overflow_restores_circadian_and_prune_state(prune_ttl: int) -> None:
    model = _model(prune_ttl)
    before = _state(model)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(FloatingPointError, match="nonfinite.*(gradient|update|parameter)"):
            model.train_epoch(np.array([[1e200, 0.0]]), np.ones((1, 1)), 1e200, 1, 0.2)
    _assert_unchanged(model, before)


def test_overflowing_circadian_diagnostic_rejected_before_commit() -> None:
    model = _model()
    model.weight_hidden_output[:] = 3e-154
    before = _state(model)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(FloatingPointError, match="nonfinite.*diagnostic"):
            model.train_epoch(np.zeros((1, 2)), np.ones((1, 1)), 0.01, 1, 1e308)
    _assert_unchanged(model, before)


def test_nonfinite_circadian_model_rejected_before_prune_decay() -> None:
    model = _model(prune_ttl=1)
    model.weight_hidden_output[0, 0] = np.nan
    before = _state(model)
    with pytest.raises(FloatingPointError, match="nonfinite.*model"):
        model.train_epoch(np.array([[0.5, 0.2]]), np.ones((1, 1)), 0.03, 1, 0.2)
    _assert_unchanged(model, before)


def test_valid_gradual_prune_step_still_advances_and_trains() -> None:
    model = _model(prune_ttl=2)
    result = model.train_epoch(np.array([[0.5, 0.2]]), np.ones((1, 1)), 0.03, 1, 0.2)
    assert np.isfinite(result.energy)
    assert model._prune_ttl[0] == 1
    assert model._epoch_count == 1
    assert model._traffic_steps == 1
    assert len(model._replay_memory) == 1
