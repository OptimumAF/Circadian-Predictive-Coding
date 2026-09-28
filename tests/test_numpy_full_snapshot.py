"""NumPy circadian snapshots detach and restore every model-owned state channel."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork


def _config() -> CircadianConfig:
    return CircadianConfig(
        sleep_mode="components",
        use_dual_chemical=True,
        use_reward_modulated_learning=True,
        max_split_per_sleep=1,
        max_prune_per_sleep=0,
        split_threshold=0.8,
        split_noise_scale=0.03,
        replay_steps=1,
        replay_memory_size=2,
        sleep_enable_homeostasis=False,
        sleep_enable_prune=False,
    )


def _model(
    seed: int = 181, config: CircadianConfig | None = None
) -> CircadianPredictiveCodingNetwork:
    return CircadianPredictiveCodingNetwork(
        2,
        4,
        seed=seed,
        min_hidden_dim=3,
        max_hidden_dim=6,
        hidden_dims=(3, 4),
        circadian_config=config or _config(),
    )


def _assert_equivalent(
    actual: CircadianPredictiveCodingNetwork,
    expected: CircadianPredictiveCodingNetwork,
) -> None:
    assert actual.hidden_dim == expected.hidden_dim
    assert actual.get_sleep_clocks() == expected.get_sleep_clocks()
    assert actual._traffic_steps == expected._traffic_steps
    assert actual._energy_history == expected._energy_history
    assert actual._reward_error_ema == expected._reward_error_ema
    assert actual._last_reward_scale == expected._last_reward_scale
    assert actual._rng.bit_generator.state == expected._rng.bit_generator.state
    for field in (
        "weight_input_hidden",
        "bias_hidden",
        "weight_hidden_output",
        "bias_output",
        "_hidden_chemical",
        "_hidden_chemical_fast",
        "_hidden_chemical_slow",
        "_neuron_age",
        "_traffic_sum",
        "_importance_ema",
        "_prune_ttl",
        "_prune_marked",
        "_split_cooldown",
        "_prune_cooldown",
    ):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    for field in ("_pre_hidden_weights", "_pre_hidden_biases"):
        actual_layers = getattr(actual, field)
        expected_layers = getattr(expected, field)
        assert len(actual_layers) == len(expected_layers)
        for current, saved in zip(actual_layers, expected_layers):
            np.testing.assert_array_equal(current, saved)
    assert len(actual._replay_memory) == len(expected._replay_memory)
    for current, saved in zip(actual._replay_memory, expected._replay_memory):
        np.testing.assert_array_equal(current.input_batch, saved.input_batch)
        np.testing.assert_array_equal(current.target_batch, saved.target_batch)
        assert current.priority == saved.priority
        assert current.positive_fraction == saved.positive_fraction


def test_numpy_snapshot_restores_topology_replay_rng_and_future_split() -> None:
    model = _model()
    features = np.array([[0.3, -0.2], [-0.1, 0.4]])
    targets = np.array([[1.0], [0.0]])
    model.train_epoch(features, targets, 0.03, 2, 0.2)
    model.set_chemical_state(np.array([0.95, 0.6, 0.4, 0.0]))
    expected = deepcopy(model)
    saved = model.snapshot_state()

    model.weight_input_hidden[0, 0] += 3.0
    model._replay_memory[0].input_batch[0, 0] += 5.0
    split = model.sleep_event(force_sleep=True)
    assert split.new_hidden_dim == 5
    assert model.get_sleep_clocks().replay_updates == 1
    model.restore_state(saved)
    _assert_equivalent(model, expected)

    # The restored model owns fresh arrays, even if the saved artifact changes later.
    saved.state["weight_input_hidden"][0, 0] += 7.0
    saved.state["_replay_memory"][0].input_batch[0, 0] += 7.0
    _assert_equivalent(model, expected)

    first = model.sleep_event(force_sleep=True)
    second = expected.sleep_event(force_sleep=True)
    assert first == second
    _assert_equivalent(model, expected)


def test_numpy_snapshot_rejects_wrong_version_or_config_without_mutation() -> None:
    model = _model(seed=191)
    saved = model.snapshot_state()
    before = deepcopy(model)
    with pytest.raises(ValueError, match="version"):
        model.restore_state(replace(saved, format_version=99))
    _assert_equivalent(model, before)

    other = CircadianPredictiveCodingNetwork(
        2,
        4,
        seed=193,
        min_hidden_dim=3,
        max_hidden_dim=6,
        hidden_dims=(3, 4),
        circadian_config=replace(model.config, replay_memory_size=1),
    )
    other_before = deepcopy(other)
    with pytest.raises(ValueError, match="config"):
        other.restore_state(saved)
    _assert_equivalent(other, other_before)

    malformed = replace(saved, state=deepcopy(saved.state))
    malformed.state["weight_hidden_output"] = np.zeros((2, 1))
    with pytest.raises(ValueError, match="topology"):
        model.restore_state(malformed)
    _assert_equivalent(model, before)


def test_numpy_snapshot_restores_active_gradual_prune_state() -> None:
    config = replace(
        _config(),
        max_split_per_sleep=0,
        max_prune_per_sleep=1,
        sleep_enable_prune=True,
        prune_threshold=0.2,
        prune_decay_steps=3,
    )
    model = _model(seed=197, config=config)
    model.set_chemical_state(np.array([0.95, 0.6, 0.4, 0.0]))
    event = model.sleep_event(force_sleep=True)
    assert event.pruned_indices
    assert np.any(model._prune_marked)
    before = deepcopy(model)
    saved = model.snapshot_state()
    features = np.array([[0.3, -0.2], [-0.1, 0.4]])
    targets = np.array([[1.0], [0.0]])
    model.train_epoch(features, targets, 0.03, 2, 0.2)
    model.restore_state(saved)
    _assert_equivalent(model, before)
