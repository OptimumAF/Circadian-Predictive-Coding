"""NumPy sleep components can run independently of structural budgets."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
)


def _component_config(**changes: Any) -> CircadianConfig:
    options: dict[str, Any] = dict(
        sleep_mode="components",
        sleep_enable_chemical_reset=False,
        sleep_enable_replay=False,
        sleep_enable_homeostasis=False,
        sleep_enable_split=False,
        sleep_enable_prune=False,
        max_split_per_sleep=0,
        max_prune_per_sleep=0,
        replay_steps=0,
        sleep_reset_factor=0.5,
        homeostatic_downscale_factor=0.8,
        homeostasis_target_input_norm=0.0,
        homeostasis_target_output_norm=0.0,
    )
    options.update(changes)
    return CircadianConfig(**options)


def _model(config: CircadianConfig, width: int = 3) -> CircadianPredictiveCodingNetwork:
    return CircadianPredictiveCodingNetwork(
        2, width, seed=83, min_hidden_dim=2, max_hidden_dim=width + 2, circadian_config=config
    )


def test_zero_structural_budgets_still_reset_numpy_chemical_only() -> None:
    model = _model(_component_config(sleep_enable_chemical_reset=True))
    model.set_chemical_state(np.array([0.2, 0.6, 1.0]))
    model._epochs_since_sleep = 4
    weights_before = model.weight_input_hidden.copy()
    event = model.sleep_event(force_sleep=True)
    assert event.performed is True
    assert event.old_hidden_dim == event.new_hidden_dim == 3
    assert event.split_indices == event.pruned_indices == ()
    np.testing.assert_array_equal(model.get_chemical_state(), [0.1, 0.3, 0.5])
    np.testing.assert_array_equal(model.weight_input_hidden, weights_before)
    assert model._epochs_since_sleep == 0


def test_zero_structural_budgets_still_run_numpy_replay_only() -> None:
    config = _component_config(sleep_enable_replay=True, replay_steps=1, replay_memory_size=2)
    model = _model(config)
    features = np.array([[0.3, -0.2], [-0.1, 0.4]])
    targets = np.array([[1.0], [0.0]])
    model.train_epoch(features, targets, 0.03, 2, 0.2)
    assert model._traffic_steps == 1
    assert len(model._replay_memory) == 1
    epoch_count = model._epoch_count
    event = model.sleep_event(force_sleep=True)
    assert event.performed is True
    assert event.split_indices == event.pruned_indices == ()
    assert model._traffic_steps == 2
    assert model._epoch_count == epoch_count
    assert len(model._replay_memory) == 1
    assert model._epochs_since_sleep == 0


def test_zero_structural_budgets_still_run_numpy_homeostasis_only() -> None:
    model = _model(_component_config(sleep_enable_homeostasis=True))
    model.set_chemical_state(np.array([0.2, 0.6, 1.0]))
    before_weight = model.weight_input_hidden.copy()
    before_chemical = model.get_chemical_state()
    event = model.sleep_event(force_sleep=True)
    assert event.performed is True
    assert event.split_indices == event.pruned_indices == ()
    np.testing.assert_allclose(model.weight_input_hidden, 0.8 * before_weight, atol=0, rtol=1e-15)
    np.testing.assert_array_equal(model.get_chemical_state(), before_chemical)


@pytest.mark.parametrize("kind,expected_width", [("split", 5), ("prune", 3)])
def test_numpy_split_and_prune_switches_act_independently(kind: str, expected_width: int) -> None:
    config = _component_config(
        sleep_enable_split=kind == "split",
        sleep_enable_prune=kind == "prune",
        max_split_per_sleep=1,
        max_prune_per_sleep=1,
        split_threshold=0.8,
        prune_threshold=0.2,
        split_noise_scale=0.0,
    )
    model = _model(config, width=4)
    model.set_chemical_state(np.array([0.95, 0.6, 0.4, 0.0]))
    event = model.sleep_event(force_sleep=True)
    assert event.performed is True
    assert event.old_hidden_dim == 4
    assert event.new_hidden_dim == expected_width
    assert len(event.split_indices) == (1 if kind == "split" else 0)
    assert len(event.pruned_indices) == (1 if kind == "prune" else 0)
    assert model.get_chemical_state()[0] > 0.4


@pytest.mark.parametrize("mode", ["disabled", "legacy"])
def test_numpy_disabled_and_legacy_zero_budget_routes_stay_noop(mode: str) -> None:
    config = _component_config(
        sleep_mode=mode,
        sleep_enable_chemical_reset=True,
        sleep_enable_replay=True,
        sleep_enable_homeostasis=True,
        sleep_enable_split=True,
        sleep_enable_prune=True,
        max_split_per_sleep=0,
        max_prune_per_sleep=0,
    )
    model = _model(config)
    model.set_chemical_state(np.array([0.2, 0.6, 1.0]))
    model._epochs_since_sleep = 4
    before_chemical = model.get_chemical_state()
    before_weight = model.weight_input_hidden.copy()
    event = model.sleep_event(force_sleep=True)
    assert event.performed is False
    assert event.split_indices == event.pruned_indices == ()
    np.testing.assert_array_equal(model.get_chemical_state(), before_chemical)
    np.testing.assert_array_equal(model.weight_input_hidden, before_weight)
    assert model._epochs_since_sleep == 4


def test_numpy_disabled_mode_ignores_forced_structural_budget() -> None:
    config = _component_config(
        sleep_mode="disabled",
        max_split_per_sleep=1,
        sleep_enable_split=True,
    )
    model = _model(config)
    model.set_chemical_state(np.array([0.95, 0.2, 0.1]))
    before = model.weight_input_hidden.copy()
    event = model.sleep_event(force_sleep=True)
    assert event.performed is False
    assert event.new_hidden_dim == 3
    np.testing.assert_array_equal(model.weight_input_hidden, before)


def test_numpy_invalid_mode_and_legacy_component_mix_fail_at_construction() -> None:
    with pytest.raises(ValueError, match="sleep_mode"):
        _model(replace(_component_config(), sleep_mode="unknown"))
    with pytest.raises(ValueError, match="sleep_mode.*components"):
        _model(replace(CircadianConfig(), sleep_enable_replay=False))


@pytest.mark.parametrize("mode", ["components", "disabled"])
def test_numpy_sleep_mode_does_not_change_valid_wake_step(mode: str) -> None:
    legacy = _model(CircadianConfig(max_split_per_sleep=0, max_prune_per_sleep=0))
    alternate = _model(
        replace(
            legacy.config,
            sleep_mode=mode,
            sleep_enable_chemical_reset=False,
            sleep_enable_replay=False,
            sleep_enable_homeostasis=False,
            sleep_enable_split=False,
            sleep_enable_prune=False,
        )
    )
    features = np.array([[0.3, -0.2], [-0.1, 0.4]])
    targets = np.array([[1.0], [0.0]])
    first = legacy.train_epoch(features, targets, 0.03, 2, 0.2)
    second = alternate.train_epoch(features, targets, 0.03, 2, 0.2)
    assert first == second
    for field in ("weight_input_hidden", "weight_hidden_output", "bias_hidden", "bias_output"):
        np.testing.assert_array_equal(getattr(legacy, field), getattr(alternate, field))
    np.testing.assert_array_equal(legacy.get_chemical_state(), alternate.get_chemical_state())
    assert legacy._epochs_since_sleep == alternate._epochs_since_sleep
