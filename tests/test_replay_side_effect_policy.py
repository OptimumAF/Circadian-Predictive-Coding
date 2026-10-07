"""Opt-in replay updates weights without advancing wake adaptive state."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
    ReplayRetentionBudget,
)
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.sleep_clocks import SleepEpochProgress


POLICY = "wake_only_adaptive_v1"
ADAPTIVE_ARRAYS = (
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


def _model(*, opt_in: bool = True) -> CircadianPredictiveCodingNetwork:
    config = CircadianConfig(
        sleep_mode="components",
        sleep_enable_replay=True,
        sleep_enable_chemical_reset=False,
        sleep_enable_homeostasis=False,
        sleep_enable_split=False,
        sleep_enable_prune=False,
        max_split_per_sleep=0,
        max_prune_per_sleep=0,
        replay_steps=1,
        replay_prioritized=False,
        replay_class_balanced=False,
        use_dual_chemical=True,
        use_reward_modulated_learning=True,
        prune_decay_steps=3,
    )
    model = CircadianPredictiveCodingNetwork(
        2, 4, seed=41, circadian_config=config, min_hidden_dim=2
    )
    model.configure_replay_retention(
        ReplayRetentionBudget(4, 96), policy=ReplayRetentionPolicy("recent_fifo")
    )
    if opt_in:
        model.configure_replay_side_effect_policy(POLICY)
    inputs = np.array([[0.2, -0.1], [0.3, 0.4]], dtype=np.float64)
    targets = np.array([[0.0], [1.0]], dtype=np.float64)
    model.train_epoch(inputs, targets, 0.02, 2, 0.1)
    model._split_cooldown[:] = 3
    model._prune_cooldown[:] = 3
    model._prune_marked[0] = True
    model._prune_ttl[0] = 3
    return model


def test_should_require_explicit_policy_before_wake_and_preserve_default_identity() -> None:
    default = _model(opt_in=False)
    assert default.get_replay_side_effect_policy() == "historical"
    assert "_replay_side_effect_policy" not in default.snapshot_state().state
    with pytest.raises(ValueError, match="before training"):
        default.configure_replay_side_effect_policy(POLICY)

    fresh = CircadianPredictiveCodingNetwork(2, 4, seed=41)
    with pytest.raises(ValueError, match="unsupported"):
        fresh.configure_replay_side_effect_policy("unknown")
    fresh.configure_replay_side_effect_policy(POLICY)
    with pytest.raises(ValueError, match="once before training"):
        fresh.configure_replay_side_effect_policy(POLICY)


def test_should_update_replay_weights_from_pre_row_gate_without_wake_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    before = model.snapshot_state().state
    retention = model.get_replay_retention()
    order = model.get_replay_retained_order_ids()
    clocks = model.get_sleep_clocks()
    exposure = model.get_replay_exposure()
    gate = model.get_plasticity_state()
    seen_gates: list[np.ndarray] = []
    original_gate = model.get_plasticity_state

    def capture_gate() -> np.ndarray:
        seen_gates.append(original_gate())
        np.testing.assert_array_equal(model._hidden_chemical, before["_hidden_chemical"])
        return seen_gates[-1]

    monkeypatch.setattr(model, "get_plasticity_state", capture_gate)
    examples, updates = model._run_replay_consolidation()
    after = model.snapshot_state().state

    assert (examples, updates) == (1, 1)
    assert len(seen_gates) == 1
    np.testing.assert_array_equal(seen_gates[0], gate)
    assert not np.array_equal(after["weight_hidden_output"], before["weight_hidden_output"])
    for name in ADAPTIVE_ARRAYS:
        np.testing.assert_array_equal(after[name], before[name], err_msg=name)
    assert after["_traffic_steps"] == before["_traffic_steps"]
    assert after["_reward_error_ema"] == before["_reward_error_ema"]
    assert after["_last_reward_scale"] == before["_last_reward_scale"]
    assert after["_energy_history"] == before["_energy_history"]
    assert model.get_replay_retention() == retention
    assert model.get_replay_retained_order_ids() == order
    assert model.get_replay_exposure().observed_ids == exposure.observed_ids
    assert model.get_replay_exposure().replay_updates == exposure.replay_updates + 1
    assert model.get_sleep_clocks().wake_batches == clocks.wake_batches
    assert model.get_sleep_clocks().wake_examples == clocks.wake_examples
    assert model.get_sleep_clocks().wake_batches_since_sleep == clocks.wake_batches_since_sleep
    assert model.get_sleep_clocks().replay_updates == clocks.replay_updates + 1


def test_should_bind_policy_to_snapshot_and_restore_continuation() -> None:
    model = _model()
    saved = model.snapshot_state()
    control = deepcopy(model)
    model._run_replay_consolidation()
    model.restore_state(saved)
    model._run_replay_consolidation()
    control._run_replay_consolidation()
    np.testing.assert_array_equal(model.weight_hidden_output, control.weight_hidden_output)
    assert model.get_replay_exposure() == control.get_replay_exposure()
    assert model.get_sleep_clocks() == control.get_sleep_clocks()

    historical = _model(opt_in=False)
    with pytest.raises(ValueError, match="fields are incompatible"):
        historical.restore_state(saved)
    assert historical.get_replay_side_effect_policy() == "historical"


def test_should_use_pre_row_reward_baseline_without_advancing_it() -> None:
    hard = _model()
    easy = deepcopy(hard)
    hard._reward_error_ema = 0.1
    easy._reward_error_ema = 1.0
    hard_bias = hard.bias_output.copy()
    easy_bias = easy.bias_output.copy()

    hard._run_replay_consolidation()
    easy._run_replay_consolidation()

    hard_delta = hard_bias - hard.bias_output
    easy_delta = easy_bias - easy.bias_output
    np.testing.assert_allclose(hard_delta, easy_delta * 2.0, rtol=1e-12, atol=1e-12)
    assert hard._reward_error_ema == 0.1
    assert easy._reward_error_ema == 1.0


def test_should_keep_adaptive_state_frozen_in_accepted_sleep() -> None:
    model = _model()
    before = model.snapshot_state().state
    event = model.sleep_event(force_sleep=True, epoch_progress=SleepEpochProgress(1, 2))
    after = model.snapshot_state().state

    assert event.telemetry is not None
    assert event.telemetry.replay.applied_updates == 1
    for name in ADAPTIVE_ARRAYS:
        np.testing.assert_array_equal(after[name], before[name], err_msg=name)
    assert after["_reward_error_ema"] == before["_reward_error_ema"]
    assert model.get_sleep_clocks().replay_updates == 1
    assert model.get_sleep_clocks().sleep_events == 1
    assert model.get_sleep_clocks().wake_batches_since_sleep == 0


def test_should_roll_back_policy_replay_after_sleep_error(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    before = model.snapshot_state().state
    clock = model.get_sleep_clocks()
    exposure = model.get_replay_exposure()
    original = model._run_replay_consolidation

    def fail_after_replay() -> tuple[int, int]:
        original()
        raise RuntimeError("injected after replay")

    monkeypatch.setattr(model, "_run_replay_consolidation", fail_after_replay)
    with pytest.raises(RuntimeError, match="injected after replay"):
        model.sleep_event(force_sleep=True, epoch_progress=SleepEpochProgress(1, 2))
    after = model.snapshot_state().state
    for name in (*ADAPTIVE_ARRAYS, "weight_hidden_output", "bias_output"):
        np.testing.assert_array_equal(after[name], before[name], err_msg=name)
    assert model.get_sleep_clocks() == clock
    assert model.get_replay_exposure() == exposure
    assert model.get_replay_side_effect_policy() == POLICY
