"""Observe current NumPy replay mutations apart from the sleep wrapper."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np

from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
    ReplayRetentionBudget,
)
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.sleep_clocks import SleepEpochProgress


def _woken_model() -> CircadianPredictiveCodingNetwork:
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
        min_epochs_between_sleep=1,
        sleep_energy_window=2,
        sleep_plateau_delta=10.0,
        sleep_chemical_variance_threshold=0.0,
        use_adaptive_sleep_trigger=True,
    )
    model = CircadianPredictiveCodingNetwork(
        2, 4, seed=41, circadian_config=config, min_hidden_dim=2
    )
    model.configure_replay_retention(
        ReplayRetentionBudget(4, 96), policy=ReplayRetentionPolicy("recent_fifo")
    )
    inputs = np.array([[0.2, -0.1], [0.3, 0.4]], dtype=np.float64)
    targets = np.array([[0.0], [1.0]], dtype=np.float64)
    model.train_epoch(inputs, targets, 0.02, 2, 0.1)
    model.train_epoch(inputs, targets, 0.02, 2, 0.1)
    model._split_cooldown[:] = 3
    model._prune_cooldown[:] = 3
    model._prune_marked[0] = True
    model._prune_ttl[0] = 3
    return model


def test_should_audit_direct_replay_adaptive_state_without_wake_or_memory_refill() -> None:
    model = _woken_model()
    before = model.snapshot_state().state
    retention = model.get_replay_retention()
    retained_order = model.get_replay_retained_order_ids()
    exposure = model.get_replay_exposure()
    clocks = model.get_sleep_clocks()
    selected_ids = model.preview_unprioritized_replay_ids(1)
    adaptive_before = model.should_trigger_sleep()

    examples, updates = model._run_replay_consolidation()
    after = model.snapshot_state().state

    assert (examples, updates) == (1, 1)
    assert model.get_replay_retention() == retention
    assert model.get_replay_retained_order_ids() == retained_order
    assert model.get_replay_exposure().exposed_ids == selected_ids
    assert model.get_replay_exposure().observed_ids == exposure.observed_ids
    assert model.get_replay_exposure().replay_updates == exposure.replay_updates + 1
    assert model.get_sleep_clocks().wake_batches == clocks.wake_batches
    assert model.get_sleep_clocks().wake_examples == clocks.wake_examples
    assert model.get_sleep_clocks().wake_batches_since_sleep == clocks.wake_batches_since_sleep
    assert model.get_sleep_clocks().sleep_events == clocks.sleep_events
    assert model.get_sleep_clocks().replay_updates == clocks.replay_updates + 1
    assert after["_energy_history"] == before["_energy_history"]
    np.testing.assert_array_equal(after["_neuron_age"], before["_neuron_age"])
    assert after["_prune_ttl"][0] == before["_prune_ttl"][0] - 1
    np.testing.assert_array_equal(after["_split_cooldown"], before["_split_cooldown"] - 1)
    np.testing.assert_array_equal(after["_prune_cooldown"], before["_prune_cooldown"] - 1)
    for name in (
        "_hidden_chemical",
        "_hidden_chemical_fast",
        "_hidden_chemical_slow",
        "_importance_ema",
        "_traffic_sum",
    ):
        assert not np.array_equal(after[name], before[name]), name
    assert after["_reward_error_ema"] != before["_reward_error_ema"]
    assert after["_traffic_steps"] == before["_traffic_steps"] + 1
    assert model.should_trigger_sleep() == adaptive_before


def test_should_separate_replay_mutations_from_sleep_wrapper_clocks() -> None:
    before = _woken_model()
    direct = deepcopy(before)
    with_sleep = deepcopy(before)

    direct._run_replay_consolidation()
    event = with_sleep.sleep_event(force_sleep=True, epoch_progress=SleepEpochProgress(2, 3))

    assert event.telemetry is not None
    assert event.telemetry.replay.applied_updates == 1
    direct_state = direct.snapshot_state().state
    sleep_state = with_sleep.snapshot_state().state
    for name in (
        "_hidden_chemical",
        "_hidden_chemical_fast",
        "_hidden_chemical_slow",
        "_importance_ema",
        "_traffic_sum",
        "_neuron_age",
        "_split_cooldown",
        "_prune_cooldown",
        "_prune_ttl",
    ):
        np.testing.assert_array_equal(sleep_state[name], direct_state[name])
    assert sleep_state["_reward_error_ema"] == direct_state["_reward_error_ema"]
    assert with_sleep.get_replay_retention() == direct.get_replay_retention()
    assert with_sleep.get_replay_exposure() == direct.get_replay_exposure()
    assert with_sleep.get_sleep_clocks().wake_batches == direct.get_sleep_clocks().wake_batches
    assert with_sleep.get_sleep_clocks().replay_updates == direct.get_sleep_clocks().replay_updates
    assert with_sleep.get_sleep_clocks().wake_batches_since_sleep == 0
    assert direct.get_sleep_clocks().wake_batches_since_sleep == 2
    assert with_sleep.get_sleep_clocks().sleep_events == direct.get_sleep_clocks().sleep_events + 1


def test_should_show_chemical_variance_can_change_adaptive_sleep_readiness() -> None:
    model = _woken_model()
    before_variance = float(np.var(model.snapshot_state().state["_hidden_chemical"]))
    control = deepcopy(model)
    control._run_replay_consolidation()
    after_variance = float(np.var(control.snapshot_state().state["_hidden_chemical"]))
    assert before_variance != after_variance

    # Why this: a threshold between observed values proves the replay-only
    # chemistry mutation can affect scheduling; it is not a benchmark setting.
    model.config = replace(
        model.config,
        sleep_chemical_variance_threshold=0.5 * (before_variance + after_variance),
    )
    readiness_before = model.should_trigger_sleep()
    model._run_replay_consolidation()
    readiness_after = model.should_trigger_sleep()
    assert readiness_before != readiness_after
