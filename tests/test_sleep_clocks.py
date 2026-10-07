"""Core sleep clocks separate wake batches, replay, and executed events."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.core.sleep_clocks import SleepEpochProgress


@pytest.mark.parametrize(
    "completed,total",
    [(-1, 3), (0, 0), (4, 3), (True, 3), (1, False)],
)
def test_sleep_epoch_progress_rejects_invalid_counts(completed: Any, total: Any) -> None:
    with pytest.raises(ValueError, match="epoch"):
        SleepEpochProgress(completed_epochs=completed, total_epochs=total)


def test_numpy_clocks_separate_replay_from_wake_and_skipped_from_executed_sleep() -> None:
    config = CircadianConfig(
        sleep_mode="components",
        sleep_enable_chemical_reset=True,
        sleep_enable_replay=True,
        sleep_enable_homeostasis=False,
        sleep_enable_split=False,
        sleep_enable_prune=False,
        max_split_per_sleep=0,
        max_prune_per_sleep=0,
        sleep_warmup_steps=1,
        replay_steps=1,
        replay_memory_size=2,
    )
    model = CircadianPredictiveCodingNetwork(
        2, 3, seed=113, min_hidden_dim=3, circadian_config=config
    )
    features = np.array([[0.3, -0.2], [-0.1, 0.4]])
    targets = np.array([[1.0], [0.0]])
    initial = model.get_sleep_clocks()
    assert (initial.wake_batches, initial.wake_examples, initial.replay_updates) == (0, 0, 0)
    assert initial.wake_batches_since_sleep == initial.sleep_events == 0

    model.train_epoch(features, targets, 0.03, 2, 0.2)
    skipped = model.sleep_event(force_sleep=True, epoch_progress=SleepEpochProgress(1, 3))
    assert skipped.performed is False
    assert model.get_sleep_clocks().wake_batches_since_sleep == 1
    model.train_epoch(features, targets, 0.03, 2, 0.2)
    executed = model.sleep_event(force_sleep=True, epoch_progress=SleepEpochProgress(2, 3))
    assert executed.performed is True
    clocks = model.get_sleep_clocks()
    assert (clocks.wake_batches, clocks.wake_examples) == (2, 4)
    assert (clocks.replay_updates, clocks.sleep_events, clocks.wake_batches_since_sleep) == (
        1,
        1,
        0,
    )


def test_numpy_typed_progress_rejects_ambiguous_legacy_clock_arguments() -> None:
    config = replace(CircadianConfig.matched_pc_control(), sleep_mode="components")
    model = CircadianPredictiveCodingNetwork(
        2, 3, seed=127, min_hidden_dim=3, circadian_config=config
    )
    with pytest.raises(ValueError, match="epoch_progress"):
        model.sleep_event(
            current_step=1,
            total_steps=2,
            epoch_progress=SleepEpochProgress(1, 2),
        )


def test_numpy_adaptive_minimum_counts_wake_batches_within_one_runner_epoch() -> None:
    config = CircadianConfig(
        sleep_mode="components",
        max_split_per_sleep=0,
        max_prune_per_sleep=0,
        use_adaptive_sleep_trigger=True,
        min_epochs_between_sleep=2,
        sleep_energy_window=2,
        sleep_plateau_delta=1e9,
        sleep_chemical_variance_threshold=0.0,
    )
    model = CircadianPredictiveCodingNetwork(
        2, 4, seed=129, min_hidden_dim=4, circadian_config=config
    )
    features = np.array([[0.3, -0.2], [-0.1, 0.4]])
    targets = np.array([[1.0], [0.0]])
    model.train_epoch(features, targets, 0.03, 2, 0.2)
    assert model.should_trigger_sleep() is False
    model.train_epoch(features, targets, 0.03, 2, 0.2)
    assert model.should_trigger_sleep() is True
    event = model.sleep_event(force_sleep=False, epoch_progress=SleepEpochProgress(1, 3))
    assert event.performed is True
    assert model.get_sleep_clocks().wake_batches_since_sleep == 0
    assert model.should_trigger_sleep() is False


def test_numpy_legacy_clock_arguments_match_typed_epoch_progress() -> None:
    config = CircadianConfig(
        sleep_mode="components",
        max_split_per_sleep=0,
        max_prune_per_sleep=0,
        sleep_warmup_steps=1,
    )
    legacy_input = CircadianPredictiveCodingNetwork(
        2, 4, seed=133, min_hidden_dim=4, circadian_config=config
    )
    typed_input = CircadianPredictiveCodingNetwork(
        2, 4, seed=133, min_hidden_dim=4, circadian_config=config
    )
    first = legacy_input.sleep_event(force_sleep=True, current_step=2, total_steps=3)
    second = typed_input.sleep_event(force_sleep=True, epoch_progress=SleepEpochProgress(2, 3))
    assert first == second
    assert legacy_input.get_sleep_clocks() == typed_input.get_sleep_clocks()
    np.testing.assert_array_equal(
        legacy_input.get_chemical_state(), typed_input.get_chemical_state()
    )


def test_torch_clocks_count_wake_batches_and_restore_after_sleep() -> None:
    torch = pytest.importorskip("torch")
    from src.core.resnet50_variants import CircadianHeadConfig, CircadianPredictiveCodingHead

    config = CircadianHeadConfig(
        sleep_mode="components",
        sleep_enable_homeostasis=False,
        sleep_enable_split=False,
        sleep_enable_prune=False,
        max_split_per_sleep=0,
        max_prune_per_sleep=0,
    )
    head = CircadianPredictiveCodingHead(
        2, 3, 3, torch.device("cpu"), seed=131, min_hidden_dim=3, config=config
    )
    features = torch.tensor([[0.3, -0.2], [-0.1, 0.4]])
    targets = torch.tensor([1, 0], dtype=torch.int64)
    for _ in range(2):
        head.train_step(features, targets, 0.03, 2, 0.2)
    before = head.get_sleep_clocks()
    assert (before.wake_batches, before.wake_examples, before.wake_batches_since_sleep) == (
        2,
        4,
        2,
    )
    assert before.replay_updates == before.sleep_events == 0
    snapshot = head.snapshot_state()
    event = head.sleep_event(force_sleep=True, epoch_progress=SleepEpochProgress(1, 2))
    assert event.performed is True
    assert head.get_sleep_clocks().sleep_events == 1
    assert head.get_sleep_clocks().wake_batches_since_sleep == 0
    head.restore_state(snapshot)
    assert head.get_sleep_clocks() == before


def test_torch_typed_progress_rejects_ambiguous_legacy_clock_arguments() -> None:
    torch = pytest.importorskip("torch")
    from src.core.resnet50_variants import CircadianHeadConfig, CircadianPredictiveCodingHead

    head = CircadianPredictiveCodingHead(
        2, 3, 3, torch.device("cpu"), seed=137, min_hidden_dim=3, config=CircadianHeadConfig()
    )
    with pytest.raises(ValueError, match="epoch_progress"):
        head.sleep_event(
            current_step=1,
            total_steps=2,
            epoch_progress=SleepEpochProgress(1, 2),
        )


def test_torch_adaptive_minimum_counts_wake_batches_within_one_runner_epoch() -> None:
    torch = pytest.importorskip("torch")
    from src.core.resnet50_variants import CircadianHeadConfig, CircadianPredictiveCodingHead

    config = CircadianHeadConfig(
        sleep_mode="components",
        max_split_per_sleep=0,
        max_prune_per_sleep=0,
        use_adaptive_sleep_trigger=True,
        min_sleep_steps=2,
        sleep_energy_window=2,
        sleep_plateau_delta=1e9,
        sleep_chemical_variance_threshold=0.0,
    )
    head = CircadianPredictiveCodingHead(
        2, 4, 3, torch.device("cpu"), seed=139, min_hidden_dim=4, config=config
    )
    features = torch.tensor([[0.3, -0.2], [-0.1, 0.4]])
    targets = torch.tensor([1, 0], dtype=torch.int64)
    head.train_step(features, targets, 0.03, 2, 0.2)
    assert head.should_trigger_sleep() is False
    head.train_step(features, targets, 0.03, 2, 0.2)
    assert head.should_trigger_sleep() is True
    event = head.sleep_event(force_sleep=False, epoch_progress=SleepEpochProgress(1, 3))
    assert event.performed is True
    assert head.get_sleep_clocks().wake_batches_since_sleep == 0
    assert head.should_trigger_sleep() is False


def test_torch_legacy_clock_arguments_match_typed_epoch_progress() -> None:
    torch = pytest.importorskip("torch")
    from src.core.resnet50_variants import CircadianHeadConfig, CircadianPredictiveCodingHead

    config = CircadianHeadConfig(
        sleep_mode="components",
        max_split_per_sleep=0,
        max_prune_per_sleep=0,
        sleep_warmup_steps=1,
    )
    legacy_input = CircadianPredictiveCodingHead(
        2, 4, 3, torch.device("cpu"), seed=149, min_hidden_dim=4, config=config
    )
    typed_input = CircadianPredictiveCodingHead(
        2, 4, 3, torch.device("cpu"), seed=149, min_hidden_dim=4, config=config
    )
    first = legacy_input.sleep_event(force_sleep=True, current_step=2, total_steps=3)
    second = typed_input.sleep_event(force_sleep=True, epoch_progress=SleepEpochProgress(2, 3))
    assert first == second
    assert legacy_input.get_sleep_clocks() == typed_input.get_sleep_clocks()
    torch.testing.assert_close(legacy_input._chemical, typed_input._chemical)
