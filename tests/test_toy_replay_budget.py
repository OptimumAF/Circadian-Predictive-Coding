"""A total replay-example ceiling stops before a transactional toy sleep."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from src.app import experiment_runner
from src.app.experiment_runner import ExperimentConfig, run_experiment
from src.app.toy_execution_budget import ToyExecutionBudget, ToyExecutionStopped
from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
    ReplaySnapshot,
    SleepReplayLimitExceeded,
)
from src.infra.circadian_checkpoint_files import TrustedLocalToyCheckpointStore


def _replay_config(epochs: int = 2) -> ExperimentConfig:
    return ExperimentConfig(
        sample_count=80,
        hidden_dim=4,
        epoch_count=epochs,
        pc_inference_steps=2,
        circadian_inference_steps=2,
        circadian_sleep_interval=1,
        random_seed=29,
        circadian_config=CircadianConfig(
            sleep_mode="components",
            sleep_enable_split=False,
            sleep_enable_prune=False,
            max_split_per_sleep=0,
            max_prune_per_sleep=0,
            replay_steps=1,
            replay_memory_size=2,
            replay_prioritized=False,
        ),
    )


def _assert_same_scientific_result(actual: Any, expected: Any) -> None:
    assert actual.protocol_id == expected.protocol_id
    assert actual.split_hashes == expected.split_hashes
    assert actual.training_order == expected.training_order
    for name in ("backprop", "predictive_coding", "circadian_predictive_coding"):
        left, right = getattr(actual, name), getattr(expected, name)
        assert left.loss_history == right.loss_history
        assert left.validation_accuracy == right.validation_accuracy
        assert left.test_accuracy == right.test_accuracy
    left_sleep, right_sleep = actual.circadian_sleep, expected.circadian_sleep
    assert (
        left_sleep.event_count,
        left_sleep.total_splits,
        left_sleep.total_prunes,
        left_sleep.hidden_dim_end,
    ) == (
        right_sleep.event_count,
        right_sleep.total_splits,
        right_sleep.total_prunes,
        right_sleep.hidden_dim_end,
    )
    assert [event.replay for event in left_sleep.events] == [
        event.replay for event in right_sleep.events
    ]


def test_should_stop_before_first_replay_without_opening_final_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _replay_config(epochs=1)
    store = TrustedLocalToyCheckpointStore(tmp_path / "replay.checkpoint")
    original_split = experiment_runner.split_training_validation
    final_reads = 0

    class SealedFinal:
        @property
        def input(self) -> Any:
            nonlocal final_reads
            final_reads += 1
            raise AssertionError("replay stop opened final input")

        @property
        def target(self) -> Any:
            nonlocal final_reads
            final_reads += 1
            raise AssertionError("replay stop opened final labels")

    def sealed_split(*args: Any, **kwargs: Any) -> Any:
        roles = original_split(*args, **kwargs)
        return SimpleNamespace(
            train=roles.train,
            validation=roles.validation,
            test=SealedFinal(),
            split_hashes=roles.split_hashes,
        )

    with monkeypatch.context() as scoped:
        scoped.setattr(experiment_runner, "split_training_validation", sealed_split)
        with pytest.raises(ToyExecutionStopped) as stopped:
            run_experiment(
                config,
                checkpoint_store=store,
                execution_budget=ToyExecutionBudget(max_replay_examples=0),
            )

    stop = stopped.value.stop
    assert (stop.reason, stop.updates_completed, stop.replay_examples_completed) == (
        "max_replay_examples",
        3,
        0,
    )
    assert stop.checkpoint_position == store.load().combined.position
    assert stop.checkpoint_position is not None
    assert stop.checkpoint_position.stage == "before_sleep"
    assert store.load().sleep_events == ()
    assert final_reads == 0
    _assert_same_scientific_result(
        run_experiment(
            config,
            checkpoint_store=store,
            resume_from_checkpoint=True,
            execution_budget=ToyExecutionBudget(max_replay_examples=52),
        ),
        run_experiment(config),
    )


def test_should_count_applied_replay_across_checked_resume(tmp_path: Path) -> None:
    config = _replay_config()
    expected = run_experiment(config)
    assert [event.replay.applied_examples for event in expected.circadian_sleep.events] == [
        52,
        52,
    ]
    store = TrustedLocalToyCheckpointStore(tmp_path / "replay.checkpoint")

    with pytest.raises(ToyExecutionStopped) as stopped:
        run_experiment(
            config,
            checkpoint_store=store,
            execution_budget=ToyExecutionBudget(max_replay_examples=52),
        )

    stop = stopped.value.stop
    assert (stop.reason, stop.updates_completed, stop.replay_examples_completed) == (
        "max_replay_examples",
        6,
        52,
    )
    checkpoint = store.load()
    assert checkpoint.combined.position.stage == "before_sleep"
    assert checkpoint.combined.position.completed_epoch == 2
    assert [event.replay.applied_examples for event in checkpoint.sleep_events] == [52]

    actual = run_experiment(
        config,
        checkpoint_store=store,
        resume_from_checkpoint=True,
        execution_budget=ToyExecutionBudget(max_replay_examples=104),
    )
    _assert_same_scientific_result(actual, expected)


@pytest.mark.parametrize(
    "mode", ["no_replay", "component_replay_off", "disabled", "legacy_zero_structure"]
)
def test_should_allow_zero_cap_when_sleep_applies_no_replay(mode: str) -> None:
    config = _replay_config(epochs=1)
    circadian = config.circadian_config
    assert circadian is not None
    if mode == "no_replay":
        circadian = replace(circadian, replay_steps=0)
    elif mode == "component_replay_off":
        circadian = replace(circadian, sleep_enable_replay=False)
    elif mode == "disabled":
        circadian = replace(circadian, sleep_mode="disabled")
    else:
        circadian = replace(
            circadian, sleep_mode="legacy", sleep_enable_split=True, sleep_enable_prune=True
        )
    config = replace(config, circadian_config=circadian)

    actual = run_experiment(config, execution_budget=ToyExecutionBudget(max_replay_examples=0))

    _assert_same_scientific_result(actual, run_experiment(config))
    assert sum(event.replay.applied_examples for event in actual.circadian_sleep.events) == 0


def test_core_should_count_selected_batch_lengths_before_any_sleep_mutation() -> None:
    config = _replay_config(epochs=1).circadian_config
    assert config is not None
    config = replace(config, replay_steps=2)
    model = CircadianPredictiveCodingNetwork(2, 4, seed=37, circadian_config=config)
    features = np.array([[0.3, -0.2], [-0.1, 0.4], [0.2, 0.1]])
    targets = np.array([[1.0], [0.0], [1.0]])
    model.train_epoch(features[:2], targets[:2], 0.03, 2, 0.2)
    model.train_epoch(features, targets, 0.03, 2, 0.2)
    before_clocks = model.get_sleep_clocks()
    before_weights = model.weight_input_hidden.copy()

    with pytest.raises(SleepReplayLimitExceeded) as caught:
        model.sleep_event(force_sleep=True, max_replay_examples=4)

    assert (caught.value.planned_examples, caught.value.remaining_examples) == (5, 4)
    assert model.get_sleep_clocks() == before_clocks
    np.testing.assert_array_equal(model.weight_input_hidden, before_weights)
    applied = model.sleep_event(force_sleep=True, max_replay_examples=5)
    assert applied.telemetry is not None
    assert applied.telemetry.replay.applied_examples == 5


def test_core_should_preflight_the_selected_priority_batch_size() -> None:
    config = _replay_config(epochs=1).circadian_config
    assert config is not None
    model = CircadianPredictiveCodingNetwork(
        2, 4, seed=37, circadian_config=replace(config, replay_prioritized=True)
    )
    features = np.array([[0.3, -0.2], [-0.1, 0.4], [0.2, 0.1]])
    targets = np.array([[1.0], [0.0], [1.0]])
    model._replay_memory.extend(
        (
            ReplaySnapshot(features, targets, priority=10.0, positive_fraction=2 / 3),
            ReplaySnapshot(features[:2], targets[:2], priority=1.0, positive_fraction=0.5),
        )
    )

    with pytest.raises(SleepReplayLimitExceeded) as caught:
        model.sleep_event(max_replay_examples=2)

    assert caught.value.planned_examples == 3
    applied = model.sleep_event(max_replay_examples=3)
    assert applied.telemetry is not None
    assert applied.telemetry.replay.applied_examples == 3


def test_should_refuse_resume_with_cap_below_checked_replay_progress(tmp_path: Path) -> None:
    config = _replay_config()
    store = TrustedLocalToyCheckpointStore(tmp_path / "replay.checkpoint")
    with pytest.raises(ToyExecutionStopped):
        run_experiment(
            config,
            checkpoint_store=store,
            execution_budget=ToyExecutionBudget(max_replay_examples=52),
        )
    before = store.path.read_bytes()

    with pytest.raises(ToyExecutionStopped) as stopped:
        run_experiment(
            config,
            checkpoint_store=store,
            resume_from_checkpoint=True,
            execution_budget=ToyExecutionBudget(max_replay_examples=51),
        )

    assert (stopped.value.stop.reason, stopped.value.stop.replay_examples_completed) == (
        "max_replay_examples",
        52,
    )
    assert stopped.value.stop.checkpoint_position == store.load().combined.position
    assert store.path.read_bytes() == before


def test_should_reject_invalid_checkpoint_replay_count_before_resume(tmp_path: Path) -> None:
    config = _replay_config()
    store = TrustedLocalToyCheckpointStore(tmp_path / "replay.checkpoint")
    with pytest.raises(ToyExecutionStopped):
        run_experiment(
            config,
            checkpoint_store=store,
            execution_budget=ToyExecutionBudget(max_replay_examples=52),
        )
    checkpoint = store.load()
    object.__setattr__(checkpoint.sleep_events[0].replay, "applied_examples", -1)
    store.save(checkpoint)

    with pytest.raises(ValueError, match="incompatible toy checkpoint replay usage"):
        run_experiment(
            config,
            checkpoint_store=store,
            resume_from_checkpoint=True,
            execution_budget=ToyExecutionBudget(max_replay_examples=104),
        )


@pytest.mark.parametrize("invalid", [True, -1, 1.5, float("nan")])
def test_should_reject_invalid_replay_caps_before_dataset(
    invalid: object,
) -> None:
    with pytest.raises(ValueError, match="max_replay_examples"):
        ToyExecutionBudget(max_replay_examples=cast(Any, invalid))
