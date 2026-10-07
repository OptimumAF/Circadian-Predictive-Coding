"""Opt-in toy work limits stop at resumable, final-test-sealed boundaries."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from src.app import experiment_runner
from src.app.experiment_runner import ExperimentConfig, run_experiment
from src.app.toy_execution_budget import ToyExecutionBudget, ToyExecutionStopped
from src.infra.circadian_checkpoint_files import TrustedLocalToyCheckpointStore


def small_config(*, epochs: int = 2) -> ExperimentConfig:
    return ExperimentConfig(
        sample_count=80,
        hidden_dim=4,
        epoch_count=epochs,
        random_seed=29,
        pc_inference_steps=2,
        circadian_inference_steps=2,
        circadian_sleep_interval=0,
    )


def assert_same_scores_and_work(actual: Any, expected: Any) -> None:
    assert actual.protocol_id == expected.protocol_id
    assert actual.split_hashes == expected.split_hashes
    assert actual.training_order == expected.training_order
    for name in ("backprop", "predictive_coding", "circadian_predictive_coding"):
        left, right = getattr(actual, name), getattr(expected, name)
        assert left.loss_history == right.loss_history
        assert left.validation_accuracy == right.validation_accuracy
        assert left.test_accuracy == right.test_accuracy


@pytest.mark.parametrize("reversed_order", [False, True])
def test_should_stop_at_exact_total_wake_update_and_resume_same_result(
    tmp_path: Path, reversed_order: bool
) -> None:
    config = small_config()
    if reversed_order:
        config = replace(config, model_order=tuple(reversed(config.model_order)))
    expected = run_experiment(config)
    store = TrustedLocalToyCheckpointStore(tmp_path / "toy.checkpoint")

    with pytest.raises(ToyExecutionStopped) as caught:
        run_experiment(
            config,
            checkpoint_store=store,
            execution_budget=ToyExecutionBudget(max_training_updates=2),
        )

    stop = caught.value.stop
    assert (stop.status, stop.reason, stop.updates_completed) == (
        "incomplete",
        "max_training_updates",
        2,
    )
    assert stop.resumable
    assert stop.checkpoint_position == store.load().combined.position
    assert store.load().runner_config_digest == experiment_runner.toy_config_digest(config)
    assert (stop.checkpoint_position.stage, stop.checkpoint_position.next_batch_index) == (
        "wake",
        2,
    )
    with pytest.raises(ToyExecutionStopped) as unchanged:
        run_experiment(
            config,
            checkpoint_store=store,
            resume_from_checkpoint=True,
            execution_budget=ToyExecutionBudget(max_training_updates=2),
        )
    assert unchanged.value.stop.checkpoint_position == stop.checkpoint_position
    assert unchanged.value.stop.updates_completed == 2

    actual = run_experiment(
        config,
        checkpoint_store=store,
        resume_from_checkpoint=True,
        execution_budget=ToyExecutionBudget(max_training_updates=6),
    )
    assert_same_scores_and_work(actual, expected)


def test_should_allow_exact_full_update_limit_to_complete() -> None:
    config = small_config(epochs=1)

    actual = run_experiment(config, execution_budget=ToyExecutionBudget(max_training_updates=3))
    expected = run_experiment(config)

    assert_same_scores_and_work(actual, expected)


def test_should_choose_update_reason_when_both_limits_reach_update_boundary() -> None:
    samples = iter((0.0, 0.1, 1.0))

    with pytest.raises(ToyExecutionStopped) as caught:
        run_experiment(
            small_config(epochs=1),
            execution_budget=ToyExecutionBudget(max_training_updates=1, max_wall_seconds=1.0),
            execution_clock=lambda: next(samples),
        )

    assert caught.value.stop.reason == "max_training_updates"
    assert caught.value.stop.updates_completed == 1


def test_should_stop_at_wall_deadline_before_next_update_and_resume(tmp_path: Path) -> None:
    config = small_config()
    expected = run_experiment(config)
    store = TrustedLocalToyCheckpointStore(tmp_path / "wall.checkpoint")
    samples = iter((0.0, 0.1, 1.0))

    with pytest.raises(ToyExecutionStopped) as caught:
        run_experiment(
            config,
            checkpoint_store=store,
            execution_budget=ToyExecutionBudget(max_wall_seconds=1.0),
            execution_clock=lambda: next(samples),
        )

    stop = caught.value.stop
    assert (stop.reason, stop.updates_completed, stop.elapsed_seconds) == (
        "max_wall_seconds",
        1,
        1.0,
    )
    assert stop.checkpoint_position == store.load().combined.position
    assert_same_scores_and_work(
        run_experiment(config, checkpoint_store=store, resume_from_checkpoint=True), expected
    )


def test_should_stop_before_sleep_with_checked_cursor(tmp_path: Path) -> None:
    config = small_config(epochs=1)
    store = TrustedLocalToyCheckpointStore(tmp_path / "before-sleep.checkpoint")
    samples = iter((0.0, 0.1, 0.2, 0.3, 1.0))

    with pytest.raises(ToyExecutionStopped) as caught:
        run_experiment(
            config,
            checkpoint_store=store,
            execution_budget=ToyExecutionBudget(max_wall_seconds=1.0),
            execution_clock=lambda: next(samples),
        )

    assert caught.value.stop.updates_completed == 3
    assert caught.value.stop.checkpoint_position == store.load().combined.position
    assert caught.value.stop.checkpoint_position.stage == "before_sleep"
    assert_same_scores_and_work(
        run_experiment(config, checkpoint_store=store, resume_from_checkpoint=True),
        run_experiment(config),
    )


def test_should_stop_before_final_release_without_opening_final_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = small_config(epochs=1)
    store = TrustedLocalToyCheckpointStore(tmp_path / "after-sleep.checkpoint")
    original_split = experiment_runner.split_training_validation
    final_reads = 0

    class SealedFinal:
        @property
        def input(self) -> Any:
            nonlocal final_reads
            final_reads += 1
            raise AssertionError("final inputs opened before completed training")

        @property
        def target(self) -> Any:
            nonlocal final_reads
            final_reads += 1
            raise AssertionError("final labels opened before completed training")

    def sealed_split(*args: Any, **kwargs: Any) -> Any:
        roles = original_split(*args, **kwargs)
        return SimpleNamespace(
            train=roles.train,
            validation=roles.validation,
            test=SealedFinal(),
            split_hashes=roles.split_hashes,
        )

    monkeypatch.setattr(experiment_runner, "split_training_validation", sealed_split)
    samples = iter((0.0, 0.1, 0.2, 0.3, 0.4, 0.5))

    with pytest.raises(ToyExecutionStopped) as caught:
        run_experiment(
            config,
            checkpoint_store=store,
            execution_budget=ToyExecutionBudget(max_wall_seconds=0.5),
            execution_clock=lambda: next(samples),
        )

    assert caught.value.stop.reason == "max_wall_seconds"
    position = caught.value.stop.checkpoint_position
    assert position is not None and position.stage == "after_sleep"
    assert final_reads == 0


def test_should_explain_nonresumable_stop_without_checkpoint() -> None:
    config = small_config(epochs=1)

    with pytest.raises(ToyExecutionStopped) as caught:
        run_experiment(config, execution_budget=ToyExecutionBudget(max_training_updates=0))

    assert caught.value.stop.reason == "max_training_updates"
    assert caught.value.stop.updates_completed == 0
    assert not caught.value.stop.resumable
    assert caught.value.stop.checkpoint_position is None


def test_should_explain_zero_wall_time_stop_without_checkpoint() -> None:
    with pytest.raises(ToyExecutionStopped) as caught:
        run_experiment(
            small_config(epochs=1),
            execution_budget=ToyExecutionBudget(max_wall_seconds=0.0),
            execution_clock=lambda: 0.0,
        )

    assert caught.value.stop.reason == "max_wall_seconds"
    assert caught.value.stop.updates_completed == 0
    assert not caught.value.stop.resumable


@pytest.mark.parametrize(
    "budget",
    [
        ToyExecutionBudget(max_training_updates=1),
        ToyExecutionBudget(max_wall_seconds=1.0),
    ],
)
def test_should_reject_nonmonotonic_clock(budget: ToyExecutionBudget) -> None:
    samples = iter((2.0, 1.0))

    with pytest.raises(ValueError, match="clock"):
        run_experiment(
            small_config(epochs=1),
            execution_budget=budget,
            execution_clock=lambda: next(samples),
        )


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"max_training_updates": True},
        {"max_training_updates": -1},
        {"max_wall_seconds": True},
        {"max_wall_seconds": -1.0},
        {"max_wall_seconds": float("nan")},
        {"max_wall_seconds": float("inf")},
    ],
)
def test_should_reject_invalid_budget(settings: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="budget"):
        ToyExecutionBudget(**cast(Any, settings))


@pytest.mark.parametrize(
    "arguments",
    [
        {"execution_budget": object()},
        {"execution_clock": lambda: 0.0},
        {
            "execution_budget": ToyExecutionBudget(max_wall_seconds=1.0),
            "execution_clock": lambda: float("nan"),
        },
    ],
)
def test_should_reject_bad_execution_inputs_before_dataset(
    arguments: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        experiment_runner,
        "generate_two_cluster_dataset",
        lambda **_kwargs: pytest.fail("invalid input reached dataset construction"),
    )

    with pytest.raises(ValueError, match="execution"):
        run_experiment(small_config(), **cast(Any, arguments))
