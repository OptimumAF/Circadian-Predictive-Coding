from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.app import experiment_runner
from src.app.experiment_runner import (
    TOY_LEGACY_PROTOCOL,
    TOY_VALIDATION_PROTOCOL,
    ExperimentConfig,
    format_experiment_result,
    run_experiment,
)
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianConfig
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork


def test_should_run_experiment_and_return_reports() -> None:
    config = ExperimentConfig(
        sample_count=220,
        noise_scale=0.8,
        hidden_dim=8,
        epoch_count=60,
        random_seed=5,
    )
    result = run_experiment(config)

    assert len(result.backprop.loss_history) == config.epoch_count
    assert len(result.predictive_coding.loss_history) == config.epoch_count
    assert len(result.circadian_predictive_coding.loss_history) == config.epoch_count
    assert 0.0 <= result.backprop.test_accuracy <= 1.0
    assert 0.0 <= result.predictive_coding.test_accuracy <= 1.0
    assert 0.0 <= result.circadian_predictive_coding.test_accuracy <= 1.0
    assert len(result.backprop.traffic_by_layer) == 1
    assert len(result.predictive_coding.traffic_by_layer) == 1
    assert len(result.circadian_predictive_coding.traffic_by_layer) == 2
    assert result.circadian_sleep.hidden_dim_start == config.hidden_dim
    assert result.circadian_sleep.hidden_dim_end >= 4


def test_should_support_adaptive_circadian_configuration_in_experiment() -> None:
    config = ExperimentConfig(
        sample_count=220,
        noise_scale=0.8,
        hidden_dim=8,
        epoch_count=40,
        circadian_sleep_interval=5,
        circadian_force_sleep=False,
        circadian_config=CircadianConfig(
            use_adaptive_thresholds=True,
            adaptive_split_percentile=80.0,
            adaptive_prune_percentile=20.0,
            use_adaptive_sleep_trigger=True,
            min_epochs_between_sleep=3,
            sleep_energy_window=3,
            sleep_plateau_delta=1.0,
            sleep_chemical_variance_threshold=0.0,
            replay_steps=1,
            replay_memory_size=4,
        ),
        random_seed=9,
    )
    result = run_experiment(config)

    assert len(result.circadian_predictive_coding.loss_history) == config.epoch_count
    assert 0.0 <= result.circadian_predictive_coding.test_accuracy <= 1.0
    assert result.circadian_sleep.hidden_dim_end >= 4
    assert result.backprop.training_metric_id == "numpy_binary_bce_preupdate_v1"
    assert result.predictive_coding.training_metric_id == "numpy_pc_bce_plus_half_mean_all_hidden_error_sq_v1"
    assert result.circadian_predictive_coding.training_metric_id == "numpy_circadian_bce_plus_half_mean_final_hidden_error_sq_v1"
    assert "training diagnostic" in format_experiment_result(result)


def test_toy_validation_protocol_has_stable_roles_and_explicit_legacy_route() -> None:
    config = ExperimentConfig(sample_count=80, hidden_dim=6, epoch_count=3, random_seed=7)
    first = run_experiment(config)
    second = run_experiment(config)
    legacy = run_experiment(replace(config, protocol_id=TOY_LEGACY_PROTOCOL))

    assert first.protocol_id == TOY_VALIDATION_PROTOCOL
    assert first.split_hashes == second.split_hashes
    assert set(first.split_hashes) == {"train", "validation", "test"}
    assert first.backprop.validation_accuracy is not None
    assert first.predictive_coding.validation_accuracy is not None
    assert first.circadian_predictive_coding.validation_accuracy is not None
    assert legacy.protocol_id == TOY_LEGACY_PROTOCOL
    assert legacy.split_hashes == {}
    assert legacy.backprop.validation_accuracy is None
    assert legacy.predictive_coding.validation_accuracy is None
    assert legacy.circadian_predictive_coding.validation_accuracy is None


def test_toy_reversal_preserves_replay_splits_and_trained_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ExperimentConfig(
        sample_count=80,
        hidden_dim=4,
        epoch_count=6,
        random_seed=29,
        pc_inference_steps=2,
        circadian_inference_steps=2,
        circadian_sleep_interval=2,
        circadian_config=CircadianConfig(
            split_threshold=0.0,
            max_split_per_sleep=1,
            max_prune_per_sleep=0,
            split_noise_scale=0.05,
            replay_steps=2,
            replay_memory_size=4,
            replay_prioritized=True,
            replay_class_balanced=True,
        ),
    )
    original_classes = {
        "backprop": BackpropMLP,
        "predictive_coding": PredictiveCodingNetwork,
        "circadian_predictive_coding": CircadianPredictiveCodingNetwork,
    }
    original_select = CircadianPredictiveCodingNetwork._select_replay_snapshots
    original_sleep = CircadianPredictiveCodingNetwork.sleep_event
    models: dict[str, Any] = {}
    replay_choices: list[tuple[float, ...]] = []
    sleep_decisions: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def capture(name: str, model_class: Any) -> Any:
        def build(*args: Any, **kwargs: Any) -> Any:
            model = model_class(*args, **kwargs)
            models[name] = model
            return model

        return build

    for name, model_class in original_classes.items():
        class_name = model_class.__name__
        monkeypatch.setattr(experiment_runner, class_name, capture(name, model_class))

    def select_replay(
        model: CircadianPredictiveCodingNetwork, replay_count: int
    ) -> Any:
        chosen = original_select(model, replay_count)
        replay_choices.append(tuple(snapshot.priority for snapshot in chosen))
        return chosen

    def sleep(model: CircadianPredictiveCodingNetwork, *args: Any, **kwargs: Any) -> Any:
        result = original_sleep(model, *args, **kwargs)
        sleep_decisions.append((result.split_indices, result.pruned_indices))
        return result

    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "_select_replay_snapshots", select_replay)
    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "sleep_event", sleep)

    def run_order(order: tuple[str, ...]) -> tuple[Any, Any, Any, dict[str, str]]:
        models.clear()
        replay_choices.clear()
        sleep_decisions.clear()
        result = run_experiment(replace(config, model_order=order))
        hashes = {
            name: sha256(pickle.dumps(model, protocol=5)).hexdigest()
            for name, model in models.items()
        }
        return result, tuple(replay_choices), tuple(sleep_decisions), hashes

    forward = run_order(experiment_runner.TOY_MODEL_ORDER)
    _ = np.random.normal(size=257)
    reverse = run_order(tuple(reversed(experiment_runner.TOY_MODEL_ORDER)))

    assert forward[0].training_order == experiment_runner.TOY_MODEL_ORDER
    assert reverse[0].training_order == tuple(reversed(forward[0].training_order))
    assert forward[0].split_hashes == reverse[0].split_hashes
    assert forward[1] and all(len(choice) == 2 for choice in forward[1])
    assert any(split for split, _ in forward[2])
    assert forward[1:] == reverse[1:]
    assert forward[0].circadian_sleep == reverse[0].circadian_sleep
    for name in original_classes:
        left = getattr(forward[0], name)
        right = getattr(reverse[0], name)
        assert left.loss_history == right.loss_history
        assert left.validation_accuracy == right.validation_accuracy
        assert left.test_accuracy == right.test_accuracy


def test_toy_rejects_invalid_order_before_loading_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        experiment_runner,
        "generate_two_cluster_dataset",
        lambda **kwargs: pytest.fail("Invalid model order reached data loading"),
    )
    with pytest.raises(ValueError, match="permutation"):
        run_experiment(
            ExperimentConfig(model_order=("backprop", "backprop", "predictive_coding"))
        )


def test_toy_final_test_labels_cannot_change_training_or_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ExperimentConfig(sample_count=80, hidden_dim=6, epoch_count=3, random_seed=11)
    generator = experiment_runner.generate_two_cluster_dataset
    model_classes = {
        "BackpropMLP": BackpropMLP,
        "PredictiveCodingNetwork": PredictiveCodingNetwork,
        "CircadianPredictiveCodingNetwork": CircadianPredictiveCodingNetwork,
    }

    def run_with_shift(shift: bool) -> tuple[Any, dict[str, bytes]]:
        created: dict[str, Any] = {}
        for name, model_class in model_classes.items():
            def capture(*args: Any, _class: Any = model_class,
                        _name: str = name, **kwargs: Any) -> Any:
                model = _class(*args, **kwargs)
                created[_name] = model
                return model
            monkeypatch.setattr(experiment_runner, name, capture)

        def dataset_with_shift(*args: Any, **kwargs: Any) -> Any:
            dataset = generator(*args, **kwargs)
            return replace(dataset, test_target=1.0 - dataset.test_target) if shift else dataset

        monkeypatch.setattr(experiment_runner, "generate_two_cluster_dataset", dataset_with_shift)
        result = run_experiment(config)
        return result, {name: pickle.dumps(model) for name, model in created.items()}

    original, original_models = run_with_shift(False)
    changed, changed_models = run_with_shift(True)
    assert original_models == changed_models
    assert original.split_hashes["train"] == changed.split_hashes["train"]
    assert original.split_hashes["validation"] == changed.split_hashes["validation"]
    assert original.split_hashes["test"] != changed.split_hashes["test"]
    for model_name in ("backprop", "predictive_coding", "circadian_predictive_coding"):
        original_report = getattr(original, model_name)
        changed_report = getattr(changed, model_name)
        assert original_report.loss_history == changed_report.loss_history
        assert original_report.validation_accuracy == changed_report.validation_accuracy


def test_toy_training_boundary_cannot_open_final_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_split = experiment_runner.split_training_validation
    original_train = experiment_runner._train_toy_models
    training_complete = False
    final_reads = 0
    train_role: Any = None

    class SealedFinalTest:
        def __init__(self, actual: Any) -> None:
            self.actual = actual

        @property
        def input(self) -> Any:
            nonlocal final_reads
            if not training_complete:
                raise AssertionError("Toy training opened final-test inputs")
            final_reads += 1
            return self.actual.input

        @property
        def target(self) -> Any:
            nonlocal final_reads
            if not training_complete:
                raise AssertionError("Toy training opened final-test labels")
            final_reads += 1
            return self.actual.target

    def sealed_split(*args: Any, **kwargs: Any) -> Any:
        nonlocal train_role
        roles = original_split(*args, **kwargs)
        train_role = roles.train
        return SimpleNamespace(
            train=roles.train, validation=roles.validation,
            test=SealedFinalTest(roles.test), split_hashes=roles.split_hashes,
        )

    def train_without_test(**kwargs: Any) -> Any:
        nonlocal training_complete
        assert set(kwargs) == {"config", "policy", "train_data", "validation_data"}
        assert kwargs["train_data"] is train_role
        outcome = original_train(**kwargs)
        training_complete = True
        return outcome

    monkeypatch.setattr(experiment_runner, "split_training_validation", sealed_split)
    monkeypatch.setattr(experiment_runner, "_train_toy_models", train_without_test)

    result = run_experiment(
        ExperimentConfig(sample_count=80, hidden_dim=6, epoch_count=2, random_seed=23)
    )

    assert training_complete
    assert final_reads == 6
    assert 0.0 <= result.backprop.test_accuracy <= 1.0


def test_toy_wake_replay_and_adaptive_sleep_only_use_training_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roles_seen: list[Any] = []
    original_split = experiment_runner.split_training_validation

    def capture_split(*args: Any, **kwargs: Any) -> Any:
        roles = original_split(*args, **kwargs)
        roles_seen.append(roles)
        return roles

    monkeypatch.setattr(experiment_runner, "split_training_validation", capture_split)
    wake_calls = {"BackpropMLP": 0, "PredictiveCodingNetwork": 0}
    for model_class in (BackpropMLP, PredictiveCodingNetwork):
        original_train = model_class.train_epoch
        name = model_class.__name__

        def checked_train(self: Any, input_batch: Any, target_batch: Any,
                          *args: Any, _original: Any = original_train,
                          _name: str = name, **kwargs: Any) -> Any:
            assert input_batch is roles_seen[0].train.input
            assert target_batch is roles_seen[0].train.target
            wake_calls[_name] += 1
            return _original(self, input_batch, target_batch, *args, **kwargs)

        monkeypatch.setattr(model_class, "train_epoch", checked_train)

    step_calls = {"wake": 0, "replay": 0}
    threshold_calls = 0
    original_step = CircadianPredictiveCodingNetwork._run_training_step
    original_thresholds = CircadianPredictiveCodingNetwork._resolve_split_prune_thresholds

    def checked_step(self: Any, input_batch: Any, target_batch: Any,
                     *args: Any, **kwargs: Any) -> Any:
        if kwargs["update_epoch_state"]:
            assert input_batch is roles_seen[0].train.input
            assert target_batch is roles_seen[0].train.target
            step_calls["wake"] += 1
        else:
            assert np.array_equal(input_batch, roles_seen[0].train.input)
            assert np.array_equal(target_batch, roles_seen[0].train.target)
            step_calls["replay"] += 1
        return original_step(self, input_batch, target_batch, *args, **kwargs)

    def checked_thresholds(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal threshold_calls
        threshold_calls += 1
        return original_thresholds(self, *args, **kwargs)

    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "_run_training_step", checked_step)
    monkeypatch.setattr(
        CircadianPredictiveCodingNetwork, "_resolve_split_prune_thresholds",
        checked_thresholds,
    )
    config = ExperimentConfig(
        sample_count=80, hidden_dim=6, epoch_count=2, circadian_sleep_interval=1,
        circadian_config=CircadianConfig(
            split_threshold=0.0, use_adaptive_thresholds=True,
            adaptive_split_percentile=0.0, max_split_per_sleep=1,
            max_prune_per_sleep=0, replay_steps=1, replay_memory_size=4,
        ),
    )
    run_experiment(config)
    assert wake_calls == {"BackpropMLP": 2, "PredictiveCodingNetwork": 2}
    assert step_calls["wake"] == 2
    assert step_calls["replay"] > 0
    assert threshold_calls > 0
