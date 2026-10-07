from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from importlib import import_module
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.app import continual_shift_benchmark
from src.app.continual_shift_benchmark import (
    CONTINUAL_LEGACY_PROTOCOL,
    CONTINUAL_VALIDATION_PROTOCOL,
    ContinualShiftConfig,
    format_continual_shift_benchmark,
    run_continual_shift_benchmark,
)
from src.core.circadian_predictive_coding import CircadianConfig
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork

continual_script = import_module("scripts.run_continual_shift_benchmark")


def test_should_run_continual_shift_benchmark_and_return_metrics() -> None:
    config = ContinualShiftConfig(
        sample_count_phase_a=180,
        sample_count_phase_b=180,
        phase_b_train_fraction=0.20,
        hidden_dim=8,
        phase_a_epochs=25,
        phase_b_epochs=20,
        circadian_sleep_interval_phase_a=10,
        circadian_sleep_interval_phase_b=4,
        circadian_config=CircadianConfig(
            split_threshold=0.35,
            max_split_per_sleep=1,
            max_prune_per_sleep=0,
            replay_steps=1,
            replay_memory_size=4,
        ),
    )
    result = run_continual_shift_benchmark(config=config, seeds=[3, 7])

    assert result.seeds == [3, 7]
    assert len(result.seed_results) == 2
    assert result.aggregate.run_count == 2

    for seed_result in result.seed_results:
        assert 0.0 <= seed_result.backprop.phase_a_pre_accuracy <= 1.0
        assert 0.0 <= seed_result.backprop.phase_a_post_accuracy <= 1.0
        assert 0.0 <= seed_result.backprop.phase_b_post_accuracy <= 1.0
        assert seed_result.backprop.retention_ratio >= 0.0
        assert 0.0 <= seed_result.backprop.balanced_score <= 1.0

        assert 0.0 <= seed_result.predictive_coding.phase_a_pre_accuracy <= 1.0
        assert 0.0 <= seed_result.predictive_coding.phase_a_post_accuracy <= 1.0
        assert 0.0 <= seed_result.predictive_coding.phase_b_post_accuracy <= 1.0
        assert seed_result.predictive_coding.retention_ratio >= 0.0
        assert 0.0 <= seed_result.predictive_coding.balanced_score <= 1.0

        assert 0.0 <= seed_result.circadian_predictive_coding.phase_a_pre_accuracy <= 1.0
        assert 0.0 <= seed_result.circadian_predictive_coding.phase_a_post_accuracy <= 1.0
        assert 0.0 <= seed_result.circadian_predictive_coding.phase_b_post_accuracy <= 1.0
        assert seed_result.circadian_predictive_coding.retention_ratio >= 0.0
        assert 0.0 <= seed_result.circadian_predictive_coding.balanced_score <= 1.0
        assert seed_result.circadian_predictive_coding.hidden_dim_start == config.hidden_dim
        assert seed_result.circadian_predictive_coding.hidden_dim_end >= 4

    report_text = format_continual_shift_benchmark(result)
    assert "Continual Shift Benchmark" in report_text
    assert "Training order:" in report_text
    assert "Circadian predictive coding" in report_text


def test_should_fail_when_seeds_are_empty() -> None:
    with pytest.raises(ValueError, match="seeds cannot be empty"):
        run_continual_shift_benchmark(config=ContinualShiftConfig(), seeds=[])


def test_should_fail_for_invalid_phase_b_train_fraction() -> None:
    with pytest.raises(ValueError, match="phase_b_train_fraction"):
        run_continual_shift_benchmark(
            config=ContinualShiftConfig(phase_b_train_fraction=1.2),
            seeds=[7],
        )


def test_should_fail_when_hidden_dim_does_not_match_hidden_dims_tail() -> None:
    with pytest.raises(ValueError, match="hidden_dim must match the last value in hidden_dims"):
        run_continual_shift_benchmark(
            config=ContinualShiftConfig(hidden_dim=8, hidden_dims=(12, 10)),
            seeds=[7],
        )


def test_continual_test_scoring_waits_until_all_training_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ContinualShiftConfig(
        sample_count_phase_a=40,
        sample_count_phase_b=40,
        hidden_dim=4,
        phase_a_epochs=2,
        phase_b_epochs=2,
        circadian_sleep_interval_phase_a=0,
        circadian_sleep_interval_phase_b=0,
    )
    model_classes = (BackpropMLP, PredictiveCodingNetwork, CircadianPredictiveCodingNetwork)
    counts = {model_class.__name__: {"train": 0, "score": 0} for model_class in model_classes}
    for model_class in model_classes:
        train_epoch = model_class.train_epoch
        compute_accuracy = model_class.compute_accuracy
        calls = counts[model_class.__name__]

        def tracked_train(
            self: Any,
            *args: Any,
            _original: Any = train_epoch,
            _calls: dict[str, int] = calls,
            **kwargs: Any,
        ) -> Any:
            _calls["train"] += 1
            return _original(self, *args, **kwargs)

        def tracked_score(
            self: Any,
            *args: Any,
            _original: Any = compute_accuracy,
            _calls: dict[str, int] = calls,
            **kwargs: Any,
        ) -> float:
            assert all(
                model_calls["train"] == config.phase_a_epochs + config.phase_b_epochs
                for model_calls in counts.values()
            )
            _calls["score"] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(model_class, "train_epoch", tracked_train)
        monkeypatch.setattr(model_class, "compute_accuracy", tracked_score)

    result = run_continual_shift_benchmark(config=config, seeds=[7])
    assert len(result.seed_results) == 1
    assert all(calls == {"train": 4, "score": 3} for calls in counts.values())


def test_continual_final_test_labels_cannot_change_any_trained_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ContinualShiftConfig(
        sample_count_phase_a=80,
        sample_count_phase_b=80,
        hidden_dim=6,
        phase_a_epochs=2,
        phase_b_epochs=2,
        circadian_sleep_interval_phase_a=1,
        circadian_sleep_interval_phase_b=1,
        circadian_config=CircadianConfig(replay_steps=1, replay_memory_size=4),
    )
    generator = continual_shift_benchmark.generate_two_cluster_dataset_with_transform
    model_classes = {
        "BackpropMLP": BackpropMLP,
        "PredictiveCodingNetwork": PredictiveCodingNetwork,
        "CircadianPredictiveCodingNetwork": CircadianPredictiveCodingNetwork,
    }

    def run_with_shift(shift: bool) -> tuple[Any, dict[str, bytes]]:
        created: dict[str, Any] = {}
        for name, model_class in model_classes.items():

            def capture(
                *args: Any, _class: Any = model_class, _name: str = name, **kwargs: Any
            ) -> Any:
                model = _class(*args, **kwargs)
                created[_name] = model
                return model

            monkeypatch.setattr(continual_shift_benchmark, name, capture)

        def dataset_with_shift(*args: Any, **kwargs: Any) -> Any:
            split = generator(*args, **kwargs)
            return replace(split, test_target=1.0 - split.test_target) if shift else split

        monkeypatch.setattr(
            continual_shift_benchmark,
            "generate_two_cluster_dataset_with_transform",
            dataset_with_shift,
        )
        result = run_continual_shift_benchmark(config=config, seeds=[7])
        return result.seed_results[0], {key: pickle.dumps(model) for key, model in created.items()}

    original_result, original_models = run_with_shift(False)
    shifted_result, shifted_models = run_with_shift(True)
    assert original_models.keys() == shifted_models.keys() == model_classes.keys()
    assert original_models == shifted_models
    assert (
        original_result.split_hashes["phase_a_train"]
        == shifted_result.split_hashes["phase_a_train"]
    )
    assert (
        original_result.split_hashes["phase_b_validation"]
        == shifted_result.split_hashes["phase_b_validation"]
    )
    assert (
        original_result.split_hashes["phase_a_test"] != shifted_result.split_hashes["phase_a_test"]
    )
    assert (
        original_result.backprop.phase_a_pre_accuracy
        != shifted_result.backprop.phase_a_pre_accuracy
    )


@pytest.mark.parametrize(
    "order",
    [
        ("backprop", "predictive_coding", "circadian_predictive_coding"),
        ("circadian_predictive_coding", "predictive_coding", "backprop"),
    ],
)
def test_continual_training_boundaries_cannot_open_either_final_test(
    monkeypatch: pytest.MonkeyPatch,
    order: tuple[str, ...],
) -> None:
    original_split = continual_shift_benchmark.split_training_validation
    original_build_b = continual_shift_benchmark._build_phase_b_roles
    original_train_a = continual_shift_benchmark._train_phase_a_models
    original_train_b = continual_shift_benchmark._train_phase_b_models
    phase_a_complete = False
    training_complete = False
    final_reads = 0
    train_roles: list[Any] = []

    class SealedFinalTest:
        def __init__(self, actual: Any) -> None:
            self.actual = actual

        @property
        def input(self) -> Any:
            nonlocal final_reads
            if not training_complete:
                raise AssertionError("Continual training opened final-test inputs")
            final_reads += 1
            return self.actual.input

        @property
        def target(self) -> Any:
            nonlocal final_reads
            if not training_complete:
                raise AssertionError("Continual training opened final-test labels")
            final_reads += 1
            return self.actual.target

    def sealed_a(*args: Any, **kwargs: Any) -> Any:
        roles = original_split(*args, **kwargs)
        if phase_a_complete:
            return roles  # The B builder performs a second internal split before sealing.
        train_roles.append(roles.train)
        return SimpleNamespace(
            train=roles.train,
            validation=roles.validation,
            test=SealedFinalTest(roles.test),
            split_hashes=roles.split_hashes,
        )

    def sealed_b(*args: Any, **kwargs: Any) -> Any:
        assert phase_a_complete
        roles = original_build_b(*args, **kwargs)
        train_roles.append(roles.train)
        return SimpleNamespace(
            train=roles.train,
            validation=roles.validation,
            test=SealedFinalTest(roles.test),
            split_hashes=roles.split_hashes,
        )

    def train_a(**kwargs: Any) -> Any:
        nonlocal phase_a_complete
        assert set(kwargs) == {"config", "seed", "phase_a_train", "on_sleep_event"}
        assert callable(kwargs["on_sleep_event"])
        assert kwargs["phase_a_train"] is train_roles[0]
        state = original_train_a(**kwargs)
        phase_a_complete = True
        return state

    def train_b(**kwargs: Any) -> Any:
        nonlocal training_complete
        assert set(kwargs) == {"config", "phase_b_train", "state", "on_sleep_event"}
        assert callable(kwargs["on_sleep_event"])
        assert kwargs["phase_b_train"] is train_roles[1]
        state = original_train_b(**kwargs)
        training_complete = True
        return state

    monkeypatch.setattr(continual_shift_benchmark, "split_training_validation", sealed_a)
    monkeypatch.setattr(continual_shift_benchmark, "_build_phase_b_roles", sealed_b)
    monkeypatch.setattr(continual_shift_benchmark, "_train_phase_a_models", train_a)
    monkeypatch.setattr(continual_shift_benchmark, "_train_phase_b_models", train_b)

    result = run_continual_shift_benchmark(
        config=ContinualShiftConfig(
            sample_count_phase_a=40,
            sample_count_phase_b=40,
            hidden_dim=4,
            phase_a_epochs=2,
            phase_b_epochs=2,
            circadian_sleep_interval_phase_a=0,
            circadian_sleep_interval_phase_b=0,
            model_order=order,
        ),
        seeds=[7],
    )

    assert phase_a_complete and training_complete
    assert final_reads == 18
    assert len(result.seed_results) == 1


def test_continual_protocols_keep_legacy_route_and_hash_corrected_roles() -> None:
    config = ContinualShiftConfig(
        sample_count_phase_a=80,
        sample_count_phase_b=80,
        phase_b_train_fraction=0.5,
        hidden_dim=6,
        phase_a_epochs=2,
        phase_b_epochs=2,
        circadian_sleep_interval_phase_a=0,
        circadian_sleep_interval_phase_b=0,
    )
    corrected = run_continual_shift_benchmark(config=config, seeds=[7])
    repeated = run_continual_shift_benchmark(config=config, seeds=[7])
    legacy = run_continual_shift_benchmark(
        config=replace(config, protocol_id=CONTINUAL_LEGACY_PROTOCOL), seeds=[7]
    )

    assert corrected.config.protocol_id == CONTINUAL_VALIDATION_PROTOCOL
    assert corrected.seed_results[0].split_hashes == repeated.seed_results[0].split_hashes
    assert set(corrected.seed_results[0].split_hashes) == {
        "phase_a_train",
        "phase_a_validation",
        "phase_a_test",
        "phase_b_train",
        "phase_b_validation",
        "phase_b_test",
    }
    assert legacy.seed_results[0].split_hashes == {}
    phase_b_roles = continual_shift_benchmark._build_phase_b_roles(config, seed=108)
    phase_b_legacy = continual_shift_benchmark._build_phase_b_dataset(config, seed=108)
    assert phase_b_roles.train.input.shape[0] < phase_b_legacy.train_input.shape[0]
    assert (phase_b_roles.test.input == phase_b_legacy.test_input).all()
    assert "Protocol: continual_validation_v1" in format_continual_shift_benchmark(corrected)


def test_continual_reversal_preserves_both_phase_states_replay_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ContinualShiftConfig(
        sample_count_phase_a=80,
        sample_count_phase_b=80,
        phase_b_train_fraction=0.5,
        hidden_dim=4,
        phase_a_epochs=3,
        phase_b_epochs=3,
        pc_inference_steps=2,
        circadian_inference_steps=2,
        circadian_sleep_interval_phase_a=2,
        circadian_sleep_interval_phase_b=1,
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
    original_a = continual_shift_benchmark._train_phase_a_models
    original_b = continual_shift_benchmark._train_phase_b_models
    original_select = CircadianPredictiveCodingNetwork._select_replay_snapshots
    original_sleep = CircadianPredictiveCodingNetwork.sleep_event
    phase = "A"
    phase_a_hashes: dict[str, str] = {}
    final_hashes: dict[str, str] = {}
    replay_choices: list[tuple[str, tuple[tuple[float, str], ...]]] = []
    sleep_decisions: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    def state_hashes(state: Any, after_a: bool) -> dict[str, str]:
        suffix = "_after_a" if after_a else "_model"
        names = {
            "backprop": f"backprop{suffix}",
            "predictive_coding": f"predictive{suffix}",
            "circadian_predictive_coding": f"circadian{suffix}",
        }
        return {
            name: sha256(pickle.dumps(getattr(state, field), protocol=5)).hexdigest()
            for name, field in names.items()
        }

    def train_a(**kwargs: Any) -> Any:
        nonlocal phase, phase_a_hashes
        state = original_a(**kwargs)
        phase_a_hashes = state_hashes(state, after_a=True)
        phase = "B"
        return state

    def train_b(**kwargs: Any) -> Any:
        nonlocal final_hashes
        state = original_b(**kwargs)
        final_hashes = state_hashes(state, after_a=False)
        return state

    def select_replay(model: CircadianPredictiveCodingNetwork, replay_count: int) -> Any:
        chosen = original_select(model, replay_count)
        replay_choices.append(
            (
                phase,
                tuple(
                    (snapshot.priority, sha256(snapshot.input_batch.tobytes()).hexdigest())
                    for snapshot in chosen
                ),
            )
        )
        return chosen

    def sleep(model: CircadianPredictiveCodingNetwork, *args: Any, **kwargs: Any) -> Any:
        result = original_sleep(model, *args, **kwargs)
        sleep_decisions.append((phase, result.split_indices, result.pruned_indices))
        return result

    monkeypatch.setattr(continual_shift_benchmark, "_train_phase_a_models", train_a)
    monkeypatch.setattr(continual_shift_benchmark, "_train_phase_b_models", train_b)
    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "_select_replay_snapshots", select_replay)
    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "sleep_event", sleep)

    def run_order(order: tuple[str, ...]) -> tuple[Any, Any, Any, Any, Any]:
        nonlocal phase, phase_a_hashes, final_hashes
        phase = "A"
        phase_a_hashes = {}
        final_hashes = {}
        replay_choices.clear()
        sleep_decisions.clear()
        result = run_continual_shift_benchmark(
            config=replace(config, model_order=order), seeds=[13]
        )
        return (
            result.seed_results[0],
            tuple(replay_choices),
            tuple(sleep_decisions),
            dict(phase_a_hashes),
            dict(final_hashes),
        )

    forward = run_order(continual_shift_benchmark.CONTINUAL_MODEL_ORDER)
    _ = np.random.normal(size=257)
    reverse = run_order(tuple(reversed(continual_shift_benchmark.CONTINUAL_MODEL_ORDER)))

    assert forward[0].training_order == continual_shift_benchmark.CONTINUAL_MODEL_ORDER
    assert reverse[0].training_order == tuple(reversed(forward[0].training_order))
    assert forward[0].split_hashes == reverse[0].split_hashes
    assert {item[0] for item in forward[1]} == {"A", "B"}
    assert {item[0] for item in forward[2] if item[1]} == {"A", "B"}
    assert forward[1:] == reverse[1:]
    for name in ("backprop", "predictive_coding", "circadian_predictive_coding"):
        assert getattr(forward[0], name) == getattr(reverse[0], name)


def test_continual_rejects_invalid_order_before_data_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        continual_shift_benchmark,
        "generate_two_cluster_dataset_with_transform",
        lambda **kwargs: pytest.fail("Invalid order reached data loading"),
    )
    with pytest.raises(ValueError, match="permutation"):
        run_continual_shift_benchmark(
            config=ContinualShiftConfig(model_order=("backprop", "backprop", "predictive_coding")),
            seeds=[13],
        )


def test_continual_cli_refuses_existing_output_before_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    historical = tmp_path / "historical.txt"
    historical.write_text("historical", encoding="utf-8")
    args = SimpleNamespace(output_file=str(historical))
    monkeypatch.setattr(
        continual_script, "build_parser", lambda: SimpleNamespace(parse_args=lambda: args)
    )
    monkeypatch.setattr(
        continual_script,
        "run_continual_shift_benchmark",
        lambda **kwargs: pytest.fail("Training must not start before output preflight"),
    )

    with pytest.raises(FileExistsError, match="output already exists"):
        continual_script.main()
    assert historical.read_text(encoding="utf-8") == "historical"


def test_continual_adaptive_sleep_and_replay_use_only_seen_training_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase_train: list[Any] = []
    original_split = continual_shift_benchmark.split_training_validation
    original_phase_b = continual_shift_benchmark._build_phase_b_roles

    def capture_phase_a(*args: Any, **kwargs: Any) -> Any:
        roles = original_split(*args, **kwargs)
        if not phase_train:
            phase_train.append(roles.train)
        return roles

    def capture_phase_b(*args: Any, **kwargs: Any) -> Any:
        roles = original_phase_b(*args, **kwargs)
        phase_train.append(roles.train)
        return roles

    monkeypatch.setattr(continual_shift_benchmark, "split_training_validation", capture_phase_a)
    monkeypatch.setattr(continual_shift_benchmark, "_build_phase_b_roles", capture_phase_b)
    original_step = CircadianPredictiveCodingNetwork._run_training_step
    original_trigger = CircadianPredictiveCodingNetwork.should_trigger_sleep
    original_thresholds = CircadianPredictiveCodingNetwork._resolve_split_prune_thresholds
    calls = {"wake": 0, "replay": 0, "trigger": 0, "threshold": 0}

    def checked_step(
        self: Any, input_batch: Any, target_batch: Any, *args: Any, **kwargs: Any
    ) -> Any:
        allowed = phase_train[:1] if calls["wake"] < 2 else phase_train
        if kwargs["update_epoch_state"]:
            expected = phase_train[0] if calls["wake"] < 2 else phase_train[1]
            assert input_batch is expected.input
            assert target_batch is expected.target
            calls["wake"] += 1
        else:
            assert any(
                np.array_equal(input_batch, role.input)
                and np.array_equal(target_batch, role.target)
                for role in allowed
            )
            calls["replay"] += 1
        return original_step(self, input_batch, target_batch, *args, **kwargs)

    def checked_trigger(self: Any) -> bool:
        calls["trigger"] += 1
        return original_trigger(self)

    def checked_thresholds(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls["threshold"] += 1
        return original_thresholds(self, *args, **kwargs)

    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "_run_training_step", checked_step)
    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "should_trigger_sleep", checked_trigger)
    monkeypatch.setattr(
        CircadianPredictiveCodingNetwork,
        "_resolve_split_prune_thresholds",
        checked_thresholds,
    )
    config = ContinualShiftConfig(
        sample_count_phase_a=80,
        sample_count_phase_b=80,
        phase_b_train_fraction=0.5,
        hidden_dim=6,
        phase_a_epochs=2,
        phase_b_epochs=2,
        circadian_sleep_interval_phase_a=2,
        circadian_sleep_interval_phase_b=2,
        circadian_force_sleep=False,
        circadian_config=CircadianConfig(
            use_adaptive_sleep_trigger=True,
            min_epochs_between_sleep=0,
            sleep_energy_window=2,
            sleep_plateau_delta=1e6,
            sleep_chemical_variance_threshold=0.0,
            split_threshold=0.0,
            use_adaptive_thresholds=True,
            adaptive_split_percentile=0.0,
            max_split_per_sleep=1,
            max_prune_per_sleep=0,
            replay_steps=1,
            replay_memory_size=4,
        ),
    )
    run_continual_shift_benchmark(config=config, seeds=[7])
    assert len(phase_train) == 2
    assert calls["wake"] == 4
    assert calls["replay"] > 0
    assert calls["trigger"] > 0
    assert calls["threshold"] > 0
