"""Phase-label arrival canaries for the corrected continual benchmark.

The checkpointed path still binds both phase roles before Phase A. These
tests make that limit visible without changing the offline protocol.
"""

from __future__ import annotations

from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.app import continual_shift_benchmark as continual
from src.app.continual_shift_benchmark import ContinualShiftConfig
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from src.infra import datasets as dataset_roles
from src.infra.circadian_checkpoint_files import TrustedLocalContinualCheckpointStore
from src.infra.datasets import LabeledData, make_role_separated_dataset


class FuturePhaseUnavailable(Exception):
    """The experimental canary detected Phase B source access before arrival."""


class PhaseAInterruption(Exception):
    """Stop after one committed Phase A model update."""


class InterruptingCheckpointStore:
    def __init__(
        self,
        store: TrustedLocalContinualCheckpointStore,
        *,
        phase: str,
        stage: str,
        epoch: int,
        next_model: int,
    ) -> None:
        self.store = store
        self.phase = phase
        self.stage = stage
        self.epoch = epoch
        self.next_model = next_model
        self.interrupted = False

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        self.store.save(checkpoint)
        if (
            not self.interrupted
            and checkpoint.phase == self.phase
            and checkpoint.phase_epoch_completed == self.epoch
            and checkpoint.combined.position.stage == self.stage
            and checkpoint.combined.position.next_batch_index == self.next_model
        ):
            self.interrupted = True
            raise PhaseAInterruption()


def _tiny_config() -> ContinualShiftConfig:
    return ContinualShiftConfig(
        sample_count_phase_a=40,
        sample_count_phase_b=40,
        hidden_dim=4,
        phase_a_epochs=2,
        phase_b_epochs=2,
        pc_inference_steps=2,
        circadian_inference_steps=2,
        circadian_sleep_interval_phase_a=0,
        circadian_sleep_interval_phase_b=0,
    )


def _assert_same_checkpoint_state(actual: Any, expected: Any) -> None:
    for name in (
        "backprop_model",
        "predictive_model",
        "backprop_after_a",
        "predictive_after_a",
    ):
        assert pickle.dumps(getattr(actual, name)) == pickle.dumps(getattr(expected, name))
    for name in ("sleep_event_count", "total_splits", "total_prunes", "hidden_dim_start"):
        assert getattr(actual, name) == getattr(expected, name)
    left = actual.circadian_after_a
    right = expected.circadian_after_a
    assert left is not None and right is not None
    assert left.format_version == right.format_version
    assert left.config == right.config
    assert left.state.keys() == right.state.keys()
    for name in left.state:
        left_value, right_value = left.state[name], right.state[name]
        if name == "_replay_memory":
            assert len(left_value) == len(right_value)
            for left_sample, right_sample in zip(left_value, right_value, strict=True):
                np.testing.assert_array_equal(left_sample.input_batch, right_sample.input_batch)
                np.testing.assert_array_equal(left_sample.target_batch, right_sample.target_batch)
                assert left_sample.priority == right_sample.priority
                assert left_sample.positive_fraction == right_sample.positive_fraction
        elif isinstance(left_value, np.ndarray):
            np.testing.assert_array_equal(left_value, right_value, err_msg=name)
        else:
            assert pickle.dumps(left_value) == pickle.dumps(right_value), name


@pytest.mark.parametrize("reverse_order", [False, True])
def test_ordinary_continual_phase_b_source_arrives_after_all_phase_a_training(
    monkeypatch: pytest.MonkeyPatch, reverse_order: bool
) -> None:
    config = _tiny_config()
    if reverse_order:
        config = replace(config, model_order=tuple(reversed(config.model_order)))
    model_classes = (BackpropMLP, PredictiveCodingNetwork, CircadianPredictiveCodingNetwork)
    phase_a_updates = {model_class.__name__: 0 for model_class in model_classes}
    phase_b_source_calls = 0
    phase_a_finished = False

    for model_class in model_classes:
        original_train = model_class.train_epoch

        def count_train(
            self: Any,
            *args: Any,
            _name: str = model_class.__name__,
            _original: Any = original_train,
            **kwargs: Any,
        ) -> Any:
            if not phase_a_finished:
                phase_a_updates[_name] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(model_class, "train_epoch", count_train)

    original_train_a = continual._train_phase_a_models
    original_source_b = continual._generate_phase_b_source

    def finish_phase_a(**kwargs: Any) -> Any:
        nonlocal phase_a_finished
        trained = original_train_a(**kwargs)
        assert all(count == config.phase_a_epochs for count in phase_a_updates.values())
        phase_a_finished = True
        return trained

    def require_arrival(*args: Any, **kwargs: Any) -> Any:
        nonlocal phase_b_source_calls
        if not phase_a_finished:
            raise FuturePhaseUnavailable("Phase B source requested during Phase A")
        phase_b_source_calls += 1
        return original_source_b(*args, **kwargs)

    monkeypatch.setattr(continual, "_train_phase_a_models", finish_phase_a)
    monkeypatch.setattr(continual, "_generate_phase_b_source", require_arrival)
    result = continual.run_continual_shift_benchmark(config, seeds=[7])

    assert phase_a_finished
    assert phase_b_source_calls == 1
    assert phase_a_updates == {name: config.phase_a_epochs for name in phase_a_updates}
    assert result.seed_results[0].training_order == config.model_order


def test_checkpointed_continual_phase_b_source_is_currently_requested_before_phase_a(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    phase_a_training_started = False
    original_train_phase = continual._train_checkpoint_phase
    original_source_b = continual._generate_phase_b_source

    def track_phase_training(*args: Any, **kwargs: Any) -> Any:
        nonlocal phase_a_training_started
        if kwargs["phase"] == "a":
            phase_a_training_started = True
        return original_train_phase(*args, **kwargs)

    def require_arrival(*args: Any, **kwargs: Any) -> Any:
        if not phase_a_training_started:
            raise FuturePhaseUnavailable("Phase B source requested before Phase A training")
        return original_source_b(*args, **kwargs)

    monkeypatch.setattr(continual, "_train_checkpoint_phase", track_phase_training)
    monkeypatch.setattr(continual, "_generate_phase_b_source", require_arrival)
    store = TrustedLocalContinualCheckpointStore(tmp_path / "arrival.ckpt")

    # Why this: a passing expected-failure test records a real boundary gap.
    # P1.3b must change this assertion when phase-specific identity is added.
    with pytest.raises(FuturePhaseUnavailable, match="before Phase A training"):
        continual.run_continual_shift_benchmark(_tiny_config(), seeds=[7], checkpoint_store=store)
    assert not phase_a_training_started
    assert not (tmp_path / "arrival.ckpt").exists()


@pytest.mark.parametrize("reverse_order", [False, True])
def test_phase_arrival_checkpoint_resumes_phase_a_without_future_phase_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, reverse_order: bool
) -> None:
    config = replace(_tiny_config(), protocol_id="continual_phase_arrival_v2")
    if reverse_order:
        config = replace(config, model_order=tuple(reversed(config.model_order)))
    expected_store = TrustedLocalContinualCheckpointStore(tmp_path / "expected.ckpt")
    expected = continual.run_continual_shift_benchmark(
        config, seeds=[7], checkpoint_store=expected_store
    )

    store = TrustedLocalContinualCheckpointStore(tmp_path / "resumed.ckpt")
    interrupting = InterruptingCheckpointStore(
        store, phase="a", stage="wake", epoch=0, next_model=1
    )
    original_source_b = continual._generate_phase_b_source
    original_train_phase = continual._train_checkpoint_phase
    phase_a_complete = False
    phase_b_source_calls = 0

    def require_arrival(*args: Any, **kwargs: Any) -> Any:
        nonlocal phase_b_source_calls
        if not phase_a_complete:
            raise FuturePhaseUnavailable("Phase B source requested before Phase A finished")
        phase_b_source_calls += 1
        return original_source_b(*args, **kwargs)

    def track_phase_training(*args: Any, **kwargs: Any) -> Any:
        nonlocal phase_a_complete
        trained = original_train_phase(*args, **kwargs)
        if kwargs["phase"] == "a":
            phase_a_complete = True
        return trained

    monkeypatch.setattr(continual, "_generate_phase_b_source", require_arrival)
    monkeypatch.setattr(continual, "_train_checkpoint_phase", track_phase_training)

    with pytest.raises(PhaseAInterruption):
        continual.run_continual_shift_benchmark(config, seeds=[7], checkpoint_store=interrupting)
    assert interrupting.interrupted
    assert phase_b_source_calls == 0
    checkpoint = store.load()
    assert checkpoint.format_version == 2
    assert checkpoint.phase == "a"
    assert {role for role, _ in checkpoint.split_hashes} == {
        "phase_a_train",
        "phase_a_validation",
    }

    original_generate = continual.generate_two_cluster_dataset_with_transform

    def changed_phase_a(*args: Any, **kwargs: Any) -> Any:
        source = original_generate(*args, **kwargs)
        if kwargs["seed"] != 7:
            return source
        changed = np.array(source.train_input, copy=True)
        changed[0, 0] += 0.01
        return replace(source, train_input=changed)

    monkeypatch.setattr(continual, "generate_two_cluster_dataset_with_transform", changed_phase_a)
    with pytest.raises(ValueError, match="incompatible continual checkpoint"):
        continual.run_continual_shift_benchmark(
            config, seeds=[7], checkpoint_store=store, resume_from_checkpoint=True
        )
    assert not phase_a_complete
    assert phase_b_source_calls == 0
    monkeypatch.setattr(continual, "generate_two_cluster_dataset_with_transform", original_generate)

    actual = continual.run_continual_shift_benchmark(
        config, seeds=[7], checkpoint_store=store, resume_from_checkpoint=True
    )
    assert phase_a_complete
    assert phase_b_source_calls == 1
    assert actual.seed_results == expected.seed_results
    assert actual.aggregate == expected.aggregate
    _assert_same_checkpoint_state(store.load().state, expected_store.load().state)


@pytest.mark.parametrize("reverse_order", [False, True])
def test_phase_arrival_checkpoint_rejects_changed_arrived_phase_b_before_update(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, reverse_order: bool
) -> None:
    config = replace(_tiny_config(), protocol_id="continual_phase_arrival_v2")
    if reverse_order:
        config = replace(config, model_order=tuple(reversed(config.model_order)))
    store = TrustedLocalContinualCheckpointStore(tmp_path / "phase_b.ckpt")
    interrupting = InterruptingCheckpointStore(
        store, phase="b", stage="after_sleep", epoch=0, next_model=0
    )
    with pytest.raises(PhaseAInterruption):
        continual.run_continual_shift_benchmark(config, seeds=[7], checkpoint_store=interrupting)
    checkpoint = store.load()
    assert checkpoint.format_version == 2
    assert {role for role, _ in checkpoint.split_hashes} == {
        "phase_a_train",
        "phase_a_validation",
        "phase_b_train",
        "phase_b_validation",
    }

    original_b = continual._build_phase_b_roles
    original_train = continual.BackpropMLP.train_epoch
    updates = 0

    def changed_b(*args: Any, **kwargs: Any) -> Any:
        roles = original_b(*args, **kwargs)
        changed = np.array(roles.train.input, copy=True)
        changed[0, 0] += 0.01
        return make_role_separated_dataset(
            train=LabeledData(changed, roles.train.target),
            validation=roles.validation,
            test=roles.test,
        )

    def count_train(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal updates
        updates += 1
        return original_train(self, *args, **kwargs)

    monkeypatch.setattr(continual, "_build_phase_b_roles", changed_b)
    monkeypatch.setattr(continual.BackpropMLP, "train_epoch", count_train)
    with pytest.raises(ValueError, match="incompatible continual checkpoint"):
        continual.run_continual_shift_benchmark(
            config, seeds=[7], checkpoint_store=store, resume_from_checkpoint=True
        )
    assert updates == 0

    monkeypatch.setattr(continual, "_build_phase_b_roles", original_b)
    expected_store = TrustedLocalContinualCheckpointStore(tmp_path / "expected_b.ckpt")
    expected = continual.run_continual_shift_benchmark(
        config, seeds=[7], checkpoint_store=expected_store
    )
    actual = continual.run_continual_shift_benchmark(
        config, seeds=[7], checkpoint_store=store, resume_from_checkpoint=True
    )
    assert actual.seed_results == expected.seed_results
    _assert_same_checkpoint_state(store.load().state, expected_store.load().state)


def test_phase_arrival_checkpoint_keeps_both_final_tests_sealed_through_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    config = replace(_tiny_config(), protocol_id="continual_phase_arrival_v2")
    original_phase_a = continual._build_checkpoint_phase_a_data
    original_complete = continual._complete_checkpoint_seed_data
    original_train_phase = continual._train_checkpoint_phase
    original_hash = dataset_roles._split_hash
    training_complete = False
    final_reads = 0

    class SealedFinalTest:
        def __init__(self, actual: Any) -> None:
            self.actual = actual

        @property
        def input(self) -> Any:
            nonlocal final_reads
            if not training_complete:
                raise AssertionError("final-test inputs opened during training")
            final_reads += 1
            return self.actual.input

        @property
        def target(self) -> Any:
            nonlocal final_reads
            if not training_complete:
                raise AssertionError("final-test labels opened during training")
            final_reads += 1
            return self.actual.target

    def sealed_phase_a(*args: Any, **kwargs: Any) -> Any:
        data = original_phase_a(*args, **kwargs)
        return replace(data, phase_a_test=SealedFinalTest(data.phase_a_test))

    def sealed_complete(*args: Any, **kwargs: Any) -> Any:
        data = original_complete(*args, **kwargs)
        return replace(data, phase_b_test=SealedFinalTest(data.phase_b_test))

    def train_phase(*args: Any, **kwargs: Any) -> Any:
        nonlocal training_complete
        assert not hasattr(args[0].data, "phase_a_test")
        assert not hasattr(args[0].data, "phase_b_test")
        trained = original_train_phase(*args, **kwargs)
        if kwargs["phase"] == "b":
            training_complete = True
        return trained

    def sealed_hash(role: str, samples: Any) -> str:
        if role == "test" and not training_complete:
            raise AssertionError("final-test role hashed during training")
        return original_hash(role, samples)

    monkeypatch.setattr(continual, "_build_checkpoint_phase_a_data", sealed_phase_a)
    monkeypatch.setattr(continual, "_complete_checkpoint_seed_data", sealed_complete)
    monkeypatch.setattr(continual, "_train_checkpoint_phase", train_phase)
    monkeypatch.setattr(dataset_roles, "_split_hash", sealed_hash)
    store = TrustedLocalContinualCheckpointStore(tmp_path / "sealed_v2.ckpt")
    result = continual.run_continual_shift_benchmark(config, seeds=[7], checkpoint_store=store)

    assert len(result.seed_results) == 1
    assert training_complete
    assert final_reads >= 22


def test_phase_arrival_checkpoint_resumes_next_seed_without_future_phase_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    config = replace(_tiny_config(), protocol_id="continual_phase_arrival_v2")
    seeds = [7, 11]
    expected_store = TrustedLocalContinualCheckpointStore(tmp_path / "two_expected.ckpt")
    expected = continual.run_continual_shift_benchmark(
        config, seeds=seeds, checkpoint_store=expected_store
    )
    store = TrustedLocalContinualCheckpointStore(tmp_path / "two_resumed.ckpt")
    interrupting = InterruptingCheckpointStore(
        store, phase="seed_complete", stage="after_sleep", epoch=2, next_model=0
    )
    with pytest.raises(PhaseAInterruption):
        continual.run_continual_shift_benchmark(config, seeds=seeds, checkpoint_store=interrupting)
    assert store.load().seed_index == 0

    original_source_b = continual._generate_phase_b_source
    original_train_phase = continual._train_checkpoint_phase
    original_backprop_train = continual.BackpropMLP.train_epoch
    second_phase_a_complete = False
    second_b_calls = 0
    resumed_backprop_updates = 0

    def require_second_arrival(*args: Any, **kwargs: Any) -> Any:
        nonlocal second_b_calls
        phase_seed = kwargs.get("seed", args[1] if len(args) > 1 else None)
        if phase_seed == seeds[1] + 101:
            if not second_phase_a_complete:
                raise FuturePhaseUnavailable("second seed Phase B arrived before its Phase A")
            second_b_calls += 1
        return original_source_b(*args, **kwargs)

    def track_second_phase(*args: Any, **kwargs: Any) -> Any:
        nonlocal second_phase_a_complete
        trained = original_train_phase(*args, **kwargs)
        if args[0].seed_index == 1 and kwargs["phase"] == "a":
            second_phase_a_complete = True
        return trained

    def count_backprop(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal resumed_backprop_updates
        resumed_backprop_updates += 1
        return original_backprop_train(self, *args, **kwargs)

    monkeypatch.setattr(continual, "_generate_phase_b_source", require_second_arrival)
    monkeypatch.setattr(continual, "_train_checkpoint_phase", track_second_phase)
    monkeypatch.setattr(continual.BackpropMLP, "train_epoch", count_backprop)
    actual = continual.run_continual_shift_benchmark(
        config, seeds=seeds, checkpoint_store=store, resume_from_checkpoint=True
    )

    assert second_phase_a_complete
    assert second_b_calls == 1
    assert resumed_backprop_updates == config.phase_a_epochs + config.phase_b_epochs
    assert actual.seed_results == expected.seed_results
    _assert_same_checkpoint_state(store.load().state, expected_store.load().state)
