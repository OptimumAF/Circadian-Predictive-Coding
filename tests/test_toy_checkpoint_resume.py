"""File resume of the real three-model NumPy toy comparison."""

from __future__ import annotations

from dataclasses import replace
import pickle
import random
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.app import experiment_runner
from src.app.experiment_runner import TOY_LEGACY_PROTOCOL, ExperimentConfig, run_experiment
from src.core.circadian_predictive_coding import CircadianConfig, CircadianNetworkSnapshot
from src.infra.circadian_checkpoint_files import TrustedLocalToyCheckpointStore


class IntentionalInterruption(Exception):
    pass


class InterruptingStore:
    def __init__(
        self,
        store: TrustedLocalToyCheckpointStore,
        *,
        stage: str,
        epoch: int,
        next_model: int = 0,
    ) -> None:
        self.store = store
        self.stage = stage
        self.epoch = epoch
        self.next_model = next_model
        self.interrupted = False

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        self.store.save(checkpoint)
        position = checkpoint.combined.position
        if (
            not self.interrupted
            and position.stage == self.stage
            and position.completed_epoch == self.epoch
            and position.next_batch_index == self.next_model
        ):
            self.interrupted = True
            raise IntentionalInterruption()


@pytest.fixture(autouse=True)
def preserve_random_streams() -> Any:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        sample_count=80,
        hidden_dim=4,
        epoch_count=4,
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


def _same_result(actual: Any, expected: Any) -> None:
    assert actual.protocol_id == expected.protocol_id
    assert actual.split_hashes == expected.split_hashes
    assert actual.training_order == expected.training_order
    assert actual.circadian_sleep == expected.circadian_sleep
    for name in experiment_runner.TOY_MODEL_ORDER:
        left = getattr(actual, name)
        right = getattr(expected, name)
        assert left.loss_history == right.loss_history
        assert left.validation_accuracy == right.validation_accuracy
        assert left.test_accuracy == right.test_accuracy
        for left_layer, right_layer in zip(left.traffic_by_layer, right.traffic_by_layer):
            np.testing.assert_array_equal(
                left_layer.mean_abs_activation, right_layer.mean_abs_activation
            )


@pytest.mark.parametrize("reversed_order", [False, True])
@pytest.mark.parametrize(
    ("stage", "epoch", "next_model"),
    [("wake", 0, 1), ("before_sleep", 2, 0), ("after_sleep", 2, 0)],
)
def test_toy_file_resume_matches_uninterrupted_split_and_replay(
    tmp_path: Any, stage: str, epoch: int, next_model: int, reversed_order: bool
) -> None:
    config = _config()
    if reversed_order:
        config = replace(config, model_order=tuple(reversed(config.model_order)))
    random.seed(700)
    np.random.seed(701)
    control_store = TrustedLocalToyCheckpointStore(tmp_path / "control.checkpoint")
    expected = run_experiment(config, checkpoint_store=control_store)
    control_checkpoint = control_store.load()
    expected_draw = (random.random(), float(np.random.random()))

    random.seed(700)
    np.random.seed(701)
    store = TrustedLocalToyCheckpointStore(tmp_path / "toy.checkpoint")
    interrupting = InterruptingStore(store, stage=stage, epoch=epoch, next_model=next_model)
    with pytest.raises(IntentionalInterruption):
        run_experiment(config, checkpoint_store=interrupting)
    assert interrupting.interrupted
    if stage == "after_sleep":
        state = store.load().combined.model_state
        assert isinstance(state, CircadianNetworkSnapshot)
        assert state.state["weight_hidden_output"].shape[0] > config.hidden_dim

    # A fresh call constructs fresh models and regenerates the same roles.
    actual = run_experiment(config, checkpoint_store=store, resume_from_checkpoint=True)
    _same_result(actual, expected)
    assert (random.random(), float(np.random.random())) == expected_draw
    final_checkpoint = store.load()
    assert pickle.dumps(final_checkpoint.backprop_model) == pickle.dumps(
        control_checkpoint.backprop_model
    )
    assert pickle.dumps(final_checkpoint.predictive_model) == pickle.dumps(
        control_checkpoint.predictive_model
    )
    final_state = final_checkpoint.combined.model_state
    control_state = control_checkpoint.combined.model_state
    assert isinstance(final_state, CircadianNetworkSnapshot)
    assert isinstance(control_state, CircadianNetworkSnapshot)
    assert final_state.state.keys() == control_state.state.keys()
    for name in final_state.state:
        left = final_state.state[name]
        right = control_state.state[name]
        if name == "_replay_memory":
            assert len(left) == len(right)
            for left_sample, right_sample in zip(left, right):
                np.testing.assert_array_equal(left_sample.input_batch, right_sample.input_batch)
                np.testing.assert_array_equal(left_sample.target_batch, right_sample.target_batch)
                assert left_sample.priority == right_sample.priority
                assert left_sample.positive_fraction == right_sample.positive_fraction
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right, err_msg=name)
        else:
            assert pickle.dumps(left) == pickle.dumps(right), name
    assert final_checkpoint.combined.position.stage == "after_sleep"
    assert final_checkpoint.combined.position.completed_epoch == config.epoch_count
    assert final_checkpoint.sleep_event_count == expected.circadian_sleep.event_count
    assert len(final_checkpoint.losses[0]) == config.epoch_count


def test_toy_checkpoint_rejects_mismatched_order_and_corrupt_file(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    store = TrustedLocalToyCheckpointStore(tmp_path / "toy.checkpoint")
    with pytest.raises(IntentionalInterruption):
        run_experiment(
            config,
            checkpoint_store=InterruptingStore(store, stage="before_sleep", epoch=2),
        )
    trained = 0
    original_train = experiment_runner.BackpropMLP.train_epoch

    def tracked_train(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal trained
        trained += 1
        return original_train(self, *args, **kwargs)

    monkeypatch.setattr(experiment_runner.BackpropMLP, "train_epoch", tracked_train)
    with pytest.raises(ValueError, match="incompatible"):
        run_experiment(
            replace(config, model_order=tuple(reversed(config.model_order))),
            checkpoint_store=store,
            resume_from_checkpoint=True,
        )
    assert trained == 0

    generator = experiment_runner.generate_two_cluster_dataset

    def changed_data(*args: Any, **kwargs: Any) -> Any:
        data = generator(*args, **kwargs)
        changed = data.train_input.copy()
        changed[0, 0] += 0.01
        return replace(data, train_input=changed)

    monkeypatch.setattr(experiment_runner, "generate_two_cluster_dataset", changed_data)
    with pytest.raises(ValueError, match="incompatible"):
        run_experiment(config, checkpoint_store=store, resume_from_checkpoint=True)
    assert trained == 0
    monkeypatch.setattr(experiment_runner, "generate_two_cluster_dataset", generator)

    checkpoint = store.load()
    store.save(replace(checkpoint, sleep_event_count=100))
    with pytest.raises(ValueError, match="counters"):
        run_experiment(config, checkpoint_store=store, resume_from_checkpoint=True)
    assert trained == 0
    store.save(checkpoint)

    malformed = pickle.loads(pickle.dumps(checkpoint))
    malformed.backprop_model._hidden_weights[0] = np.zeros((1, 1))
    store.save(malformed)
    with pytest.raises(ValueError, match="baseline"):
        run_experiment(config, checkpoint_store=store, resume_from_checkpoint=True)
    assert trained == 0
    store.save(checkpoint)

    raw = store.path.read_bytes()
    store.path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        run_experiment(config, checkpoint_store=store, resume_from_checkpoint=True)
    assert trained == 0


def test_toy_checkpoint_does_not_score_final_test_before_training(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    original_train = experiment_runner._train_toy_models
    original_split = experiment_runner.split_training_validation
    training_complete = False
    final_reads = 0

    class SealedFinalTest:
        def __init__(self, actual: Any) -> None:
            self.actual = actual

        @property
        def input(self) -> Any:
            nonlocal final_reads
            if not training_complete:
                raise AssertionError("final-test inputs opened before training")
            final_reads += 1
            return self.actual.input

        @property
        def target(self) -> Any:
            nonlocal final_reads
            if not training_complete:
                raise AssertionError("final-test labels opened before training")
            final_reads += 1
            return self.actual.target

    def sealed_split(*args: Any, **kwargs: Any) -> Any:
        roles = original_split(*args, **kwargs)
        return SimpleNamespace(
            train=roles.train,
            validation=roles.validation,
            test=SealedFinalTest(roles.test),
            split_hashes=roles.split_hashes,
        )

    def train(*args: Any, **kwargs: Any) -> Any:
        nonlocal training_complete
        outcome = original_train(*args, **kwargs)
        training_complete = True
        return outcome

    monkeypatch.setattr(experiment_runner, "split_training_validation", sealed_split)
    monkeypatch.setattr(experiment_runner, "_train_toy_models", train)
    store = TrustedLocalToyCheckpointStore(tmp_path / "toy.checkpoint")
    with pytest.raises(IntentionalInterruption):
        run_experiment(
            config,
            checkpoint_store=InterruptingStore(store, stage="before_sleep", epoch=2),
        )
    assert not training_complete
    assert final_reads == 0
    run_experiment(config, checkpoint_store=store, resume_from_checkpoint=True)
    assert training_complete
    assert final_reads == 6


def test_toy_legacy_protocol_can_resume_without_validation_role(tmp_path: Any) -> None:
    config = replace(_config(), protocol_id=TOY_LEGACY_PROTOCOL)
    expected = run_experiment(config)
    store = TrustedLocalToyCheckpointStore(tmp_path / "legacy.checkpoint")
    with pytest.raises(IntentionalInterruption):
        run_experiment(
            config,
            checkpoint_store=InterruptingStore(store, stage="after_sleep", epoch=2),
        )
    actual = run_experiment(config, checkpoint_store=store, resume_from_checkpoint=True)
    _same_result(actual, expected)
    assert actual.split_hashes == {}
