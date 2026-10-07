"""Trusted-file recovery of the real two-phase, multi-seed NumPy runner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import pickle
import random
from typing import Any, cast

import numpy as np
import pytest

from src.app import continual_shift_benchmark as continual
from src.app.continual_shift_benchmark import (
    CONTINUAL_LEGACY_PROTOCOL,
    ContinualShiftConfig,
    run_continual_shift_benchmark,
)
from src.core.circadian_predictive_coding import CircadianConfig, CircadianNetworkSnapshot
from src.infra.circadian_checkpoint_files import TrustedLocalContinualCheckpointStore
from src.infra.datasets import LabeledData, make_role_separated_dataset


class IntentionalInterruption(Exception):
    pass


class InterruptingStore:
    def __init__(
        self,
        store: TrustedLocalContinualCheckpointStore,
        *,
        seed_index: int,
        phase: str,
        stage: str,
        phase_epoch: int,
        next_model: int = 0,
    ) -> None:
        self.store = store
        self.seed_index = seed_index
        self.phase = phase
        self.stage = stage
        self.phase_epoch = phase_epoch
        self.next_model = next_model
        self.interrupted = False

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        self.store.save(checkpoint)
        position = checkpoint.combined.position
        if (
            not self.interrupted
            and checkpoint.seed_index == self.seed_index
            and checkpoint.phase == self.phase
            and checkpoint.phase_epoch_completed == self.phase_epoch
            and position.stage == self.stage
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


def _config() -> ContinualShiftConfig:
    return ContinualShiftConfig(
        sample_count_phase_a=80,
        sample_count_phase_b=80,
        phase_b_train_fraction=0.5,
        hidden_dim=4,
        phase_a_epochs=2,
        phase_b_epochs=2,
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


def _same_circadian_snapshot(
    actual: CircadianNetworkSnapshot, expected: CircadianNetworkSnapshot
) -> None:
    assert actual.format_version == expected.format_version
    assert actual.config == expected.config
    assert actual.state.keys() == expected.state.keys()
    for name in actual.state:
        left = actual.state[name]
        right = expected.state[name]
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


@pytest.mark.parametrize("reversed_order", [False, True])
@pytest.mark.parametrize(
    ("phase", "stage", "phase_epoch", "next_model"),
    [
        ("a", "before_sleep", 2, 0),
        ("a", "after_sleep", 2, 0),
        ("b", "after_sleep", 0, 0),
        ("b", "wake", 0, 1),
        ("b", "before_sleep", 1, 0),
        ("b", "after_sleep", 1, 0),
    ],
)
def test_continual_file_resume_matches_uninterrupted_phases_and_replay(
    tmp_path: Any,
    reversed_order: bool,
    phase: str,
    stage: str,
    phase_epoch: int,
    next_model: int,
) -> None:
    config = _config()
    if reversed_order:
        config = replace(config, model_order=tuple(reversed(config.model_order)))
    seeds = [13, 17]
    random.seed(700)
    np.random.seed(701)
    control_store = TrustedLocalContinualCheckpointStore(tmp_path / "control.ckpt")
    expected = run_continual_shift_benchmark(config, seeds, checkpoint_store=control_store)
    expected_draw = (random.random(), float(np.random.random()))

    random.seed(700)
    np.random.seed(701)
    store = TrustedLocalContinualCheckpointStore(tmp_path / "resume.ckpt")
    interrupting = InterruptingStore(
        store,
        seed_index=0,
        phase=phase,
        stage=stage,
        phase_epoch=phase_epoch,
        next_model=next_model,
    )
    with pytest.raises(IntentionalInterruption):
        run_continual_shift_benchmark(config, seeds, checkpoint_store=interrupting)
    assert interrupting.interrupted
    if phase == "b":
        assert store.load().state.circadian_after_a is not None

    _ = random.random(), np.random.random()

    actual = run_continual_shift_benchmark(
        config, seeds, checkpoint_store=store, resume_from_checkpoint=True
    )
    assert actual.seed_results == expected.seed_results
    assert actual.aggregate == expected.aggregate
    assert (random.random(), float(np.random.random())) == expected_draw
    final = store.load()
    control_final = control_store.load()
    assert final.phase == "seed_complete"
    assert final.seed_index == 1
    assert len(final.completed_results) == 2
    assert final.state.total_splits > 0
    assert pickle.dumps(final.state.backprop_model) == pickle.dumps(
        control_final.state.backprop_model
    )
    assert pickle.dumps(final.state.predictive_model) == pickle.dumps(
        control_final.state.predictive_model
    )
    assert pickle.dumps(final.state.backprop_after_a) == pickle.dumps(
        control_final.state.backprop_after_a
    )
    assert pickle.dumps(final.state.predictive_after_a) == pickle.dumps(
        control_final.state.predictive_after_a
    )
    assert final.state.circadian_after_a is not None
    assert control_final.state.circadian_after_a is not None
    _same_circadian_snapshot(final.state.circadian_after_a, control_final.state.circadian_after_a)
    assert isinstance(final.combined.model_state, CircadianNetworkSnapshot)
    assert isinstance(control_final.combined.model_state, CircadianNetworkSnapshot)
    _same_circadian_snapshot(final.combined.model_state, control_final.combined.model_state)


def test_continual_resume_after_first_seed_does_not_repeat_its_training(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    seeds = [13, 17]
    random.seed(755)
    np.random.seed(756)
    expected = run_continual_shift_benchmark(config, seeds)
    expected_draw = (random.random(), float(np.random.random()))
    random.seed(755)
    np.random.seed(756)
    store = TrustedLocalContinualCheckpointStore(tmp_path / "seeds.ckpt")
    with pytest.raises(IntentionalInterruption):
        run_continual_shift_benchmark(
            config,
            seeds,
            checkpoint_store=InterruptingStore(
                store,
                seed_index=0,
                phase="seed_complete",
                stage="after_sleep",
                phase_epoch=config.phase_b_epochs,
            ),
        )
    trained = 0
    original = continual.BackpropMLP.train_epoch

    def tracked(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal trained
        trained += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(continual.BackpropMLP, "train_epoch", tracked)
    _ = random.random(), np.random.random()
    actual = run_continual_shift_benchmark(
        config, seeds, checkpoint_store=store, resume_from_checkpoint=True
    )
    assert actual.seed_results == expected.seed_results
    assert trained == config.phase_a_epochs + config.phase_b_epochs
    assert (random.random(), float(np.random.random())) == expected_draw


def test_continual_terminal_checkpoint_returns_committed_report_without_rescoring(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    random.seed(811)
    np.random.seed(812)
    expected = run_continual_shift_benchmark(config, [13])
    expected_draw = (random.random(), float(np.random.random()))
    random.seed(811)
    np.random.seed(812)
    store = TrustedLocalContinualCheckpointStore(tmp_path / "terminal.ckpt")
    with pytest.raises(IntentionalInterruption):
        run_continual_shift_benchmark(
            config,
            [13],
            checkpoint_store=InterruptingStore(
                store,
                seed_index=0,
                phase="seed_complete",
                stage="after_sleep",
                phase_epoch=config.phase_b_epochs,
            ),
        )

    def unexpected_work(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("terminal resume repeated training or held-out scoring")

    monkeypatch.setattr(continual.BackpropMLP, "train_epoch", unexpected_work)
    monkeypatch.setattr(continual.BackpropMLP, "compute_accuracy", unexpected_work)
    _ = random.random(), np.random.random()
    actual = run_continual_shift_benchmark(
        config, [13], checkpoint_store=store, resume_from_checkpoint=True
    )
    assert actual.seed_results == expected.seed_results
    assert (random.random(), float(np.random.random())) == expected_draw


def test_continual_file_rejects_incompatible_state_before_training(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    seeds = [13, 17]
    store = TrustedLocalContinualCheckpointStore(tmp_path / "invalid.ckpt")
    with pytest.raises(IntentionalInterruption):
        run_continual_shift_benchmark(
            config,
            seeds,
            checkpoint_store=InterruptingStore(
                store, seed_index=0, phase="b", stage="before_sleep", phase_epoch=1
            ),
        )
    trained = 0
    original = continual.BackpropMLP.train_epoch

    def tracked(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal trained
        trained += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(continual.BackpropMLP, "train_epoch", tracked)
    for incompatible_config, incompatible_seeds in (
        (replace(config, model_order=tuple(reversed(config.model_order))), seeds),
        (config, list(reversed(seeds))),
        (replace(config, phase_b_epochs=3), seeds),
    ):
        with pytest.raises(ValueError, match="incompatible"):
            run_continual_shift_benchmark(
                incompatible_config,
                incompatible_seeds,
                checkpoint_store=store,
                resume_from_checkpoint=True,
            )
    assert trained == 0

    checkpoint = store.load()
    invalid = deepcopy(checkpoint)
    invalid.state.sleep_event_count = -1
    store.save(invalid)
    with pytest.raises(ValueError, match="counters"):
        run_continual_shift_benchmark(
            config, seeds, checkpoint_store=store, resume_from_checkpoint=True
        )
    assert trained == 0
    store.save(checkpoint)

    original_b = continual._build_phase_b_roles

    def changed_phase_b(*args: Any, **kwargs: Any) -> Any:
        roles = original_b(*args, **kwargs)
        altered = roles.train.input.copy()
        altered[0, 0] += 0.01
        return make_role_separated_dataset(
            train=LabeledData(altered, roles.train.target),
            validation=roles.validation,
            test=roles.test,
        )

    monkeypatch.setattr(continual, "_build_phase_b_roles", changed_phase_b)
    with pytest.raises(ValueError, match="incompatible"):
        run_continual_shift_benchmark(
            config, seeds, checkpoint_store=store, resume_from_checkpoint=True
        )
    assert trained == 0
    monkeypatch.setattr(continual, "_build_phase_b_roles", original_b)

    malformed = deepcopy(checkpoint)
    assert malformed.state.circadian_after_a is not None
    malformed.state.circadian_after_a.state["weight_hidden_output"] = np.zeros((1, 1))
    store.save(malformed)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    with pytest.raises(ValueError):
        run_continual_shift_benchmark(
            config, seeds, checkpoint_store=store, resume_from_checkpoint=True
        )
    assert trained == 0
    assert random.getstate() == python_state
    np.testing.assert_array_equal(cast(Any, np.random.get_state())[1], cast(Any, numpy_state)[1])
    store.save(checkpoint)

    raw = store.path.read_bytes()
    store.path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        run_continual_shift_benchmark(
            config, seeds, checkpoint_store=store, resume_from_checkpoint=True
        )
    assert trained == 0


def test_continual_file_rejects_changed_completed_seed_data(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    store = TrustedLocalContinualCheckpointStore(tmp_path / "prefix.ckpt")
    with pytest.raises(IntentionalInterruption):
        run_continual_shift_benchmark(
            config,
            [13, 17],
            checkpoint_store=InterruptingStore(
                store, seed_index=1, phase="a", stage="before_sleep", phase_epoch=1
            ),
        )
    original = continual.generate_two_cluster_dataset_with_transform

    def changed_first_seed(*args: Any, **kwargs: Any) -> Any:
        source = original(*args, **kwargs)
        if kwargs["seed"] != 13:
            return source
        altered = source.train_input.copy()
        altered[0, 0] += 0.01
        return replace(source, train_input=altered)

    monkeypatch.setattr(
        continual, "generate_two_cluster_dataset_with_transform", changed_first_seed
    )
    with pytest.raises(ValueError, match="completed seed data"):
        run_continual_shift_benchmark(
            config, [13, 17], checkpoint_store=store, resume_from_checkpoint=True
        )


def test_continual_checkpoint_keeps_both_final_tests_sealed_until_seed_training_ends(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    original_data = continual._build_checkpoint_seed_data
    original_score = continual._score_seed_models
    training_complete = False
    final_reads = 0

    class SealedFinalTest:
        def __init__(self, actual: Any) -> None:
            self.actual = actual

        @property
        def input(self) -> Any:
            nonlocal final_reads
            if not training_complete:
                raise AssertionError("continual final-test inputs opened during training")
            final_reads += 1
            return self.actual.input

        @property
        def target(self) -> Any:
            nonlocal final_reads
            if not training_complete:
                raise AssertionError("continual final-test labels opened during training")
            final_reads += 1
            return self.actual.target

    def sealed_data(*args: Any, **kwargs: Any) -> Any:
        data = original_data(*args, **kwargs)
        return replace(
            data,
            phase_a_test=SealedFinalTest(data.phase_a_test),
            phase_b_test=SealedFinalTest(data.phase_b_test),
        )

    def score(*args: Any, **kwargs: Any) -> Any:
        nonlocal training_complete
        training_complete = True
        return original_score(*args, **kwargs)

    monkeypatch.setattr(continual, "_build_checkpoint_seed_data", sealed_data)
    monkeypatch.setattr(continual, "_score_seed_models", score)
    store = TrustedLocalContinualCheckpointStore(tmp_path / "sealed.ckpt")
    with pytest.raises(IntentionalInterruption):
        run_continual_shift_benchmark(
            config,
            [13],
            checkpoint_store=InterruptingStore(
                store, seed_index=0, phase="b", stage="before_sleep", phase_epoch=1
            ),
        )
    assert not training_complete
    assert final_reads == 0
    run_continual_shift_benchmark(config, [13], checkpoint_store=store, resume_from_checkpoint=True)
    assert training_complete
    # The 18 scoring reads are followed by four reads to bind both completed
    # test roles before this seed result can be reused from a later file.
    assert final_reads == 22


def test_continual_legacy_protocol_resumes_without_validation_roles(tmp_path: Any) -> None:
    config = replace(_config(), protocol_id=CONTINUAL_LEGACY_PROTOCOL)
    expected = run_continual_shift_benchmark(config, [13])
    store = TrustedLocalContinualCheckpointStore(tmp_path / "legacy.ckpt")
    with pytest.raises(IntentionalInterruption):
        run_continual_shift_benchmark(
            config,
            [13],
            checkpoint_store=InterruptingStore(
                store, seed_index=0, phase="a", stage="after_sleep", phase_epoch=2
            ),
        )
    actual = run_continual_shift_benchmark(
        config, [13], checkpoint_store=store, resume_from_checkpoint=True
    )
    assert actual.seed_results == expected.seed_results
    assert actual.seed_results[0].split_hashes == {}
