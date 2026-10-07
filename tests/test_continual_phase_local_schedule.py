"""Phase A sleep decisions cannot use a phase that has not arrived."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pytest

from src.app import continual_shift_benchmark as continual
from src.app.continual_shift_benchmark import ContinualShiftConfig
from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
)
from src.infra.circadian_checkpoint_files import TrustedLocalContinualCheckpointStore


class PhaseAComplete(Exception):
    """Stop before Phase B arrives so only the trained Phase A state is compared."""


class StopAfterPhaseACheckpoint:
    def __init__(self, path: Path, phase_a_epochs: int) -> None:
        self.store = TrustedLocalContinualCheckpointStore(path)
        self.phase_a_epochs = phase_a_epochs

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        self.store.save(checkpoint)
        if (
            checkpoint.phase == "a"
            and checkpoint.phase_epoch_completed == self.phase_a_epochs
            and checkpoint.combined.position.stage == "after_sleep"
        ):
            raise PhaseAComplete()


def _state_hash(value: Any) -> str:
    return sha256(pickle.dumps(value, protocol=5)).hexdigest()


def _assert_same_circadian_snapshot(left: Any, right: Any) -> None:
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


def _config(protocol_id: str, phase_b_epochs: int, reverse_order: bool) -> ContinualShiftConfig:
    config = ContinualShiftConfig(
        protocol_id=protocol_id,
        sample_count_phase_a=40,
        sample_count_phase_b=40,
        hidden_dim=4,
        phase_a_epochs=2,
        phase_b_epochs=phase_b_epochs,
        pc_inference_steps=2,
        circadian_inference_steps=2,
        circadian_sleep_interval_phase_a=1,
        circadian_sleep_interval_phase_b=1,
        circadian_config=CircadianConfig(
            split_threshold=0.0,
            max_split_per_sleep=1,
            max_prune_per_sleep=0,
            sleep_split_only_until_fraction=0.5,
            sleep_prune_only_after_fraction=0.6,
            split_noise_scale=0.05,
            replay_steps=1,
            replay_memory_size=4,
        ),
    )
    return (
        replace(config, model_order=tuple(reversed(config.model_order)))
        if reverse_order
        else config
    )


def _capture_phase_a(
    config: ContinualShiftConfig,
    *,
    checkpointed: bool,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_path: Path,
) -> tuple[tuple[tuple[int, int, tuple[int, ...], tuple[int, ...]], ...], tuple[Any, ...]]:
    sleeps: list[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = []
    original_sleep = CircadianPredictiveCodingNetwork.sleep_event
    ordinary_state: list[tuple[Any, ...]] = []

    def record_sleep(model: CircadianPredictiveCodingNetwork, *args: Any, **kwargs: Any) -> Any:
        result = original_sleep(model, *args, **kwargs)
        progress = kwargs["epoch_progress"]
        sleeps.append(
            (
                progress.completed_epochs,
                progress.total_epochs,
                result.split_indices,
                result.pruned_indices,
            )
        )
        return result

    def stop_before_phase_b(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Phase B source arrived before the Phase A isolation gate")

    with monkeypatch.context() as patch:
        patch.setattr(CircadianPredictiveCodingNetwork, "sleep_event", record_sleep)
        patch.setattr(continual, "_generate_phase_b_source", stop_before_phase_b)
        if checkpointed:
            store = StopAfterPhaseACheckpoint(checkpoint_path, config.phase_a_epochs)
            with pytest.raises(PhaseAComplete):
                continual.run_continual_shift_benchmark(config, [17], checkpoint_store=store)
            checkpoint = store.load()
            assert checkpoint.phase == "a"
            assert checkpoint.format_version == (
                3 if config.protocol_id == "continual_phase_local_schedule_v3" else 2
            )
            assert {role for role, _ in checkpoint.split_hashes} == {
                "phase_a_train",
                "phase_a_validation",
            }
            state = checkpoint.state
            hashes = (
                _state_hash(state.backprop_model),
                _state_hash(state.predictive_model),
                _state_hash(checkpoint.combined.model_state),
                state.sleep_event_count,
                state.total_splits,
                state.total_prunes,
            )
        else:
            original_train = continual._train_phase_a_models

            def capture_and_stop(**kwargs: Any) -> Any:
                state = original_train(**kwargs)
                ordinary_state.append(
                    (
                        _state_hash(state.backprop_after_a),
                        _state_hash(state.predictive_after_a),
                        _state_hash(state.circadian_after_a.snapshot_state()),
                        state.sleep_event_count,
                        state.total_splits,
                        state.total_prunes,
                    )
                )
                raise PhaseAComplete()

            patch.setattr(continual, "_train_phase_a_models", capture_and_stop)
            with pytest.raises(PhaseAComplete):
                continual.run_continual_shift_benchmark(config, [17])
            hashes = ordinary_state[0]

    assert len(sleeps) == config.phase_a_epochs
    return tuple(sleeps), hashes


@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_phase_local_route_keeps_phase_a_sleep_and_state_independent_of_future_duration(
    checkpointed: bool,
    reverse_order: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol = "continual_phase_local_schedule_v3"
    short = _capture_phase_a(
        _config(protocol, phase_b_epochs=1, reverse_order=reverse_order),
        checkpointed=checkpointed,
        monkeypatch=monkeypatch,
        checkpoint_path=tmp_path / "short.ckpt",
    )
    long = _capture_phase_a(
        _config(protocol, phase_b_epochs=7, reverse_order=reverse_order),
        checkpointed=checkpointed,
        monkeypatch=monkeypatch,
        checkpoint_path=tmp_path / "long.ckpt",
    )

    assert short[0] == long[0]
    assert all(total == 2 for _, total, _, _ in short[0])
    assert any(split_indices for _, _, split_indices, _ in short[0])
    assert short[1] == long[1]


@pytest.mark.parametrize("checkpointed", [False, True])
def test_phase_arrival_v2_keeps_reviewed_full_horizon(
    checkpointed: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    short = _capture_phase_a(
        _config("continual_phase_arrival_v2", phase_b_epochs=1, reverse_order=False),
        checkpointed=checkpointed,
        monkeypatch=monkeypatch,
        checkpoint_path=tmp_path / "v2-short.ckpt",
    )
    long = _capture_phase_a(
        _config("continual_phase_arrival_v2", phase_b_epochs=7, reverse_order=False),
        checkpointed=checkpointed,
        monkeypatch=monkeypatch,
        checkpoint_path=tmp_path / "v2-long.ckpt",
    )
    assert [total for _, total, _, _ in short[0]] == [3, 3]
    assert [total for _, total, _, _ in long[0]] == [9, 9]
    assert short[0][-1][2] == ()
    assert len(long[0][-1][2]) == 1
    assert short[1] != long[1]


@pytest.mark.parametrize("reverse_order", [False, True])
def test_phase_local_route_resumes_after_phase_a_and_matches_ordinary_result(
    reverse_order: bool, tmp_path: Path
) -> None:
    config = _config(
        "continual_phase_local_schedule_v3", phase_b_epochs=1, reverse_order=reverse_order
    )
    ordinary = continual.run_continual_shift_benchmark(config, [17])
    expected_store = TrustedLocalContinualCheckpointStore(tmp_path / "expected.ckpt")
    expected = continual.run_continual_shift_benchmark(
        config, [17], checkpoint_store=expected_store
    )
    interrupting = StopAfterPhaseACheckpoint(tmp_path / "resumed.ckpt", config.phase_a_epochs)
    with pytest.raises(PhaseAComplete):
        continual.run_continual_shift_benchmark(config, [17], checkpoint_store=interrupting)
    assert interrupting.store.load().format_version == 3
    actual = continual.run_continual_shift_benchmark(
        config, [17], checkpoint_store=interrupting.store, resume_from_checkpoint=True
    )

    assert actual == expected == ordinary
    expected_checkpoint = expected_store.load()
    actual_checkpoint = interrupting.store.load()
    assert expected_checkpoint.format_version == actual_checkpoint.format_version == 3
    assert _state_hash(expected_checkpoint.state) == _state_hash(actual_checkpoint.state)
    _assert_same_circadian_snapshot(
        expected_checkpoint.combined.model_state, actual_checkpoint.combined.model_state
    )
