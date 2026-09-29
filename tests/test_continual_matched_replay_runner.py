"""Actual matched replay updates use one train-only selection and guard outcome."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from scripts.run_continual_matched_replay_schedule_smoke import _manifest
from src.app import continual_arrived_benchmark as arrived
from src.app import continual_matched_replay_schedule as schedule
from src.app import continual_shift_benchmark as base
from src.app.continual_matched_replay_runner import (
    MATCHED_REPLAY_TRAINING_PROTOCOL,
    run_matched_replay_training,
)
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork, replay_sample_id
from src.core.predictive_coding import PredictiveCodingNetwork
from src.core.replay_retention import ReplayRetentionPolicy


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize("seed", [17, 19])
@pytest.mark.parametrize(
    "policy",
    [ReplayRetentionPolicy("recent_fifo"), ReplayRetentionPolicy("seeded_reservoir", 53)],
)
def test_should_apply_same_selected_rows_and_account_actual_work(
    policy: ReplayRetentionPolicy,
    seed: int,
    reverse_order: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(policy)
    training = manifest.arrived.training
    if reverse_order:
        training = replace(training, model_order=tuple(reversed(training.model_order)))
    manifest = replace(
        manifest,
        arrived=replace(manifest.arrived, training=training, guard_drop_tolerance=1.0),
    )
    seen: dict[str, list[str]] = {
        "backprop": [],
        "predictive_coding": [],
        "circadian_predictive_coding": [],
    }
    replay_inputs: dict[str, list[Any]] = {method: [] for method in seen}
    inference_steps: dict[str, list[int]] = {method: [] for method in seen}
    original_backprop = BackpropMLP.train_epoch
    original_pc = PredictiveCodingNetwork.train_epoch
    original_circadian = CircadianPredictiveCodingNetwork._run_training_step

    def backprop_step(
        self: Any, input_batch: Any, target_batch: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if len(input_batch) == 1:
            seen["backprop"].append(replay_sample_id(input_batch, target_batch))
            replay_inputs["backprop"].append(input_batch)
        return original_backprop(self, input_batch, target_batch, *args, **kwargs)

    def pc_step(self: Any, input_batch: Any, target_batch: Any, *args: Any, **kwargs: Any) -> Any:
        if len(input_batch) == 1:
            seen["predictive_coding"].append(replay_sample_id(input_batch, target_batch))
            replay_inputs["predictive_coding"].append(input_batch)
            inference_steps["predictive_coding"].append(kwargs["inference_steps"])
        return original_pc(self, input_batch, target_batch, *args, **kwargs)

    def circadian_step(self: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("store_replay_snapshot") is False:
            seen["circadian_predictive_coding"].append(
                replay_sample_id(kwargs["input_batch"], kwargs["target_batch"])
            )
            replay_inputs["circadian_predictive_coding"].append(kwargs["input_batch"])
            inference_steps["circadian_predictive_coding"].append(kwargs["inference_steps"])
        return original_circadian(self, *args, **kwargs)

    monkeypatch.setattr(BackpropMLP, "train_epoch", backprop_step)
    monkeypatch.setattr(PredictiveCodingNetwork, "train_epoch", pc_step)
    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "_run_training_step", circadian_step)
    result = run_matched_replay_training(manifest, seed=seed)
    assert result.protocol_id == MATCHED_REPLAY_TRAINING_PROTOCOL
    assert len(result.boundaries) == 4
    expected_ids = tuple(sample_id for item in result.boundaries for sample_id in item.selected_ids)
    assert len(expected_ids) == 8
    assert all(tuple(values) == expected_ids for values in seen.values())
    assert inference_steps["predictive_coding"] == [2] * 8
    assert inference_steps["circadian_predictive_coding"] == [3] * 8
    for index in range(8):
        assert not np.shares_memory(
            replay_inputs["backprop"][index], replay_inputs["predictive_coding"][index]
        )
        assert not np.shares_memory(
            replay_inputs["backprop"][index], replay_inputs["circadian_predictive_coding"][index]
        )
    assert result.pending.state.circadian_model.get_sleep_clocks().wake_batches == 4
    assert result.pending.state.circadian_model.get_sleep_clocks().replay_updates == 8
    for boundary in result.boundaries:
        assert boundary.sleep_outcome == "accepted"
        assert len(boundary.selected_ids) == 2
        for work in boundary.applied_by_method:
            assert work.sample_ids == boundary.selected_ids
            assert work.examples == work.optimizer_updates == 2
            assert (
                work.inference_iterations
                == {
                    "backprop": 0,
                    "predictive_coding": 4,
                    "circadian_predictive_coding": 6,
                }[work.method]
            )
    assert all(event.role != "final_test" for event in result.pending.audit.accesses)
    assert all(
        event.action in {"source_release", "label_release"}
        for event in result.pending.audit.accesses
        if event.role == "outer_selection"
    )


def test_should_skip_all_baseline_replay_when_guard_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(ReplayRetentionPolicy("recent_fifo"))
    scores = iter([1.0, 0.0] * 4)
    monkeypatch.setattr(
        CircadianPredictiveCodingNetwork, "compute_accuracy", lambda *_, **__: next(scores)
    )
    original_backprop = BackpropMLP.train_epoch
    original_pc = PredictiveCodingNetwork.train_epoch
    replay_calls: list[str] = []

    def backprop_step(
        self: Any, input_batch: Any, target_batch: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if len(input_batch) == 1:
            replay_calls.append("backprop")
        return original_backprop(self, input_batch, target_batch, *args, **kwargs)

    def pc_step(self: Any, input_batch: Any, target_batch: Any, *args: Any, **kwargs: Any) -> Any:
        if len(input_batch) == 1:
            replay_calls.append("predictive_coding")
        return original_pc(self, input_batch, target_batch, *args, **kwargs)

    monkeypatch.setattr(BackpropMLP, "train_epoch", backprop_step)
    monkeypatch.setattr(PredictiveCodingNetwork, "train_epoch", pc_step)
    result = run_matched_replay_training(manifest, seed=17)
    assert replay_calls == []
    assert result.pending.state.circadian_model.get_sleep_clocks().replay_updates == 0
    assert len(result.boundaries) == 4
    assert all(item.sleep_outcome == "rolled_back" for item in result.boundaries)
    assert all(
        work.sample_ids == () and work.optimizer_updates == 0
        for item in result.boundaries
        for work in item.applied_by_method
    )


def test_should_preflight_selected_ids_before_circadian_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(ReplayRetentionPolicy("recent_fifo"))
    original_plan = schedule.MatchedReplayScheduleSession.complete_wake_epoch

    def forged_plan(self: Any, *args: Any, **kwargs: Any) -> Any:
        boundary = original_plan(self, *args, **kwargs)
        if boundary is None:
            return None
        forged = replace(boundary.selection, sample_ids=("f" * 64, *boundary.selected_ids[1:]))
        return replace(boundary, selection=forged)

    monkeypatch.setattr(schedule.MatchedReplayScheduleSession, "complete_wake_epoch", forged_plan)
    monkeypatch.setattr(
        CircadianPredictiveCodingNetwork,
        "sleep_event",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("slept before preflight")),
    )
    with pytest.raises(ValueError, match="selected replay IDs"):
        run_matched_replay_training(manifest, seed=17)


def test_should_preflight_retention_before_circadian_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(ReplayRetentionPolicy("recent_fifo"))
    original_plan = schedule.MatchedReplayScheduleSession.complete_wake_epoch

    def forged_plan(self: Any, *args: Any, **kwargs: Any) -> Any:
        boundary = original_plan(self, *args, **kwargs)
        if boundary is None:
            return None
        return replace(
            boundary,
            retention=replace(
                boundary.retention, retained_bytes=boundary.retention.retained_bytes + 8
            ),
        )

    monkeypatch.setattr(schedule.MatchedReplayScheduleSession, "complete_wake_epoch", forged_plan)
    monkeypatch.setattr(
        CircadianPredictiveCodingNetwork,
        "sleep_event",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("slept before preflight")),
    )
    with pytest.raises(ValueError, match="retained IDs or caps"):
        run_matched_replay_training(manifest, seed=17)


def test_should_only_replay_at_shared_periodic_boundaries() -> None:
    manifest = _manifest(ReplayRetentionPolicy("recent_fifo"))
    training = replace(
        manifest.arrived.training,
        circadian_sleep_interval_phase_a=2,
        circadian_sleep_interval_phase_b=2,
    )
    manifest = replace(
        manifest,
        arrived=replace(manifest.arrived, training=training, guard_drop_tolerance=1.0),
    )
    result = run_matched_replay_training(manifest, seed=17)
    assert [(item.phase, item.epoch) for item in result.boundaries] == [("a", 2), ("b", 2)]
    assert [item.outcome for item in result.pending.sleep_events] == [
        "skipped",
        "accepted",
        "skipped",
        "accepted",
    ]
    assert result.pending.state.circadian_model.get_sleep_clocks().wake_batches == 4
    assert result.pending.state.circadian_model.get_sleep_clocks().replay_updates == 4


def test_should_not_replay_baselines_after_sleep_core_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(ReplayRetentionPolicy("recent_fifo"))
    replay_calls: list[str] = []
    original_backprop = BackpropMLP.train_epoch

    def backprop_step(
        self: Any, input_batch: Any, target_batch: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if len(input_batch) == 1:
            replay_calls.append("backprop")
        return original_backprop(self, input_batch, target_batch, *args, **kwargs)

    monkeypatch.setattr(BackpropMLP, "train_epoch", backprop_step)
    monkeypatch.setattr(
        CircadianPredictiveCodingNetwork,
        "sleep_event",
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError("injected sleep failure")),
    )
    with pytest.raises(RuntimeError, match="injected sleep failure"):
        run_matched_replay_training(manifest, seed=17)
    assert replay_calls == []


def test_should_keep_b_and_final_sources_sealed_until_arrival(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(ReplayRetentionPolicy("recent_fifo"))
    original_a = arrived.generate_two_cluster_dataset_with_transform
    original_b = arrived._generate_phase_b_source
    original_update = base._train_named_model_epoch
    wake_updates = 0

    class SealedSource:
        def __init__(self, source: Any) -> None:
            self.train_input = source.train_input
            self.train_target = source.train_target

        @property
        def test_input(self) -> Any:
            raise AssertionError("final input opened")

        @property
        def test_target(self) -> Any:
            raise AssertionError("final label opened")

    def update(*args: Any, **kwargs: Any) -> Any:
        nonlocal wake_updates
        result = original_update(*args, **kwargs)
        wake_updates += 1
        return result

    def build_b(*args: Any, **kwargs: Any) -> Any:
        assert wake_updates == 6
        return SealedSource(original_b(*args, **kwargs))

    monkeypatch.setattr(base, "_train_named_model_epoch", update)
    monkeypatch.setattr(
        arrived,
        "generate_two_cluster_dataset_with_transform",
        lambda **kwargs: SealedSource(original_a(**kwargs)),
    )
    monkeypatch.setattr(arrived, "_generate_phase_b_source", build_b)
    result = run_matched_replay_training(
        replace(manifest, arrived=replace(manifest.arrived, guard_drop_tolerance=1.0)),
        seed=17,
    )
    assert wake_updates == 12
    assert all(event.role != "final_test" for event in result.pending.audit.accesses)
