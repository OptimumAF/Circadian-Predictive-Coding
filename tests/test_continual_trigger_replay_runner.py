"""V14 replay is applied only after an accepted guarded circadian event."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_trigger_replay_schedule as schedule
from src.app.continual_trigger_replay_runner import (
    TRIGGER_REPLAY_TRAINING_PROTOCOL,
    run_trigger_replay_training,
)
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork, replay_sample_id
from src.core.predictive_coding import PredictiveCodingNetwork


@pytest.mark.parametrize("seed", [47, 53])
def test_should_train_all_arms_on_equal_arrived_wake_roles_and_seal_final(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = schedule.fixed_trigger_replay_manifest()
    original_a = arrived.generate_two_cluster_dataset_with_transform
    original_b = arrived._generate_phase_b_source
    b_arrivals: list[int] = []

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

    monkeypatch.setattr(
        arrived,
        "generate_two_cluster_dataset_with_transform",
        lambda **kwargs: SealedSource(original_a(**kwargs)),
    )

    def build_b(*args: Any, **kwargs: Any) -> Any:
        b_arrivals.append(1)
        return SealedSource(original_b(*args, **kwargs))

    monkeypatch.setattr(arrived, "_generate_phase_b_source", build_b)
    trials = [run_trigger_replay_training(manifest, seed=seed, arm=arm) for arm in manifest.arms]
    assert len(b_arrivals) == 3
    assert {trial.protocol_id for trial in trials} == {TRIGGER_REPLAY_TRAINING_PROTOCOL}
    assert len({trial.initial_parameter_digests for trial in trials}) == 1
    assert len({trial.wake_work for trial in trials}) == 1
    assert [
        (work.method, work.optimizer_updates, work.examples, work.inference_iterations)
        for work in trials[0].wake_work
    ] == [
        ("backprop", 24, 1296, 0),
        ("predictive_coding", 24, 1296, 48),
        ("circadian_predictive_coding", 24, 1296, 48),
    ]
    for trial in trials:
        assert len(trial.opportunities) == len(trial.pending.sleep_events) == 24
        assert not trial.pending.phase_a.final_released
        assert not trial.pending.phase_b.final_released
        assert all(event.role != "final_test" for event in trial.pending.audit.accesses)
        assert all(
            event.action in {"source_release", "label_release"}
            for event in trial.pending.audit.accesses
            if event.role == "outer_selection"
        )
        assert trial.pending.state.circadian_model.get_sleep_clocks().wake_examples == 1296
        assert sum(item.event.replay.applied_updates for item in trial.opportunities) == (
            trial.pending.state.circadian_model.get_sleep_clocks().replay_updates
        )
        assert all(
            manifest.min_width <= item.width <= manifest.max_width
            and item.cumulative_splits <= manifest.max_applied_splits
            and item.cumulative_prunes <= manifest.max_applied_prunes
            for item in trial.opportunities
        )
    assert (
        len(
            {
                tuple(
                    (
                        item.train_role_hash,
                        item.retention,
                        item.retained_order_ids,
                        item.selected_ids,
                    )
                    for item in trial.opportunities
                )
                for trial in trials
            }
        )
        == 1
    )
    assert [
        (item.epoch, item.event.outcome)
        for item in trials[0].opportunities
        if item.event.outcome != "skipped"
    ] == [
        (4, "accepted"),
        (8, "accepted"),
        (12, "accepted"),
        (4, "accepted"),
        (8, "accepted"),
        (12, "accepted"),
    ]
    assert all(
        item.event.outcome == "skipped" for trial in trials[1:] for item in trial.opportunities
    )
    assert all(
        work.optimizer_updates == 0
        for trial in trials[1:]
        for item in trial.opportunities
        for work in item.applied_by_method
    )


def test_should_apply_exact_detached_rows_after_guard_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = schedule.fixed_trigger_replay_manifest()
    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "compute_accuracy", lambda *_, **__: 0.5)
    seen: dict[str, list[str]] = {
        "backprop": [],
        "predictive_coding": [],
        "circadian_predictive_coding": [],
    }
    arrays: dict[str, list[np.ndarray]] = {method: [] for method in seen}
    inference: dict[str, list[int]] = {method: [] for method in seen}
    original_backprop = BackpropMLP.train_epoch
    original_pc = PredictiveCodingNetwork.train_epoch
    original_circadian = CircadianPredictiveCodingNetwork._run_training_step

    def backprop_step(
        self: Any, input_batch: Any, target_batch: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if len(input_batch) == 1:
            seen["backprop"].append(replay_sample_id(input_batch, target_batch))
            arrays["backprop"].append(input_batch)
        return original_backprop(self, input_batch, target_batch, *args, **kwargs)

    def pc_step(self: Any, input_batch: Any, target_batch: Any, *args: Any, **kwargs: Any) -> Any:
        if len(input_batch) == 1:
            seen["predictive_coding"].append(replay_sample_id(input_batch, target_batch))
            arrays["predictive_coding"].append(input_batch)
            inference["predictive_coding"].append(kwargs["inference_steps"])
        return original_pc(self, input_batch, target_batch, *args, **kwargs)

    def circadian_step(self: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("store_replay_snapshot") is False:
            seen["circadian_predictive_coding"].append(
                replay_sample_id(kwargs["input_batch"], kwargs["target_batch"])
            )
            arrays["circadian_predictive_coding"].append(kwargs["input_batch"])
            inference["circadian_predictive_coding"].append(kwargs["inference_steps"])
        return original_circadian(self, *args, **kwargs)

    monkeypatch.setattr(BackpropMLP, "train_epoch", backprop_step)
    monkeypatch.setattr(PredictiveCodingNetwork, "train_epoch", pc_step)
    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "_run_training_step", circadian_step)
    trial = run_trigger_replay_training(manifest, seed=47, arm="periodic")
    expected = tuple(
        sample_id
        for item in trial.opportunities
        if item.event.outcome == "accepted"
        for sample_id in item.selected_ids
    )
    assert len(expected) == 12
    assert all(tuple(ids) == expected for ids in seen.values())
    assert inference["predictive_coding"] == inference["circadian_predictive_coding"] == [2] * 12
    assert all(
        not np.shares_memory(arrays["backprop"][index], arrays["predictive_coding"][index])
        and not np.shares_memory(
            arrays["backprop"][index], arrays["circadian_predictive_coding"][index]
        )
        for index in range(12)
    )
    assert all(
        work.sample_ids == item.selected_ids and work.optimizer_updates == 2
        for item in trial.opportunities
        if item.event.outcome == "accepted"
        for work in item.applied_by_method
    )


def test_should_apply_no_baseline_replay_after_guard_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = schedule.fixed_trigger_replay_manifest()
    scores = iter([1.0, 0.0] * 6)
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
    trial = run_trigger_replay_training(manifest, seed=47, arm="periodic")
    assert replay_calls == []
    assert [item.event.outcome for item in trial.opportunities].count("rolled_back") == 6
    assert trial.pending.state.circadian_model.get_sleep_clocks().replay_updates == 0
    assert all(
        work.sample_ids == () and work.optimizer_updates == 0
        for item in trial.opportunities
        for work in item.applied_by_method
    )


def test_should_reject_forged_selection_before_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = schedule.fixed_trigger_replay_manifest()
    original = schedule.TriggerReplayScheduleSession.complete_wake_epoch

    def forged(self: Any, *args: Any, **kwargs: Any) -> Any:
        offered = original(self, *args, **kwargs)
        selected = replace(offered.selection, sample_ids=("f" * 64, *offered.selected_ids[1:]))
        return replace(offered, selection=selected)

    monkeypatch.setattr(schedule.TriggerReplayScheduleSession, "complete_wake_epoch", forged)
    monkeypatch.setattr(
        CircadianPredictiveCodingNetwork,
        "sleep_event",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("slept before preflight")),
    )
    with pytest.raises(ValueError, match="selected IDs"):
        run_trigger_replay_training(manifest, seed=47, arm="periodic")


def test_should_reject_changed_manifest_before_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = schedule.fixed_trigger_replay_manifest()
    changed = replace(manifest, max_applied_prunes=7)
    monkeypatch.setattr(
        arrived,
        "generate_two_cluster_dataset_with_transform",
        lambda **_: (_ for _ in ()).throw(AssertionError("source opened")),
    )
    with pytest.raises(ValueError, match="fixed prospective manifest"):
        run_trigger_replay_training(changed, seed=47, arm="periodic")
    with pytest.raises(ValueError, match="arm"):
        run_trigger_replay_training(manifest, seed=47, arm="experimental")


def test_should_not_replay_baselines_after_sleep_core_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = schedule.fixed_trigger_replay_manifest()
    original = BackpropMLP.train_epoch
    replay_calls: list[str] = []

    def backprop_step(
        self: Any, input_batch: Any, target_batch: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if len(input_batch) == 1:
            replay_calls.append("backprop")
        return original(self, input_batch, target_batch, *args, **kwargs)

    monkeypatch.setattr(BackpropMLP, "train_epoch", backprop_step)
    monkeypatch.setattr(
        CircadianPredictiveCodingNetwork,
        "sleep_event",
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError("injected core error")),
    )
    with pytest.raises(RuntimeError, match="injected core error"):
        run_trigger_replay_training(manifest, seed=47, arm="periodic")
    assert replay_calls == []
