"""A shared, train-only replay schedule precedes matched baseline updates."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app.continual_matched_replay_schedule import (
    MatchedReplayScheduleManifest,
    MatchedReplayScheduleSession,
)
from src.app.continual_shift_benchmark import ContinualGlobalSealConfig
from src.core.circadian_predictive_coding import CircadianConfig, replay_sample_id
from src.core.replay_retention import ReplayRetentionPolicy
from src.infra.datasets import LabeledData


def _manifest(
    policy: ReplayRetentionPolicy = ReplayRetentionPolicy("recent_fifo"),
    *,
    reverse_order: bool = False,
) -> MatchedReplayScheduleManifest:
    training = ContinualGlobalSealConfig(
        sample_count_phase_a=40,
        sample_count_phase_b=40,
        hidden_dim=4,
        phase_a_epochs=2,
        phase_b_epochs=2,
        pc_inference_steps=2,
        circadian_inference_steps=2,
        circadian_sleep_interval_phase_a=1,
        circadian_sleep_interval_phase_b=1,
        circadian_force_sleep=True,
        model_order=(
            ("circadian_predictive_coding", "predictive_coding", "backprop")
            if reverse_order
            else ("backprop", "predictive_coding", "circadian_predictive_coding")
        ),
        circadian_config=CircadianConfig(
            sleep_mode="components",
            max_split_per_sleep=0,
            max_prune_per_sleep=0,
            replay_steps=2,
            replay_memory_size=1,
            replay_prioritized=False,
            replay_class_balanced=False,
            replay_inference_steps=3,
        ),
        replay_max_examples=4,
        replay_max_bytes=96,
    )
    return MatchedReplayScheduleManifest(
        arrived=arrived.ContinualArrivedRolesConfig(training, 0.2, 0.2),
        seeds=(17,),
        policy=policy,
        replay_updates_per_sleep=2,
        pc_replay_inference_steps=2,
    )


def _content_ids(inputs: np.ndarray, targets: np.ndarray) -> tuple[str, ...]:
    return tuple(
        replay_sample_id(inputs[index : index + 1], targets[index : index + 1])
        for index in range(len(inputs))
    )


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize(
    "policy",
    [ReplayRetentionPolicy("recent_fifo"), ReplayRetentionPolicy("seeded_reservoir", 53)],
)
def test_should_schedule_identical_train_rows_and_separate_work_for_all_methods(
    policy: ReplayRetentionPolicy,
    reverse_order: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(policy, reverse_order=reverse_order)
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
    session = MatchedReplayScheduleSession(manifest, seed=17)
    assert b_arrivals == []
    boundaries = []
    for _ in range(2):
        boundary = session.complete_wake_epoch(manifest, source_role="train")
        assert boundary is not None
        boundaries.append(boundary)
    assert b_arrivals == []
    session.arrive_phase_b(manifest)
    assert b_arrivals == [1]
    for _ in range(2):
        boundary = session.complete_wake_epoch(manifest, source_role="train")
        assert boundary is not None
        boundaries.append(boundary)
    assert [(item.phase, item.epoch) for item in boundaries] == [
        ("a", 1),
        ("a", 2),
        ("b", 1),
        ("b", 2),
    ]
    for boundary in boundaries:
        assert len(boundary.selected_ids) == 2
        assert boundary.retention.example_count <= 4
        assert boundary.retention.retained_bytes <= 96
        assert boundary.selected_ids == boundary.retained_order_ids[-2:]
        assert (
            tuple(work.method for work in boundary.method_work)
            == manifest.arrived.training.model_order
        )
        for work in boundary.method_work:
            assert work.sample_ids == boundary.selected_ids
            assert work.planned_examples == 2
            assert work.planned_optimizer_updates == 2
            assert (
                work.planned_inference_iterations
                == {
                    "backprop": 0,
                    "predictive_coding": 4,
                    "circadian_predictive_coding": 6,
                }[work.method]
            )
        first = boundary.selection.training_batches()
        second = boundary.selection.training_batches()
        assert (
            _content_ids(
                np.concatenate([item[0] for item in first]),
                np.concatenate([item[1] for item in first]),
            )
            == boundary.selected_ids
        )
        first[0][0][0, 0] += 100.0
        assert first[0][0][0, 0] != second[0][0][0, 0]
    if policy.name == "recent_fifo":
        assert not set(boundaries[-1].retention.sample_ids) & set(
            boundaries[0].retention.sample_ids
        )
    else:
        assert set(boundaries[-1].retention.sample_ids) & set(boundaries[0].retention.sample_ids)


@pytest.mark.parametrize("change", ["cap", "policy", "seed", "replay_budget", "source_role"])
def test_should_reject_changed_schedule_identity_before_observing_another_epoch(
    change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    session = MatchedReplayScheduleSession(manifest, seed=17)
    original = session.retention
    if change == "cap":
        training = replace(manifest.arrived.training, replay_max_examples=3)
        changed = replace(manifest, arrived=replace(manifest.arrived, training=training))
    elif change == "policy":
        changed = replace(manifest, policy=ReplayRetentionPolicy("seeded_reservoir", 53))
    elif change == "seed":
        changed = replace(manifest, seeds=(19,))
    elif change == "replay_budget":
        changed = replace(manifest, replay_updates_per_sleep=1)
    else:
        changed = manifest
    with pytest.raises(ValueError, match="schedule|role|budget"):
        session.complete_wake_epoch(
            changed, source_role="inner_guard" if change == "source_role" else "train"
        )
    assert session.retention == original
    assert session.phase == "a"
    assert session.completed_epochs == 0


def test_should_reject_model_dependent_sampling_before_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    training = replace(
        manifest.arrived.training,
        circadian_config=replace(
            manifest.arrived.training.circadian_config, replay_prioritized=True
        ),
    )
    changed = replace(manifest, arrived=replace(manifest.arrived, training=training))
    monkeypatch.setattr(
        arrived,
        "generate_two_cluster_dataset_with_transform",
        lambda **_: (_ for _ in ()).throw(AssertionError("source opened")),
    )
    with pytest.raises(ValueError, match="unprioritized"):
        MatchedReplayScheduleSession(changed, seed=17)


@pytest.mark.parametrize("change", ["label", "extra_row"])
def test_should_reject_mutated_arrived_train_role_and_early_b_arrival(change: str) -> None:
    manifest = _manifest()
    session = MatchedReplayScheduleSession(manifest, seed=17)
    with pytest.raises(ValueError, match="before Phase A"):
        session.arrive_phase_b(manifest)
    original = session.retention
    if change == "label":
        session._roles.train.target[0, 0] = 1.0 - session._roles.train.target[0, 0]
    else:
        train = session._roles.train
        extended = LabeledData(
            np.concatenate((train.input, train.input[:1])),
            np.concatenate((train.target, train.target[:1])),
        )
        session._roles = replace(session._roles, train=extended)
    with pytest.raises(ValueError, match="arrived train role"):
        session.complete_wake_epoch(manifest, source_role="train")
    assert session.retention == original
    assert session.completed_epochs == 0


def test_should_never_read_guard_or_outer_role_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    original_build = arrived._build_phase_a_roles

    class SealedDecisionRole:
        @property
        def input(self) -> Any:
            raise AssertionError("decision input opened")

        @property
        def target(self) -> Any:
            raise AssertionError("decision label opened")

    def sealed_roles(*args: Any, **kwargs: Any) -> Any:
        sealed = cast(LabeledData, SealedDecisionRole())
        return replace(
            original_build(*args, **kwargs),
            inner_guard=sealed,
            outer_selection=sealed,
        )

    monkeypatch.setattr(arrived, "_build_phase_a_roles", sealed_roles)
    session = MatchedReplayScheduleSession(manifest, seed=17)
    boundary = session.complete_wake_epoch(manifest, source_role="train")
    assert boundary is not None and boundary.selected_ids
