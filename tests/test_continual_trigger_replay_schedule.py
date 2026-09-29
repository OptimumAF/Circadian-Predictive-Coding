"""V14 all-opportunity replay supply stays sealed and matches v9 periodic rows."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Any, cast

import numpy as np
import pytest

from scripts.run_continual_trigger_replay_schedule import build_payload
from src.app import continual_arrived_benchmark as arrived
from src.app.continual_matched_replay_schedule import (
    MatchedReplayScheduleManifest,
    MatchedReplayScheduleSession,
)
from src.app.continual_trigger_replay_schedule import (
    TriggerReplayScheduleSession,
    fixed_trigger_replay_manifest,
)
from src.core.circadian_predictive_coding import replay_sample_id
from src.core.replay_retention import ReplayRetentionPolicy
from src.infra.datasets import LabeledData


def _v9_manifest() -> MatchedReplayScheduleManifest:
    fixed = fixed_trigger_replay_manifest()
    return MatchedReplayScheduleManifest(
        arrived=fixed.arrived,
        seeds=fixed.seeds,
        policy=fixed.policy,
        replay_updates_per_sleep=fixed.replay_updates_per_sleep,
        pc_replay_inference_steps=fixed.pc_replay_inference_steps,
    )


@pytest.mark.parametrize("seed", [47, 53])
def test_should_offer_every_wake_epoch_and_match_v9_periodic_subset(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = fixed_trigger_replay_manifest()
    v9_manifest = _v9_manifest()
    original_a = arrived.generate_two_cluster_dataset_with_transform
    original_b = arrived._generate_phase_b_source
    b_arrivals: list[int] = []

    class SealedSource:
        def __init__(self, source: Any) -> None:
            self.train_input = source.train_input
            self.train_target = source.train_target

        @property
        def test_input(self) -> Any:
            raise AssertionError("final input opened before global freeze")

        @property
        def test_target(self) -> Any:
            raise AssertionError("final label opened before global freeze")

    monkeypatch.setattr(
        arrived,
        "generate_two_cluster_dataset_with_transform",
        lambda **kwargs: SealedSource(original_a(**kwargs)),
    )

    def build_b(*args: Any, **kwargs: Any) -> Any:
        b_arrivals.append(1)
        return SealedSource(original_b(*args, **kwargs))

    monkeypatch.setattr(arrived, "_generate_phase_b_source", build_b)
    session = TriggerReplayScheduleSession(manifest, seed=seed)
    historical = MatchedReplayScheduleSession(v9_manifest, seed=seed)
    assert b_arrivals == []
    with pytest.raises(ValueError, match="before Phase A"):
        session.arrive_phase_b(manifest)
    opportunities = []
    for phase in ("a", "b"):
        if phase == "b":
            assert b_arrivals == []
            session.arrive_phase_b(manifest)
            historical.arrive_phase_b(v9_manifest)
            assert b_arrivals == [1, 1]
        for epoch in range(1, 13):
            item = session.complete_wake_epoch(manifest, source_role="train")
            previous = historical.complete_wake_epoch(v9_manifest, source_role="train")
            opportunities.append(item)
            assert (item.phase, item.epoch) == (phase, epoch)
            assert item.global_epoch == epoch + (12 if phase == "b" else 0)
            assert item.train_role_hash == session._roles.split_hashes["train"]
            assert item.train_role_ids == session._roles.sample_ids["train"]
            assert item.retention.example_count == 8
            assert item.retention.retained_bytes == 192
            assert item.selected_ids == item.retained_order_ids[-2:]
            assert item.selected_ids == tuple(
                replay_sample_id(inputs, targets)
                for inputs, targets in item.selection.training_batches()
            )
            if epoch % 4:
                assert previous is None
            else:
                assert previous is not None
                assert item.train_role_hash == previous.train_role_hash
                assert item.retention == previous.retention
                assert item.retained_order_ids == previous.retained_order_ids
                assert item.selected_ids == previous.selected_ids
                assert item.method_work == previous.method_work
    assert len(opportunities) == 24
    assert len({item.train_role_hash for item in opportunities}) == 2
    with pytest.raises(ValueError, match="remaining wake epochs"):
        session.complete_wake_epoch(manifest, source_role="train")


def test_should_give_each_method_detached_selected_rows() -> None:
    manifest = fixed_trigger_replay_manifest()
    session = TriggerReplayScheduleSession(manifest, seed=47)
    item = session.complete_wake_epoch(manifest, source_role="train")
    first = item.selection.training_batches()
    second = item.selection.training_batches()
    assert len(first) == 2
    assert (
        tuple(replay_sample_id(inputs, targets) for inputs, targets in first) == item.selected_ids
    )
    first[0][0][0, 0] += 100.0
    first[0][1][0, 0] = 1.0 - first[0][1][0, 0]
    assert not np.array_equal(first[0][0], second[0][0])
    assert not np.array_equal(first[0][1], second[0][1])
    assert tuple(work.method for work in item.method_work) == manifest.arrived.training.model_order
    assert [
        (work.planned_examples, work.planned_optimizer_updates, work.planned_inference_iterations)
        for work in item.method_work
    ] == [(2, 2, 0), (2, 2, 4), (2, 2, 4)]
    assert all(work.sample_ids == item.selected_ids for work in item.method_work)


@pytest.mark.parametrize("change", ["seed", "policy", "cap", "arm", "source_role"])
def test_should_reject_changed_manifest_or_non_train_role_before_observation(change: str) -> None:
    manifest = fixed_trigger_replay_manifest()
    session = TriggerReplayScheduleSession(manifest, seed=47)
    if change == "seed":
        changed = replace(manifest, seeds=(47, 59))
    elif change == "policy":
        changed = replace(manifest, policy=ReplayRetentionPolicy("seeded_reservoir", 53))
    elif change == "cap":
        training = replace(manifest.arrived.training, replay_max_bytes=168)
        changed = replace(manifest, arrived=replace(manifest.arrived, training=training))
    elif change == "arm":
        changed = replace(manifest, arms=("periodic", "no_sleep"))
    else:
        changed = manifest
    with pytest.raises(ValueError, match="manifest|source role"):
        session.complete_wake_epoch(
            changed, source_role="inner_guard" if change == "source_role" else "train"
        )
    assert session.completed_epochs == 0
    assert session.retention.example_count == 0


def test_should_reject_nonfixed_manifest_before_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = fixed_trigger_replay_manifest()
    changed = replace(manifest, replay_updates_per_sleep=3)
    monkeypatch.setattr(
        arrived,
        "generate_two_cluster_dataset_with_transform",
        lambda **_: (_ for _ in ()).throw(AssertionError("source opened")),
    )
    with pytest.raises(ValueError, match="fixed prospective manifest"):
        TriggerReplayScheduleSession(changed, seed=47)


@pytest.mark.parametrize("change", ["label", "extra_row"])
def test_should_reject_mutated_arrived_train_before_buffer_mutation(change: str) -> None:
    manifest = fixed_trigger_replay_manifest()
    session = TriggerReplayScheduleSession(manifest, seed=47)
    if change == "label":
        session._roles.train.target[0, 0] = 1.0 - session._roles.train.target[0, 0]
    else:
        train = session._roles.train
        session._roles = replace(
            session._roles,
            train=LabeledData(
                np.concatenate((train.input, train.input[:1])),
                np.concatenate((train.target, train.target[:1])),
            ),
        )
    with pytest.raises(ValueError, match="arrived train role"):
        session.complete_wake_epoch(manifest, source_role="train")
    assert session.completed_epochs == 0
    assert session.retention.example_count == 0


def test_should_never_read_guard_or_outer_values(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = fixed_trigger_replay_manifest()
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
        return replace(original_build(*args, **kwargs), inner_guard=sealed, outer_selection=sealed)

    monkeypatch.setattr(arrived, "_build_phase_a_roles", sealed_roles)
    session = TriggerReplayScheduleSession(manifest, seed=47)
    assert session.complete_wake_epoch(manifest, source_role="train").selected_ids


def test_should_repeat_exact_unscored_payload_without_final_fields() -> None:
    first = build_payload()
    assert first == build_payload()
    parsed = json.loads(first)
    assert len(parsed["rows"]) == 2
    assert all(len(row["opportunities"]) == 24 for row in parsed["rows"])
    assert all(
        "final_test" not in opportunity and "selection" not in opportunity
        for row in parsed["rows"]
        for opportunity in row["opportunities"]
    )
