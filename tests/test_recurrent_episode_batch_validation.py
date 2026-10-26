"""New step/episode clocks reject invalid admission before native validation."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

from src.adapters import torch_recurrent_episode_batch as batch
from src.adapters import torch_recurrent_learner as native
from test_torch_feature_learner import forbidden
from test_torch_recurrent_learner import episode


def snapshot(**changes):
    values: dict[str, Any] = dict(
        format_version=1,
        policy=native.TorchRecurrentPolicy(2, 3, 3, 17, max_episode_steps=4),
        parameters={},
        optimizer_state={},
        gradients={},
        completed_updates=0,
        training_episodes=0,
        training_examples=0,
    )
    values.update(changes)
    return batch.TorchRecurrentEpisodeBatchSnapshot(**values)


@pytest.mark.parametrize(
    "changes",
    [
        {"completed_updates": True},
        {"training_episodes": False},
        {"training_examples": True},
        {"completed_updates": -1},
        {"training_episodes": -1},
        {"training_examples": -1},
        {"completed_updates": 2**53},
        {"training_episodes": 2**53},
        {"training_examples": 2**53},
        {"completed_updates": 1.0},
        {"training_episodes": 1.0},
        {"training_examples": 1.0},
        {"completed_updates": 2, "training_episodes": 1, "training_examples": 2},
        {"completed_updates": 1, "training_episodes": 3, "training_examples": 3},
        {"completed_updates": 1, "training_episodes": 2, "training_examples": 1},
        {"completed_updates": 1, "training_episodes": 2, "training_examples": 9},
        {"completed_updates": 0, "training_episodes": 1, "training_examples": 1},
        {"completed_updates": 1, "training_episodes": 0, "training_examples": 1},
        {"training_examples": 1},
    ],
)
def test_should_refuse_invalid_clock_relations_before_native_validation(monkeypatch, changes):
    monkeypatch.setattr(native, "_snapshot", forbidden)
    with pytest.raises(ValueError):
        batch._snapshot(snapshot(**changes))


@pytest.mark.parametrize("version", [True, 0, 2, 1.0])
def test_should_refuse_unsupported_snapshot_version_before_native_work(monkeypatch, version):
    monkeypatch.setattr(native, "_snapshot", forbidden)
    with pytest.raises(ValueError):
        batch._snapshot(snapshot(format_version=version))


def test_should_refuse_extra_snapshot_fields_before_native_copy(monkeypatch):
    incoming = snapshot()
    object.__setattr__(incoming, "unobserved", True)
    monkeypatch.setattr(native, "_snapshot", forbidden)
    with pytest.raises(ValueError):
        batch._snapshot(incoming)


def test_should_refuse_foreign_objects_without_invoking_their_deepcopy_hook(monkeypatch):
    class Foreign:
        def __deepcopy__(self, memo):
            raise AssertionError("foreign object must be refused before copying")

    monkeypatch.setattr(native, "require_torch", forbidden)
    incoming: Any = Foreign()
    with pytest.raises(ValueError):
        batch._snapshot(incoming)
    with pytest.raises(ValueError):
        batch.TorchRecurrentEpisodeBatchLearner.from_single_episode_snapshot(incoming)


def test_should_keep_new_and_legacy_snapshot_admission_separate(monkeypatch):
    legacy: Any = native.TorchRecurrentSnapshot(
        1, native.TorchRecurrentPolicy(2, 3, 3, 17), {}, {}, {}, 0, 0
    )
    monkeypatch.setattr(native, "require_torch", forbidden)
    with pytest.raises(ValueError):
        batch._snapshot(legacy)
    with pytest.raises(ValueError):
        native._snapshot(snapshot())


@pytest.mark.parametrize(
    "steps,episodes,rows", [(0, 0, 0), (1, 1, 1), (1, 2, 8), (2, 3, 9), (129, 130, 520)]
)
def test_should_use_episode_clock_only_in_private_validation_bridge(
    monkeypatch, steps, episodes, rows
):
    bridges = []

    def validate(bridge):
        assert type(bridge) is native.TorchRecurrentSnapshot
        bridges.append(deepcopy(bridge))
        return deepcopy(bridge)

    monkeypatch.setattr(native, "_snapshot", validate)
    incoming = snapshot(
        completed_updates=steps,
        training_episodes=episodes,
        training_examples=rows,
        parameters={"test_only_sentinel": [1]},
    )
    owned = batch._snapshot(incoming)
    assert type(owned) is batch.TorchRecurrentEpisodeBatchSnapshot
    assert (owned.completed_updates, owned.training_episodes, owned.training_examples) == (
        steps,
        episodes,
        rows,
    )
    assert bridges[-1].completed_updates == episodes
    assert bridges[-1].training_examples == rows
    owned.parameters["test_only_sentinel"][0] = 99
    assert incoming.parameters["test_only_sentinel"] == [1]


@pytest.mark.parametrize("length", [0, 5])
def test_should_bound_episode_length_before_scanning_or_copying_transitions(monkeypatch, length):
    incoming = episode()
    object.__setattr__(incoming, "transitions", tuple(object() for _ in range(length)))
    exact = batch._exact

    def bounded_record(value, kind):
        assert kind is type(incoming), "out-of-bounds transition container must not be scanned"
        exact(value, kind)

    monkeypatch.setattr(batch, "_exact", bounded_record)
    monkeypatch.setattr(batch, "copy_episode", forbidden)
    with pytest.raises(ValueError):
        batch._pair((incoming, incoming), snapshot().policy)


@pytest.mark.parametrize("role", ["final_test", "probe_train", "probe_test"])
def test_should_refuse_reserved_episode_roles_before_native_work(monkeypatch, role):
    monkeypatch.setattr(native, "require_torch", forbidden)
    with pytest.raises(ValueError):
        batch._pair(
            (episode(role=role), replace(episode(), episode_id="second")), snapshot().policy
        )


@pytest.mark.parametrize("fault", ["width", "classes", "length", "duplicate"])
def test_should_refuse_episode_geometry_and_duplicate_identifiers_before_native_work(
    monkeypatch, fault
):
    monkeypatch.setattr(native, "require_torch", forbidden)
    first, second, policy = episode(), replace(episode(), episode_id="second"), snapshot().policy
    if fault == "width":
        first = episode(dimension=1)
    elif fault == "classes":
        first = episode(classes=4)
    elif fault == "length":
        policy = replace(policy, max_episode_steps=2)
    else:
        second = first
    with pytest.raises(ValueError):
        batch._pair((first, second), policy)


@pytest.mark.parametrize("container", [[], (), (None, None, None)])
def test_should_refuse_non_pair_containers_before_native_work(monkeypatch, container):
    monkeypatch.setattr(native, "require_torch", forbidden)
    with pytest.raises(ValueError):
        batch._pair(container, snapshot().policy)


def test_should_refuse_extra_episode_fields_before_native_work(monkeypatch):
    first = episode()
    object.__setattr__(first, "foreign", True)
    monkeypatch.setattr(native, "require_torch", forbidden)
    with pytest.raises(ValueError):
        batch._pair((first, replace(episode(), episode_id="second")), snapshot().policy)
