"""Derived train weights and typed release retain ordinary native ownership."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

from src.adapters import torch_demonstration_reward_learner as port
from src.adapters import torch_recurrent_learner as native
from src.adapters import torch_recurrent_weighted as weighted
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from src.core.learner_ports import TrainingDiagnostic
from test_torch_demonstration_reward_learner import Native, reward
from test_torch_feature_rehearsal_pilot import forbidden


def episode(role="train", classes=3):
    def observation(index):
        return (index / 8.0, -index / 8.0)

    return DemonstrationEpisode(
        "weighted-1",
        "A",
        role,
        tuple(
            DemonstrationTransition(
                i, observation(i), action, float(i == 3), observation(i + 1), i == 3, classes
            )
            for i, action in enumerate((0, 0, 1, 2))
        ),
        classes,
    )


class WeightedNative(Native):
    def train_weighted_episode(self, packet, weights):
        self.calls.append(("weighted", packet, weights))
        if self.failure is not None:
            raise self.failure
        self.state.parameters[0] += 1
        self.state.optimizer[0] += 1
        self.state.gradients[0] += 1
        self.state.completed_updates += 1
        self.state.training_examples += len(packet.transitions)
        return TrainingDiagnostic("fake_weighted_episode", 0.75)

    def fork(self):
        owned = WeightedNative(self.policy)
        owned.restore_state(self.snapshot_state())
        return owned


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(port, "TorchRecurrentLearner", WeightedNative)
    return port.TorchDemonstrationRewardLearner(native.TorchRecurrentPolicy(2, 3, 3, 17))


def test_should_derive_owned_frozen_weights_from_only_this_train_episode():
    original = episode()
    packet = port.ClassWeightedDemonstrationUpdate(original)
    assert packet.class_weights == (2 / 3, 4 / 3, 4 / 3)
    assert all(type(w) is float for w in packet.class_weights)
    assert packet.episode == original and packet.episode is not original
    object.__setattr__(original.transitions[0], "action", 2)
    assert packet.episode.transitions[0].action == 0
    with pytest.raises(FrozenInstanceError):
        packet.class_weights = (1.0, 1.0, 1.0)  # type: ignore[misc]
    with pytest.raises(TypeError):
        port.ClassWeightedDemonstrationUpdate(original, class_weights=(1.0,) * 3)  # type: ignore[call-arg]


@pytest.mark.parametrize("role", ["final_test", "probe_train", "probe_test"])
def test_should_refuse_heldout_or_probe_roles_at_packet_construction(role):
    with pytest.raises(ValueError):
        port.ClassWeightedDemonstrationUpdate(episode(role))


def test_should_dispatch_named_weighted_update_on_same_owner_without_normalizing_again(setup):
    packet = port.ClassWeightedDemonstrationUpdate(episode())
    before, owner = deepcopy(packet), setup._learner
    result = setup.train_batch(packet, "weighted-1")
    assert result == TrainingDiagnostic("fake_weighted_episode", 0.75)
    assert owner is setup._learner and packet == before
    assert len(owner.calls) == 1
    kind, owned, weights = owner.calls[0]
    assert kind == "weighted" and owned == packet.episode and owned is not packet.episode
    assert weights == packet.class_weights
    assert (setup.completed_updates, setup.training_examples) == (1, 4)


@pytest.mark.parametrize("release", [None, True, "wrong", " weighted-1", np.zeros((1, 1))])
def test_should_require_matching_release_before_weighted_native_entry(setup, release):
    before = setup.snapshot_state()
    with pytest.raises(ValueError, match="release identifier"):
        setup.train_batch(port.ClassWeightedDemonstrationUpdate(episode()), release)
    assert setup._learner.calls == [] and setup.snapshot_state() == before


@pytest.mark.parametrize(
    "fault", ["extra", "weights", "list", "integer", "labels", "role", "classes"]
)
def test_should_refuse_changed_packet_fields_weights_or_role_before_native_entry(setup, fault):
    packet = port.ClassWeightedDemonstrationUpdate(episode(classes=4 if fault == "classes" else 3))
    if fault == "extra":
        object.__setattr__(packet, "extra", 1)
    elif fault == "weights":
        object.__setattr__(packet, "class_weights", (1.0,) * 3)
    elif fault == "list":
        object.__setattr__(packet, "class_weights", list(packet.class_weights))
    elif fault == "integer":
        object.__setattr__(packet, "class_weights", (1, 1, 1))
    elif fault == "labels":
        object.__setattr__(packet.episode.transitions[0], "action", 2)
    elif fault == "role":
        object.__setattr__(packet.episode, "role", "final_test")
    before = setup.snapshot_state()
    with pytest.raises(ValueError):
        setup.train_batch(packet, "weighted-1")
    assert setup._learner.calls == [] and setup.snapshot_state() == before


def test_should_preserve_same_optimizer_across_weighted_mean_and_reward_updates(setup):
    owner = setup._learner
    setup.train_batch(port.ClassWeightedDemonstrationUpdate(episode()), "weighted-1")
    packet = reward(setup)
    setup.train_batch(packet, packet.episode.episode_id)
    setup.train_batch(port.DemonstrationUpdate(episode()), "weighted-1")
    assert owner is setup._learner
    assert (setup.completed_updates, setup.training_examples) == (3, 9)
    assert set(vars(setup)) == {"_learner"}


def test_should_make_prior_reward_behavior_stale_after_weighted_update(setup):
    stale = reward(setup)
    setup.train_batch(port.ClassWeightedDemonstrationUpdate(episode()), "weighted-1")
    before = setup.snapshot_state()
    with pytest.raises(ValueError, match="stale"):
        setup.train_batch(stale, stale.episode.episode_id)
    assert setup.snapshot_state() == before


def test_should_preserve_failure_identity_and_owned_packet(setup):
    packet = port.ClassWeightedDemonstrationUpdate(episode())
    before, old_packet = setup.snapshot_state(), deepcopy(packet)
    error = RuntimeError("weighted native refused")
    setup._learner.failure = error
    with pytest.raises(RuntimeError) as caught:
        setup.train_batch(packet, "weighted-1")
    assert caught.value is error and packet == old_packet and setup.snapshot_state() == before


def test_should_fork_and_restore_next_weighted_update_without_policy_or_format_changes(setup):
    packet = port.ClassWeightedDemonstrationUpdate(episode())
    setup.train_batch(packet, "weighted-1")
    before = setup.snapshot_state()
    child = setup.fork()
    first = setup.train_batch(packet, "weighted-1")
    second = child.train_batch(packet, "weighted-1")
    assert first == second and setup.snapshot_state() == child.snapshot_state()
    child.restore_state(before)
    assert child.snapshot_state() == before and setup.snapshot_state() != before


@pytest.mark.parametrize("role", ["final_test", "probe_train", "probe_test"])
def test_should_refuse_native_weighted_heldout_roles_before_torch_or_payload(monkeypatch, role):
    learner: Any = object.__new__(native.TorchRecurrentLearner)
    learner._policy = native.TorchRecurrentPolicy(2, 3, 3, 17)
    monkeypatch.setattr(native, "require_torch", forbidden)
    monkeypatch.setattr(weighted, "require_torch", forbidden)
    with pytest.raises(ValueError, match="train-role"):
        learner.train_weighted_episode(episode(role), (1.0,) * 3)


def test_should_refuse_weighted_dto_subclasses():
    class Derived(port.ClassWeightedDemonstrationUpdate):
        pass

    with pytest.raises(ValueError, match="exact"):
        Derived(episode())
