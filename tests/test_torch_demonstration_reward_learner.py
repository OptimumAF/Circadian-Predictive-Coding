"""Fake-native phase dispatch preserves one complete optimizer/state owner."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from typing import Any, cast

import numpy as np
import pytest

import src.adapters.torch_demonstration_reward_learner as port
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy
from src.adapters.torch_reward_update_learner import RewardUpdate
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from src.core.learner_ports import TrainingDiagnostic
from src.core.policy_gradient_episode import PolicyGradientEpisode
from test_torch_reward_update_learner import Native as RewardNative, State


class Native(RewardNative):
    def train_episode(self, episode):
        self.calls.append(("demonstration", episode))
        if self.failure is not None:
            raise self.failure
        assert episode.role == "train" and episode.num_classes == self.policy.num_classes
        self.state.parameters[0] += 1
        self.state.optimizer[0] += 1
        self.state.gradients[0] += 1
        self.state.completed_updates += 1
        self.state.training_examples += len(episode.transitions)
        return TrainingDiagnostic("fake_episode_ce_mean", 0.5)

    def fork(self):
        owned = Native(self.policy)
        owned.restore_state(self.snapshot_state())
        return owned


def demonstration(role="train", classes=3):
    transition = DemonstrationTransition(0, (0.0, 0.0), classes - 1, 1.0, (0.5, 0.0), True, classes)
    return DemonstrationEpisode("demonstration-1", "A", role, (transition,), classes)


def reward(learner, *, role="train", classes=3):
    demo = demonstration(role, classes)
    episode = PolicyGradientEpisode(
        "reward-1", "A", role, demo.transitions, ((1.0 / classes,) * classes,), "terminal", classes
    )
    return RewardUpdate(episode, learner.snapshot_state())


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(port, "TorchRecurrentLearner", Native)
    policy = TorchRecurrentPolicy(2, 3, 3, 17)
    learner = port.TorchDemonstrationRewardLearner(policy)
    return learner, learner._learner, policy


def test_should_keep_one_native_optimizer_owner_across_both_objectives(setup):
    learner, native, _ = setup
    demo = port.DemonstrationUpdate(demonstration())
    first = learner.train_batch(demo, demo.episode.episode_id)
    behavior = learner.snapshot_state()
    packet = reward(learner)
    second = learner.train_batch(packet, packet.episode.episode_id)
    assert learner._learner is native
    assert first.definition == "fake_episode_ce_mean" and second.definition == "fake_reward_sum"
    assert native.calls[0][0] == "demonstration"
    assert native.calls[0][1] == demo.episode and native.calls[0][1] is not demo.episode
    assert native.calls[1][0] == packet.episode and native.calls[1][0] is not packet.episode
    assert native.calls[1][1] == behavior and native.calls[1][1] is not packet.behavior_state
    assert native.calls[1][2] == 1.0 and type(native.calls[1][2]) is float
    assert (learner.completed_updates, learner.training_examples) == (2, 2)
    after = learner.snapshot_state()
    assert after.parameters == [2.25] and after.optimizer == [2.1] and after.gradients == [2.2]
    assert set(vars(learner)) == {"_learner"}


def test_should_own_demonstration_packets_and_preserve_immutable_contract():
    original = demonstration()
    packet = port.DemonstrationUpdate(original)
    object.__setattr__(original.transitions[0], "reward", -1.0)
    assert packet.episode.transitions[0].reward == 1.0
    assert packet.episode is not original
    with pytest.raises(FrozenInstanceError):
        packet.episode = original  # type: ignore[misc]


@pytest.mark.parametrize("role", ["final_test", "probe_train", "probe_test"])
def test_should_refuse_nontraining_demonstration_construction(role):
    with pytest.raises(ValueError, match="train-role"):
        port.DemonstrationUpdate(demonstration(role))


@pytest.mark.parametrize("role", ["final_test", "probe_train", "probe_test"])
def test_should_refuse_nontraining_reward_packets_before_native_entry(setup, role):
    learner, native, _ = setup
    with pytest.raises(ValueError, match="train-role"):
        learner.train_batch(reward(learner, role=role), "reward-1")
    assert native.calls == []


@pytest.mark.parametrize("kind", ["demonstration", "reward"])
@pytest.mark.parametrize("target", [None, True, "wrong", " demonstration-1", np.zeros((1, 1))])
def test_should_require_exact_matching_string_release_identifier(setup, kind, target):
    learner, native, _ = setup
    packet = (
        port.DemonstrationUpdate(demonstration()) if kind == "demonstration" else reward(learner)
    )
    with pytest.raises(ValueError, match="release identifier"):
        learner.train_batch(packet, target)
    assert native.calls == []


@pytest.mark.parametrize("kind", ["demonstration", "reward"])
def test_should_bind_class_count_before_native_training(setup, kind):
    learner, native, _ = setup
    packet = (
        port.DemonstrationUpdate(demonstration(classes=2))
        if kind == "demonstration"
        else reward(learner, classes=2)
    )
    with pytest.raises(ValueError, match="class count"):
        learner.train_batch(packet, packet.episode.episode_id)
    assert native.calls == []


@pytest.mark.parametrize("packet", [None, {}, np.zeros((1, 2)), demonstration()])
def test_should_refuse_implicit_array_or_record_fallback(setup, packet):
    learner, native, _ = setup
    with pytest.raises(ValueError, match="exact complete"):
        learner.train_batch(packet, "demonstration-1")
    assert native.calls == []


@pytest.mark.parametrize("kind", ["demonstration", "reward"])
@pytest.mark.parametrize("field,value", [("extra", 1), ("episode", None)])
def test_should_refuse_forged_dto_fields_before_native_entry(setup, kind, field, value):
    learner, native, _ = setup
    packet = (
        port.DemonstrationUpdate(demonstration()) if kind == "demonstration" else reward(learner)
    )
    object.__setattr__(packet, field, value)
    with pytest.raises(ValueError):
        learner.train_batch(packet, "demonstration-1")
    assert native.calls == []


@pytest.mark.parametrize("field,value", [("extra", 1), ("role", "final_test"), ("num_classes", 2)])
def test_should_revalidate_frozen_demonstration_corruption(setup, field, value):
    learner, native, _ = setup
    packet = port.DemonstrationUpdate(demonstration())
    object.__setattr__(packet.episode, field, value)
    with pytest.raises(ValueError):
        learner.train_batch(packet, "demonstration-1")
    assert native.calls == []


def test_should_keep_full_state_staleness_guard_after_demonstration_update(setup):
    learner, _, _ = setup
    stale = reward(learner)
    learner.train_batch(port.DemonstrationUpdate(demonstration()), "demonstration-1")
    before = learner.snapshot_state()
    with pytest.raises(ValueError, match="stale"):
        learner.train_batch(stale, "reward-1")
    assert learner.snapshot_state() == before


@pytest.mark.parametrize("kind", ["demonstration", "reward"])
def test_should_preserve_primary_native_failure_and_owned_inputs(setup, kind):
    learner, native, _ = setup
    packet = (
        port.DemonstrationUpdate(demonstration()) if kind == "demonstration" else reward(learner)
    )
    before, saved_packet = learner.snapshot_state(), deepcopy(packet)
    error = RuntimeError("native phase update failed")
    native.failure = error
    with pytest.raises(RuntimeError) as caught:
        learner.train_batch(packet, packet.episode.episode_id)
    assert caught.value is error and packet == saved_packet
    assert learner.snapshot_state() == before


def test_should_fork_and_restore_same_opaque_native_state_without_phase_counters(setup):
    learner, _, policy = setup
    learner.train_batch(port.DemonstrationUpdate(demonstration()), "demonstration-1")
    original = learner.snapshot_state()
    owned = learner.fork()
    assert type(owned) is port.TorchDemonstrationRewardLearner
    assert type(owned.snapshot_state()) is State
    owned.train_batch(reward(owned), "reward-1")
    assert learner.snapshot_state() == original
    assert set(vars(owned)) == {"_learner"}
    state = cast(Any, owned.snapshot_state())
    learner.restore_state(state)
    state.optimizer[0] = 99
    assert learner.snapshot_state() == owned.snapshot_state()
    object.__setattr__(policy, "hidden_dim", 99)
    assert learner.policy.hidden_dim == owned.policy.hidden_dim == 3


def test_should_delegate_predictions_and_independent_zero_reset_sessions(setup):
    learner, _, _ = setup
    before = learner.snapshot_state()
    features = np.zeros((1, 2))
    np.testing.assert_array_equal(learner.predict(features), [[0.2, 0.3, 0.5]])
    first, second = learner.start_episode(), learner.start_episode()
    first.predict(features)
    assert first.completed_steps == 1 and second.completed_steps == 0
    learner.train_batch(port.DemonstrationUpdate(demonstration()), "demonstration-1")
    assert first.snapshot_state() == second.snapshot_state() == before


def test_should_refuse_dto_and_wrapper_subclasses(setup):
    learner, _, policy = setup

    class DerivedDTO(port.DemonstrationUpdate):
        pass

    class DerivedLearner(port.TorchDemonstrationRewardLearner):
        pass

    with pytest.raises(ValueError, match="exact"):
        DerivedDTO(demonstration())
    derived = DerivedLearner(policy)
    with pytest.raises(ValueError, match="exact"):
        derived.train_batch(port.DemonstrationUpdate(demonstration()), "demonstration-1")
    with pytest.raises(ValueError, match="exact"):
        derived.fork()
    before = learner.snapshot_state()
    with pytest.raises(ValueError, match="snapshot"):
        learner.restore_state(None)
    assert learner.snapshot_state() == before
