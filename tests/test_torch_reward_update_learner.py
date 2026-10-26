"""Fake-native boundary checks for the explicit reward-only learner wrapper."""

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
import pytest

import src.adapters.torch_reward_update_learner as port
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy
from src.core.demonstration_episode import DemonstrationTransition
from src.core.learner_ports import TrainingDiagnostic
from src.core.policy_gradient_episode import PolicyGradientEpisode


@dataclass
class State:
    policy: TorchRecurrentPolicy
    parameters: list[float]
    optimizer: list[float]
    gradients: list[float]
    completed_updates: int = 0
    training_examples: int = 0


class Session:
    def __init__(self, native):
        self.native = native.fork()
        self.completed_steps = 0

    def predict(self, features):
        self.completed_steps += 1
        return self.native.predict(features)

    def snapshot_state(self):
        return self.native.snapshot_state()


class Native:
    def __init__(self, policy):
        self.state = State(deepcopy(policy), [0.25], [0.1], [0.2])
        self.calls = []
        self.failure = None

    @property
    def policy(self):
        return deepcopy(self.state.policy)

    @property
    def completed_updates(self):
        return self.state.completed_updates

    @property
    def training_examples(self):
        return self.state.training_examples

    def snapshot_state(self):
        return deepcopy(self.state)

    def restore_state(self, state):
        if type(state) is not State or state.policy != self.policy:
            raise ValueError("fake full snapshot differs")
        self.state = deepcopy(state)

    def fork(self):
        owned = Native(self.policy)
        owned.restore_state(self.snapshot_state())
        return owned

    def train_reward_episode(self, episode, behavior_state, *, discount):
        self.calls.append((episode, behavior_state, discount))
        if self.failure is not None:
            raise self.failure
        if (
            episode.role != "train"
            or type(behavior_state) is not State
            or behavior_state != self.state
        ):
            raise ValueError("fake full behavior state is stale or differs")
        self.state.parameters[0] += 1
        self.state.optimizer[0] += 1
        self.state.gradients[0] += 1
        self.state.completed_updates += 1
        self.state.training_examples += len(episode.transitions)
        return TrainingDiagnostic("fake_reward_sum", 0.25)

    def predict(self, features):
        return np.tile(np.array([[0.2, 0.3, 0.5]]), (len(features), 1))

    def start_episode(self):
        return Session(self)


def packet(role="train"):
    step = DemonstrationTransition(0, (0.0, 0.0), 2, 1.0, (0.5, 0.0), True, 3)
    return PolicyGradientEpisode("reward-1", "A", role, (step,), ((0.2, 0.3, 0.5),), "terminal", 3)


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(port, "TorchRecurrentLearner", Native)
    policy = TorchRecurrentPolicy(2, 3, 3, 17)
    learner = port.TorchRewardUpdateLearner(policy)
    return learner, learner._learner, policy


def update(learner, role="train"):
    return port.RewardUpdate(packet(role), learner.snapshot_state())


def test_should_delegate_exact_owned_episode_and_full_state_with_discount_one(setup):
    learner, native, _ = setup
    record = update(learner)
    original = deepcopy(record)
    diagnostic = learner.train_batch(record, record.episode.episode_id)
    assert diagnostic == TrainingDiagnostic("fake_reward_sum", 0.25)
    episode, state, discount = native.calls[-1]
    assert episode == record.episode and episode is not record.episode
    assert state == record.behavior_state and state is not record.behavior_state
    assert discount == 1.0 and type(discount) is float
    assert record == original
    assert (learner.completed_updates, learner.training_examples) == (1, 1)


@pytest.mark.parametrize("role", ["final_test", "probe_train", "probe_test"])
def test_should_refuse_nontraining_roles_before_native_entry(setup, role):
    learner, native, _ = setup
    record = update(learner, role)
    with pytest.raises(ValueError, match="train-role"):
        learner.train_batch(record, "reward-1")
    assert native.calls == []


@pytest.mark.parametrize("target", [None, 1, True, "wrong", " reward-1", np.array([[2.0]])])
def test_should_refuse_array_labels_and_nonmatching_release_identifiers(setup, target):
    learner, native, _ = setup
    with pytest.raises(ValueError, match="release identifier"):
        learner.train_batch(update(learner), target)
    assert native.calls == []


@pytest.mark.parametrize("features", [None, {}, np.zeros((1, 2)), packet()])
def test_should_refuse_non_dto_training_features(setup, features):
    learner, native, _ = setup
    with pytest.raises(ValueError, match="RewardUpdate"):
        learner.train_batch(features, "reward-1")
    assert native.calls == []


def test_should_own_constructor_inputs_and_bound_configuration(setup):
    learner, _, policy = setup
    episode, state = packet(), learner.snapshot_state()
    record = port.RewardUpdate(episode, state)
    object.__setattr__(episode.transitions[0], "reward", -1.0)
    state.parameters[0] = 99
    object.__setattr__(policy, "hidden_dim", 99)
    assert record.episode.transitions[0].reward == 1.0
    assert cast(Any, record.behavior_state).parameters[0] == 0.25
    assert learner.policy.hidden_dim == 3


@pytest.mark.parametrize(
    "field",
    ["parameters", "optimizer", "gradients", "completed_updates", "training_examples", "policy"],
)
def test_should_delegate_full_state_staleness_refusal_without_native_commit(setup, field):
    learner, native, _ = setup
    record = update(learner)
    before = learner.snapshot_state()
    value: Any = (
        [99.0]
        if field in ("parameters", "optimizer", "gradients")
        else replace(before.policy, momentum=0.5)
        if field == "policy"
        else 1
    )
    setattr(record.behavior_state, field, value)
    with pytest.raises(ValueError, match="full behavior state"):
        learner.train_batch(record, "reward-1")
    assert learner.snapshot_state() == before
    assert len(native.calls) == 1


def test_should_refuse_a_preceding_behavior_checkpoint_after_one_success(setup):
    learner, _, _ = setup
    record = update(learner)
    learner.train_batch(record, "reward-1")
    before = learner.snapshot_state()
    with pytest.raises(ValueError, match="stale"):
        learner.train_batch(record, "reward-1")
    assert learner.snapshot_state() == before


@pytest.mark.parametrize("field,value", [("extra", 1), ("episode", None)])
def test_should_refuse_corrupted_frozen_dto_before_native_entry(setup, field, value):
    learner, native, _ = setup
    record = update(learner)
    object.__setattr__(record, field, value)
    with pytest.raises(ValueError):
        learner.train_batch(record, "reward-1")
    assert native.calls == []


def test_should_refuse_corrupted_episode_and_wrong_snapshot_without_reinterpretation(setup):
    learner, native, _ = setup
    record = update(learner)
    object.__setattr__(record.episode, "role", "fake")
    with pytest.raises(ValueError):
        learner.train_batch(record, "reward-1")
    assert native.calls == []
    record = update(learner)
    object.__setattr__(record, "behavior_state", {})
    with pytest.raises(ValueError, match="full behavior state"):
        learner.train_batch(record, "reward-1")


def test_should_preserve_primary_native_error_and_original_packet(setup):
    learner, native, _ = setup
    record = update(learner)
    before = deepcopy(record)
    error = RuntimeError("native update failed")
    native.failure = error
    with pytest.raises(RuntimeError) as caught:
        learner.train_batch(record, "reward-1")
    assert caught.value is error and record == before


def test_should_fork_own_state_configuration_and_pass_through_opaque_snapshots(setup):
    learner, _, _ = setup
    learner.train_batch(update(learner), "reward-1")
    owned = learner.fork()
    before = learner.snapshot_state()
    assert type(owned) is port.TorchRewardUpdateLearner
    assert type(before) is State and owned.snapshot_state() == before
    owned.train_batch(update(owned), "reward-1")
    assert learner.snapshot_state() == before
    state = cast(Any, owned.snapshot_state())
    state.parameters[0] = 55
    assert cast(Any, owned.snapshot_state()).parameters[0] != 55
    learner.restore_state(owned.snapshot_state())
    assert learner.snapshot_state() == owned.snapshot_state()
    assert set(vars(owned)) == {"_learner"}


def test_should_delegate_prediction_and_fresh_independent_zero_reset_sessions(setup):
    learner, _, _ = setup
    features = np.zeros((1, 2))
    before = learner.snapshot_state()
    result = learner.predict(features)
    np.testing.assert_array_equal(result, [[0.2, 0.3, 0.5]])
    first, second = learner.start_episode(), learner.start_episode()
    first.predict(features)
    assert first.completed_steps == 1 and second.completed_steps == 0
    learner.train_batch(update(learner), "reward-1")
    assert first.snapshot_state() == second.snapshot_state() == before


def test_should_preserve_restore_refusal_and_decline_wrapper_subclasses(setup):
    learner, _, policy = setup
    before = learner.snapshot_state()
    with pytest.raises(ValueError, match="snapshot"):
        learner.restore_state(None)
    assert learner.snapshot_state() == before

    class Derived(port.TorchRewardUpdateLearner):
        pass

    derived = Derived(policy)
    with pytest.raises(ValueError, match="exact"):
        derived.train_batch(update(derived), "reward-1")
    with pytest.raises(ValueError, match="exact"):
        derived.fork()
