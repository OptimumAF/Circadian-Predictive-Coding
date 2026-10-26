"""Sequential weighted adapter behavior on an owned fake native learner."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.adapters import torch_train_weighted_learner as weighted
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy, TorchRecurrentSnapshot
from src.app.coupled_attribution import _same_value
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    episode_arrays,
)
from src.core.learner_ports import TrainingDiagnostic


def episode(name="first", labels=(0, 0, 1), *, role="train", classes=3):
    observations = tuple((float(i) / 10, 0.0) for i in range(len(labels) + 1))
    return DemonstrationEpisode(
        name,
        "A",
        role,
        tuple(
            DemonstrationTransition(
                i, observations[i], action, 0.0, observations[i + 1], i == len(labels) - 1, classes
            )
            for i, action in enumerate(labels)
        ),
        classes,
    )


def schedule():
    return (episode(), episode("second", (2, 2, 2, 0)))


class Native:
    births = 0
    failure = None

    def __init__(self, policy):
        type(self).births += 1
        self.policy = deepcopy(policy)
        self.state = TorchRecurrentSnapshot(
            1, deepcopy(policy), {"weight": 1.0}, {"momentum": 0.5}, {"weight": 0.25}, 0, 0
        )
        self.calls = []
        self.events = []

    def snapshot_state(self):
        return deepcopy(self.state)

    def restore_state(self, state):
        self.events.append("restore")
        self.state = deepcopy(state)

    def fork(self):
        self.events.append("fork")
        child = Native(self.policy)
        child.state = deepcopy(self.state)
        return child

    def train_weighted_episode(self, record, weights):
        self.calls.append((deepcopy(record), weights))
        if self.failure == "native":
            raise RuntimeError("native update failed")
        self.state = replace(
            self.state,
            parameters={"weight": self.state.parameters["weight"] + 1},
            optimizer_state={"momentum": self.state.optimizer_state["momentum"] + 0.5},
            gradients={"weight": 0.75},
            completed_updates=self.state.completed_updates + 1,
            training_examples=self.state.training_examples + len(record.transitions),
        )
        if self.failure == "counter":
            self.state = replace(self.state, training_examples=self.state.training_examples + 1)
        definition = (
            "wrong_objective"
            if self.failure == "objective"
            else weighted.TORCH_RECURRENT_WEIGHTED_CE_ID
        )
        return TrainingDiagnostic(definition, 0.75)

    def predict(self, features):
        self.events.append("predict")
        return np.full((len(features), self.policy.num_classes), 1 / self.policy.num_classes)

    def start_episode(self):
        self.events.append("session")
        return Session(self.state)


class Session:
    def __init__(self, state):
        self.state, self.completed_steps = deepcopy(state), 0

    def snapshot_state(self):
        return deepcopy(self.state)


@pytest.fixture(autouse=True)
def fake_native(monkeypatch):
    monkeypatch.setattr(weighted, "TorchRecurrentLearner", Native)
    monkeypatch.setattr(weighted, "_same_state", _same_value)
    Native.births, Native.failure = 0, None


def learner(episodes=None):
    return weighted.TorchTrainWeightedLearner(
        TorchRecurrentPolicy(2, 2, 3, 17), schedule() if episodes is None else episodes
    )


def test_should_weight_only_each_actual_current_episode_and_keep_native_diagnostic_and_state():
    source = learner()
    first, second = schedule()
    initial = source.snapshot_state()
    result = source.train_batch(*episode_arrays(first))
    assert result == TrainingDiagnostic(weighted.TORCH_RECURRENT_WEIGHTED_CE_ID, 0.75)
    assert source._learner.calls[0] == (first, (0.75, 1.5, 1.0))
    assert source.completed_updates == 1 and source.training_examples == 3
    source.train_batch(*episode_arrays(second))
    assert source._learner.calls[1] == (second, (2.0, 1.0, 2 / 3))
    final = source.snapshot_state()
    assert type(final) is TorchRecurrentSnapshot and final.format_version == initial.format_version
    assert source.completed_updates == 2 and source.training_examples == 7
    assert final.optimizer_state == {"momentum": 1.5} and final.gradients == {"weight": 0.75}
    assert initial.parameters == {"weight": 1.0} and final.parameters == {"weight": 3.0}


def test_should_own_schedule_and_training_arrays_without_inventing_transition_evidence():
    originals = schedule()
    source = learner(originals)
    first = deepcopy(originals[0])
    object.__setattr__(originals[0], "role", "final_test")
    features, targets = episode_arrays(first)
    source.train_batch(features, targets)
    features.fill(-1)
    targets.fill(2)
    supplied, weights = source._learner.calls[0]
    assert supplied == first and supplied is not originals[0]
    assert weights == (0.75, 1.5, 1.0)
    assert supplied.transitions[0].reward == 0.0
    assert supplied.transitions[0].next_observation == first.transitions[0].next_observation


@pytest.mark.parametrize(
    "fault",
    [
        "empty",
        "list",
        "huge",
        "final",
        "probe",
        "classes",
        "dimension",
        "tampered",
        "extra",
        "policy",
    ],
)
def test_should_refuse_invalid_schedule_before_creating_native_resources(fault):
    episodes: Any = schedule()
    policy: Any = TorchRecurrentPolicy(2, 2, 3, 17)
    if fault == "empty":
        episodes = ()
    elif fault == "list":
        episodes = list(episodes)
    elif fault == "huge":
        episodes = (episodes[0],) * 513
    elif fault == "final":
        episodes = (episode(role="final_test"),)
    elif fault == "probe":
        episodes = (episode(role="probe_train"),)
    elif fault == "classes":
        episodes = (episode(classes=4),)
    elif fault == "dimension":
        policy = TorchRecurrentPolicy(3, 2, 3, 17)
    elif fault == "tampered":
        object.__setattr__(episodes[1].transitions[-1], "terminal", False)
    elif fault == "extra":
        object.__setattr__(episodes[1], "unowned", 1)
    else:
        policy = {"feature_dim": 2}
    with pytest.raises(ValueError):
        weighted.TorchTrainWeightedLearner(policy, episodes)
    assert Native.births == 0


@pytest.mark.parametrize("fault", ["features", "targets", "order", "dtype", "list", "shape", "nan"])
def test_should_refuse_nonmatching_arrays_before_sgd_and_preserve_full_owner_state(fault):
    source = learner()
    before = source.snapshot_state()
    features, targets = episode_arrays(schedule()[0])
    if fault == "features":
        features[0, 0] += 0.1
    elif fault == "targets":
        targets[0, 0] = 2
    elif fault == "order":
        features, targets = episode_arrays(schedule()[1])
    elif fault == "dtype":
        features = features.astype(np.float32)
    elif fault == "list":
        features = features.tolist()
    elif fault == "shape":
        targets = targets[:, 0]
    else:
        features[0, 0] = np.nan
    with pytest.raises(ValueError, match="exactly match"):
        source.train_batch(features, targets)
    assert _same_value(source.snapshot_state(), before) and source._learner.calls == []


@pytest.mark.parametrize(
    "fault",
    ["schedule", "role", "future_label", "extra", "objective", "owner", "policy", "native_state"],
)
def test_should_refuse_owned_schedule_objective_or_native_owner_mutation_before_work(fault):
    source = learner()
    owner = source._learner
    if fault == "schedule":
        source._episodes = tuple(deepcopy(schedule()))
    elif fault == "role":
        object.__setattr__(source._episodes[1], "role", "final_test")
    elif fault == "future_label":
        object.__setattr__(source._episodes[1].transitions[0], "action", True)
    elif fault == "extra":
        object.__setattr__(source._episodes[1].transitions[0], "unowned", 1)
    elif fault == "objective":
        source._objective_id = "mean_ce"
    elif fault == "owner":
        source._learner = Native(source._policy)
    elif fault == "policy":
        object.__setattr__(source._policy, "momentum", 0.5)
    else:
        source._learner.state.parameters["weight"] = 7.0
    with pytest.raises(ValueError, match="changed"):
        source.train_batch(*episode_arrays(schedule()[0]))
    assert owner.calls == []


@pytest.mark.parametrize("fault", ["native", "counter", "objective"])
def test_should_preserve_owner_state_and_schedule_position_on_failed_update(fault):
    source = learner()
    before = source.snapshot_state()
    source._learner.failure = fault
    with pytest.raises((RuntimeError, ValueError)):
        source.train_batch(*episode_arrays(schedule()[0]))
    assert _same_value(source.snapshot_state(), before)
    source._learner.failure = None
    source.train_batch(*episode_arrays(schedule()[0]))
    assert source.completed_updates == 1


def test_should_restore_native_checkpoints_and_select_their_next_schedule_entry():
    source = learner()
    initial = source.snapshot_state()
    for record in schedule():
        source.train_batch(*episode_arrays(record))
    final = source.snapshot_state()
    source.restore_state(initial)
    source.train_batch(*episode_arrays(schedule()[0]))
    assert source.completed_updates == 1
    source.restore_state(final)
    with pytest.raises(ValueError, match="exhausted"):
        source.train_batch(*episode_arrays(schedule()[1]))
    assert _same_value(source.snapshot_state(), final)


@pytest.mark.parametrize("fault", ["negative", "bool", "beyond", "rows", "policy"])
def test_should_refuse_misaligned_restore_before_native_state_changes(fault):
    source = learner()
    before = source.snapshot_state()
    snapshot = replace(before)
    if fault == "negative":
        snapshot = replace(before, completed_updates=-1)
    elif fault == "bool":
        snapshot = replace(before, completed_updates=False)
    elif fault == "beyond":
        snapshot = replace(before, completed_updates=3, training_examples=10)
    elif fault == "rows":
        snapshot = replace(before, completed_updates=1, training_examples=4)
    else:
        snapshot = replace(before, policy=TorchRecurrentPolicy(2, 3, 3, 17))
    with pytest.raises(ValueError, match="schedule prefix"):
        source.restore_state(snapshot)
    assert source._learner.events == [] and _same_value(source.snapshot_state(), before)


def test_should_fork_full_native_state_with_independent_schedule_and_predict_without_sgd():
    source = learner()
    source.train_batch(*episode_arrays(schedule()[0]))
    before = source.snapshot_state()
    child = source.fork()
    assert child is not source and child._learner is not source._learner
    assert child._episodes is not source._episodes
    assert child._episodes[0] is not source._episodes[0]
    child.train_batch(*episode_arrays(schedule()[1]))
    assert source.completed_updates == 1 and child.completed_updates == 2
    assert _same_value(source.snapshot_state(), before)
    child_before = child.snapshot_state()
    assert child.predict(np.ones((1, 2))).shape == (1, 3)
    session = child.start_episode()
    assert session.completed_steps == 0 and _same_value(session.snapshot_state(), child_before)
    assert _same_value(child.snapshot_state(), child_before)
    object.__setattr__(child._episodes[1], "role", "probe_test")
    with pytest.raises(ValueError, match="changed"):
        child.snapshot_state()
    assert _same_value(source.snapshot_state(), before)


def test_should_deepcopy_ownership_aliases_without_sharing_native_or_schedule_objects():
    source = learner()
    child = deepcopy(source)
    assert child._native_owner is child._learner and child._schedule_owner is child._episodes
    assert child._learner is not source._learner
    child.train_batch(*episode_arrays(schedule()[0]))
    assert source.completed_updates == 0 and child.completed_updates == 1


def test_should_rollback_admitted_owner_after_update_replaces_mutable_owner_binding(monkeypatch):
    source = learner()
    admitted = source._learner
    before = admitted.snapshot_state()
    original = admitted.train_weighted_episode

    def replace_binding(record, weights):
        result = original(record, weights)
        source._native_owner = Native(source._policy)
        return result

    monkeypatch.setattr(admitted, "train_weighted_episode", replace_binding)
    with pytest.raises(ValueError, match="owner changed"):
        source.train_batch(*episode_arrays(schedule()[0]))
    assert _same_value(admitted.snapshot_state(), before)
    assert len(admitted.calls) == 1  # Execution happened; rollback does not erase work evidence.
    assert source._native_owner is not admitted


def test_should_rollback_admitted_owner_after_restore_replaces_mutable_owner_binding(monkeypatch):
    source = learner()
    initial = source.snapshot_state()
    source.train_batch(*episode_arrays(schedule()[0]))
    admitted = source._learner
    before = admitted.snapshot_state()
    original = admitted.restore_state
    replaced = False

    def replace_binding(snapshot):
        nonlocal replaced
        original(snapshot)
        if not replaced:
            replaced = True
            source._native_owner = Native(source._policy)

    monkeypatch.setattr(admitted, "restore_state", replace_binding)
    with pytest.raises(ValueError, match="owner changed"):
        source.restore_state(initial)
    assert _same_value(admitted.snapshot_state(), before)
    assert admitted.events.count("restore") == 2
