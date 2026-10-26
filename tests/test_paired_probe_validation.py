"""Pure role/admission and typed paid-failure boundaries for paired probes."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.adapters import torch_paired_probe as adapter
from src.adapters import torch_recurrent_episode_batch as native
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy, TorchRecurrentSnapshot
from test_paired_probe_native import pair_arrays, probe_episodes
from test_paired_episode_stream import FakeNative, fake_snapshot
from test_minigrid_learning_pilot import same_state


def forbidden(*args, **kwargs):
    raise AssertionError("preadmission reached Torch or native construction")


def header():
    return native.TorchRecurrentEpisodeBatchSnapshot(
        1,
        TorchRecurrentPolicy(2, 3, 3, 17),
        {"weight": np.zeros((1, 1), dtype=np.float32)},
        {"state": {}, "param_groups": []},
        {"weight": None},
        0,
        0,
        0,
    )


@pytest.fixture
def no_native(monkeypatch):
    monkeypatch.setattr(native, "require_torch", forbidden)
    monkeypatch.setattr(native.native, "require_torch", forbidden)


@pytest.mark.parametrize(
    "method,role",
    [
        ("train_episode_pair", "probe_train"),
        ("train_probe_episode_pair", "train"),
        ("train_probe_episode_pair", "final_test"),
        ("train_probe_episode_pair", "probe_test"),
    ],
)
def test_should_refuse_wrong_role_at_public_native_entry_before_torch(no_native, method, role):
    source = object.__new__(native.TorchRecurrentEpisodeBatchLearner)
    source._policy = TorchRecurrentPolicy(2, 3, 3, 17)
    episodes = tuple(replace(episode, role=role) for episode in probe_episodes()[:2])
    with pytest.raises(ValueError):
        getattr(source, method)(episodes)


def test_should_preserve_default_train_pair_validation_and_original_probe_role_without_relabel(
    no_native,
):
    episodes = probe_episodes()[:2]
    policy = TorchRecurrentPolicy(2, 3, 3, 17)
    with pytest.raises(ValueError):
        native._pair(episodes, policy)
    owned = native._pair(episodes, policy, role="probe_train")
    assert owned == episodes and all(item.role == "probe_train" for item in owned)
    assert all(left is not right for left, right in zip(owned, episodes, strict=True))
    train = tuple(replace(item, role="train") for item in episodes)
    assert native._pair(train, policy) == train
    assert all(item.role == "probe_train" for item in episodes)


@pytest.mark.parametrize(
    "fault",
    [
        "foreign",
        "legacy",
        "snapshot_extra",
        "count_bool",
        "empty",
        "odd",
        "overlong",
        "list",
        "train",
        "final_test",
        "probe_test",
        "duplicate_ids",
        "width",
        "classes",
        "extra_episode",
    ],
)
def test_should_reject_invalid_probe_admission_before_torch_copy_or_native_constructor(
    monkeypatch, no_native, fault
):
    state: Any = header()
    episodes: Any = probe_episodes()
    if fault == "foreign":

        class Foreign:
            def __deepcopy__(self, memo):
                pytest.fail("foreign snapshot was copied")

        state = Foreign()
    elif fault == "legacy":
        state = TorchRecurrentSnapshot(
            1, state.policy, state.parameters, state.optimizer_state, state.gradients, 0, 0
        )
    elif fault == "snapshot_extra":
        object.__setattr__(state, "unrecorded", True)
    elif fault == "count_bool":
        state = replace(state, completed_updates=True)
    elif fault == "empty":
        episodes = ()
    elif fault == "odd":
        episodes = episodes[:3]
    elif fault == "overlong":
        episodes = episodes * 9
    elif fault == "list":
        episodes = list(episodes)
    elif fault in ("train", "final_test", "probe_test"):
        episodes = (replace(episodes[0], role=fault), *episodes[1:])
    elif fault == "duplicate_ids":
        episodes = (episodes[0], episodes[0])
    elif fault == "width":
        transition = episodes[0].transitions[0]
        object.__setattr__(transition, "observation", (0.0,))
    elif fault == "classes":
        object.__setattr__(episodes[0], "num_classes", 4)
    else:
        object.__setattr__(episodes[0], "unexpected", True)
    monkeypatch.setattr(adapter, "TorchRecurrentEpisodeBatchLearner", forbidden)
    with pytest.raises(ValueError):
        adapter.TorchPairedProbeLearner(state, episodes)


@pytest.fixture
def fake_probe(monkeypatch):
    work: dict[str, Any] = dict(owners=[], trains=[], predictions=[])
    monkeypatch.setattr(FakeNative, "work", work)
    monkeypatch.setattr(adapter, "TorchRecurrentEpisodeBatchLearner", FakeNative)
    monkeypatch.setattr(adapter, "_same_state", same_state)
    monkeypatch.setattr(native, "_snapshot", fake_snapshot)

    def train(owner, episodes):
        assert all(episode.role == "probe_train" for episode in episodes)
        rows = sum(len(episode.transitions) for episode in episodes)
        work["trains"].append(deepcopy(episodes))
        owner.state = replace(
            owner.state,
            parameters={"weight": owner.state.parameters["weight"] + 0.1},
            optimizer_state={"momentum": owner.state.optimizer_state["momentum"] + 0.5},
            gradients={"weight": np.ones((1, 1), dtype=np.float32)},
            completed_updates=owner.state.completed_updates + 1,
            training_episodes=owner.state.training_episodes + 2,
            training_examples=owner.state.training_examples + rows,
        )
        callback = work.get("after_train")
        if callback:
            callback()
        observed = dict(
            forward_calls_attempted=2,
            forward_calls_completed=2,
            forward_rows_attempted=rows,
            forward_rows_completed=rows,
            backward_calls_attempted=1,
            backward_calls_completed=1,
            backward_rows_attempted=rows,
            backward_rows_completed=rows,
            optimizer_steps_attempted=1,
            optimizer_steps=1,
            published_updates=1,
            failure_work_may_be_unobserved=False,
        )
        return native.EpisodePairDiagnostic(
            native.TORCH_RECURRENT_EPISODE_PAIR_CE_ID,
            0.5,
            tuple(episode.episode_id for episode in episodes),
            (0.4, 0.6),
            observed,
        )

    monkeypatch.setattr(FakeNative, "train_probe_episode_pair", train, raising=False)
    state = replace(
        header(),
        completed_updates=1,
        training_episodes=2,
        training_examples=5,
        optimizer_state={"momentum": 0.5},
        gradients={"weight": np.ones((1, 1), dtype=np.float32)},
    )
    episodes = probe_episodes()
    source = adapter.TorchPairedProbeLearner(state, episodes)
    return work, state, episodes, source


def test_should_select_probe_pairs_by_local_offset_and_own_forks_restores_and_exhaustion(
    fake_probe,
):
    work, state, episodes, source = fake_probe
    before = source.snapshot_state()
    child = source.fork()
    for index in range(2):
        child.train_batch(*pair_arrays(episodes[2 * index : 2 * index + 2]))
        assert work["trains"][-1] == episodes[2 * index : 2 * index + 2]
    assert (child.completed_updates, child.training_episodes, child.training_examples) == (3, 6, 11)
    assert same_state(source.snapshot_state(), before)
    assert child._episodes is not source._episodes
    with pytest.raises(ValueError):
        child.train_batch(*pair_arrays(episodes[:2]))
    child.restore_state(state)
    assert same_state(child.snapshot_state(), before)
    assert len(work["trains"]) == 2


@pytest.mark.parametrize("fault", ["features", "targets", "dtype", "next_pair"])
def test_should_refuse_arrays_that_do_not_match_next_owned_pair_before_native_call(
    fake_probe, fault
):
    work, _, episodes, source = fake_probe
    features, targets = pair_arrays(episodes[:2])
    if fault == "features":
        features[0, 0] += 0.1
    elif fault == "targets":
        targets[0, 0] = 2
    elif fault == "dtype":
        features = features.astype(np.float32)
    else:
        features, targets = pair_arrays(episodes[2:])
    with pytest.raises(ValueError):
        source.train_batch(features, targets)
    assert work["trains"] == []


@pytest.mark.parametrize("fault", ["base", "schedule", "state", "prefix_restore"])
def test_should_refuse_changed_warm_base_schedule_state_or_restore_prefix(fake_probe, fault):
    work, state, episodes, source = fake_probe
    restores = source._learner.events.count("restore")
    if fault == "base":
        source._base = tuple(list(source._base))
    elif fault == "schedule":
        object.__setattr__(source._episodes[0], "role", "final_test")
    elif fault == "state":
        source._learner.state = replace(source._learner.state, training_episodes=3)
    else:
        invalid = replace(state, completed_updates=2, training_episodes=4, training_examples=9)
        with pytest.raises(ValueError):
            source.restore_state(invalid)
        assert source._learner.events.count("restore") == restores
        return
    with pytest.raises(ValueError):
        source.train_batch(*pair_arrays(episodes[:2]))
    assert work["trains"] == [] and source._learner.events.count("restore") == restores


@pytest.mark.parametrize("fault", ["arrays", "schedule"])
def test_should_preserve_published_native_work_on_post_update_refusal_without_rollback(
    fake_probe, fault
):
    work, state, episodes, source = fake_probe
    features, targets = pair_arrays(episodes[:2])
    restores = source._learner.events.count("restore")

    def corrupt():
        if fault == "arrays":
            features[0, 0] += 0.1
        else:
            object.__setattr__(source._episodes[0], "role", "final_test")

    work["after_train"] = corrupt
    with pytest.raises(native.EpisodePairUpdateFailed) as error:
        source.train_batch(features, targets)
    assert error.value.work["optimizer_steps"] == error.value.work["published_updates"] == 1
    assert error.value.work["failure_work_may_be_unobserved"] is True
    assert (
        source._learner.state.completed_updates == 2
        and source._learner.state.training_examples == 8
    )
    assert source._learner.events.count("restore") == restores
    assert state.completed_updates == 1 and len(work["trains"]) == 1
    with pytest.raises(ValueError):
        source.snapshot_state()


def test_should_reprove_parent_after_child_snapshot_callback_mutates_it(monkeypatch, fake_probe):
    work, _, _, source = fake_probe
    parent = source._learner
    snapshot = FakeNative.snapshot_state
    child_reads = [0]

    def mutate_parent(owner):
        state = snapshot(owner)
        if owner is not parent:
            child_reads[0] += 1
            if child_reads[0] == 2:
                parent.state.parameters["weight"][0, 0] += 1
        return state

    monkeypatch.setattr(FakeNative, "snapshot_state", mutate_parent)
    with pytest.raises(ValueError):
        source.fork()
    assert child_reads[0] == 2 and parent.state.parameters["weight"][0, 0] == 1.0
    assert work["trains"] == []
