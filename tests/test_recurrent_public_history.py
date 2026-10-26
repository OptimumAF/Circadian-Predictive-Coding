"""Resource-free admission and episode composition for recurrent public history."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_learning_pilot as fixed
from src.adapters import torch_recurrent_episode_batch as paired
from src.adapters import torch_recurrent_learner as native
from src.adapters import torch_recurrent_public_history as history
from src.adapters import torch_train_episode_batch as arrays
from src.core.demonstration_episode import episode_arrays
from src.core.minigrid_public_history import encode_public_history_episode
from test_minigrid_learning_pilot import source_batch


def forbidden(*args, **kwargs):
    raise AssertionError("invalid admission reached Torch or native construction")


def header():
    return paired.TorchRecurrentEpisodeBatchSnapshot(
        1,
        native.TorchRecurrentPolicy(148, 8, 7, 17),
        {},
        {},
        {},
        0,
        0,
        0,
    )


@pytest.mark.parametrize("width", [1, 2, 16, 148, 296])
def test_should_admit_only_declared_toy_raw_and_history_policy_widths(width):
    policy = native.TorchRecurrentPolicy(width, 8, 7, 17)
    assert native._policy(policy).feature_dim == width
    assert policy.max_episode_steps == 16


@pytest.mark.parametrize("width", [0, 17, 147, 149, 295, 297, True, 296.0, np.int64(296)])
def test_should_refuse_nearby_or_non_native_policy_widths(width):
    with pytest.raises(ValueError):
        native.TorchRecurrentPolicy(width, 8, 7, 17)


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "dict",
        "foreign",
        "subclass",
        "legacy",
        "extra",
        "missing",
        "version",
        "version_bool",
        "counter_bool",
        "negative_counter",
        "policy_extra",
    ],
)
def test_should_reject_malformed_snapshot_headers_before_torch(monkeypatch, fault):
    state: Any = header()
    if fault == "none":
        state = None
    elif fault == "dict":
        state = vars(state)
    elif fault == "foreign":
        state = SimpleNamespace(**vars(state))
    elif fault == "subclass":

        class Subclass(paired.TorchRecurrentEpisodeBatchSnapshot):
            pass

        state = Subclass(**vars(state))
    elif fault == "legacy":
        state = native.TorchRecurrentSnapshot(1, state.policy, {}, {}, {}, 0, 0)
    elif fault == "extra":
        object.__setattr__(state, "unexpected", None)
    elif fault == "missing":
        object.__delattr__(state, "gradients")
    elif fault == "version":
        state = replace(state, format_version=2)
    elif fault == "version_bool":
        state = replace(state, format_version=True)
    elif fault == "counter_bool":
        state = replace(state, training_episodes=False)
    elif fault == "negative_counter":
        state = replace(state, completed_updates=-1)
    else:
        object.__setattr__(state.policy, "unexpected", True)
    monkeypatch.setattr(native, "require_torch", forbidden)
    monkeypatch.setattr(history, "require_torch", forbidden)
    with pytest.raises(ValueError):
        history.initialize_public_history_control(state)


def test_should_keep_raw_fixed_protocol_counter_admission_at148(monkeypatch):
    # Only bypass tensor validation here: this assertion targets the unchanged
    # fixed experiment's policy/clock law, not generic native snapshot validity.
    monkeypatch.setattr(paired, "_snapshot", deepcopy)
    raw = header()
    fixed._paired_counter(raw, 0)
    wider = replace(raw, policy=replace(raw.policy, feature_dim=296))
    with pytest.raises(ValueError, match="policy"):
        fixed._paired_counter(wider, 0)


def test_should_encode_owned_train_records_without_role_or_outcome_changes():
    original = source_batch("A", "blue").episode
    assert original is not None
    encoded = encode_public_history_episode(original)
    policy = native.TorchRecurrentPolicy(296, 8, 7, 17)
    partner = replace(encoded, episode_id="another-owned-A-episode")
    schedule = arrays._schedule((encoded, partner), policy)
    assert schedule[0] == encoded and schedule[0] is not encoded
    assert (encoded.episode_id, encoded.context, encoded.role, encoded.num_classes) == (
        original.episode_id,
        "A",
        "train",
        7,
    )
    initial = original.transitions[0].observation
    for before, after in zip(original.transitions, encoded.transitions, strict=True):
        assert after.observation == initial + before.observation
        assert after.next_observation == initial + before.next_observation
        assert (after.action, after.reward, after.terminal, after.step, after.num_classes) == (
            before.action,
            before.reward,
            before.terminal,
            before.step,
            before.num_classes,
        )
    raw_features, raw_targets = episode_arrays(original)
    history_features, history_targets = episode_arrays(encoded)
    assert history_features.shape == (9, 296)
    np.testing.assert_array_equal(history_features[:, 148:], raw_features)
    np.testing.assert_array_equal(history_targets, raw_targets)
    assert not np.shares_memory(history_features, raw_features)
    assert not np.shares_memory(history_targets, raw_targets)


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_should_not_relabel_encoded_non_train_records_to_enter_paired_training(role):
    original = source_batch("A", "blue").episode
    assert original is not None
    encoded = encode_public_history_episode(replace(original, role=role))
    assert encoded.role == role
    partner = replace(encoded, episode_id="second-distinct-episode")
    with pytest.raises(ValueError, match="TRAIN"):
        arrays._schedule((encoded, partner), native.TorchRecurrentPolicy(296, 8, 7, 17))


def test_should_keep_raw_and_history_schedule_geometry_distinct():
    original = source_batch("A", "blue").episode
    assert original is not None
    encoded = encode_public_history_episode(original)
    for policy_width, episode in ((148, encoded), (296, original)):
        with pytest.raises(ValueError, match="policy"):
            arrays._schedule(
                (episode, replace(episode, episode_id="second")),
                native.TorchRecurrentPolicy(policy_width, 8, 7, 17),
            )
