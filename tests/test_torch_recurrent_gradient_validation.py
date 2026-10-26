"""Reject invalid TRAIN gradient pairs before any optional native work.

Snapshot tensor payloads are deliberately non-native placeholders: every case
must be refused by pair/metadata admission before full native validation.
"""

from copy import deepcopy
from dataclasses import replace
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters import torch_recurrent_gradient_diagnostics as diagnostics
from src.adapters import torch_recurrent_learner as native
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition


class PairTuple(tuple):
    pass


class EpisodeSubclass(DemonstrationEpisode):
    pass


class SnapshotSubclass(native.TorchRecurrentSnapshot):
    pass


def episode(identity, *, role="train", context="A", dimension=2, classes=2):
    return DemonstrationEpisode(
        identity,
        context,
        role,
        tuple(
            DemonstrationTransition(
                index,
                (index / 4,) + (0.0,) * (dimension - 1),
                index % classes,
                0.5 if index == 1 else 0.0,
                ((index + 1) / 4,) + (0.0,) * (dimension - 1),
                index == 1,
                classes,
            )
            for index in range(2)
        ),
        classes,
    )


@pytest.fixture
def admission(monkeypatch):
    policy = native.TorchRecurrentPolicy(2, 2, 2, 17)
    shapes = dict(
        weight_input=(2, 2),
        weight_recurrent=(2, 2),
        bias_hidden=(1, 2),
        weight_output=(2, 2),
        bias_output=(1, 2),
    )
    snapshot = native.TorchRecurrentSnapshot(
        1,
        policy,
        {key: np.zeros(shape, dtype=np.float32) for key, shape in shapes.items()},
        {},
        {key: None for key in shapes},
        0,
        0,
    )
    calls: list[str] = []

    def forbidden(*args, **kwargs):
        calls.append("native work")
        pytest.fail("invalid pair must be refused before Torch or native snapshot/forward work")

    monkeypatch.setattr(diagnostics, "require_torch", forbidden)
    monkeypatch.setattr(native, "require_torch", forbidden)
    monkeypatch.setattr(native, "_snapshot", forbidden)
    monkeypatch.setattr(native, "_forward", forbidden)
    return SimpleNamespace(
        snapshot=snapshot, episodes=(episode("A-blue"), episode("A-red")), calls=calls
    )


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "tuple_subclass",
        "one",
        "three",
        "foreign_episode",
        "episode_subclass",
        "final_test",
        "probe_train",
        "probe_test",
        "context",
        "duplicate_id",
        "empty_id",
        "width",
        "classes",
        "policy_length",
        "overlong",
        "empty_transitions",
        "transition_list",
        "episode_extra",
        "transition_extra",
        "bad_action",
        "bad_terminal",
        "bad_order",
        "nonfinite_observation",
    ],
)
def test_should_refuse_malformed_or_unavailable_episode_pair_without_native_access(
    admission, fault
):
    snapshot: Any = admission.snapshot
    pair: Any = deepcopy(admission.episodes)
    if fault == "list":
        pair = list(pair)
    elif fault == "tuple_subclass":
        pair = PairTuple(pair)
    elif fault == "one":
        pair = pair[:1]
    elif fault == "three":
        pair = (*pair, episode("A-other"))
    elif fault == "foreign_episode":
        pair = (dict(vars(pair[0])), pair[1])
    elif fault == "episode_subclass":
        pair = (EpisodeSubclass(**vars(pair[0])), pair[1])
    elif fault in ("final_test", "probe_train", "probe_test"):
        pair = (replace(pair[0], role=fault), pair[1])
    elif fault == "context":
        pair = (pair[0], replace(pair[1], context="B"))
    elif fault == "duplicate_id":
        pair = (pair[0], replace(pair[1], episode_id=pair[0].episode_id))
    elif fault == "empty_id":
        object.__setattr__(pair[0], "episode_id", "")
    elif fault == "width":
        pair = (episode("A-blue", dimension=3), pair[1])
    elif fault == "classes":
        pair = (episode("A-blue", classes=3), pair[1])
    elif fault == "policy_length":
        snapshot = replace(snapshot, policy=replace(snapshot.policy, max_episode_steps=1))
    elif fault == "overlong":
        object.__setattr__(pair[0], "transitions", pair[0].transitions * 9)
    elif fault == "empty_transitions":
        object.__setattr__(pair[0], "transitions", ())
    elif fault == "transition_list":
        object.__setattr__(pair[0], "transitions", list(pair[0].transitions))
    elif fault == "episode_extra":
        object.__setattr__(pair[0], "unobserved_context", "future")
    elif fault == "transition_extra":
        object.__setattr__(pair[0].transitions[0], "teacher_hint", 1)
    elif fault == "bad_action":
        object.__setattr__(pair[0].transitions[0], "action", 2)
    elif fault == "bad_terminal":
        object.__setattr__(pair[0].transitions[-1], "terminal", False)
    elif fault == "bad_order":
        object.__setattr__(pair[0].transitions[0], "step", 9)
    else:
        object.__setattr__(pair[0].transitions[0], "observation", (float("nan"), 0.0))
    before = pickle.dumps((snapshot, pair))
    with pytest.raises(ValueError):
        diagnostics.diagnose_episode_gradients(snapshot, pair)
    assert admission.calls == [] and pickle.dumps((snapshot, pair)) == before


@pytest.mark.parametrize(
    "fault",
    [
        "foreign_snapshot",
        "snapshot_subclass",
        "snapshot_extra",
        "foreign_policy",
        "policy_extra",
        "policy_width",
    ],
)
def test_should_refuse_snapshot_identity_and_metadata_tampering_before_native_validation(
    admission, fault
):
    snapshot: Any = admission.snapshot
    if fault == "foreign_snapshot":
        snapshot = SimpleNamespace(**vars(snapshot))
    elif fault == "snapshot_subclass":
        snapshot = SnapshotSubclass(**vars(snapshot))
    elif fault == "snapshot_extra":
        object.__setattr__(snapshot, "unrecorded_optimizer", {})
    elif fault == "foreign_policy":
        snapshot = replace(snapshot, policy=SimpleNamespace(**vars(snapshot.policy)))
    elif fault == "policy_extra":
        object.__setattr__(snapshot.policy, "hidden_context", "A")
    else:
        snapshot = replace(snapshot, policy=replace(snapshot.policy, feature_dim=3))
    before = pickle.dumps((snapshot, admission.episodes))
    with pytest.raises(ValueError):
        diagnostics.diagnose_episode_gradients(snapshot, admission.episodes)
    assert admission.calls == []
    assert pickle.dumps((snapshot, admission.episodes)) == before
