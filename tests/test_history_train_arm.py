"""The explicit 296 paired lane uses the ordinary delayed TRAIN stream."""

from dataclasses import replace
import pickle
from typing import Any

import pytest

from src.adapters import minigrid_learning_pilot as shared
from src.adapters import torch_train_episode_batch as paired
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy, TorchRecurrentSnapshot
from src.core.minigrid_public_history import encode_public_history_episode
from test_paired_episode_stream import case as case, packets
from test_torch_feature_rehearsal_pilot import Sampler, forbidden


@pytest.fixture
def history_arm(case):
    episodes = tuple(encode_public_history_episode(episode) for episode in case.episodes)
    policy = replace(case.policy, feature_dim=296)
    learner = paired.TorchTrainEpisodeBatchLearner(policy, episodes)
    case.history_episodes, case.history_learner = episodes, learner
    case.history_packets = packets(episodes)
    return case


def train(case, *, row=None, **options):
    if row is None:
        row = shared._row("recurrent_paired_bc")
    return shared._train_arm(
        row,
        case.history_packets,
        Sampler(),
        100.0,
        lambda: None,
        learner_factory=lambda method: case.history_learner,
        checkpoint_updates=(0, 4),
        public_history=True,
        **options,
    )


def test_should_complete_four_delayed_history_pairs_with_original_budget_and_owned_states(
    history_arm,
):
    case = history_arm
    original = pickle.dumps(
        (case.history_episodes, case.history_packets, case.history_learner.snapshot_state())
    )
    row = shared._row("recurrent_paired_bc")
    arm = train(case, row=row)
    arm["guard"]()
    assert row["status"] == "trained"
    assert (
        row["charged_updates"],
        row["native_training_episodes"],
        row["native_training_examples"],
    ) == (4, 8, 72)
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    result = row["stream_result"]
    assert result["evaluation_scope"] == "training_only"
    assert result["score_forward_calls"] == 0 and result["checkpoints"] == []
    assert len(result["applied_updates"]) == 4
    assert [point for point, _ in arm["states"]] == [0, 4]
    assert all(state.policy.feature_dim == 296 for _, state in arm["states"])
    final = arm["states"][-1][1]
    assert (final.completed_updates, final.training_episodes, final.training_examples) == (4, 8, 72)
    assert arm["budget"].updates_completed == 4
    assert arm["budget"].budget.max_training_updates == 4
    assert len(case.work["trains"]) == 4 and not case.work["predictions"]
    observed = [episode for call in case.work["trains"] for episode in call["episodes"]]
    assert observed == list(case.history_episodes)
    assert all(
        episode.role == "train"
        and len(episode.transitions) == 9
        and len(episode.transitions[0].observation) == 296
        for episode in observed
    )
    assert (
        pickle.dumps(
            (case.history_episodes, case.history_packets, case.history_learner.snapshot_state())
        )
        == original
    )


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_should_refuse_non_bool_history_selection_before_factory(history_arm, value):
    case = history_arm
    with pytest.raises(ValueError):
        shared._train_arm(
            shared._row("recurrent_paired_bc"),
            case.history_packets,
            Sampler(),
            100.0,
            forbidden,
            learner_factory=forbidden,
            checkpoint_updates=(0, 4),
            public_history=value,
        )
    assert not case.work["trains"]


@pytest.mark.parametrize(
    "fault", ["method", "factory", "raw_width", "role", "metadata", "default_width"]
)
def test_should_refuse_wrong_history_transport_or_mode_before_native_work(history_arm, fault):
    case = history_arm
    supplied = case.history_packets
    row = shared._row("recurrent_paired_bc")
    factory: Any = forbidden
    history = True
    if fault == "method":
        row["method"] = "recurrent_bc"
    elif fault == "factory":
        factory = None
    elif fault == "raw_width":
        supplied = packets(case.episodes)
    elif fault == "role":
        supplied = (replace(supplied[0], role="final_test"), *supplied[1:])
    elif fault == "metadata":
        supplied = (replace(supplied[0], episode=case.history_episodes[0]), *supplied[1:])
    else:
        history = False
    with pytest.raises(ValueError):
        shared._train_arm(
            row,
            supplied,
            Sampler(),
            100.0,
            forbidden,
            learner_factory=factory,
            checkpoint_updates=(0, 4),
            public_history=history,
        )
    assert not case.work["trains"]


@pytest.mark.parametrize("fault", ["raw_policy", "point", "steps", "episodes", "rows", "legacy"])
def test_should_keep_history_clock_law_separate_from_raw_paired_state(history_arm, fault):
    case = history_arm
    state: Any = case.history_learner.snapshot_state()
    point: Any = 0
    if fault == "raw_policy":
        state = replace(state, policy=TorchRecurrentPolicy(148, 8, 7, 17))
    elif fault == "point":
        point = False
    elif fault == "steps":
        state = replace(state, completed_updates=1)
    elif fault == "episodes":
        state = replace(state, training_episodes=1)
    elif fault == "rows":
        state = replace(state, training_examples=1)
    else:
        state = TorchRecurrentSnapshot(
            1, state.policy, state.parameters, state.optimizer_state, state.gradients, 0, 0
        )
    with pytest.raises(ValueError):
        shared._paired_history_counter(state, point)
    with pytest.raises(ValueError):
        shared._paired_counter(case.history_learner.snapshot_state(), 0)
    assert not case.work["trains"]


def test_should_preserve_committed_history_charge_when_preceding_initialization_proof_changes(
    history_arm,
):
    case = history_arm
    row = shared._row("recurrent_paired_bc")
    proof = dict(initialization_valid=True)

    def corrupt(owner):
        proof["initialization_valid"] = False

    def boundary():
        if proof["initialization_valid"] is not True:
            raise ValueError("preceding initialization proof changed")

    case.work["after_train"] = corrupt
    with pytest.raises(ValueError):
        shared._train_arm(
            row,
            case.history_packets,
            Sampler(),
            100.0,
            boundary,
            learner_factory=lambda method: case.history_learner,
            checkpoint_updates=(0, 4),
            public_history=True,
        )
    assert len(case.work["trains"]) == 1
    assert row["status"] == "incomplete"
    assert row["charged_updates"] == 1 and row["native_training_examples"] == 18
    assert row["native_training_episodes"] == 2
    assert len(row["partial_stream"]["applied_updates"]) == 1
    assert proof["initialization_valid"] is False
