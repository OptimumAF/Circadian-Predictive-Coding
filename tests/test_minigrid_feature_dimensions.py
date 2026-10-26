"""Explicit full normalized MiniGrid geometry; no environment or native training."""

from dataclasses import asdict, fields, replace
from typing import Any

import numpy as np
import pytest

from src.adapters.torch_feature_learner import TorchFeaturePolicy
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy, TorchRecurrentSnapshot
from src.app.coupled_attribution import AttributionBatch, _validate_batch
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    EpisodeReplaySchema,
    copy_episode,
    decode_episode,
    encode_episode,
    episode_arrays,
)
from src.core.policy_gradient_episode import PolicyGradientEpisode, copy_policy_gradient_episode


def episode(*, width=148, steps=16, role="train"):
    observations = tuple(
        tuple(float((column + step) % 3 - 1) for column in range(width))
        for step in range(steps + 1)
    )
    return DemonstrationEpisode(
        "minigrid-fixed",
        "memory-s7",
        role,
        tuple(
            DemonstrationTransition(
                step,
                observations[step],
                step % 7,
                1.0 if step == steps - 1 else 0.0,
                observations[step + 1],
                step == steps - 1,
                7,
            )
            for step in range(steps)
        ),
        7,
    )


def test_should_retain_all148_features_seven_actions_and_exact_replay_byte_cost():
    original = episode()
    features, targets = episode_arrays(original)
    assert features.shape == (16, 148) and targets.shape == (16, 1)
    assert features.dtype == targets.dtype == np.float64
    assert features[0].tolist() == list(original.transitions[0].observation)
    assert features[-1].tolist() != list(original.transitions[-1].next_observation)
    assert set(targets[:, 0]) == set(range(7))
    schema = EpisodeReplaySchema(148, 16, 7)
    encoded, dummy = encode_episode(original, schema)
    assert schema.encoded_dim == 4883 and encoded.shape == (1, 4883)
    assert encoded.nbytes + dummy.nbytes == 39072
    decoded = decode_episode(encoded, dummy, schema)
    assert decoded == original and decoded is not original
    encoded.fill(0)
    features.fill(0)
    targets.fill(0)
    assert copy_episode(decoded) == original


def test_should_admit148_complete_train_arrays_through_existing_stream_validation():
    original = episode()
    features, targets = episode_arrays(original)
    batch = AttributionBatch(
        "train",
        original.context,
        tuple(
            f"{original.episode_id}:step:{transition.step}" for transition in original.transitions
        ),
        features,
        targets,
        original,
        7,
    )
    seen: set[str] = set()
    _validate_batch(batch, "train", seen)
    assert len(seen) == 16 and batch.num_classes == 7
    altered = features.copy()
    altered[0, -1] = 0.5
    with pytest.raises(ValueError, match="must agree"):
        _validate_batch(replace(batch, features=altered), "train", set())


def test_should_preserve_full148_reward_episode_and_behavior_arrays_without_changing_math():
    original = episode()
    reward = PolicyGradientEpisode(
        original.episode_id,
        original.context,
        "train",
        original.transitions,
        ((1 / 7,) * 7,) * 16,
        "terminal",
        7,
    )
    copied = copy_policy_gradient_episode(reward)
    assert copied == reward and copied is not reward
    assert len(copied.transitions[0].observation) == 148
    assert all(len(row) == 7 for row in copied.behavior_probabilities)


@pytest.mark.parametrize("role", ["final_test", "probe_train", "probe_test"])
def test_should_keep_nontraining148_episodes_out_of_replay(role):
    original = episode(role=role)
    with pytest.raises(ValueError, match="only training episodes"):
        encode_episode(original, EpisodeReplaySchema(148, 16, 7))


@pytest.mark.parametrize("width", [0, 17, 147, 149])
def test_should_reject_unapproved_observation_widths_in_records_and_schema(width):
    with pytest.raises(ValueError):
        episode(width=width)
    with pytest.raises(ValueError):
        EpisodeReplaySchema(width, 16, 7)


@pytest.mark.parametrize("width", [True, False, 148.0, np.int64(148)])
def test_should_reject_coerced_or_boolean_schema_dimensions(width):
    with pytest.raises(ValueError):
        EpisodeReplaySchema(width, 16, 7)


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc", "gated_cpc", "recurrent"])
def test_should_admit_exact148_for_each_existing_native_policy_without_changing_other_bounds(
    method,
):
    if method == "recurrent":
        policy: Any = TorchRecurrentPolicy(148, 8, 7, 17)
        assert policy.max_episode_steps == 16
    else:
        policy = TorchFeaturePolicy(method, 148, 8, 17, num_classes=7)
        assert policy.inference_steps == 2 and policy.inference_learning_rate == 0.04
    assert policy.feature_dim == 148 and policy.hidden_dim == 8 and policy.num_classes == 7
    assert policy.learning_rate == 0.03 and policy.momentum == 0.9
    for changes in ({"hidden_dim": 33}, {"num_classes": 17}):
        with pytest.raises(ValueError):
            replace(policy, **changes)


@pytest.mark.parametrize("width", [17, 147, 149, 0, -1, True, False, 148.0, np.int64(148)])
@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc", "gated_cpc", "recurrent"])
def test_should_refuse_unapproved_or_non_native_feature_policy_widths_before_resources(
    width, method
):
    with pytest.raises(ValueError, match="feature_dim"):
        if method == "recurrent":
            TorchRecurrentPolicy(width, 8, 7, 17)
        else:
            TorchFeaturePolicy(method, width, 8, 17, num_classes=7)


@pytest.mark.parametrize(
    "field,value",
    [
        ("reward", 1.01),
        ("reward", -1.01),
        ("reward", True),
        ("observation", 1.01),
        ("observation", -1.01),
        ("observation", float("nan")),
        ("observation", True),
    ],
)
def test_should_keep_normalized_observation_and_reward_range_checks_unchanged(field, value):
    transition = episode(steps=1).transitions[0]
    changes = {field: value if field == "reward" else (value,) + transition.observation[1:]}
    with pytest.raises(ValueError):
        replace(transition, **changes)


def test_should_keep_step_limit_and_reject_fabricated_complete_timeout():
    with pytest.raises(ValueError):
        episode(steps=17)
    with pytest.raises(ValueError):
        EpisodeReplaySchema(148, 17, 7)
    with pytest.raises(ValueError):
        TorchRecurrentPolicy(148, 8, 7, 17, max_episode_steps=17)
    original = episode()
    transitions = (*original.transitions[:-1], replace(original.transitions[-1], terminal=False))
    with pytest.raises(ValueError, match="only the last transition"):
        replace(original, transitions=transitions)


@pytest.mark.parametrize("width", [1, 8, 16])
def test_should_preserve_existing_geometry_wire_size_and_policy_defaults(width):
    original = episode(width=width, steps=1)
    schema = EpisodeReplaySchema(width, 1, 7)
    assert EpisodeReplaySchema(width, 1).encoded_dim == 98 + 2 * width + 3
    encoded, dummy = encode_episode(original, schema)
    assert encoded.shape == (1, 99 + 2 * width + 3)
    assert decode_episode(encoded, dummy, schema) == original
    assert asdict(TorchRecurrentPolicy(width, 8, 7, 17)) == dict(
        feature_dim=width,
        hidden_dim=8,
        num_classes=7,
        seed=17,
        learning_rate=0.03,
        momentum=0.9,
        max_episode_steps=16,
    )
    assert TorchFeaturePolicy("bp", width, 8, 17).num_classes == 2
    assert [field.name for field in fields(TorchRecurrentSnapshot)] == [
        "format_version",
        "policy",
        "parameters",
        "optimizer_state",
        "gradients",
        "completed_updates",
        "training_examples",
    ]
