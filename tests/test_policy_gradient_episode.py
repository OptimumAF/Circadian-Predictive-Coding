"""Independent arithmetic and ownership checks for reward trajectory records."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from src.core.policy_gradient_episode import (
    PolicyGradientEpisode,
    copy_policy_gradient_episode,
    policy_gradient_weights,
)


def episode(rewards=(1.0, -0.5, 1.0), *, termination="terminal", role="train", classes=5):
    transitions = tuple(
        DemonstrationTransition(
            index,
            (index / 16, 0.0),
            index % classes,
            reward,
            (
                index / 16
                if termination == "illegal_action" and index == len(rewards) - 1
                else (index + 1) / 16,
                0.0,
            ),
            termination == "terminal" and index == len(rewards) - 1,
            classes,
        )
        for index, reward in enumerate(rewards)
    )
    return PolicyGradientEpisode(
        "reward-episode",
        "A",
        role,
        transitions,
        tuple((1.0 / classes,) * classes for _ in rewards),
        termination,
        classes,
    )


@pytest.mark.parametrize(
    "discount,expected", [(1, (1.5, 0.5, 1.0)), (0.5, (1.0, 0.0, 0.25)), (0, (1.0, 0.0, 0.0))]
)
def test_should_compute_absolute_discounted_reward_coefficients(discount, expected):
    assert policy_gradient_weights(episode(), discount) == expected


def test_should_keep_sum_objective_and_default_discount():
    assert policy_gradient_weights(episode((1.0, 1.0))) == (2.0, 1.0)
    assert policy_gradient_weights(episode((0.0, 0.0))) == (0.0, 0.0)


@pytest.mark.parametrize("termination", ["terminal", "truncated", "illegal_action"])
@pytest.mark.parametrize("length", [1, 16])
def test_should_accept_bounded_variable_length_endings(termination, length):
    record = episode((0.0,) * length, termination=termination)
    assert len(record.transitions) == length
    assert record.transitions[-1].terminal == (termination == "terminal")
    assert copy_policy_gradient_episode(record) == record


@pytest.mark.parametrize("role", ["train", "final_test", "probe_train", "probe_test"])
def test_should_represent_roles_without_conferring_training_authority(role):
    assert episode(role=role).role == role


def test_should_own_immutable_transition_and_probability_records():
    original = episode()
    copied = copy_policy_gradient_episode(original)
    assert copied == original and copied is not original
    assert all(left is not right for left, right in zip(original.transitions, copied.transitions))
    assert copied.behavior_probabilities is not original.behavior_probabilities
    assert all(
        left is not right
        for left, right in zip(original.behavior_probabilities, copied.behavior_probabilities)
    )
    object.__setattr__(original.transitions[0], "reward", -1.0)
    assert copied.transitions[0].reward == 1.0
    with pytest.raises(FrozenInstanceError):
        copied.context = "B"  # type: ignore[misc]


@pytest.mark.parametrize(
    "discount", [True, np.float64(0.5), float("nan"), float("inf"), -0.1, 1.1, "1"]
)
def test_should_refuse_invalid_discounts(discount):
    with pytest.raises(ValueError):
        policy_gradient_weights(episode(), discount)


@pytest.mark.parametrize(
    "field,value",
    [
        ("episode_id", ""),
        ("episode_id", " edge"),
        ("episode_id", "é" * 33),
        ("episode_id", "\ud800"),
        ("context", ""),
        ("context", "é" * 17),
        ("role", "validation"),
        ("termination", "success"),
        ("termination", None),
        ("num_classes", True),
        ("num_classes", 1),
        ("num_classes", 17),
        ("transitions", ()),
        ("transitions", []),
        ("behavior_probabilities", []),
        ("behavior_probabilities", ()),
    ],
)
def test_should_refuse_invalid_episode_fields(field, value):
    with pytest.raises(ValueError):
        replace(episode(), **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("step", 2),
        ("num_classes", 4),
        ("terminal", True),
        ("observation", (0.0,)),
        ("next_observation", (0.75, 0.0)),
    ],
)
def test_should_refuse_broken_continuity_geometry_and_ending(field, value):
    record = episode()
    steps = list(record.transitions)
    with pytest.raises(ValueError):
        steps[0] = replace(steps[0], **{field: value})
        replace(record, transitions=tuple(steps))


@pytest.mark.parametrize("termination", ["truncated", "illegal_action"])
def test_should_refuse_fabricated_terminal_for_nonterminal_endings(termination):
    with pytest.raises(ValueError):
        replace(episode(), termination=termination)


@pytest.mark.parametrize("field,value", [("reward", 1.0), ("next_observation", (1.0, 0.0))])
def test_should_refuse_applied_illegal_action_endings(field, value):
    record = episode((0.0,), termination="illegal_action")
    with pytest.raises(ValueError):
        replace(record, transitions=(replace(record.transitions[0], **{field: value}),))


@pytest.mark.parametrize(
    "row",
    [
        [0.2] * 5,
        (0.25,) * 4,
        (0.2,) * 6,
        (float("nan"), 0.2, 0.2, 0.2, 0.2),
        (float("inf"), 0.2, 0.2, 0.2, 0.2),
        (-0.1, 0.3, 0.3, 0.3, 0.2),
        (1.1, 0.0, 0.0, 0.0, 0.0),
        (True, 0.0, 0.0, 0.0, 0.0),
        (np.float64(0.2), 0.2, 0.2, 0.2, 0.2),
        (0.0, 0.25, 0.25, 0.25, 0.25),
        (0.1,) * 5,
    ],
)
def test_should_refuse_malformed_behavior_distributions(row):
    with pytest.raises(ValueError):
        replace(episode((1.0,)), behavior_probabilities=(row,))


def test_should_allow_roundoff_without_repairing_behavior_probabilities():
    row = (0.2000001, 0.2, 0.2, 0.2, 0.2)
    record = replace(episode((1.0,)), behavior_probabilities=(row,))
    assert record.behavior_probabilities[0] == row


@pytest.mark.parametrize(
    "field,value",
    [
        ("extra", 1),
        ("reward", float("nan")),
        ("action", True),
        ("num_classes", 4),
        ("observation", [0.0, 0.0]),
    ],
)
def test_should_revalidate_corrupted_frozen_transitions(field, value):
    record = episode()
    object.__setattr__(record.transitions[0], field, value)
    with pytest.raises(ValueError):
        copy_policy_gradient_episode(record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("extra", 1),
        ("role", "invalid"),
        ("termination", "fake"),
        ("num_classes", True),
        ("behavior_probabilities", ()),
    ],
)
def test_should_revalidate_every_corrupted_frozen_episode_field(field, value):
    record = episode()
    object.__setattr__(record, field, value)
    with pytest.raises(ValueError):
        policy_gradient_weights(record)


def test_should_refuse_missing_fields_and_another_episode_type():
    record = episode()
    object.__delattr__(record, "context")
    with pytest.raises(ValueError):
        copy_policy_gradient_episode(record)
    with pytest.raises(ValueError):
        copy_policy_gradient_episode(
            DemonstrationEpisode("demo", "A", "train", episode().transitions, 5)  # type: ignore[arg-type]
        )
