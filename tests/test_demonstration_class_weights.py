"""Released episode labels determine immutable explicit weights, without models."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.core.demonstration_class_weights import episode_class_weights, validate_class_weights
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition


def episode(actions=(0, 0, 0, 1), *, classes=3, role="train"):
    return DemonstrationEpisode(
        "weighted-demo",
        "A",
        role,
        tuple(
            DemonstrationTransition(
                index,
                (index / 16, 0.0),
                action,
                float(index == len(actions) - 1),
                ((index + 1) / 16, 0.0),
                index == len(actions) - 1,
                classes,
            )
            for index, action in enumerate(actions)
        ),
        classes,
    )


@pytest.mark.parametrize(
    "actions,classes,expected",
    [
        ((0, 0, 0, 1), 3, (2 / 3, 2.0, 1.0)),
        ((0, 1), 2, (1.0, 1.0)),
        ((0, 1, 1, 2, 2, 2), 3, (2.0, 1.0, 2 / 3)),
        ((2,), 4, (1.0, 1.0, 1.0, 1.0)),
        ((1,) * 16, 16, (1.0,) * 16),
        (tuple(range(16)), 16, (1.0,) * 16),
    ],
)
def test_should_derive_reciprocal_present_counts_and_neutral_absent_weights(
    actions, classes, expected
):
    result = episode_class_weights(episode(actions, classes=classes))
    assert result == expected
    assert type(result) is tuple and all(type(weight) is float for weight in result)


def test_should_assign_equal_total_mass_to_each_present_class_and_preserve_total_row_mass():
    record = episode((0, 1, 1, 2, 2, 2), classes=5)
    weights = episode_class_weights(record)
    masses = [
        sum(weights[step.action] for step in record.transitions if step.action == c)
        for c in range(5)
    ]
    assert masses == [2.0, 2.0, 2.0, 0, 0]
    assert sum(masses) == len(record.transitions)
    assert weights[3:] == (1.0, 1.0)


def test_should_use_only_action_counts_without_reordering_or_mutating_physical_records():
    record = episode()
    before = replace(record)
    weights = episode_class_weights(record)
    assert record == before
    changed_rewards = replace(
        record, transitions=tuple(replace(step, reward=-step.reward) for step in record.transitions)
    )
    assert episode_class_weights(changed_rewards) == weights
    reversed_labels = episode(tuple(reversed(tuple(step.action for step in record.transitions))))
    assert episode_class_weights(reversed_labels) == weights


def test_should_detach_returned_weights_from_later_frozen_record_tampering():
    record = episode()
    weights = episode_class_weights(record)
    object.__setattr__(record.transitions[0], "action", 1)
    assert weights == (2 / 3, 2.0, 1.0)
    assert episode_class_weights(record) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize("role", ["final_test", "probe_train", "probe_test"])
def test_should_refuse_every_nontraining_data_role(role):
    with pytest.raises(ValueError, match="train-role"):
        episode_class_weights(episode(role=role))


@pytest.mark.parametrize(
    "field,value",
    [
        ("role", "train "),
        ("num_classes", True),
        ("num_classes", 2),
        ("transitions", ()),
        ("transitions", []),
        ("extra", 1),
        ("episode_id", ""),
        ("context", "bad "),
    ],
)
def test_should_refuse_corrupted_frozen_episode_fields(field, value):
    record = episode()
    object.__setattr__(record, field, value)
    with pytest.raises(ValueError):
        episode_class_weights(record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("action", True),
        ("action", 3),
        ("reward", float("nan")),
        ("step", 1),
        ("terminal", True),
        ("num_classes", 2),
        ("observation", [0.0, 0.0]),
        ("next_observation", (1.0, 0.0)),
        ("extra", 1),
    ],
)
def test_should_refuse_corrupted_frozen_transition_fields(field, value):
    record = episode()
    object.__setattr__(record.transitions[0], field, value)
    with pytest.raises(ValueError):
        episode_class_weights(record)


def test_should_refuse_missing_fields_wrong_types_and_episode_subclasses():
    record = episode()
    object.__delattr__(record, "context")
    with pytest.raises(ValueError):
        episode_class_weights(record)
    record = episode()
    object.__delattr__(record.transitions[0], "action")
    with pytest.raises(ValueError):
        episode_class_weights(record)
    invalid: Any = None
    with pytest.raises(ValueError):
        episode_class_weights(invalid)

    class Derived(DemonstrationEpisode):
        pass

    original = episode()
    derived = Derived(original.episode_id, original.context, original.role, original.transitions, 3)
    with pytest.raises(ValueError):
        episode_class_weights(derived)


@pytest.mark.parametrize(
    "weights,classes,expected",
    [
        ((1, 2), 2, (1.0, 2.0)),
        ((1e-6, 1e6), 2, (1e-6, 1e6)),
        ((3.0,) * 16, 16, (3.0,) * 16),
    ],
)
def test_should_validate_bounded_weights_without_normalization(weights, classes, expected):
    result = validate_class_weights(weights, classes)
    assert result == expected
    assert result is not weights
    assert all(type(value) is float for value in result)


@pytest.mark.parametrize(
    "weights",
    [
        [1.0, 1.0],
        np.array([1.0, 1.0]),
        (),
        (1.0,),
        (1.0, 1.0, 1.0),
        (True, 1.0),
        (np.float64(1.0), 1.0),
        (np.int64(1), 1.0),
        (0.0, 1.0),
        (-1.0, 1.0),
        (1e-7, 1.0),
        (1e7, 1.0),
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (float("-inf"), 1.0),
        (10**1000, 1.0),
        ("1", 1.0),
    ],
)
def test_should_refuse_non_native_nonfinite_out_of_bound_or_wrong_geometry_weights(weights):
    with pytest.raises(ValueError):
        validate_class_weights(weights, 2)


@pytest.mark.parametrize("classes", [True, np.int64(2), 1, 17, 2.0, "2"])
def test_should_refuse_invalid_class_declarations_before_weight_processing(classes):
    with pytest.raises(ValueError):
        validate_class_weights((1.0, 1.0), classes)
