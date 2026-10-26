"""Hand arithmetic, support admission and detached legal-conditioned evidence."""

from dataclasses import FrozenInstanceError, fields, replace
from math import fsum
from typing import Any

import numpy as np
import pytest

from src.core.demonstration_episode import DemonstrationTransition
from src.core.legal_policy_gradient import (
    LegalPolicyGradientEpisode,
    condition_action_probabilities,
    copy_legal_policy_gradient_episode,
)
from src.core.policy_gradient_episode import PolicyGradientEpisode, policy_gradient_weights


RAW = (0.1, 0.2, 0.3, 0.15, 0.25)
LEGAL = (1, 2, 4)


def record(*, role="train", length=2, termination="terminal") -> LegalPolicyGradientEpisode:
    probabilities = condition_action_probabilities(RAW, LEGAL)
    steps = tuple(
        DemonstrationTransition(
            index,
            (index / 16, 0.0),
            1 if index % 2 == 0 else 4,
            1.0 if termination == "terminal" and index == length - 1 else 0.0,
            (
                index / 16
                if termination == "illegal_action" and index == length - 1
                else (index + 1) / 16,
                0.0,
            ),
            termination == "terminal" and index == length - 1,
            5,
        )
        for index in range(length)
    )
    base = PolicyGradientEpisode(
        "legal-reward", "A", role, steps, tuple(probabilities for _ in steps), termination, 5
    )
    return LegalPolicyGradientEpisode(base, tuple(LEGAL for _ in steps), tuple(RAW for _ in steps))


def test_should_normalize_only_declared_support_with_hand_calculated_mass():
    result = condition_action_probabilities(RAW, LEGAL)
    assert result == (0.0, 0.2 / 0.75, 0.3 / 0.75, 0.0, 0.25 / 0.75)
    assert fsum(result) == pytest.approx(1.0, abs=1e-15, rel=0)
    assert condition_action_probabilities((0.25, 0.75), (0, 1)) == (0.25, 0.75)
    assert condition_action_probabilities(RAW, (3,)) == (0.0, 0.0, 0.0, 1.0, 0.0)


def test_should_accept_native_integer_endpoints_and_small_positive_mass_without_epsilon():
    assert condition_action_probabilities((0, 1), (1,)) == (0.0, 1.0)
    assert condition_action_probabilities((5e-324, 1.0), (0,)) == (1.0, 0.0)
    raw = (0.5, 0.5000005)
    result = condition_action_probabilities(raw, (0, 1))
    assert result == (raw[0] / fsum(raw), raw[1] / fsum(raw))


class TupleSubclass(tuple):
    pass


class IntSubclass(int):
    pass


class FloatSubclass(float):
    pass


@pytest.mark.parametrize(
    "raw",
    [
        [],
        (),
        (1.0,),
        (1 / 17,) * 17,
        TupleSubclass((0.5, 0.5)),
        (True, 0),
        (np.float64(0.5), 0.5),
        (FloatSubclass(0.5), 0.5),
        (IntSubclass(1), 0),
        ("0.5", 0.5),
        (None, 1),
        (-0.1, 1.1),
        (float("nan"), 1),
        (float("inf"), 0),
        (0.2, 0.2),
        (0.5, 0.500002),
    ],
)
def test_should_refuse_nonexact_nonnative_nonfinite_or_unnormalized_raw_distribution(raw):
    with pytest.raises(ValueError):
        condition_action_probabilities(raw, (0,))


@pytest.mark.parametrize(
    "support",
    [
        [],
        (),
        TupleSubclass((0,)),
        (True,),
        (np.int64(0),),
        (IntSubclass(0),),
        (0.0,),
        ("0",),
        (-1,),
        (5,),
        (1, 0),
        (0, 0),
        (0, 1, 0),
    ],
)
def test_should_require_exact_nonempty_sorted_unique_native_support(support):
    with pytest.raises(ValueError):
        condition_action_probabilities(RAW, support)


def test_should_refuse_zero_legal_mass_without_repair_or_uniform_fallback():
    with pytest.raises(ValueError, match="positive finite probability mass"):
        condition_action_probabilities((0.0, 0.0, 1.0), (0, 1))


@pytest.mark.parametrize("role", ["train", "final_test", "probe_train", "probe_test"])
@pytest.mark.parametrize(
    "ending,length", [("terminal", 1), ("terminal", 16), ("truncated", 1), ("truncated", 16)]
)
def test_should_preserve_data_roles_and_bounded_endings_without_conveying_train_authority(
    role, ending, length
):
    result = record(role=role, length=length, termination=ending)
    assert result.episode.role == role and len(result.episode.transitions) == length
    assert result.episode.termination == ending
    assert len(result.legal_actions) == len(result.raw_probabilities) == length


def test_should_keep_exact_three_wrapper_fields_and_original_seven_base_fields():
    result = record()
    assert tuple(field.name for field in fields(result)) == (
        "episode",
        "legal_actions",
        "raw_probabilities",
    )
    assert len(fields(result.episode)) == 7
    assert result.episode.behavior_probabilities == tuple(
        condition_action_probabilities(raw, legal)
        for raw, legal in zip(result.raw_probabilities, result.legal_actions)
    )
    assert policy_gradient_weights(result.episode) == (1.0, 1.0)
    with pytest.raises(FrozenInstanceError):
        result.legal_actions = ()  # type: ignore[misc]


def test_should_own_base_transitions_raw_rows_and_support_vectors():
    original = record()
    copied = copy_legal_policy_gradient_episode(original)
    assert copied == original and copied is not original
    assert copied.episode is not original.episode
    assert copied.episode.transitions[0] is not original.episode.transitions[0]
    assert copied.raw_probabilities is not original.raw_probabilities
    assert copied.raw_probabilities[0] is not original.raw_probabilities[0]
    assert copied.legal_actions is not original.legal_actions
    assert copied.legal_actions[0] is not original.legal_actions[0]
    object.__setattr__(original.episode.transitions[0], "reward", -1.0)
    assert copied.episode.transitions[0].reward == 0.0


@pytest.mark.parametrize(
    "kind",
    [
        "legal_list",
        "legal_count",
        "raw_list",
        "raw_count",
        "raw_classes",
        "behavior",
        "selected_illegal",
        "illegal_ending",
    ],
)
def test_should_refuse_wrapper_shape_and_behavior_support_mismatch(kind):
    original = record()
    changes: dict[str, Any] = {}
    if kind == "legal_list":
        changes["legal_actions"] = list(original.legal_actions)
    elif kind == "legal_count":
        changes["legal_actions"] = original.legal_actions[:1]
    elif kind == "raw_list":
        changes["raw_probabilities"] = list(original.raw_probabilities)
    elif kind == "raw_count":
        changes["raw_probabilities"] = original.raw_probabilities[:1]
    elif kind == "raw_classes":
        changes["raw_probabilities"] = ((0.5, 0.5),) * 2
    elif kind == "behavior":
        changes["episode"] = replace(original.episode, behavior_probabilities=(RAW,) * 2)
    elif kind == "selected_illegal":
        changes["legal_actions"] = ((0, 2, 4), LEGAL)
    else:
        base = original.episode
        last = replace(
            base.transitions[-1],
            next_observation=base.transitions[-1].observation,
            reward=0,
            terminal=False,
        )
        changes["episode"] = replace(
            base, transitions=(*base.transitions[:-1], last), termination="illegal_action"
        )
    with pytest.raises(ValueError):
        replace(original, **changes)


def test_should_use_absolute_behavior_tolerance_without_replacing_saved_evidence():
    original = record()
    vector = list(original.episode.behavior_probabilities[0])
    vector[1] += 4e-13
    vector[2] -= 4e-13
    probabilities = (tuple(vector), original.episode.behavior_probabilities[1])
    accepted = replace(
        original, episode=replace(original.episode, behavior_probabilities=probabilities)
    )
    assert accepted.episode.behavior_probabilities == probabilities
    vector[1] += 4e-12
    vector[2] -= 4e-12
    with pytest.raises(ValueError, match="differ from declared legal conditioning"):
        replace(
            original,
            episode=replace(
                original.episode, behavior_probabilities=(tuple(vector), probabilities[1])
            ),
        )


def test_should_refuse_tiny_excluded_behavior_mass_even_inside_matching_tolerance():
    original = record()
    first = list(original.episode.behavior_probabilities[0])
    first[0] = 1e-13
    first[1] -= 1e-13
    base = replace(
        original.episode,
        behavior_probabilities=(tuple(first), original.episode.behavior_probabilities[1]),
    )
    with pytest.raises(ValueError, match="outside declared legal support must be zero"):
        replace(original, episode=base)


def test_should_refuse_raw_zero_selected_action_despite_tiny_positive_saved_behavior():
    original = record()
    raw = (0.1, 0.0, 0.5, 0.15, 0.25)
    conditional = list(condition_action_probabilities(raw, LEGAL))
    conditional[1] = 1e-13
    conditional[2] -= 1e-13
    base = replace(
        original.episode,
        behavior_probabilities=(tuple(conditional), original.episode.behavior_probabilities[1]),
    )
    with pytest.raises(
        ValueError, match="selected action must have positive conditional probability"
    ):
        replace(original, episode=base, raw_probabilities=(raw, RAW))


class WrapperSubclass(LegalPolicyGradientEpisode):
    pass


@pytest.mark.parametrize(
    "kind",
    [
        "subclass",
        "extra",
        "base_extra",
        "support_mutation",
        "raw_mutation",
        "base_mutation",
        "transition_extra",
    ],
)
def test_should_revalidate_exact_wrapper_and_complete_base_after_frozen_bypass(kind):
    original = record()
    if kind == "subclass":
        with pytest.raises(ValueError, match="exact LegalPolicyGradientEpisode"):
            WrapperSubclass(original.episode, original.legal_actions, original.raw_probabilities)
        return
    if kind == "extra":
        object.__setattr__(original, "unrecorded", True)
    elif kind == "base_extra":
        object.__setattr__(original.episode, "unrecorded", True)
    elif kind == "support_mutation":
        object.__setattr__(original, "legal_actions", ((1, 1, 4), LEGAL))
    elif kind == "raw_mutation":
        object.__setattr__(original, "raw_probabilities", ((0.0,) * 5, RAW))
    elif kind == "base_mutation":
        object.__setattr__(original.episode, "behavior_probabilities", (RAW,) * 2)
    else:
        object.__setattr__(original.episode.transitions[0], "unrecorded", True)
    with pytest.raises(ValueError):
        copy_legal_policy_gradient_episode(original)
