"""Logical arrival and declared role permissions are independent of model layout."""

from dataclasses import replace
from typing import Any, cast

import pytest

from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock


def sample():
    return Experience(
        sample_id="sample-1",
        episode_id="episode-1",
        observed_at=3,
        model_version="actor-0",
        features={"text": "local"},
        role="train",
        permissions=ExperiencePermissions(training=True),
        candidate_ids=("yes", "no"),
        action_id="yes",
        reward=-0.5,
    )


@pytest.mark.parametrize("bad", [True, -1, 0.5, float("nan"), "1"])
def test_should_reject_invalid_logical_time(bad):
    with pytest.raises(ValueError, match="nonnegative integer"):
        LogicalClock(bad)


def test_should_advance_monotonically_without_using_wall_time():
    clock = LogicalClock(2)
    clock.advance_to(2)
    clock.advance_to(7)
    with pytest.raises(ValueError, match="backwards"):
        clock.advance_to(6)
    assert clock.now() == 7


@pytest.mark.parametrize(
    "changes",
    [
        {"sample_id": ""},
        {"episode_id": " episode"},
        {"model_version": 1},
        {"observed_at": True},
        {"candidate_ids": ["yes"]},
        {"candidate_ids": ("yes", "yes")},
        {"action_id": "missing"},
        {"reward": float("nan")},
        {"reward": float("inf")},
        {"reward": True},
        {"role": "final"},
        {"permissions": {"training": True}},
    ],
)
def test_should_reject_malformed_experience_metadata(changes):
    with pytest.raises(ValueError):
        replace(sample(), **changes)


@pytest.mark.parametrize("role", ["inner_guard", "outer_selection", "final_test"])
def test_should_deny_training_and_replay_permissions_for_held_out_roles(role):
    for permissions in [ExperiencePermissions(training=True), ExperiencePermissions(replay=True)]:
        with pytest.raises(ValueError, match="held-out"):
            replace(sample(), role=role, permissions=permissions)
    declared = replace(sample(), role=role, permissions=ExperiencePermissions(evaluation=True))
    assert not declared.permissions.training and not declared.permissions.replay


def test_should_preserve_signed_reward_candidates_and_version_tags():
    record = sample()
    assert record.reward == -0.5
    assert record.candidate_ids == ("yes", "no") and record.action_id == "yes"
    assert record.model_version == "actor-0"


def test_should_require_exact_boolean_permissions_and_valid_label_identity():
    with pytest.raises(ValueError, match="boolean"):
        ExperiencePermissions(training=cast(Any, 1))
    with pytest.raises(ValueError, match="event_id"):
        LabelArrival("", "sample-1", "episode-1", 3, "actor-0", "label")
