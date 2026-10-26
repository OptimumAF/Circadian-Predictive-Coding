"""Frozen teacher behavior; these checks do not establish robot skill."""

from collections import Counter
from dataclasses import FrozenInstanceError

import pytest

from src.core.pick_cube_teacher import PickCubeGeometry, PickCubeTeacher


def geometry(tcp=(0.0, 0.0, 0.0), cube=(0.02, -0.03, 0.01), goal=(0.05, 0.02, 0.08)):
    return PickCubeGeometry(tcp, cube, goal)


def advance(teacher, count):
    for _ in range(count):
        teacher.action(geometry())


def test_should_follow_frozen_stage_counts_and_stop_at_fifty_actions():
    teacher = PickCubeTeacher()
    labels = []
    actions = []
    for _ in range(50):
        labels.append(teacher.stage)
        actions.append(teacher.action(geometry()))

    assert (
        labels == ["align"] * 8 + ["descend"] * 8 + ["grasp"] * 6 + ["lift"] * 18 + ["settle"] * 10
    )
    assert Counter(labels) == {"align": 8, "descend": 8, "grasp": 6, "lift": 18, "settle": 10}
    assert teacher.steps == 50 and teacher.stage == "complete"
    assert all(type(action) is tuple and len(action) == 4 for action in actions)
    assert all(-1.0 <= value <= 1.0 for action in actions for value in action)
    assert [action[3] for action in actions] == [1.0] * 16 + [-1.0] * 34
    with pytest.raises(ValueError, match="complete"):
        teacher.action(geometry())
    assert teacher.steps == 50


@pytest.mark.parametrize(
    "steps, expected",
    [
        (0, (0.2, -0.3, 0.9, 1.0)),
        (8, (0.2, -0.3, 0.1, 1.0)),
        (16, (0.0, 0.0, 0.0, -1.0)),
        (22, (0.3, 0.5, 0.7, -1.0)),
        (40, (0.0, 0.0, 0.0, -1.0)),
    ],
)
def test_should_use_root_metres_point_one_scale_and_frozen_gripper_sign(steps, expected):
    teacher = PickCubeTeacher()
    advance(teacher, steps)
    assert teacher.action(geometry()) == pytest.approx(expected)


def test_should_move_held_cube_to_goal_instead_of_assuming_cube_is_tcp():
    teacher = PickCubeTeacher()
    advance(teacher, 22)
    value = geometry(tcp=(-5.0, 9.0, 2.0), cube=(0.4, 0.5, 0.6), goal=(0.41, 0.48, 0.63))
    assert teacher.action(value) == pytest.approx((0.1, -0.2, 0.3, -1.0))


@pytest.mark.parametrize("steps", [0, 8, 22])
def test_should_clamp_translation_per_axis_without_changing_other_axes(steps):
    teacher = PickCubeTeacher()
    advance(teacher, steps)
    value = geometry(tcp=(0.0, 0.0, 0.0), cube=(10.0, -10.0, 0.0), goal=(-10.0, 10.0, 0.05))
    expected = (
        (-1.0, 1.0, 0.5, -1.0) if steps == 22 else (1.0, -1.0, 0.8 if steps == 0 else 0.0, 1.0)
    )
    assert teacher.action(value) == pytest.approx(expected)


def test_should_ignore_goal_during_alignment_and_descent():
    first = PickCubeTeacher()
    second = PickCubeTeacher()
    for _ in range(16):
        assert first.action(geometry()) == second.action(geometry(goal=(999.0, -999.0, 999.0)))


@pytest.mark.parametrize("field", ["tcp", "cube", "goal"])
@pytest.mark.parametrize(
    "invalid",
    [
        None,
        [],
        [1, 2, 3],
        (),
        (1, 2),
        (1, 2, 3, 4),
        (True, 0, 0),
        ("1", 0, 0),
        (float("nan"), 0, 0),
        (float("inf"), 0, 0),
        (-float("inf"), 0, 0),
        (10**1000, 0, 0),
    ],
)
def test_should_refuse_non_native_non_finite_or_wrong_length_geometry(field, invalid):
    values = dict(tcp=(0, 0, 0), cube=(0, 0, 0), goal=(0, 0, 0))
    values[field] = invalid
    with pytest.raises(ValueError):
        PickCubeGeometry(**values)


def test_should_normalize_native_coordinates_to_owned_frozen_float_tuples():
    value = PickCubeGeometry((0, 1, 2), (3, 4, 5), (6, 7, 8))
    assert value.tcp == (0.0, 1.0, 2.0)
    assert all(
        type(item) is float for point in (value.tcp, value.cube, value.goal) for item in point
    )
    with pytest.raises(FrozenInstanceError):
        value.tcp = (0.0, 0.0, 0.0)


@pytest.mark.parametrize("steps", [0, 8, 16, 22, 40])
@pytest.mark.parametrize("field", ["tcp", "cube", "goal"])
def test_should_reject_corrupted_geometry_without_consuming_an_action(steps, field):
    teacher = PickCubeTeacher()
    advance(teacher, steps)
    value = geometry()
    object.__setattr__(value, field, (float("nan"), 0.0, 0.0))
    with pytest.raises(ValueError):
        teacher.action(value)
    assert teacher.steps == steps
    teacher.action(geometry())
    assert teacher.steps == steps + 1


@pytest.mark.parametrize("invalid", [None, {}, (0, 0, 0)])
def test_should_refuse_wrong_geometry_type_without_consuming_schedule(invalid):
    teacher = PickCubeTeacher()
    with pytest.raises(ValueError, match="exact PickCubeGeometry"):
        teacher.action(invalid)
    assert teacher.steps == 0


def test_should_refuse_overflowing_position_difference_without_consuming_schedule():
    teacher = PickCubeTeacher()
    value = geometry(tcp=(-1e308, 0.0, 0.0), cube=(1e308, 0.0, 0.0))
    with pytest.raises(ValueError, match="not representable"):
        teacher.action(value)
    assert teacher.steps == 0
