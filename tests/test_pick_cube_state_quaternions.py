"""Pure fixed state42 quaternion representation: no learners or array kernels."""

from dataclasses import FrozenInstanceError, asdict, replace
import math
from typing import Any

import pytest

from src.core import pick_cube_state_quaternions as quaternion
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)


def observation(*, tcp=(-0.5, 0.5, -0.5, 0.5), cube=(0.0, 0.0, -1.0, 0.0)):
    values = [float(index) for index in range(42)]
    values[22:26], values[32:36] = tcp, cube
    return tuple(values)


def episode(seed=17):
    observations = (
        observation(),
        observation(tcp=(0.5, -0.5, 0.5, -0.5)),
        observation(cube=(0.0, 0.0, 1.0, 0.0)),
    )
    spec = ContinuousActionSpec(42, ("m", "constant"), (-2.0, 1.0), (6.0, 1.0))
    return ContinuousEpisode(
        f"pick-cube-readiness-teacher-{seed}",
        "pick-cube" if seed == 17 else "pick-cube-train-19",
        "train",
        spec,
        tuple(
            ContinuousTransition(
                index,
                observations[index],
                (float(index), 1.0),
                float(index == 1),
                observations[index + 1],
                index == 1,
                False,
            )
            for index in range(2)
        ),
    )


def test_should_freeze_complete_fixed_recipe_and_return_owned_validation():
    recipe = quaternion.state_quaternion_projection()
    assert type(recipe) is quaternion.PickCubeStateQuaternionProjection
    assert asdict(recipe) == dict(
        definition="pick_cube_state42_largest_component_quaternion_sign_v1",
        source_dim=42,
        feature_dim=42,
        quaternion_starts=(22, 32),
        quaternion_width=4,
        sign_rule="first_largest_abs_positive",
        unit_tolerance=1e-4,
    )
    owned = quaternion.validate_state_quaternion_projection(recipe)
    assert owned == recipe and owned is not recipe
    with pytest.raises(FrozenInstanceError):
        setattr(recipe, "source_dim", 41)


@pytest.mark.parametrize(
    "fault",
    [
        "mapping",
        "subclass",
        "bool_dim",
        "float_dim",
        "starts_list",
        "starts_bool",
        "other_start",
        "width",
        "definition",
        "sign_rule",
        "tolerance_bool",
        "tolerance_other",
    ],
)
def test_should_refuse_recipe_aliases_coercions_and_changed_semantics(fault):
    recipe: Any = quaternion.state_quaternion_projection()
    if fault == "mapping":
        recipe = asdict(recipe)
    elif fault == "subclass":

        class Derived(quaternion.PickCubeStateQuaternionProjection):
            pass

        recipe = Derived()
    else:
        field, value = {
            "bool_dim": ("source_dim", True),
            "float_dim": ("feature_dim", 42.0),
            "starts_list": ("quaternion_starts", [22, 32]),
            "starts_bool": ("quaternion_starts", (True, 32)),
            "other_start": ("quaternion_starts", (21, 32)),
            "width": ("quaternion_width", 3),
            "definition": ("definition", "learned_quaternion_projection"),
            "sign_rule": ("sign_rule", "first_nonzero_positive"),
            "tolerance_bool": ("unit_tolerance", True),
            "tolerance_other": ("unit_tolerance", 0.001),
        }[fault]
        object.__setattr__(recipe, field, value)
    with pytest.raises(ValueError):
        quaternion.validate_state_quaternion_projection(recipe)


@pytest.mark.parametrize(
    "flip_tcp,flip_cube", [(False, False), (True, False), (False, True), (True, True)]
)
def test_should_canonicalize_each_antipode_independently_keep_ties_and_every_other_coordinate(
    flip_tcp, flip_cube
):
    tcp, cube = (-0.5, 0.5, -0.5, 0.5), (0.0, 0.0, -1.0, -0.0)
    raw = observation(
        tcp=tuple(-x if flip_tcp else x for x in tcp),
        cube=tuple(-x if flip_cube else x for x in cube),
    )
    recipe = quaternion.state_quaternion_projection()
    result = quaternion.state_quaternion_observation(raw, recipe)
    assert result[22:26] == (0.5, -0.5, 0.5, -0.5)
    assert result[32:36] == (0.0, 0.0, 1.0, 0.0)
    assert all(
        result[index] == raw[index]
        for index in range(42)
        if index not in (*range(22, 26), *range(32, 36))
    )
    assert all(math.copysign(1.0, value) == 1 for value in result[32:36] if value == 0)
    assert quaternion.state_quaternion_observation(result, recipe) == result
    assert raw[22:26] == tuple(-x if flip_tcp else x for x in tcp)
    assert result is not raw


def test_should_validate_near_unit_norm_without_renormalizing_or_changing_other_signed_zero():
    raw = list(observation(tcp=(-1.00005, 0.0, -0.0, 0.0)))
    raw[0] = -0.0
    result = quaternion.state_quaternion_observation(
        tuple(raw), quaternion.state_quaternion_projection()
    )
    assert result[22] == 1.00005
    assert math.copysign(1.0, result[0]) == -1
    assert all(math.copysign(1.0, result[index]) == 1 for index in (23, 24, 25))


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "short",
        "long",
        "bool",
        "nan_other",
        "inf_quaternion",
        "zero_tcp",
        "nonunit_tcp",
        "nonunit_cube",
    ],
)
def test_should_refuse_incomplete_nonfinite_or_nonunit_raw_state_before_projection(fault):
    values: Any = list(observation())
    if fault == "short":
        values = tuple(values[:-1])
    elif fault == "long":
        values = tuple(values + [0.0])
    elif fault == "list":
        pass
    else:
        if fault == "bool":
            values[5] = True
        elif fault == "nan_other":
            values[41] = float("nan")
        elif fault == "inf_quaternion":
            values[22] = float("inf")
        elif fault == "zero_tcp":
            values[22:26] = [0.0] * 4
        elif fault == "nonunit_tcp":
            values[22:26] = [1.0002, 0.0, 0.0, 0.0]
        else:
            values[32:36] = [0.0, 0.0, 0.5, 0.0]
        values = tuple(values)
    with pytest.raises(ValueError):
        quaternion.state_quaternion_observation(values, quaternion.state_quaternion_projection())


@pytest.mark.parametrize("seed", [17, 19])
def test_should_project_owned_complete_train_episode_including_terminal_next_without_relabelling(
    seed,
):
    source = episode(seed)
    before = asdict(source)
    recipe = quaternion.state_quaternion_projection()
    result = quaternion.state_quaternion_training_episode(source, recipe)
    assert result is not source and result.spec == source.spec and result.spec is not source.spec
    assert (result.episode_id, result.context, result.role) == (
        source.episode_id,
        source.context,
        source.role,
    )
    for raw, projected in zip(source.transitions, result.transitions, strict=True):
        assert (
            projected.step,
            projected.action,
            projected.reward,
            projected.terminated,
            projected.truncated,
        ) == (raw.step, raw.action, raw.reward, raw.terminated, raw.truncated)
        assert projected.observation == quaternion.state_quaternion_observation(
            raw.observation, recipe
        )
        assert projected.next_observation == quaternion.state_quaternion_observation(
            raw.next_observation, recipe
        )
    assert asdict(source) == before
    object.__setattr__(source.transitions[0], "observation", (0.0,) * 42)
    assert result.transitions[0].observation[22:26] == (0.5, -0.5, 0.5, -0.5)


@pytest.mark.parametrize(
    "fault",
    [
        "probe_train",
        "final_test",
        "inner_guard",
        "context",
        "dimension",
        "terminal_quaternion",
        "horizon",
    ],
)
def test_should_refuse_unqualified_role_context_geometry_or_horizon(fault):
    source = episode()
    if fault in ("probe_train", "final_test", "inner_guard"):
        source = replace(source, role=fault)
    elif fault == "context":
        source = replace(source, context="pick-cube-mass-x2-train-17")
    elif fault == "dimension":
        object.__setattr__(source, "spec", replace(source.spec, observation_dim=41))
    elif fault == "terminal_quaternion":
        object.__setattr__(source.transitions[-1], "next_observation", (0.0,) * 42)
    else:
        row = source.transitions[0]
        source = replace(
            source,
            transitions=tuple(
                ContinuousTransition(
                    i, row.observation, row.action, 0.0, row.observation, i == 50, False
                )
                for i in range(51)
            ),
        )
    with pytest.raises(ValueError):
        quaternion.state_quaternion_training_episode(
            source, quaternion.state_quaternion_projection()
        )


def test_should_preserve_arbitrary_valid_action_bounds_units_in_state_spec():
    source = episode().spec
    result = quaternion.state_quaternion_spec(source, quaternion.state_quaternion_projection())
    assert result == source and result is not source
