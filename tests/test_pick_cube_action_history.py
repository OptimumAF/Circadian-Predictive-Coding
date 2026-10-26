"""Causal executed-action history using only literal bounded native records.

No arrays, models, RNG, fitting, forwards, gradients, optimizer or native calls.
Each complete episode fixture has three rows (at most47 coordinates per view).
The closing-count test uses eight independent scalar state transitions, not data.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from src.core import pick_cube_action_history as history
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)


UNITS = ("normalized_root_dx", "normalized_root_dy", "normalized_root_dz", "normalized_gripper")
ACTIONS = ((0.5, -0.25, 0.0, -1.0), (0.0, 0.25, -0.5, -1.0), (0.25, 0.0, 0.0, 1.0))


def raw(index=0, *, sign=1.0):
    values = [0.0] * 42
    values[0], values[40] = float(index), float(index + 10)
    values[22:26] = [sign, 0.0, 0.0, 0.0]
    # Use a second exactly unit quaternion with an unambiguous dominant entry.
    values[32:36] = [0.0, -sign, 0.0, 0.0]
    return tuple(values)


def spec():
    return ContinuousActionSpec(42, UNITS, (-1.0,) * 4, (1.0,) * 4)


def episode(seed=17, actions=ACTIONS):
    return ContinuousEpisode(
        f"arbitrary-train-{seed}",
        "pick-cube" if seed == 17 else "pick-cube-train-19",
        "train",
        spec(),
        tuple(
            ContinuousTransition(
                i,
                raw(i, sign=-1.0 if i % 2 else 1.0),
                action,
                float(i == 2),
                raw(i + 1, sign=-1.0 if (i + 1) % 2 else 1.0),
                i == 2,
                False,
            )
            for i, action in enumerate(actions)
        ),
    )


def test_should_own_recipe_and_preserve_explicit_equal_width_zero_control():
    recipe = history.PickCubeActionHistoryRecipe()
    owned = history.validate_action_history_recipe(recipe)
    assert owned == recipe and owned is not recipe
    assert asdict(recipe) == dict(
        mode="executed",
        canonical_quaternions=False,
        definition="pick_cube_state42_previous_executed_action4_closing_count_v1",
        source_dim=42,
        feature_dim=47,
        closing_limit=6,
    )
    zero = replace(recipe, mode="zero")
    assert history.action_history_spec(spec(), recipe) == history.action_history_spec(spec(), zero)
    assert history.action_history_spec(spec(), recipe).observation_dim == 47
    with pytest.raises(FrozenInstanceError):
        setattr(recipe, "mode", "zero")
    object.__setattr__(recipe, "feature_dim", 48)
    assert owned.feature_dim == 47


class StringAlias(str):
    pass


class IntegerAlias(int):
    pass


class FloatAlias(float):
    pass


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "teacher"),
        ("mode", StringAlias("executed")),
        ("canonical_quaternions", 1),
        ("definition", "other"),
        ("source_dim", 42.0),
        ("feature_dim", 47.0),
        ("closing_limit", 6.0),
        ("closing_limit", True),
    ],
)
def test_should_refuse_changed_or_aliased_recipe_metadata(field, value):
    recipe = history.PickCubeActionHistoryRecipe()
    object.__setattr__(recipe, field, value)
    with pytest.raises(ValueError):
        history.validate_action_history_recipe(recipe)


def test_should_record_previous_successful_action_only_and_preserve_raw_fields_and_seams():
    source = episode()
    original = deepcopy(source)
    recipe = history.PickCubeActionHistoryRecipe()
    encoded = history.action_history_training_episode(source, recipe)
    assert source == original and encoded is not source
    assert (encoded.episode_id, encoded.context, encoded.role) == (
        source.episode_id,
        source.context,
        source.role,
    )
    assert encoded.spec.action_units == UNITS and encoded.spec.lower_bounds == (-1.0,) * 4
    expected_before = ((0.0,) * 5, (*ACTIONS[0], 1 / 6), (*ACTIONS[1], 2 / 6))
    expected_after = ((*ACTIONS[0], 1 / 6), (*ACTIONS[1], 2 / 6), (*ACTIONS[2], 0.0))
    for i, (actual, original_row) in enumerate(
        zip(encoded.transitions, source.transitions, strict=True)
    ):
        assert actual.observation[:42] == original_row.observation
        assert actual.next_observation[:42] == original_row.next_observation
        assert actual.observation[42:] == expected_before[i]
        assert actual.next_observation[42:] == expected_after[i]
        assert (actual.action, actual.reward, actual.terminated, actual.truncated) == (
            original_row.action,
            original_row.reward,
            original_row.terminated,
            original_row.truncated,
        )
        if i:
            assert encoded.transitions[i - 1].next_observation == actual.observation
    validated = history.validate_action_history_training_episode(encoded, recipe)
    assert validated == encoded and validated is not encoded


def test_should_ignore_future_actions_and_reset_history_independently_for_each_episode():
    recipe = history.PickCubeActionHistoryRecipe()
    first = history.action_history_training_episode(episode(), recipe)
    changed_last = (*ACTIONS[:2], (0.0, 0.0, 0.0, -1.0))
    second = history.action_history_training_episode(episode(19, changed_last), recipe)
    assert tuple(row.observation for row in first.transitions) == tuple(
        row.observation for row in second.transitions
    )
    assert first.transitions[-1].next_observation[42:] == (*ACTIONS[-1], 0.0)
    assert second.transitions[-1].next_observation[42:] == (0.0, 0.0, 0.0, -1.0, 0.5)
    assert (
        first.transitions[0].observation[42:]
        == second.transitions[0].observation[42:]
        == (0.0,) * 5
    )
    changed_current = ((0.0, 0.0, 0.0, 1.0), *ACTIONS[1:])
    third = history.action_history_training_episode(episode(actions=changed_current), recipe)
    assert third.transitions[0].observation == first.transitions[0].observation
    assert third.transitions[1].observation[42:] == (0.0, 0.0, 0.0, 1.0, 0.0)


def test_should_count_any_executed_negative_grip_as_proxy_cap_at_six_and_reset_on_nonnegative():
    state = history.ActionHistoryState()
    before = state
    counts = []
    for _ in range(8):
        state = history.advance_action_history(state, (0.5, 0.0, 0.0, -0.5))
        counts.append(state.closing_count)
    assert counts == [1, 2, 3, 4, 5, 6, 6, 6]
    assert before.previous_action == (0.0,) * 4 and before.closing_count == 0
    for gripper in (0.0, -0.0, 1.0):
        reset = history.advance_action_history(state, (0.0, 0.0, 0.0, gripper))
        assert reset.closing_count == 0
    owned = history.validate_action_history_state(state)
    assert owned == state and owned is not state
    with pytest.raises(FrozenInstanceError):
        setattr(state, "closing_count", 0)


@pytest.mark.parametrize(
    "action",
    [
        (0.1, 0.0, 0.0, -1.0),
        (0.0, 0.0, 0.0, -1e-50),
        (0.0, 0.0, 0.0, 1e-50),
        (FloatAlias(0.5), 0.0, 0.0, -1.0),
        (True, 0.0, 0.0, -1.0),
        (float("nan"), 0.0, 0.0, -1.0),
        (float("inf"), 0.0, 0.0, -1.0),
        (2.0, 0.0, 0.0, -1.0),
        [0.0, 0.0, 0.0, -1.0],
        (0.0, 0.0, -1.0),
    ],
)
def test_should_refuse_commands_that_cannot_be_the_exact_native_float32_execution(action):
    original = history.ActionHistoryState()
    with pytest.raises(ValueError):
        history.advance_action_history(original, action)
    assert original.previous_action == (0.0,) * 4 and original.closing_count == 0


@pytest.mark.parametrize(
    "previous,count",
    [
        ((0.0,) * 4, 1),
        ((0.0, 0.0, 0.0, -1.0), 0),
        ((0.0, 0.0, 0.0, -1.0), 7),
        ((0.0, 0.0, 0.0, -1.0), True),
        ((0.0, 0.0, 0.0, -1.0), IntegerAlias(1)),
        ((0.0, 0.0, 0.0, -1.0), 1.0),
        ((0.1, 0.0, 0.0, -1.0), 1),
    ],
)
def test_should_refuse_incoherent_or_aliased_state(previous, count):
    with pytest.raises(ValueError):
        history.ActionHistoryState(previous, count)


@pytest.mark.parametrize("canonical", [False, True])
def test_should_use_identical_prefix_preprocessing_for_executed_and_zero_history(canonical):
    executed = history.PickCubeActionHistoryRecipe(canonical_quaternions=canonical)
    zero = replace(executed, mode="zero")
    state = history.ActionHistoryState((0.5, 0.0, 0.0, -1.0), 1)
    source = raw(sign=-1.0)
    actual = history.action_history_observation(source, state, executed)
    control = history.action_history_observation(source, state, zero)
    assert len(actual) == len(control) == 47 and actual[:42] == control[:42]
    assert actual[42:] == (0.5, 0.0, 0.0, -1.0, 1 / 6) and control[42:] == (0.0,) * 5
    if canonical:
        assert (
            actual[:42] == history.action_history_observation(raw(sign=1.0), state, executed)[:42]
        )
        assert actual[22:26] == (1.0, 0.0, 0.0, 0.0) and actual[32:36] == (0.0, 1.0, 0.0, 0.0)
    else:
        assert actual[:42] == source
    assert actual[40] == source[40] and source == raw(sign=-1.0)


@pytest.mark.parametrize("mode", ["executed", "zero"])
def test_should_validate_actual_execution_even_when_history_tail_is_zero(mode):
    source = episode(actions=((0.1, 0.0, 0.0, -1.0), *ACTIONS[1:]))
    with pytest.raises(ValueError):
        history.action_history_training_episode(
            source, history.PickCubeActionHistoryRecipe(mode=mode)
        )


def test_should_zero_all_five_history_coordinates_in_complete_equal_width_control():
    source = episode()
    recipe = history.PickCubeActionHistoryRecipe(mode="zero")
    encoded = history.action_history_training_episode(source, recipe)
    assert encoded.spec.observation_dim == 47
    assert all(
        row.observation[42:] == row.next_observation[42:] == (0.0,) * 5
        for row in encoded.transitions
    )
    assert tuple(row.action for row in encoded.transitions) == ACTIONS
    assert history.validate_action_history_training_episode(encoded, recipe) == encoded
    with pytest.raises(ValueError):
        history.validate_action_history_training_episode(encoded, replace(recipe, mode="executed"))


@pytest.mark.parametrize(
    "fault", ["short", "list", "bool", "nonfinite", "canonical_zero_quaternion"]
)
def test_should_validate_full_raw_state_before_constructing_history(fault):
    source = raw()
    recipe = history.PickCubeActionHistoryRecipe()
    if fault == "short":
        source = source[:-1]
    elif fault == "list":
        source = list(source)
    elif fault == "bool":
        source = (True, *source[1:])
    elif fault == "nonfinite":
        source = (*source[:40], float("inf"), source[41])
    else:
        source = (*source[:22], 0.0, 0.0, 0.0, 0.0, *source[26:])
        recipe = replace(recipe, canonical_quaternions=True)
    with pytest.raises(ValueError):
        history.action_history_observation(source, history.ActionHistoryState(), recipe)


@pytest.mark.parametrize(
    "fault",
    [
        "final_test",
        "probe_train",
        "probe_test",
        "inner_guard",
        "context",
        "units",
        "bounds",
        "width",
    ],
)
def test_should_refuse_heldout_context_or_raw_controller_spec_changes(fault):
    source = episode()
    if fault in ("final_test", "probe_train", "probe_test", "inner_guard"):
        source = replace(source, role=fault)
    elif fault == "context":
        source = replace(source, context="pick-cube-light-half-train-17")
    elif fault == "units":
        source = replace(source, spec=replace(source.spec, action_units=("x", *UNITS[1:])))
    elif fault == "bounds":
        source = replace(source, spec=replace(source.spec, lower_bounds=(-2.0,) * 4))
    else:
        source = replace(
            source,
            spec=replace(source.spec, observation_dim=47),
            transitions=tuple(
                replace(
                    row,
                    observation=(*row.observation, *(0.0,) * 5),
                    next_observation=(*row.next_observation, *(0.0,) * 5),
                )
                for row in source.transitions
            ),
        )
    with pytest.raises(ValueError):
        history.action_history_training_episode(source, history.PickCubeActionHistoryRecipe())


@pytest.mark.parametrize(
    "fault", ["before_tail", "after_tail", "count", "prefix", "action", "recipe"]
)
def test_should_refuse_encoded_temporal_or_recipe_tampering_without_changing_admitted_record(fault):
    recipe = history.PickCubeActionHistoryRecipe()
    valid = history.action_history_training_episode(episode(), recipe)
    original = deepcopy(valid)
    damaged = deepcopy(valid)
    if fault == "before_tail":
        row = damaged.transitions[0]
        object.__setattr__(row, "observation", (*row.observation[:42], *ACTIONS[0], 1 / 6))
    elif fault == "after_tail":
        row = damaged.transitions[-1]
        object.__setattr__(
            row, "next_observation", (*row.next_observation[:42], *ACTIONS[1], 2 / 6)
        )
    elif fault == "count":
        row = damaged.transitions[-1]
        object.__setattr__(row, "next_observation", (*row.next_observation[:-1], 0.5))
    elif fault == "prefix":
        row = damaged.transitions[0]
        object.__setattr__(row, "next_observation", (99.0, *row.next_observation[1:]))
    elif fault == "action":
        object.__setattr__(damaged.transitions[0], "action", (0.0, 0.0, 0.0, 1.0))
    else:
        object.__setattr__(recipe, "closing_limit", 7)
    with pytest.raises(ValueError):
        history.validate_action_history_training_episode(damaged, recipe)
    assert valid == original
