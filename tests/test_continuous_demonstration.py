"""Continuous demonstrations retain physical values and release only training roles."""

from dataclasses import FrozenInstanceError, replace

import pytest

from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
    continuous_training_pairs,
    copy_continuous_episode,
)


class ForeignInt(int):
    pass


class ForeignFloat(float):
    pass


class ForeignTuple(tuple):
    pass


def action_spec(**changes):
    values = dict(
        observation_dim=2,
        action_units=("rad/s", "N"),
        lower_bounds=(-2.5, 4),
        upper_bounds=(3, 4),
    )
    values.update(changes)
    return ContinuousActionSpec(**values)


def transition(**changes):
    values = dict(
        step=0,
        observation=(100, -50),
        action=(-2.5, 4),
        reward=37,
        next_observation=(101, -49),
        terminated=True,
        truncated=False,
    )
    values.update(changes)
    return ContinuousTransition(**values)


def episode(*, steps=2, spec=None, **changes):
    geometry = action_spec() if spec is None else spec
    records = tuple(
        transition(
            step=index,
            observation=tuple(100 * axis + index for axis in range(geometry.observation_dim)),
            next_observation=tuple(
                100 * axis + index + 1 for axis in range(geometry.observation_dim)
            ),
            action=geometry.lower_bounds,
            reward=37 if index % 2 else -42,
            terminated=index == steps - 1,
        )
        for index in range(steps)
    )
    values = dict(
        episode_id="robot-episode-1",
        context="reach",
        role="train",
        spec=geometry,
        transitions=records,
    )
    values.update(changes)
    return ContinuousEpisode(**values)


@pytest.mark.parametrize("terminated,truncated", [(True, False), (False, True), (True, True)])
def test_should_preserve_both_end_flags_and_original_physical_values(terminated, truncated):
    record = transition(terminated=terminated, truncated=truncated)
    packet = episode(transitions=(record,))
    copied = copy_continuous_episode(packet)

    assert copied == packet
    assert copied.transitions[0].terminated is terminated
    assert copied.transitions[0].truncated is truncated
    assert copied.transitions[0].reward == 37.0
    assert copied.transitions[0].observation == (100.0, -50.0)
    assert copied.transitions[0].action == (-2.5, 4.0)
    assert copied.spec.action_units == ("rad/s", "N")
    assert copied.spec.lower_bounds == (-2.5, 4.0)
    assert copied.spec.upper_bounds == (3.0, 4.0)


@pytest.mark.parametrize("dimension,actions,steps", [(1, 1, 1), (1024, 16, 256)])
def test_should_admit_declared_geometry_limits(dimension, actions, steps):
    spec = action_spec(
        observation_dim=dimension,
        action_units=("m",) * actions,
        lower_bounds=(-100,) * actions,
        upper_bounds=(100,) * actions,
    )
    packet = episode(spec=spec, steps=steps)

    features, targets = continuous_training_pairs(packet)

    assert len(features) == len(targets) == steps
    assert len(features[0]) == dimension
    assert len(targets[0]) == actions
    assert all(type(value) is float for row in features + targets for value in row)


def test_should_extract_only_current_observations_and_full_continuous_action_rows():
    packet = episode()
    features, targets = continuous_training_pairs(packet)

    assert features == ((0.0, 100.0), (1.0, 101.0))
    assert targets == ((-2.5, 4.0), (-2.5, 4.0))
    assert features[-1] != packet.transitions[-1].next_observation
    assert all(type(rows) is tuple for rows in (features, targets))
    assert all(type(row) is tuple for row in features + targets)


def test_should_own_all_copied_records_and_tuple_rows():
    packet = episode()
    copied = copy_continuous_episode(packet)
    features, targets = continuous_training_pairs(packet)

    assert copied is not packet
    assert copied.spec is not packet.spec
    assert copied.transitions is not packet.transitions
    assert copied.spec.lower_bounds is not packet.spec.lower_bounds
    assert copied.spec.upper_bounds is not packet.spec.upper_bounds
    assert copied.spec.action_units is not packet.spec.action_units
    for original, owned, feature, target in zip(
        packet.transitions, copied.transitions, features, targets, strict=True
    ):
        assert owned is not original
        assert owned.observation is not original.observation
        assert owned.next_observation is not original.next_observation
        assert owned.action is not original.action
        assert feature is not original.observation
        assert target is not original.action
    object.__setattr__(packet.transitions[0], "observation", (9.0, 9.0))
    object.__setattr__(packet.spec, "lower_bounds", (9.0, 9.0))
    assert copied.transitions[0].observation == features[0] == (0.0, 100.0)
    assert copied.spec.lower_bounds == targets[0] == (-2.5, 4.0)
    with pytest.raises(FrozenInstanceError):
        copied.role = "final_test"
    with pytest.raises(TypeError):
        features[0][0] = 9.0


@pytest.mark.parametrize("role", ["train", "probe_train", "probe_test", "final_test"])
def test_should_store_explicit_roles_without_implicitly_releasing_them(role):
    packet = episode(role=role)
    assert copy_continuous_episode(packet).role == role
    if role in ("train", "probe_train"):
        assert continuous_training_pairs(packet, role=role)[0][0] == (0.0, 100.0)
    else:
        with pytest.raises(ValueError):
            continuous_training_pairs(packet, role=role)
        with pytest.raises(ValueError):
            continuous_training_pairs(packet)


@pytest.mark.parametrize("stored,requested", [("train", "probe_train"), ("probe_train", "train")])
def test_should_refuse_cross_release_between_deployed_and_probe_training(stored, requested):
    with pytest.raises(ValueError):
        continuous_training_pairs(episode(role=stored), role=requested)


@pytest.mark.parametrize("role", ["final_test", "probe_test", "validation", "", True, None])
def test_should_refuse_non_training_release_requests(role):
    with pytest.raises(ValueError):
        continuous_training_pairs(episode(), role=role)


@pytest.mark.parametrize("dimension", [0, 1025, True, 2.0, ForeignInt(2)])
def test_should_refuse_non_native_or_unbounded_observation_dimension(dimension):
    with pytest.raises(ValueError):
        action_spec(observation_dim=dimension)


@pytest.mark.parametrize(
    "changes",
    [
        {"action_units": ()},
        {"action_units": ("m",) * 17},
        {"action_units": ["m", "N"]},
        {"action_units": ForeignTuple(("m", "N"))},
        {"action_units": ("", "N")},
        {"action_units": (1, "N")},
        {"action_units": ("é" * 17, "N")},
        {"action_units": ("\ud800", "N")},
        {"lower_bounds": [-2.5, 4]},
        {"lower_bounds": (-2.5,)},
        {"upper_bounds": (3, 4, 5)},
        {"lower_bounds": (3.1, 4)},
        {"lower_bounds": (False, 4)},
        {"upper_bounds": (ForeignFloat(3), 4)},
        {"upper_bounds": (float("inf"), 4)},
        {"lower_bounds": (float("nan"), 4)},
    ],
)
def test_should_refuse_invalid_units_bounds_or_action_geometry(changes):
    with pytest.raises(ValueError):
        action_spec(**changes)


def test_should_accept_units_at_utf8_limit_and_action_bounds_inclusively():
    spec = action_spec(action_units=("é" * 16, "N"))
    for action in (spec.lower_bounds, spec.upper_bounds):
        assert episode(spec=spec, transitions=(transition(action=action),)).spec == spec


@pytest.mark.parametrize(
    "changes",
    [
        {"step": True},
        {"step": -1},
        {"step": 256},
        {"step": 0.0},
        {"step": ForeignInt(0)},
        {"observation": []},
        {"observation": ()},
        {"observation": (1.0,) * 1025},
        {"observation": (True, 0)},
        {"next_observation": (1.0,)},
        {"action": []},
        {"action": ()},
        {"action": (1.0,) * 17},
        {"action": (True, 4)},
        {"reward": False},
        {"reward": ForeignFloat(1)},
        {"terminated": 1},
        {"truncated": 0},
    ],
)
def test_should_refuse_malformed_transition_types_shapes_and_flags(changes):
    with pytest.raises(ValueError):
        transition(**changes)


@pytest.mark.parametrize("field", ["observation", "next_observation", "action", "reward"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), ForeignInt(1)])
def test_should_refuse_nonfinite_or_foreign_scalars_in_every_numeric_field(field, value):
    replacement = value if field == "reward" else (value, 4)
    with pytest.raises(ValueError):
        transition(**{field: replacement})


@pytest.mark.parametrize("action", [(-2.500001, 4), (3.000001, 4), (0, 3.999999), (0, 4.000001)])
def test_should_refuse_out_of_bounds_actions_instead_of_clipping(action):
    record = transition(action=action)
    with pytest.raises(ValueError):
        episode(transitions=(record,))


@pytest.mark.parametrize("reward", [-1e200, 1e200])
def test_should_preserve_finite_unbounded_reward_without_toy_clipping(reward):
    packet = episode(transitions=(transition(reward=reward),))
    assert packet.transitions[0].reward == reward


@pytest.mark.parametrize("field", ["lower_bounds", "upper_bounds"])
def test_should_refuse_integer_bounds_that_would_round_during_float_ownership(field):
    with pytest.raises(ValueError):
        action_spec(**{field: (2**53 + 1, 4)})


@pytest.mark.parametrize("field", ["observation", "next_observation", "action", "reward"])
def test_should_refuse_inexact_integer_transition_values_before_rounding(field):
    value = 2**53 + 1
    replacement = value if field == "reward" else (value, 4)
    with pytest.raises(ValueError):
        transition(**{field: replacement})


def test_should_accept_exactly_representable_large_integer_values_without_changing_them():
    value = 2**53
    spec = action_spec(lower_bounds=(value, 4), upper_bounds=(value, 4))
    record = transition(
        observation=(value, -value),
        next_observation=(value, value),
        action=(value, 4),
        reward=value,
    )
    packet = episode(spec=spec, transitions=(record,))
    features, targets = continuous_training_pairs(packet)
    assert features == ((float(value), float(-value)),)
    assert targets == ((float(value), 4.0),)
    assert packet.transitions[0].reward == value


@pytest.mark.parametrize(
    "changes",
    [
        {"role": "validation"},
        {"role": True},
        {"episode_id": ""},
        {"context": ""},
        {"transitions": ()},
        {"transitions": []},
    ],
)
def test_should_refuse_invalid_episode_metadata_or_record_container(changes):
    with pytest.raises(ValueError):
        episode(**changes)


@pytest.mark.parametrize(
    "defect",
    [
        "reordered",
        "early_terminated",
        "early_truncated",
        "discontinuous",
        "incomplete",
        "wrong_observation_dim",
        "wrong_action_dim",
    ],
)
def test_should_refuse_partial_discontinuous_or_mixed_geometry_episodes(defect):
    packet = episode()
    first, last = packet.transitions
    if defect == "reordered":
        records = (last, first)
    elif defect == "early_terminated":
        records = (replace(first, terminated=True), last)
    elif defect == "early_truncated":
        records = (replace(first, truncated=True), last)
    elif defect == "discontinuous":
        records = (replace(first, next_observation=(9, 9)), last)
    elif defect == "incomplete":
        records = (first, replace(last, terminated=False))
    elif defect == "wrong_observation_dim":
        records = (transition(observation=(1,), next_observation=(2,)),)
    else:
        records = (transition(action=(0,)),)
    with pytest.raises(ValueError):
        replace(packet, transitions=records)


def test_should_refuse_more_than_maximum_complete_episode_length():
    records = tuple(
        transition(
            step=index if index < 256 else 255,
            observation=(index, 0),
            next_observation=(index + 1, 0),
            terminated=index == 256,
        )
        for index in range(257)
    )
    with pytest.raises(ValueError):
        episode(transitions=records)


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("spec", "lower_bounds", (99.0, 4.0)),
        ("transition", "action", (99.0, 4.0)),
        ("transition", "reward", float("nan")),
        ("transition", "step", 5),
        ("transition", "truncated", True),
        ("episode", "role", "validation"),
        ("episode", "transitions", ()),
    ],
)
def test_should_revalidate_corrupted_frozen_records_before_copy_or_release(target, field, value):
    packet = episode()
    record = {"spec": packet.spec, "transition": packet.transitions[0], "episode": packet}[target]
    object.__setattr__(record, field, value)
    with pytest.raises(ValueError):
        copy_continuous_episode(packet)
    with pytest.raises(ValueError):
        continuous_training_pairs(packet)


def test_should_require_exact_episode_spec_and_transition_record_types():
    class EpisodeSubclass(ContinuousEpisode):
        pass

    class SpecSubclass(ContinuousActionSpec):
        pass

    class TransitionSubclass(ContinuousTransition):
        pass

    packet = episode()
    subclass = EpisodeSubclass(
        packet.episode_id, packet.context, packet.role, packet.spec, packet.transitions
    )
    with pytest.raises(ValueError):
        copy_continuous_episode(subclass)
    with pytest.raises(ValueError):
        continuous_training_pairs(subclass)
    with pytest.raises(ValueError):
        replace(packet, spec=SpecSubclass(2, ("m", "N"), (-2.5, 4), (3, 4)))
    with pytest.raises(ValueError):
        replace(
            packet, transitions=(TransitionSubclass(0, (0, 0), (0, 4), 1, (1, 0), True, False),)
        )
