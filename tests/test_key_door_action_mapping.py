"""Movement controls may change without changing the spatial observation basis."""

from itertools import permutations

import numpy as np
import pytest

from src.core.key_door_navigation import (
    KeyDoorLayout,
    KeyDoorNavigation,
    key_door_layout_record,
    scripted_actions,
)


IDENTITY = (0, 1, 2, 3)
REMAPPED = (1, 2, 3, 0)


@pytest.mark.parametrize("movement_map", list(permutations(range(4))))
@pytest.mark.parametrize("rotation", range(4))
@pytest.mark.parametrize("cue", (1, -1))
def test_should_preserve_spatial_observations_and_goal_under_every_control_mapping(
    movement_map, rotation, cue
):
    reference_layout = KeyDoorLayout("A", rotation, 2, True, 0.25)
    mapped_layout = KeyDoorLayout("A", rotation, 2, True, 0.25, movement_map)
    reference = KeyDoorNavigation(reference_layout, cue)
    mapped = KeyDoorNavigation(mapped_layout, cue)
    original_actions = scripted_actions(reference_layout, cue)
    mapped_actions = scripted_actions(mapped_layout, cue)

    assert len(original_actions) == len(mapped_actions) == 8
    for index, (original_action, mapped_action) in enumerate(
        zip(original_actions, mapped_actions, strict=True)
    ):
        assert mapped.observation == reference.observation
        expected_legal = tuple(
            action
            for action in range(5)
            if (4 if action == 4 else movement_map[action]) in reference.legal_actions
        )
        assert mapped.legal_actions == expected_legal
        assert mapped_action in mapped.legal_actions
        if original_action == 4:
            assert mapped_action == 4
        else:
            assert movement_map[mapped_action] == original_action
        expected = reference.step(original_action)
        actual = mapped.step(mapped_action)
        assert actual == expected
        assert actual.reward == (1.0 if index == 7 else 0.0)
        assert actual.terminal is (index == 7)
        assert actual.success is (index == 7)

    assert mapped.legal_actions == reference.legal_actions == ()


def test_should_change_action_meaning_at_the_exact_same_observation():
    original = KeyDoorNavigation(KeyDoorLayout("A"), 1)
    mapped = KeyDoorNavigation(KeyDoorLayout("B", movement_action_map=REMAPPED), 1)
    assert mapped.observation == original.observation
    assert original.legal_actions == (0,)
    assert mapped.legal_actions == (3,)
    before = mapped.observation
    with pytest.raises(ValueError):
        mapped.step(0)
    assert mapped.observation == before
    assert mapped.step(3) == original.step(0)


def test_should_keep_interaction_fixed_for_pickup_unlock_and_goal():
    layout = KeyDoorLayout("B", 3, 1, False, -0.5, REMAPPED)
    environment = KeyDoorNavigation(layout, -1)
    actions = scripted_actions(layout, -1)
    interactions = 0
    for action in actions:
        if action == 4:
            assert 4 in environment.legal_actions
            interactions += 1
        environment.step(action)
    assert interactions == 2
    assert environment.legal_actions == ()


def test_should_preserve_legacy_layout_record_and_add_only_mapped_provenance():
    original = KeyDoorLayout("A", 2, 3, True, 0.5)
    expected = dict(layout_id="A", rotation=2, branch_row=3, swap_keys=True, appearance=0.5)
    assert original.movement_action_map == IDENTITY
    assert key_door_layout_record(original) == expected
    assert key_door_layout_record(KeyDoorLayout("A", 2, 3, True, 0.5, IDENTITY)) == expected
    mapped = KeyDoorLayout("A", 2, 3, True, 0.5, REMAPPED)
    assert key_door_layout_record(mapped) == {
        **expected,
        "movement_action_map": REMAPPED,
        "observation_basis": "spatial_directions_v1",
    }
    record = key_door_layout_record(mapped)
    record["movement_action_map"] = IDENTITY
    record["rotation"] = 0
    assert mapped.movement_action_map == REMAPPED and mapped.rotation == 2


class TupleSubclass(tuple):
    pass


INVALID_MAPS = (
    None,
    [0, 1, 2, 3],
    (),
    (0, 1, 2),
    (0, 1, 2, 3, 4),
    (0, 1, 1, 3),
    (-1, 1, 2, 3),
    (0, 1, 2, 4),
    (False, 1, 2, 3),
    (0.0, 1, 2, 3),
    (np.int64(0), 1, 2, 3),
    TupleSubclass(IDENTITY),
)


@pytest.mark.parametrize("invalid", INVALID_MAPS)
def test_should_refuse_malformed_control_maps_at_construction(invalid):
    with pytest.raises(ValueError):
        KeyDoorLayout("B", movement_action_map=invalid)


@pytest.mark.parametrize("invalid", INVALID_MAPS)
def test_should_revalidate_bypassed_control_maps_before_using_layouts(invalid):
    layout = KeyDoorLayout("B")
    object.__setattr__(layout, "movement_action_map", invalid)
    for consume in (
        lambda: KeyDoorNavigation(layout, 1),
        lambda: scripted_actions(layout, 1),
        lambda: key_door_layout_record(layout),
    ):
        with pytest.raises(ValueError):
            consume()


def test_should_own_control_mapping_for_an_episode_after_caller_layout_changes():
    layout = KeyDoorLayout("B", movement_action_map=REMAPPED)
    actions = scripted_actions(layout, 1)
    environment = KeyDoorNavigation(layout, 1)
    expected = KeyDoorNavigation(KeyDoorLayout("B", movement_action_map=REMAPPED), 1)
    object.__setattr__(layout, "movement_action_map", IDENTITY)
    object.__setattr__(layout, "rotation", 2)
    for action in actions:
        assert environment.legal_actions == expected.legal_actions
        assert environment.step(action) == expected.step(action)
