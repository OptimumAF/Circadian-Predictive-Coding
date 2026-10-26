"""Finite instance catalogue and physical control parity without learners."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import src.core.key_door_instances as instances
from src.core.key_door_instances import seeded_action_remapping_layouts
from src.core.key_door_navigation import KeyDoorNavigation, scripted_actions


def physical_descriptor(layout):
    return layout.rotation, layout.branch_row, layout.swap_keys, layout.appearance


@pytest.mark.parametrize("environment_seed", range(24))
def test_should_cover_every_distinct_catalogue_pair_with_fixed_context_relationships(
    environment_seed,
):
    expected = (
        environment_seed % 4,
        1 + (environment_seed // 4) % 3,
        bool(environment_seed // 12),
        0.0,
    )
    for heldout_seed in range(24):
        if heldout_seed == environment_seed:
            continue
        layouts = seeded_action_remapping_layouts(environment_seed, heldout_seed)
        assert layouts == seeded_action_remapping_layouts(environment_seed, heldout_seed)
        assert tuple(layout.layout_id for layout in layouts) == ("A", "B", "C", "D")
        a, b, c, d = layouts
        assert (
            physical_descriptor(a) == physical_descriptor(b) == physical_descriptor(c) == expected
        )
        assert physical_descriptor(d) == (
            heldout_seed % 4,
            1 + (heldout_seed // 4) % 3,
            bool(heldout_seed // 12),
            0.0,
        )
        assert physical_descriptor(a) != physical_descriptor(d)
        assert (
            a.movement_action_map == c.movement_action_map == d.movement_action_map == (0, 1, 2, 3)
        )
        assert b.movement_action_map == (1, 2, 3, 0)


def test_should_enumerate_exactly_twenty_four_distinct_physical_instances():
    descriptors = {
        physical_descriptor(seeded_action_remapping_layouts(seed, (seed + 1) % 24)[0])
        for seed in range(24)
    }
    assert descriptors == {
        (rotation, branch_row, swap_keys, 0.0)
        for rotation in range(4)
        for branch_row in (1, 2, 3)
        for swap_keys in (False, True)
    }


@pytest.mark.parametrize("environment_seed", range(24))
@pytest.mark.parametrize("cue", [-1, 1])
def test_should_reach_each_context_goal_and_preserve_spatial_observations_under_remapping(
    environment_seed, cue
):
    layouts = seeded_action_remapping_layouts(environment_seed, (environment_seed + 1) % 24)
    environments = [KeyDoorNavigation(layout, cue) for layout in layouts]
    scripts = [scripted_actions(layout, cue) for layout in layouts]
    assert all(len(script) == 8 for script in scripts)
    a, b, c, _d = environments
    b_mapping = layouts[1].movement_action_map
    for step in range(8):
        assert a.observation == b.observation == c.observation
        assert a.legal_actions == c.legal_actions
        mapped_legal = {action if action == 4 else b_mapping[action] for action in b.legal_actions}
        assert set(a.legal_actions) == mapped_legal
        assert scripts[0][step] == scripts[2][step]
        assert scripts[0][step] == (4 if scripts[1][step] == 4 else b_mapping[scripts[1][step]])
        for environment, script in zip(environments, scripts):
            assert script[step] in environment.legal_actions
            result = environment.step(script[step])
            assert result.reward == (1.0 if step == 7 else 0.0)
            assert result.terminal is (step == 7)
    assert a.observation == b.observation == c.observation
    assert all(environment.success and environment.terminal for environment in environments)
    assert all(environment.legal_actions == () for environment in environments)


def test_should_return_fresh_frozen_layout_objects_without_cross_context_aliasing():
    first = seeded_action_remapping_layouts(0, 23)
    second = seeded_action_remapping_layouts(0, 23)
    assert all(left is not right for left, right in zip(first, second))
    assert len({id(layout) for layout in first}) == 4
    with pytest.raises(FrozenInstanceError):
        setattr(first[0], "rotation", 1)
    object.__setattr__(first[0], "rotation", 3)
    assert first[1].rotation == first[2].rotation == second[0].rotation == 0


@pytest.mark.parametrize("argument", ["environment_seed", "heldout_environment_seed"])
@pytest.mark.parametrize(
    "invalid", [True, False, None, -1, 24, 1.0, "1", [], np.int64(1), np.bool_(True)]
)
def test_should_refuse_non_native_or_out_of_range_indices_before_constructing_layouts(
    monkeypatch, argument, invalid
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid catalogue index reached layout construction")

    monkeypatch.setattr(instances, "KeyDoorLayout", forbidden)
    arguments = dict(environment_seed=0, heldout_environment_seed=1)
    arguments[argument] = invalid
    with pytest.raises(ValueError, match="native integer catalogue index"):
        seeded_action_remapping_layouts(**arguments)


@pytest.mark.parametrize("seed", range(24))
def test_should_refuse_identical_train_and_heldout_indices_before_constructing_layouts(
    monkeypatch, seed
):
    def forbidden(*args, **kwargs):
        pytest.fail("identical catalogue indices reached layout construction")

    monkeypatch.setattr(instances, "KeyDoorLayout", forbidden)
    with pytest.raises(ValueError, match="distinct"):
        seeded_action_remapping_layouts(seed, seed)
