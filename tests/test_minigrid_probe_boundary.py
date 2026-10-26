"""Reserved MiniGrid probe contexts use the existing owned public episode port."""

from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

from src.core.demonstration_episode import EpisodeReplaySchema, encode_episode
from src.core.minigrid_key_door import (
    MISSION,
    MiniGridKeyDoorLayout,
    predefined_layouts,
    predefined_probe_layouts,
)
from src.infra import minigrid_key_door as boundary
from test_minigrid_memory_adapter import Environment


def public(cue=None):
    image = np.zeros((7, 7, 3), dtype=np.uint8)
    image[:, :, 0] = 1
    if cue is not None:
        image[2, 6] = (6, 2 if cue == "blue" else 0, 0)
    return dict(image=image, direction=0, mission=MISSION)


@pytest.fixture
def environment_factory(monkeypatch):
    calls: list[tuple[Any, str, int, Environment]] = []

    def create(layout, cue, max_steps):
        environment = Environment(
            initial=public(cue),
            results=[
                (public(), 0.5 if index == 8 else 0.0, index == 8, False, {}) for index in range(9)
            ],
        )
        calls.append((layout, cue, max_steps, environment))
        return environment

    monkeypatch.setattr(boundary, "_make_key_door_environment", create)
    return calls


def test_should_declare_fixed_probe_geometry_and_preserve_original_layouts():
    probes = predefined_probe_layouts()
    assert type(probes) is tuple and len(probes) == 2
    assert [asdict(layout) for layout in probes] == [
        dict(context="E-blue", rotation=1, branch_row=3, swap_keys=False, wall_color="yellow"),
        dict(context="E-red", rotation=1, branch_row=3, swap_keys=False, wall_color="yellow"),
    ]
    assert [
        (item.context, item.rotation, item.branch_row, item.swap_keys, item.wall_color)
        for item in predefined_layouts()
    ] == [
        ("A", 0, 3, False, "grey"),
        ("B", 1, 4, True, "blue"),
        ("C_train", 2, 3, True, "red"),
        ("D", 3, 4, False, "yellow"),
    ]
    with pytest.raises(ValueError):
        MiniGridKeyDoorLayout("E")


@pytest.mark.parametrize("index,cue", [(0, "blue"), (1, "red")])
@pytest.mark.parametrize("role", ["probe_train", "probe_test"])
def test_should_preserve_reserved_role_and_context_in_complete_owned_148_k7_episode(
    environment_factory, index, cue, role
):
    layout = predefined_probe_layouts()[index]
    with boundary.MiniGridKeyDoor(layout, cue, role=role, episode_id=f"{role}-{cue}") as game:
        assert len(game.observation) == 148 and all(
            type(value) is float for value in game.observation
        )
        assert game.observation == boundary.encode_key_door_observation(public(cue))
        for _ in range(9):
            game.step(0)
        report = game.report
        assert report["context"] == layout.context and report["role"] == role
        assert report["layout"] == asdict(layout) and report["cue"] == cue
        assert report["seed"] == 17 and report["max_steps"] == 16
        assert report["environment_steps"] == report["environment_steps_attempted"] == 9
        assert report["initial_observation"]["mission"] == MISSION
        episode = game.to_demonstration()
        assert (
            episode.context == layout.context and episode.role == role and episode.num_classes == 7
        )
        assert len(episode.transitions) == 9 and all(
            len(row.observation) == 148 for row in episode.transitions
        )
        with pytest.raises(ValueError, match="training"):
            encode_episode(episode, EpisodeReplaySchema(148, 9, 7))
        leaked = game.report
        leaked["layout"]["rotation"] = 3
        assert game.report["layout"] == asdict(layout)
    assert len(environment_factory) == 1
    assert environment_factory[0][3].reset_seeds == [17]
    assert environment_factory[0][3].close_calls == 1


def test_should_keep_probe_context_metadata_out_of_numeric_inputs(environment_factory):
    layouts = predefined_probe_layouts()
    with boundary.MiniGridKeyDoor(layouts[0], "blue", role="probe_train") as blue:
        with boundary.MiniGridKeyDoor(layouts[1], "red", role="probe_test") as red:
            assert blue.report["context"] != red.report["context"]
            assert blue.observation != red.observation
            blue.step(0)
            red.step(0)
            assert blue.observation == red.observation
            assert len(blue.observation) == len(red.observation) == 148


@pytest.mark.parametrize("context,cue", [("E-blue", "blue"), ("E-red", "red")])
@pytest.mark.parametrize("role", ["train", "final_test", "validation"])
def test_should_refuse_ordinary_roles_for_probe_context_before_factory(
    monkeypatch, context, cue, role
):
    def forbidden(*args, **kwargs):
        pytest.fail("reserved-role refusal reached environment/dependency work")

    monkeypatch.setattr(boundary, "_make_key_door_environment", forbidden)
    selected = next(item for item in predefined_probe_layouts() if item.context == context)
    with pytest.raises(ValueError):
        boundary.MiniGridKeyDoor(selected, cue, role=role)


@pytest.mark.parametrize("index,cue", [(0, "red"), (1, "blue")])
@pytest.mark.parametrize("role", ["probe_train", "probe_test"])
def test_should_refuse_mismatched_probe_cue_before_factory(monkeypatch, index, cue, role):
    def forbidden(*args, **kwargs):
        pytest.fail("wrong cue reached environment/dependency work")

    monkeypatch.setattr(boundary, "_make_key_door_environment", forbidden)
    with pytest.raises(ValueError):
        boundary.MiniGridKeyDoor(predefined_probe_layouts()[index], cue, role=role)


@pytest.mark.parametrize("role", ["probe_train", "probe_test"])
def test_should_keep_heldout_d_out_of_probe_roles_before_factory(monkeypatch, role):
    def forbidden(*args, **kwargs):
        pytest.fail("heldout D entered probe construction")

    monkeypatch.setattr(boundary, "_make_key_door_environment", forbidden)
    with pytest.raises(ValueError):
        boundary.MiniGridKeyDoor(predefined_layouts()[-1], "blue", role=role)
