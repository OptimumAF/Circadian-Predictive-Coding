"""Public-only teacher checks and bounded optional actual cue/key/door physics."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from importlib import metadata
from typing import Any

import numpy as np
import pytest

from src.app.coupled_attribution import AttributionBatch, _validate_batch
from src.core.demonstration_episode import (
    EpisodeReplaySchema,
    decode_episode,
    encode_episode,
    episode_arrays,
)
from src.core.minigrid_key_door import (
    CueKeyDoorTeacher,
    MiniGridKeyDoorLayout,
    MISSION,
    predefined_layouts,
)
from src.infra.minigrid_key_door import MiniGridKeyDoor, encode_key_door_observation
from src.infra.minigrid_memory import VERSIONS, encode_memory_observation


def public_frame(*, cue=None, left=None, right=None, door=None):
    """Independent native-index fixture: blue=2, red=0, key=5, ball=6, door=4."""
    image = np.zeros((7, 7, 3), dtype=np.uint8)
    image[:, :, 0] = 1  # Empty, rather than an invented hidden-grid encoding.
    if cue is not None:
        image[1, 5] = (6, cue, 0)
    if left is not None:
        image[2, 6] = (5, left, 0)
    if right is not None:
        image[4, 6] = (5, right, 0)
    if door is not None:
        image[3, 5] = (4, 5, door)
    values = []
    for x in range(7):
        for y in range(7):
            for channel, scale in enumerate((10, 5, 2)):
                values.append(float(2 * int(image[x, y, channel]) / scale - 1))
    return (*values, 1.0)


def actual_dependencies():
    for name, version in VERSIONS.items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            pytest.skip("install requirements-minigrid.txt for actual environment checks")
        if installed != version:
            pytest.skip(f"actual environment check requires {name}=={version}")


def layout(context):
    return next(item for item in predefined_layouts() if item.context == context)


def batch(episode, *, role=None):
    features, targets = episode_arrays(episode)
    return AttributionBatch(
        episode.role if role is None else role,
        episode.context,
        tuple(f"{episode.episode_id}:step:{row.step}" for row in episode.transitions),
        features,
        targets,
        episode,
        7,
    )


def test_should_freeze_declared_layouts_without_tuning_or_reordering():
    declared = predefined_layouts()
    assert type(declared) is tuple
    assert [
        (item.context, item.rotation, item.branch_row, item.swap_keys, item.wall_color)
        for item in declared
    ] == [
        ("A", 0, 3, False, "grey"),
        ("B", 1, 4, True, "blue"),
        ("C_train", 2, 3, True, "red"),
        ("D", 3, 4, False, "yellow"),
    ]
    with pytest.raises(FrozenInstanceError):
        setattr(declared[0], "rotation", 2)
    assert MISSION == "remember the cue, take its key, unlock the neutral door and reach the goal"


@pytest.mark.parametrize(
    "changes",
    [
        {"context": "E"},
        {"context": " A"},
        {"rotation": True},
        {"rotation": -1},
        {"rotation": 4},
        {"rotation": 1.0},
        {"branch_row": 2},
        {"branch_row": True},
        {"swap_keys": 1},
        {"wall_color": "green"},
    ],
)
def test_should_refuse_undeclared_layout_values(changes):
    with pytest.raises(ValueError):
        replace(MiniGridKeyDoorLayout("A"), **changes)


def test_should_encode_fixed_key_door_mission_without_relabeling_memory_contract():
    image = np.zeros((7, 7, 3), dtype=np.uint8)
    public = dict(image=image, direction=3, mission=MISSION)
    encoded = encode_key_door_observation(public)
    assert type(encoded) is tuple and len(encoded) == 148
    assert encoded == (-1.0,) * 147 + (1.0,)
    with pytest.raises(ValueError, match="mission"):
        encode_memory_observation(public)
    public["mission"] = "go to the matching object at the end of the hallway"
    with pytest.raises(ValueError, match="mission"):
        encode_key_door_observation(public)


@pytest.mark.parametrize(
    "cue,left,right,turn", [(2, 2, 0, 0), (0, 2, 0, 1), (2, 0, 2, 1), (0, 0, 2, 0)]
)
def test_should_remember_visible_cue_then_choose_matching_key_after_cue_disappears(
    cue, left, right, turn
):
    teacher = CueKeyDoorTeacher()
    assert teacher.choose_action(public_frame(cue=cue)) == 2
    assert teacher.choose_action(public_frame(left=left, right=right)) == turn


@pytest.mark.parametrize(
    "fault", ["ambiguous_keys", "wrong_front_key", "missing_held_key", "premature_door"]
)
def test_should_refuse_public_evidence_that_breaks_scripted_key_collection(fault):
    teacher = CueKeyDoorTeacher()
    teacher.choose_action(public_frame(cue=2))
    invalid: Any
    if fault == "ambiguous_keys":
        invalid = public_frame(left=2, right=2)
    elif fault == "premature_door":
        invalid = public_frame(door=2)
    else:
        assert teacher.choose_action(public_frame(left=2, right=0)) == 0
        if fault == "wrong_front_key":
            invalid = list(public_frame())
            offset = (3 * 7 + 5) * 3
            invalid[offset : offset + 3] = [0.0, -1.0, -1.0]
            invalid = tuple(invalid)
        else:
            front = list(public_frame())
            offset = (3 * 7 + 5) * 3
            front[offset : offset + 3] = [0.0, -0.2, -1.0]
            assert teacher.choose_action(tuple(front)) == 3
            invalid = public_frame()
    with pytest.raises(ValueError):
        teacher.choose_action(invalid)


@pytest.mark.parametrize("fault", ["missing", "ambiguous", "unsupported"])
def test_should_refuse_fresh_teacher_without_one_valid_public_cue(fault):
    frame = list(public_frame(cue=1 if fault == "unsupported" else 2))
    if fault == "missing":
        frame = list(public_frame())
    elif fault == "ambiguous":
        offset = (5 * 7 + 5) * 3
        frame[offset : offset + 3] = [0.2, -1.0, -1.0]  # Second red ball.
    with pytest.raises(ValueError):
        CueKeyDoorTeacher().choose_action(tuple(frame))


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "short",
        "int_scalar",
        "bool_scalar",
        "numpy_scalar",
        "nan",
        "outside",
        "object_lattice",
        "direction_lattice",
    ],
)
def test_should_refuse_malformed_normalized_teacher_inputs(fault):
    values: Any = list(public_frame(cue=2))
    if fault == "list":
        pass
    elif fault == "short":
        values = tuple(values[:-1])
    else:
        index = 147 if fault == "direction_lattice" else 0
        values[index] = {
            "int_scalar": -1,
            "bool_scalar": False,
            "numpy_scalar": np.float64(values[0]),
            "nan": float("nan"),
            "outside": 1.1,
            "object_lattice": 0.123,
            "direction_lattice": 0.5,
        }[fault]
        values = tuple(values)
    with pytest.raises(ValueError):
        CueKeyDoorTeacher().choose_action(values)


@pytest.mark.parametrize("context", ["A", "B", "C_train", "D"])
@pytest.mark.parametrize("cue", ["blue", "red"])
def test_should_finish_actual_rotated_swapped_game_using_only_public_teacher_inputs(context, cue):
    # Eight independent episodes consume at most128 actual steps in total.
    actual_dependencies()
    specification = layout(context)
    role = "final_test" if context == "D" else "train"
    teacher = CueKeyDoorTeacher()
    with MiniGridKeyDoor(
        specification, cue, seed=17, role=role, episode_id=f"{context}-{cue}"
    ) as game:
        initial = game.report["initial_observation"]
        image = np.array(initial["image"])
        assert initial["mission"] == MISSION
        assert initial["direction"] == (3 + specification.rotation) % 4
        assert np.count_nonzero(image[:, :, 0] == 6) == 1
        assert image[:, :, 1][image[:, :, 0] == 6].tolist() == [2 if cue == "blue" else 0]
        door_colors = image[:, :, 1][image[:, :, 0] == 4]
        assert len(door_colors) >= 1 and set(door_colors.tolist()) == {5}
        wall_colors = image[:, :, 1][image[:, :, 0] == 2]
        assert len(wall_colors) >= 1
        expected_wall = {"grey": 5, "blue": 2, "red": 0, "yellow": 4}[specification.wall_color]
        assert set(wall_colors.tolist()) == {expected_wall}
        key_colors = image[:, :, 1][image[:, :, 0] == 5]
        assert sorted(key_colors.tolist()) == [0, 2]
        positions = sorted(np.argwhere(image[:, :, 0] == 5).tolist())
        assert [int(image[x, y, 1]) for x, y in positions] == (
            [0, 2] if specification.swap_keys else [2, 0]
        )

        for _ in range(16):
            if game.terminal:
                break
            before = game.observation
            action = teacher.choose_action(before)
            assert type(action) is int and action in range(7)
            returned = game.step(action)
            assert game.report["transitions"][-1]["observation"] == before
            next_image = np.array(game.report["transitions"][-1]["raw_next_observation"]["image"])
            assert not np.any(next_image[:, :, 0] == 6)
            assert returned["observation"] == game.observation

        report = game.report
        assert report["status"] == "complete" and report["terminated"] is True
        assert report["truncated"] is False and report["total_return"] > 0
        assert (
            report["environment_steps"]
            == report["environment_steps_attempted"]
            == len(report["transitions"])
        )
        assert 1 <= len(report["transitions"]) <= 16
        assert any(row["action"] == 3 for row in report["transitions"])
        assert any(row["action"] == 5 for row in report["transitions"])
        assert game.legal_actions == ()
        episode = game.to_demonstration()
        assert episode.role == role and episode.context == context
        _validate_batch(batch(episode), role, set())
        schema = EpisodeReplaySchema(148, len(episode.transitions), 7)
        if role == "train":
            encoded, dummy = encode_episode(episode, schema)
            assert decode_episode(encoded, dummy, schema) == episode
        else:
            with pytest.raises(ValueError, match="training"):
                encode_episode(episode, schema)
            with pytest.raises(ValueError, match="role"):
                _validate_batch(batch(episode), "train", set())
            with pytest.raises(ValueError, match="must agree"):
                _validate_batch(batch(episode, role="train"), "train", set())


@pytest.mark.parametrize("action", list(range(7)))
def test_should_remove_initial_cue_on_every_first_native_action_and_require_history(action):
    # Two one-step episodes per action:14 actual steps, including ineffective actions.
    actual_dependencies()
    with MiniGridKeyDoor(layout("A"), "blue", seed=17) as blue:
        with MiniGridKeyDoor(layout("A"), "red", seed=17) as red:
            assert blue.observation != red.observation
            blue_result = blue.step(action)
            red_result = red.step(action)
            assert blue_result == red_result
            assert blue.observation == red.observation
            for game in (blue, red):
                report = game.report
                assert report["environment_steps"] == report["environment_steps_attempted"] == 1
                assert report["transitions"][0]["action"] == action
                image = np.array(report["transitions"][0]["raw_next_observation"]["image"])
                assert not np.any(image[:, :, 0] == 6)


def test_should_fail_actual_wrong_key_toggle_without_repair_or_success_relabeling():
    actual_dependencies()
    teacher = CueKeyDoorTeacher()
    with MiniGridKeyDoor(layout("A"), "blue", seed=17) as game:
        # Deliberately present the opposite cue to select the causal wrong-key path.
        synthetic = list(game.observation)
        image = np.array(game.report["initial_observation"]["image"])
        positions = np.argwhere(image[:, :, 0] == 6)
        assert len(positions) == 1
        x, y = (int(value) for value in positions[0])
        synthetic[(x * 7 + y) * 3 + 1] = -1.0  # Red, while the actual target is blue.
        action = teacher.choose_action(tuple(synthetic))
        for _ in range(16):
            game.step(action)
            if game.terminal:
                break
            action = teacher.choose_action(game.observation)
        report = game.report
        assert report["terminated"] is True and report["truncated"] is False
        assert report["total_return"] == -1.0
        assert report["transitions"][-1]["action"] == 5
        assert report["transitions"][-1]["reward"] == -1.0
        episode = game.to_demonstration()
        assert episode.transitions[-1].reward == -1.0
        assert len(episode.transitions) == report["environment_steps"] <= 16


def test_should_preserve_actual_timeout_and_refuse_terminal_demonstration():
    actual_dependencies()
    with MiniGridKeyDoor(layout("A"), "blue", seed=17, max_steps=1) as game:
        returned = game.step(0)
        assert returned["terminated"] is False and returned["truncated"] is True
        assert game.terminal and game.legal_actions == ()
        before = deepcopy(game.report)
        assert before["environment_steps"] == 1
        with pytest.raises(ValueError):
            game.to_demonstration()
        with pytest.raises(ValueError):
            game.step(0)
        assert game.report == before


@pytest.mark.parametrize("role", ["train", "probe_train", "probe_test"])
def test_should_refuse_heldout_layout_outside_final_role_before_environment(monkeypatch, role):
    from src.infra import minigrid_key_door as boundary

    def forbidden(*args, **kwargs):
        raise AssertionError("heldout role constructed an environment")

    monkeypatch.setattr(boundary, "_make_key_door_environment", forbidden)
    with pytest.raises(ValueError):
        MiniGridKeyDoor(layout("D"), "blue", role=role)


@pytest.mark.parametrize("cue", ["green", "Blue", "", True])
def test_should_refuse_invalid_explicit_cue_before_environment(monkeypatch, cue):
    from src.infra import minigrid_key_door as boundary

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid cue constructed an environment")

    monkeypatch.setattr(boundary, "_make_key_door_environment", forbidden)
    with pytest.raises(ValueError):
        MiniGridKeyDoor(layout("A"), cue)
