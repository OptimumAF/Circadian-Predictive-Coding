"""Fixed F source configuration/role boundaries without native physics."""

from dataclasses import replace
import pytest
from src.core.minigrid_key_door import MiniGridKeyDoorLayout, predefined_distinct_probe_layouts
from src.infra import minigrid_key_door as boundary


def forbidden(*args, **kwargs):
    raise AssertionError("source construction must not happen")


def test_should_declare_the_public_key_position_task_without_changing_native_actions():
    layouts = predefined_distinct_probe_layouts()
    assert [(v.context, v.rotation, v.branch_row, v.swap_keys, v.wall_color) for v in layouts] == [
        ("F-blue", 0, 2, False, "grey"),
        ("F-red", 0, 2, False, "grey"),
    ]
    assert predefined_distinct_probe_layouts() == layouts
    assert predefined_distinct_probe_layouts()[0] is not layouts[0]
    with pytest.raises(ValueError):
        MiniGridKeyDoorLayout("A", branch_row=2)


@pytest.mark.parametrize(
    "change",
    [
        dict(branch_row=3),
        dict(branch_row=4),
        dict(branch_row=True),
        dict(branch_row=2.0),
        dict(rotation=1),
        dict(swap_keys=True),
        dict(wall_color="blue"),
    ],
)
def test_should_refuse_mutations_of_the_fixed_f_geometry(change):
    with pytest.raises(ValueError):
        replace(predefined_distinct_probe_layouts()[0], **change)


@pytest.mark.parametrize("cue", ["blue", "red"])
@pytest.mark.parametrize("role", ["train", "validation", "final_test", "probe_train ", None])
def test_should_refuse_nonprobe_f_roles_before_any_environment(cue, role, monkeypatch):
    monkeypatch.setattr(boundary, "_make_key_door_environment", forbidden)
    layout = predefined_distinct_probe_layouts()[cue == "red"]
    with pytest.raises(ValueError):
        boundary.MiniGridKeyDoor(layout, cue, role=role)


@pytest.mark.parametrize("cue", ["blue", "red"])
def test_should_refuse_f_cue_context_mismatch_before_any_environment(cue, monkeypatch):
    monkeypatch.setattr(boundary, "_make_key_door_environment", forbidden)
    layout = predefined_distinct_probe_layouts()[cue == "blue"]
    with pytest.raises(ValueError):
        boundary.MiniGridKeyDoor(layout, cue, role="probe_train")


@pytest.mark.parametrize("cue", ["blue", "red"])
@pytest.mark.parametrize("role", ["probe_train", "probe_test"])
def test_should_bind_f_configuration_to_the_original_owned_episode_boundary(cue, role, monkeypatch):
    calls = []

    def init(self, *args, **kwargs):
        calls.append((args, kwargs))
        self._report = {}

    monkeypatch.setattr(boundary._MiniGridEpisode, "__init__", init)
    monkeypatch.setattr(boundary, "_make_key_door_environment", forbidden)
    layout = predefined_distinct_probe_layouts()[cue == "red"]
    value = boundary.MiniGridKeyDoor(
        layout, cue, seed=17, max_steps=16, role=role, episode_id="fixed"
    )
    assert len(calls) == 1 and calls[0][0] == (17, 16)
    assert calls[0][1]["role"] == role and calls[0][1]["context"] == "F-" + cue
    assert value._report["layout"]["branch_row"] == 2 and value._report["cue"] == cue


def test_should_keep_final_context_unavailable_to_probe_collection(monkeypatch):
    monkeypatch.setattr(boundary, "_make_key_door_environment", forbidden)
    with pytest.raises(ValueError, match="final_test"):
        boundary.MiniGridKeyDoor(MiniGridKeyDoorLayout("D"), "blue", role="probe_train")
