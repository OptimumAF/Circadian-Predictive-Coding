"""Public cue-relative color semantics, invertibility and causal episode state.

Fixtures are explicit normalized public cells. No model, game, teacher, target
selection or physics is needed to check this raw-input representation contract.
"""

from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.core.minigrid_cue_relative_encoding import (
    CUE_RELATIVE_HISTORY_ENCODING_ID,
    CueRelativeInitialFrameHistory,
    cue_relative_public_history,
    encode_cue_relative_history_episode,
)
from src.core.minigrid_invariant_encoding import canonicalize_public_history
from test_minigrid_invariant_encoding import change_pixel, episode, frame


def rich_frame(cue=0, key=0, *, wall=1, direction=2):
    value = frame(wall_color=wall, direction=direction, cue_color=cue, key_color=key, door_state=1)
    value = change_pixel(value, 5, (5, 2 - key, 2))
    value = change_pixel(value, 6, (8, 4, 1))
    return value


def pixel(value, position, *, half=0):
    offset = half * 148 + 3 * position
    return tuple(round((value[offset + i] + 1) * scale / 2) for i, scale in enumerate((10, 5, 2)))


@pytest.mark.parametrize("cue", [0, 2])
@pytest.mark.parametrize("key", [0, 2])
def test_should_remap_only_key_colors_using_the_original_visible_initial_cue(cue, key):
    initial = rich_frame(cue, key)
    current = rich_frame(2 - cue, 2 - key, wall=3, direction=1)
    raw = initial + current
    before = pickle.dumps(raw)
    actual = cue_relative_public_history(raw)
    canonical = canonicalize_public_history(raw)
    assert type(actual) is tuple and len(actual) == 296
    assert all(type(value) is float for value in actual)
    assert pixel(actual, 1) == (6, cue, 0)
    assert pixel(actual, 1, half=1) == (6, 2 - cue, 0)
    assert pixel(actual, 2) == (5, 2 if key == cue else 0, 0)
    assert pixel(actual, 5) == (5, 2 if 2 - key == cue else 0, 2)
    assert pixel(actual, 2, half=1) == (5, 2 if 2 - key == cue else 0, 0)
    assert pixel(actual, 5, half=1) == (5, 2 if key == cue else 0, 2)
    allowed_colors = {7, 16, 148 + 7, 148 + 16}
    assert all(actual[i] == canonical[i] for i in range(296) if i not in allowed_colors)
    assert pixel(actual, 3) == pixel(actual, 3, half=1) == (4, 5, 1)
    assert pixel(actual, 6) == pixel(actual, 6, half=1) == (8, 4, 1)
    assert pickle.dumps(raw) == before


@pytest.mark.parametrize("cue", [0, 2])
def test_should_preserve_old_canonical_information_and_document_exactly_once_input(cue):
    raw = rich_frame(cue, 0) + rich_frame(2 - cue, 2)
    canonical = canonicalize_public_history(raw)
    relative = cue_relative_public_history(raw)
    assert cue_relative_public_history(relative) == canonical
    if cue == 0:
        assert relative != canonical and cue_relative_public_history(relative) != relative
    else:
        assert relative == canonical
    assert pixel(relative, 1)[1] == cue


@pytest.mark.parametrize("cue", [0, 2])
def test_should_remove_original_wall_and_compass_nuisances_with_no_other_loss(cue):
    expected = cue_relative_public_history(rich_frame(cue, 0) + rich_frame(cue, 2))
    for wall in range(6):
        for direction in range(4):
            raw = rich_frame(cue, 0, wall=wall, direction=direction) + rich_frame(
                cue, 2, wall=5 - wall, direction=3 - direction
            )
            assert cue_relative_public_history(raw) == expected


@pytest.mark.parametrize(
    "current",
    [
        change_pixel(rich_frame(0, 0), 1, (3, 0, 0)),
        change_pixel(rich_frame(2, 0), 7, (6, 0, 2)),
        change_pixel(rich_frame(0, 0), 1, (6, 1, 1)),
    ],
)
def test_should_never_derive_or_replace_the_initial_cue_from_current_balls(current):
    result = cue_relative_public_history(rich_frame(2, 2) + current)
    assert pixel(result, 2, half=1) == (5, 0, 0)
    assert result[148:] == canonicalize_public_history(rich_frame(2, 2) + current)[148:]


@pytest.mark.parametrize(
    "initial",
    [
        change_pixel(rich_frame(), 1, (3, 0, 0)),
        change_pixel(rich_frame(), 7, (6, 0, 0)),
        change_pixel(rich_frame(), 7, (6, 2, 0)),
        change_pixel(rich_frame(), 1, (6, 1, 0)),
        change_pixel(rich_frame(), 1, (6, 5, 0)),
    ],
)
def test_should_refuse_missing_ambiguous_or_unsupported_initial_cues(initial):
    with pytest.raises(ValueError):
        cue_relative_public_history(initial + rich_frame())


@pytest.mark.parametrize("half", [0, 1])
@pytest.mark.parametrize("color", [1, 3, 4, 5])
def test_should_refuse_every_unsupported_key_color_in_either_full_frame(half, color):
    frames = [rich_frame(), rich_frame()]
    frames[half] = change_pixel(frames[half], 48, (5, color, 0))
    with pytest.raises(ValueError):
        cue_relative_public_history(frames[0] + frames[1])


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "array",
        "short",
        "frame",
        "bool",
        "numpy",
        "nan",
        "inf",
        "range",
        "late_object_grid",
        "late_color_grid",
        "late_state_grid",
        "late_direction",
    ],
)
def test_should_validate_all_raw_coordinates_and_exact_input_types(fault):
    value: Any = rich_frame() + rich_frame()
    if fault == "list":
        value = list(value)
    elif fault == "array":
        value = np.array(value)
    elif fault == "short":
        value = value[:-1]
    elif fault == "frame":
        value = value[:148]
    else:
        changed = list(value)
        if fault in ("bool", "numpy", "nan", "inf", "range"):
            changed[-3] = {
                "bool": True,
                "numpy": np.float64(value[-3]),
                "nan": float("nan"),
                "inf": float("inf"),
                "range": 1.01,
            }[fault]
        else:
            position = {
                "late_object_grid": 148 + 144,
                "late_color_grid": 148 + 145,
                "late_state_grid": 148 + 146,
                "late_direction": 295,
            }[fault]
            changed[position] += 0.01
        value = tuple(changed)
    with pytest.raises(ValueError):
        cue_relative_public_history(value)


def test_should_reuse_pre_relative_initial_state_without_double_applying_red_cue():
    history = CueRelativeInitialFrameHistory()
    initial, current = rich_frame(0, 0), rich_frame(2, 2)
    assert history.encoding_id == CUE_RELATIVE_HISTORY_ENCODING_ID
    assert (
        CUE_RELATIVE_HISTORY_ENCODING_ID == "minigrid_public_cue_relative_key_color_history_296_v1"
    )
    with pytest.raises(ValueError):
        history.encode(current)
    assert history.reset(initial) == cue_relative_public_history(initial + initial)
    expected = cue_relative_public_history(initial + current)
    assert history.encode(current) == history.encode(current) == expected
    assert history.encode(initial) == cue_relative_public_history(initial + initial)
    replacement = rich_frame(2, 2)
    history.reset(replacement)
    assert history.encode(current) == cue_relative_public_history(replacement + current)
    history.clear()
    with pytest.raises(ValueError):
        history.encode(current)
    with pytest.raises(AttributeError):
        setattr(history, "encoding_id", "foreign")


def test_should_keep_initial_after_bad_current_and_clear_it_before_bad_reset():
    history = CueRelativeInitialFrameHistory()
    initial = rich_frame(0, 0)
    history.reset(initial)
    invalid = change_pixel(rich_frame(), 48, (5, 1, 0))
    with pytest.raises(ValueError):
        history.encode(invalid)
    assert history.encode(initial) == cue_relative_public_history(initial + initial)
    with pytest.raises(ValueError):
        history.reset(change_pixel(initial, 1, (3, 0, 0)))
    with pytest.raises(ValueError):
        history.encode(initial)


@pytest.mark.parametrize("role", ["train", "probe_train", "probe_test", "final_test"])
def test_should_encode_complete_causal_episodes_preserving_every_original_metadata_field(role):
    source = episode(role)
    before = pickle.dumps(source)
    encoded = encode_cue_relative_history_episode(source)
    initial = source.transitions[0].observation
    assert encoded is not source
    assert (encoded.episode_id, encoded.context, encoded.role, encoded.num_classes) == (
        source.episode_id,
        source.context,
        source.role,
        source.num_classes,
    )
    for original, actual in zip(source.transitions, encoded.transitions, strict=True):
        assert actual is not original
        for field in ("step", "action", "reward", "terminal", "num_classes"):
            assert getattr(actual, field) == getattr(original, field)
        assert actual.observation == cue_relative_public_history(initial + original.observation)
        assert actual.next_observation == cue_relative_public_history(
            initial + original.next_observation
        )
    assert encoded.transitions[0].next_observation == encoded.transitions[1].observation
    assert pickle.dumps(source) == before


def test_should_keep_current_encoding_independent_of_labels_roles_context_and_future():
    source = episode()
    last = source.transitions[-1]
    changed = replace(
        source,
        episode_id="different",
        context="foreign",
        role="probe_test",
        transitions=tuple(
            replace(step, action=(step.action + 1) % 7, reward=0.0)
            for step in source.transitions[:-1]
        )
        + (replace(last, action=0, reward=-1.0, next_observation=rich_frame(2, 2)),),
    )
    first = encode_cue_relative_history_episode(source)
    second = encode_cue_relative_history_episode(changed)
    assert [step.observation for step in first.transitions] == [
        step.observation for step in second.transitions
    ]
    assert first.transitions[-1].next_observation != second.transitions[-1].next_observation


@pytest.mark.parametrize("fault", ["discontinuous", "next_invalid", "missing_cue", "already296"])
def test_should_refuse_invalid_original_or_later_episode_evidence(fault):
    source = episode()
    if fault == "discontinuous":
        object.__setattr__(source.transitions[0], "next_observation", rich_frame(2, 2))
    elif fault == "next_invalid":
        object.__setattr__(
            source.transitions[-1],
            "next_observation",
            (float("nan"), *source.transitions[-1].next_observation[1:]),
        )
    elif fault == "missing_cue":
        frames = tuple(change_pixel(step.observation, 1, (3, 0, 0)) for step in source.transitions)
        first = source.transitions[0]
        object.__setattr__(first, "observation", frames[0])
    else:
        source = encode_cue_relative_history_episode(source)
    with pytest.raises(ValueError):
        encode_cue_relative_history_episode(source)
