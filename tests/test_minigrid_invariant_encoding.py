"""Public nuisance invariance and causal episode encoding without game execution."""

from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    episode_arrays,
)
from src.core.minigrid_invariant_encoding import (
    INVARIANT_HISTORY_ENCODING_ID,
    InvariantInitialFrameHistory,
    canonicalize_public_frame,
    canonicalize_public_history,
    encode_invariant_history_episode,
)
from src.core.minigrid_public_history import PublicInitialFrameHistory


def frame(wall_color=5, direction=3, key_color=0, cue_color=0, door_state=2):
    """Explicit public integers: wall0, cue1, key2, door3, goal4, then floor.

    This reusable array fixture constructs no environment and encodes no target.
    """
    pixels = [
        (2, wall_color, 0),
        (6, cue_color, 0),
        (5, key_color, 0),
        (4, 5, door_state),
        (8, 1, 0),
    ] + [(3, 0, 0)] * 44
    values = [
        2 * value / scale - 1
        for pixel in pixels
        for value, scale in zip(pixel, (10, 5, 2), strict=True)
    ]
    return (*values, 2 * direction / 3 - 1)


def change_pixel(original, position, pixel):
    values = list(original)
    values[3 * position : 3 * position + 3] = [
        2 * value / scale - 1 for value, scale in zip(pixel, (10, 5, 2), strict=True)
    ]
    return tuple(values)


def episode(role="train"):
    frames = (
        frame(wall_color=0),
        frame(wall_color=2, direction=1, key_color=2),
        frame(wall_color=4, direction=2, key_color=2, door_state=0),
    )
    return DemonstrationEpisode(
        "engineering-invariant-A",
        "A",
        role,
        (
            DemonstrationTransition(0, frames[0], 6, -0.25, frames[1], False, 7),
            DemonstrationTransition(1, frames[1], 2, 0.5, frames[2], True, 7),
        ),
        7,
    )


def test_should_remove_only_wall_color_and_absolute_compass_for_all_declared_values():
    expected = frame(wall_color=5, direction=0)
    for wall_color in range(6):
        for direction in range(4):
            raw = frame(wall_color=wall_color, direction=direction)
            actual = canonicalize_public_frame(raw)
            assert type(actual) is tuple and len(actual) == 148
            assert all(type(value) is float for value in actual)
            assert actual == expected
            assert canonicalize_public_frame(actual) == actual
            assert raw[1] == 2 * wall_color / 5 - 1 and raw[-1] == 2 * direction / 3 - 1


def test_should_preserve_cue_key_door_goal_nonwall_state_and_spatial_discrimination():
    raw = frame(wall_color=1, direction=2)
    canonical = canonicalize_public_frame(raw)
    assert all(canonical[index] == raw[index] for index in range(148) if index not in (1, 147))
    variants = [
        frame(key_color=2),
        frame(cue_color=2),
        frame(door_state=0),
        frame(door_state=1),
        change_pixel(frame(), 4, (8, 4, 0)),  # Preserve goal color.
        change_pixel(frame(), 4, (3, 1, 0)),  # Preserve the goal object's presence.
        change_pixel(frame(), 5, (3, 0, 2)),  # Even non-wall state is retained.
    ]
    original = canonicalize_public_frame(frame())
    assert all(canonicalize_public_frame(value) != original for value in variants)
    swapped = list(frame())
    swapped[0:3], swapped[15:18] = swapped[15:18], swapped[0:3]
    projected = canonicalize_public_frame(tuple(swapped))
    assert projected != original
    assert projected[0:3] == original[15:18] and projected[15:18] == original[0:3]


def test_should_snap_only_accepted_integer_roundoff_then_reencode_deterministically():
    exact = frame(wall_color=1, direction=1)
    near = list(exact)
    near[3] += 1e-9  # Object-grid error is 5e-9, below the decoded tolerance.
    near[4] += 1e-9
    near[5] += 1e-9
    near[-1] += 1e-9
    assert canonicalize_public_frame(tuple(near)) == canonicalize_public_frame(exact)
    assert canonicalize_public_frame((-1,) * 148) == (-1.0,) * 148
    beyond = list(exact)
    beyond[3] += 4e-9  # Decoded object error is 2e-8.
    with pytest.raises(ValueError):
        canonicalize_public_frame(tuple(beyond))


def test_should_canonicalize_both_history_halves_independently_and_refuse_mixed_widths():
    initial, current = (
        frame(wall_color=0, cue_color=2),
        frame(wall_color=3, direction=1, key_color=2),
    )
    expected = canonicalize_public_frame(initial) + canonicalize_public_frame(current)
    assert canonicalize_public_history(initial + current) == expected
    assert canonicalize_public_history(expected) == expected
    assert expected[:148] != expected[148:]
    with pytest.raises(ValueError):
        canonicalize_public_frame(initial + current)
    with pytest.raises(ValueError):
        canonicalize_public_history(initial)


def test_should_keep_only_owned_current_episode_initial_frame_and_clear_it_explicitly():
    history = InvariantInitialFrameHistory()
    assert (
        history.encoding_id
        == INVARIANT_HISTORY_ENCODING_ID
        == "minigrid_public_wall_direction_invariant_history_296_v1"
    )
    with pytest.raises(AttributeError):
        setattr(history, "encoding_id", "raw")
    initial, current, replacement = frame(wall_color=0), frame(key_color=2), frame(cue_color=2)
    with pytest.raises(ValueError):
        history.encode(current)
    first = canonicalize_public_frame(initial)
    assert history.reset(initial) == first + first
    assert history.encode(current) == first + canonicalize_public_frame(current)
    assert history.encode(replacement) == first + canonicalize_public_frame(replacement)
    second = canonicalize_public_frame(replacement)
    assert history.reset(replacement) == second + second
    assert history.encode(current) == second + canonicalize_public_frame(current)
    history.clear()
    with pytest.raises(ValueError):
        history.encode(current)


def test_should_keep_initial_on_rejected_current_but_clear_it_on_rejected_reset():
    history = InvariantInitialFrameHistory()
    initial = frame(cue_color=2)
    history.reset(initial)
    bad: Any = list(frame())
    with pytest.raises(ValueError):
        history.encode(bad)
    assert history.encode(frame()) == canonicalize_public_frame(
        initial
    ) + canonicalize_public_frame(frame())
    bad = list(frame())
    bad[3] += 0.01
    with pytest.raises(ValueError):
        history.reset(tuple(bad))
    with pytest.raises(ValueError):
        history.encode(frame())


class TupleSubclass(tuple):
    pass


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "tuple_subclass",
        "short",
        "double",
        "nan",
        "inf",
        "range",
        "bool",
        "numpy",
        "object_grid",
        "color_grid",
        "state_grid",
        "direction_grid",
    ],
)
def test_should_refuse_malformed_or_nondiscrete_public_frames_without_coercion(fault):
    value: Any = frame()
    if fault == "list":
        value = list(value)
    elif fault == "tuple_subclass":
        value = TupleSubclass(value)
    elif fault == "short":
        value = value[:-1]
    elif fault == "double":
        value = value + value
    else:
        changed = list(value)
        if fault in ("nan", "inf", "range", "bool", "numpy"):
            changed[9] = {
                "nan": float("nan"),
                "inf": float("inf"),
                "range": 1.01,
                "bool": False,
                "numpy": np.float64(value[9]),
            }[fault]
        else:
            changed[
                {"object_grid": 3, "color_grid": 4, "state_grid": 5, "direction_grid": 147}[fault]
            ] += 0.01
        value = tuple(changed)
    with pytest.raises(ValueError):
        canonicalize_public_frame(value)


def test_should_validate_the_whole_history_envelope_including_the_later_frame():
    valid = frame(wall_color=0) + frame(wall_color=1, direction=1)
    late_bool = list(valid)
    late_bool[148 + 9] = True
    late_nondiscrete = list(valid)
    late_nondiscrete[148 + 3] += 0.01
    invalid: Any
    for invalid in (list(valid), TupleSubclass(valid), tuple(late_bool), tuple(late_nondiscrete)):
        before = pickle.dumps(invalid)
        with pytest.raises(ValueError):
            canonicalize_public_history(invalid)
        assert pickle.dumps(invalid) == before


@pytest.mark.parametrize("role", ["train", "final_test", "probe_train", "probe_test"])
def test_should_detach_episode_encoding_preserving_all_roles_actions_rewards_and_metadata(role):
    original = episode(role)
    before = pickle.dumps(original)
    encoded = encode_invariant_history_episode(original)
    assert encoded is not original and encoded.transitions is not original.transitions
    assert (encoded.episode_id, encoded.context, encoded.role, encoded.num_classes) == (
        original.episode_id,
        "A",
        role,
        7,
    )
    first = canonicalize_public_frame(original.transitions[0].observation)
    for old, new in zip(original.transitions, encoded.transitions, strict=True):
        assert new is not old
        assert new.observation == first + canonicalize_public_frame(old.observation)
        assert new.next_observation == first + canonicalize_public_frame(old.next_observation)
        assert (new.step, new.action, new.reward, new.terminal, new.num_classes) == (
            old.step,
            old.action,
            old.reward,
            old.terminal,
            old.num_classes,
        )
    x, y = episode_arrays(encoded)
    assert x.shape == (2, 296) and y.shape == (2, 1)
    np.testing.assert_array_equal(y[:, 0], [6.0, 2.0])
    x.fill(0)
    y.fill(0)
    assert pickle.dumps(original) == before
    object.__setattr__(original.transitions[0], "reward", 1.0)
    assert encoded.transitions[0].reward == -0.25


def test_should_not_use_metadata_actions_or_future_terminal_frames_for_current_features():
    original = episode()
    changed = replace(
        original,
        episode_id="different-provenance",
        context="another-context",
        role="probe_test",
        transitions=(
            replace(original.transitions[0], action=1, reward=0.0),
            replace(
                original.transitions[1],
                action=6,
                reward=-1.0,
                next_observation=frame(key_color=4, cue_color=4),
            ),
        ),
    )
    left, right = map(encode_invariant_history_episode, (original, changed))
    assert [step.observation for step in left.transitions] == [
        step.observation for step in right.transitions
    ]
    assert left.transitions[0].next_observation == right.transitions[0].next_observation
    assert left.transitions[-1].next_observation != right.transitions[-1].next_observation
    raw = PublicInitialFrameHistory()
    assert raw.reset(frame(wall_color=0)) == frame(wall_color=0) * 2
    assert raw.encode(frame(wall_color=1, direction=1)) == frame(wall_color=0) + frame(
        wall_color=1, direction=1
    )


@pytest.mark.parametrize("fault", ["double", "discontinuous", "nondiscrete", "wrong_type"])
def test_should_refuse_previously_encoded_or_corrupted_complete_episodes(fault):
    value: Any = episode()
    if fault == "double":
        value = encode_invariant_history_episode(value)
    elif fault == "discontinuous":
        object.__setattr__(value.transitions[1], "observation", frame(cue_color=2))
    elif fault == "nondiscrete":
        bad = list(value.transitions[-1].next_observation)
        bad[3] += 0.01
        object.__setattr__(value.transitions[-1], "next_observation", tuple(bad))
    else:
        value = object()
    with pytest.raises(ValueError):
        encode_invariant_history_episode(value)
