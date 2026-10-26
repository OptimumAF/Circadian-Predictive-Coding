"""Pure public-history causality, ownership, admission and complete-record checks."""

from dataclasses import replace
import pickle

import numpy as np
import pytest

from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    episode_arrays,
)
from src.core.key_door_navigation import KeyDoorLayout, KeyDoorNavigation, scripted_actions
from src.core.key_door_public_history import (
    PublicInitialFrameHistory,
    encode_public_history_episode,
)


def frame(step=0):
    return tuple(float((column + step) % 5 - 2) / 2 for column in range(8))


def episode(role="train"):
    return DemonstrationEpisode(
        "history-local-A",
        "A",
        role,
        tuple(
            DemonstrationTransition(
                index,
                frame(index),
                index % 5,
                0.5 if index == 2 else -0.25,
                frame(index + 1),
                index == 2,
                5,
            )
            for index in range(3)
        ),
        5,
    )


def test_should_keep_only_initial_and_current_frames_and_reset_each_episode():
    history = PublicInitialFrameHistory()
    with pytest.raises(ValueError, match="reset"):
        history.encode(frame())
    assert history.reset(frame()) == frame() + frame()
    assert history.encode(frame(1)) == frame() + frame(1)
    assert history.encode(frame(2)) == frame() + frame(2)
    # Prior current frames never advance or accumulate the retained history.
    assert history.encode(frame(1)) == frame() + frame(1)
    assert history.reset(frame(3)) == frame(3) + frame(3)
    assert history.encode(frame(2)) == frame(3) + frame(2)
    history.clear()
    history.clear()
    with pytest.raises(ValueError, match="reset"):
        history.encode(frame())


def test_should_own_frame_tuple_and_normalize_exact_native_integer_coordinates():
    history = PublicInitialFrameHistory()
    initial = (0, 1, -1, 0, 1, -1, 0, 1)
    encoded = history.reset(initial)
    assert encoded == initial + initial
    assert type(encoded) is tuple and encoded is not initial and len(encoded) == 16
    assert all(type(value) is float for value in encoded)
    assert history.encode(frame()) == tuple(float(value) for value in initial) + frame()


class TupleSubclass(tuple):
    pass


class FloatSubclass(float):
    pass


class IntSubclass(int):
    pass


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        [],
        [0.0] * 8,
        (),
        (0.0,) * 7,
        (0.0,) * 9,
        (0.0,) * 16,
        (0.0,) * 148,
        TupleSubclass((0.0,) * 8),
        (float("nan"),) + (0.0,) * 7,
        (float("inf"),) + (0.0,) * 7,
        (-float("inf"),) + (0.0,) * 7,
        (1.01,) + (0.0,) * 7,
        (-1.01,) + (0.0,) * 7,
        (True,) + (0.0,) * 7,
        (False,) + (0.0,) * 7,
        (np.float64(0.0),) + (0.0,) * 7,
        (FloatSubclass(0.0),) + (0.0,) * 7,
        (IntSubclass(0),) + (0.0,) * 7,
        ("0",) + (0.0,) * 7,
    ],
)
def test_should_clear_previous_initial_before_refusing_invalid_reset(invalid):
    history = PublicInitialFrameHistory()
    history.reset(frame())
    with pytest.raises(ValueError):
        history.reset(invalid)
    with pytest.raises(ValueError, match="reset"):
        history.encode(frame(1))
    assert history.reset(frame(2)) == frame(2) + frame(2)


@pytest.mark.parametrize(
    "invalid", [(float("nan"),) + (0.0,) * 7, (False,) * 8, (0.0,) * 16, TupleSubclass((0.0,) * 8)]
)
def test_should_refuse_invalid_current_without_losing_this_episode_initial(invalid):
    history = PublicInitialFrameHistory()
    history.reset(frame())
    with pytest.raises(ValueError):
        history.encode(invalid)
    assert history.encode(frame(1)) == frame() + frame(1)


def test_should_separate_real_A_cue_alias_using_only_the_initial_public_frame():
    layout = KeyDoorLayout("A")
    red, blue = KeyDoorNavigation(layout, 1), KeyDoorNavigation(layout, -1)
    red_history, blue_history = PublicInitialFrameHistory(), PublicInitialFrameHistory()
    red_initial, blue_initial = red.observation, blue.observation
    assert red_initial[6] == 1.0 and blue_initial[6] == -1.0
    red_history.reset(red_initial)
    blue_history.reset(blue_initial)
    for _ in range(2):
        red.step(0)
        blue.step(0)
    assert red.observation == blue.observation
    assert red.observation[6] == 0.0
    assert scripted_actions(layout, 1)[2] != scripted_actions(layout, -1)[2]
    red_encoded = red_history.encode(red.observation)
    blue_encoded = blue_history.encode(blue.observation)
    assert red_encoded[8:] == blue_encoded[8:]
    assert red_encoded[:8] == red_initial and blue_encoded[:8] == blue_initial
    assert red_encoded != blue_encoded


@pytest.mark.parametrize("role", ["train", "final_test", "probe_train", "probe_test"])
def test_should_preserve_every_complete_record_field_and_detach_original_episode(role):
    original = episode(role)
    before = pickle.dumps(original)
    encoded = encode_public_history_episode(original)
    assert encoded is not original and encoded.transitions is not original.transitions
    assert (encoded.episode_id, encoded.context, encoded.role, encoded.num_classes) == (
        original.episode_id,
        original.context,
        role,
        5,
    )
    initial = original.transitions[0].observation
    for old, new in zip(original.transitions, encoded.transitions, strict=True):
        assert new is not old
        assert new.observation == initial + old.observation
        assert new.next_observation == initial + old.next_observation
        assert (new.step, new.action, new.reward, new.terminal, new.num_classes) == (
            old.step,
            old.action,
            old.reward,
            old.terminal,
            old.num_classes,
        )
    assert all(
        left.next_observation == right.observation
        for left, right in zip(encoded.transitions, encoded.transitions[1:])
    )
    assert pickle.dumps(original) == before
    first, labels = episode_arrays(encoded)
    second, other_labels = episode_arrays(encoded)
    assert first.shape == (3, 16) and labels.shape == (3, 1)
    assert not np.shares_memory(first, second) and not np.shares_memory(labels, other_labels)
    first.fill(0)
    labels.fill(4)
    np.testing.assert_array_equal(second[0], initial + initial)
    np.testing.assert_array_equal(other_labels[:, 0], [0.0, 1.0, 2.0])
    object.__setattr__(original.transitions[0], "reward", 0.75)
    object.__setattr__(original.transitions[0], "observation", frame(4))
    assert encoded.transitions[0].reward == -0.25
    assert encoded.transitions[0].observation == initial + initial


def test_should_ignore_ids_context_roles_actions_and_rewards_when_encoding_inputs():
    original = episode()
    altered = replace(
        original,
        episode_id="different-labels",
        context="B",
        role="probe_test",
        transitions=tuple(
            replace(row, action=(row.action + 1) % 5, reward=-row.reward)
            for row in original.transitions
        ),
    )
    before, after = map(encode_public_history_episode, (original, altered))
    assert [row.observation for row in before.transitions] == [
        row.observation for row in after.transitions
    ]
    assert [row.next_observation for row in before.transitions] == [
        row.next_observation for row in after.transitions
    ]
    assert [row.action for row in before.transitions] != [row.action for row in after.transitions]


def test_should_never_use_future_terminal_observation_for_current_encoding():
    original = episode()
    changed = replace(
        original,
        transitions=(
            *original.transitions[:-1],
            replace(original.transitions[-1], next_observation=frame(4), action=4, reward=-1.0),
        ),
    )
    before, after = map(encode_public_history_episode, (original, changed))
    assert [row.observation for row in before.transitions] == [
        row.observation for row in after.transitions
    ]
    assert [row.next_observation for row in before.transitions[:-1]] == [
        row.next_observation for row in after.transitions[:-1]
    ]
    assert before.transitions[-1].next_observation != after.transitions[-1].next_observation


@pytest.mark.parametrize("width", [1, 7, 9, 16, 148, 296])
def test_should_refuse_nonlocal_or_previously_encoded_episode_widths(width):
    record = DemonstrationEpisode(
        "wrong-width",
        "A",
        "train",
        (DemonstrationTransition(0, (0.0,) * width, 0, 0.0, (0.0,) * width, True, 5),),
        5,
    )
    with pytest.raises(ValueError, match="eight-coordinate"):
        encode_public_history_episode(record)


@pytest.mark.parametrize(
    "fault",
    [
        "not_episode",
        "subclass",
        "discontinuous",
        "bool",
        "nan",
        "early_terminal",
        "missing_terminal",
        "double_encoding",
    ],
)
def test_should_revalidate_complete_episode_and_refuse_corrupted_record(fault):
    record = episode()
    if fault == "not_episode":
        record = object()
    elif fault == "subclass":

        class EpisodeSubclass(DemonstrationEpisode):
            pass

        record = EpisodeSubclass(
            record.episode_id, record.context, record.role, record.transitions, record.num_classes
        )
    elif fault == "discontinuous":
        object.__setattr__(record.transitions[1], "observation", frame(4))
    elif fault in ("bool", "nan"):
        bad = True if fault == "bool" else float("nan")
        object.__setattr__(record.transitions[0], "observation", (bad,) + frame()[1:])
    elif fault == "early_terminal":
        object.__setattr__(record.transitions[0], "terminal", True)
    elif fault == "missing_terminal":
        object.__setattr__(record.transitions[-1], "terminal", False)
    else:
        record = encode_public_history_episode(record)
    with pytest.raises(ValueError):
        encode_public_history_episode(record)
