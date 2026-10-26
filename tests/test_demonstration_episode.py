"""Whole demonstration packets stay complete, isolated, and canonically encoded."""

from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pytest

from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    EpisodeReplaySchema,
    copy_episode,
    decode_episode,
    encode_episode,
    episode_arrays,
)


def episode(*, dim=2, steps=3, **changes):
    def observation(step):
        return tuple(float(step + column) / (steps + dim) for column in range(dim))

    values: dict[str, Any] = dict(
        episode_id="episode-A",
        context="context-A",
        role="train",
        transitions=tuple(
            DemonstrationTransition(
                step=step,
                observation=observation(step),
                action=step % 2,
                reward=0.25 if step % 2 else -0.25,
                next_observation=observation(step + 1),
                terminal=step == steps - 1,
            )
            for step in range(steps)
        ),
    )
    values.update(changes)
    return DemonstrationEpisode(**values)


@pytest.mark.parametrize("dim,steps", [(1, 1), (2, 3), (16, 16)])
def test_should_roundtrip_every_transition_in_order_with_schema_bounds(dim, steps):
    original = episode(dim=dim, steps=steps)
    schema = EpisodeReplaySchema(observation_dim=dim, steps_per_episode=steps)

    encoded, dummy = encode_episode(original, schema)
    restored = decode_episode(encoded, dummy, schema)

    assert restored == original
    assert restored.role == "train"
    assert encoded.shape == (1, 98 + steps * (2 * dim + 3))
    assert encoded.shape[1] == schema.encoded_dim
    assert encoded.dtype == dummy.dtype == np.float64
    np.testing.assert_array_equal(dummy, np.zeros((1, 1), dtype=np.float64))
    assert [item.step for item in restored.transitions] == list(range(steps))
    assert [item.terminal for item in restored.transitions] == [False] * (steps - 1) + [True]


def test_should_roundtrip_unicode_identifiers_at_utf8_byte_limits():
    original = episode(episode_id="é" * 32, context="界" * 10 + "ab")
    schema = EpisodeReplaySchema(2, 3)

    encoded, dummy = encode_episode(original, schema)

    assert decode_episode(encoded, dummy, schema) == original
    assert encoded[0, 0] == 64
    assert encoded[0, 65] == 32


def test_should_extract_ordered_observations_and_actions_without_next_state_leakage():
    original = episode()

    features, targets = episode_arrays(original)

    np.testing.assert_array_equal(
        features, np.array([item.observation for item in original.transitions], dtype=np.float64)
    )
    np.testing.assert_array_equal(targets, np.array([[0.0], [1.0], [0.0]]))
    assert features.shape == (3, 2)
    assert targets.shape == (3, 1)
    assert features.dtype == targets.dtype == np.float64
    assert not np.array_equal(features[-1], original.transitions[-1].next_observation)


def test_should_return_owned_arrays_and_immutable_detached_decoded_records():
    original = episode()
    schema = EpisodeReplaySchema(2, 3)
    encoded, dummy = encode_episode(original, schema)
    restored = decode_episode(encoded, dummy, schema)
    first_features, first_targets = episode_arrays(original)
    second_features, second_targets = episode_arrays(original)

    first_features.fill(-1.0)
    first_targets.fill(1.0)
    encoded.fill(0.0)
    dummy.fill(1.0)

    assert restored == original
    assert not np.shares_memory(first_features, second_features)
    assert not np.shares_memory(first_targets, second_targets)
    np.testing.assert_array_equal(second_targets, [[0.0], [1.0], [0.0]])
    np.testing.assert_array_equal(second_features[0], original.transitions[0].observation)
    with pytest.raises(FrozenInstanceError):
        setattr(restored, "role", "final_test")


def test_should_revalidate_frozen_records_when_copying_or_extracting_arrays():
    original = episode()
    copied = copy_episode(original)
    assert copied == original
    assert copied is not original
    object.__setattr__(original.transitions[1], "step", 0)

    with pytest.raises((TypeError, ValueError)):
        copy_episode(original)
    with pytest.raises((TypeError, ValueError)):
        episode_arrays(original)
    with pytest.raises((TypeError, ValueError)):
        encode_episode(original, EpisodeReplaySchema(2, 3))
    assert copied.transitions[1].step == 1


@pytest.mark.parametrize(
    "dim,steps",
    [(0, 3), (17, 3), (True, 3), (2, 0), (2, 17), (2, 1.0), (np.int64(2), 3)],
)
def test_should_refuse_non_native_or_out_of_bounds_schema_dimensions(dim, steps):
    with pytest.raises((TypeError, ValueError)):
        EpisodeReplaySchema(dim, steps)


@pytest.mark.parametrize(
    "changes",
    [
        {"episode_id": ""},
        {"episode_id": "é" * 33},
        {"episode_id": 3},
        {"context": ""},
        {"context": "界" * 11},
        {"role": "validation"},
        {"transitions": ()},
        {"transitions": []},
    ],
)
def test_should_refuse_invalid_packet_identity_role_or_transition_container(changes):
    with pytest.raises((TypeError, ValueError)):
        episode(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"step": True},
        {"action": True},
        {"action": 0.0},
        {"action": 2},
        {"reward": float("nan")},
        {"reward": 1.01},
        {"observation": (float("inf"), 0.0)},
        {"observation": (np.float64(0.0), 0.0)},
        {"next_observation": (0.0,)},
        {"terminal": 1},
    ],
)
def test_should_refuse_invalid_transition_values_without_numeric_coercion(changes):
    with pytest.raises((TypeError, ValueError)):
        replace(episode().transitions[0], **changes)


@pytest.mark.parametrize("failure", ["reordered", "discontinuous", "truncated", "early_terminal"])
def test_should_refuse_incomplete_or_disconnected_episode_sequences(failure):
    original = episode()
    first, middle, last = original.transitions
    transitions: tuple[Any, ...]
    with pytest.raises((TypeError, ValueError)):
        if failure == "reordered":
            transitions = (middle, first, last)
        elif failure == "discontinuous":
            transitions = (first, replace(middle, observation=(-0.5, -0.5)), last)
        elif failure == "truncated":
            transitions = (first, middle)
        else:
            transitions = (replace(first, terminal=True), middle, last)
        replace(original, transitions=transitions)


def test_should_keep_final_packets_readable_but_refuse_replay_encoding():
    final_packet = episode(role="final_test")
    assert copy_episode(final_packet) == final_packet
    features, targets = episode_arrays(final_packet)
    assert features.shape == (3, 2)
    assert targets.shape == (3, 1)
    with pytest.raises((TypeError, ValueError)):
        encode_episode(final_packet, EpisodeReplaySchema(2, 3))


@pytest.mark.parametrize("dim,steps", [(1, 3), (2, 2)])
def test_should_refuse_schema_that_would_drop_observations_or_transitions(dim, steps):
    with pytest.raises((TypeError, ValueError)):
        encode_episode(episode(), EpisodeReplaySchema(dim, steps))


@pytest.mark.parametrize("failure", ["flat", "two_rows", "float32", "dummy_shape", "dummy_nonzero"])
def test_should_refuse_wrong_array_shapes_dtypes_or_dummy_targets(failure):
    schema = EpisodeReplaySchema(2, 3)
    encoded, dummy = encode_episode(episode(), schema)
    if failure == "flat":
        encoded = encoded[0]
    elif failure == "two_rows":
        encoded = np.concatenate((encoded, encoded))
    elif failure == "float32":
        encoded = encoded.astype(np.float32)
    elif failure == "dummy_shape":
        dummy = np.zeros((1,))
    else:
        dummy[0, 0] = 1.0

    with pytest.raises((TypeError, ValueError)):
        decode_episode(encoded, dummy, schema)


@pytest.mark.parametrize(
    "column,value",
    [
        (0, 1.5),  # A length cannot be rounded to an integer.
        (0, 65.0),
        (1, 255.0),  # Invalid UTF-8 must not be replaced or ignored.
        (1, 101.5),  # Byte values must be exact integers too.
        (64, 1.0),  # Unused identifier padding is canonical zero.
        (97, 1.0),  # Context padding is covered independently.
        (100, 0.5),  # First binary action.
        (101, float("nan")),  # Reward.
        (104, 1.0),  # Premature terminal.
        (118, 0.0),  # Last transition must terminate the complete packet.
        (105, -0.5),  # Next observation must match the prior next state.
    ],
)
def test_should_refuse_corrupt_wire_fields_instead_of_admitting_partial_replay(column, value):
    schema = EpisodeReplaySchema(2, 3)
    encoded, dummy = encode_episode(episode(), schema)
    encoded[0, column] = value

    with pytest.raises((TypeError, ValueError)):
        decode_episode(encoded, dummy, schema)
