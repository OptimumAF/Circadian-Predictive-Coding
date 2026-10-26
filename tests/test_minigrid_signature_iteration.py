"""Every immutable episode leaf and mutable array byte remains qualified.

Synthetic complete sources exercise the existing signature boundary only. No
learner, native update, prediction or environment is involved.
"""

from copy import deepcopy

import numpy as np
import pytest

from src.adapters import minigrid_learning_pilot as pilot
from test_minigrid_learning_pilot import source_batch


def test_should_accept_unchanged_packets_and_preserve_the_existing_array_value_contract():
    packets = tuple(source_batch(context, cue) for context, cue in (("A", "blue"), ("B", "red")))
    captured = pilot._signature(packets)
    assert pilot._check_signature(packets, captured) is True
    # Array values/dtype/shape are bound, while immutable episode leaves and
    # containers retain their original stronger identity contract.
    object.__setattr__(packets[-1], "features", packets[-1].features.copy())
    object.__setattr__(packets[-1], "targets", packets[-1].targets.copy())
    assert pilot._check_signature(packets, captured) is True


@pytest.mark.parametrize(
    "fault", ["last_class_type", "equal_reward_owner", "equal_last_observation_owner"]
)
def test_should_visit_the_final_transition_and_reject_equal_valued_leaf_replacements(fault):
    packet = source_batch("A", "blue")
    captured = pilot._signature((packet,))
    assert packet.episode is not None
    last = packet.episode.transitions[-1]
    if fault == "last_class_type":
        # Equal numerical value must not substitute another native type at the
        # final leaf of the final transition.
        object.__setattr__(last, "num_classes", float(last.num_classes))
    elif fault == "equal_reward_owner":
        reward_copy = float(str(last.reward))
        assert reward_copy == last.reward and reward_copy is not last.reward
        object.__setattr__(last, "reward", reward_copy)
    else:
        observation_copy = tuple(list(last.next_observation))
        assert (
            observation_copy == last.next_observation
            and observation_copy is not last.next_observation
        )
        object.__setattr__(last, "next_observation", observation_copy)
    assert pilot._check_signature((packet,), captured) is False


@pytest.mark.parametrize("fault", ["last_feature_byte", "last_target_byte", "dtype", "shape"])
def test_should_keep_full_array_content_dtype_and_shape_checks(fault):
    packet = source_batch("A", "blue")
    captured = pilot._signature((packet,))
    if fault == "last_feature_byte":
        packet.features[-1, -1] = 0.25
    elif fault == "last_target_byte":
        packet.targets[-1, 0] = 2.0
    elif fault == "dtype":
        object.__setattr__(packet, "features", packet.features.astype(np.float32))
    else:
        object.__setattr__(packet, "features", packet.features.reshape(3, -1))
    assert pilot._check_signature((packet,), captured) is False


@pytest.mark.parametrize("owner", ["episode", "transitions"])
def test_should_refuse_equal_episode_or_transition_tuple_owner_replacements(owner):
    packet = source_batch("A", "blue")
    captured = pilot._signature((packet,))
    assert packet.episode is not None
    if owner == "episode":
        episode_copy = deepcopy(packet.episode)
        assert episode_copy == packet.episode and episode_copy is not packet.episode
        object.__setattr__(packet, "episode", episode_copy)
    else:
        transitions = packet.episode.transitions
        transitions_copy = tuple(list(transitions))
        assert transitions_copy == transitions and transitions_copy is not transitions
        object.__setattr__(packet.episode, "transitions", transitions_copy)
    assert pilot._check_signature((packet,), captured) is False


def test_should_refuse_truncated_or_extra_captured_leaves_instead_of_matching_a_prefix():
    packet = source_batch("A", "blue")
    captured = pilot._signature((packet,))
    binding = captured[0][11]
    for leaves in (binding[2][:-1], (*binding[2], binding[2][-1])):
        altered = (*captured[0][:11], (binding[0], binding[1], leaves))
        assert pilot._check_signature((packet,), (altered,)) is False


def test_should_refuse_a_mutable_transition_container_at_initial_capture():
    packet = source_batch("A", "blue")
    assert packet.episode is not None
    object.__setattr__(packet.episode, "transitions", list(packet.episode.transitions))
    with pytest.raises(ValueError):
        pilot._signature((packet,))
