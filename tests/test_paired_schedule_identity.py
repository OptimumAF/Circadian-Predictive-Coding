"""Pure preservation checks for the measured identity-comparison optimization.

No learner, Torch, environment or experiment is constructed. Record ownership
and immutable leaf identity remain distinct from equal array-byte contents.
"""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_learning_pilot as shared
from src.adapters import torch_train_episode_batch as paired
from test_minigrid_learning_pilot import source_batch


def fixture_records():
    packet = source_batch("A", "blue")
    red = source_batch("A", "red").episode
    assert packet.episode is not None and red is not None
    return packet, (packet.episode, red)


def frozen(scope, packet, episodes):
    return paired._binding(episodes) if scope == "schedule" else shared._signature((packet,))


def assert_refused(scope, packet, episodes, binding):
    if scope == "schedule":
        assert paired._same_schedule(episodes, binding) is False
    else:
        # Legacy signature admission may fail loudly when a formerly present
        # required field/episode disappears. Either refusal prevents acceptance.
        try:
            valid = shared._check_signature((packet,), binding)
        except (AttributeError, ValueError):
            return
        assert valid is False


@pytest.mark.parametrize("scope", ["schedule", "signature"])
def test_should_accept_original_records_without_resealing_or_changing_them(scope):
    packet, episodes = fixture_records()
    binding = frozen(scope, packet, episodes)
    if scope == "schedule":
        assert paired._same_schedule(episodes, binding) is True
    else:
        assert shared._check_signature((packet,), binding) is True


@pytest.mark.parametrize("scope", ["schedule", "signature"])
@pytest.mark.parametrize("field", ["episode_id", "context", "role", "transitions", "num_classes"])
def test_should_refuse_mutation_of_every_episode_field(scope, field):
    packet, episodes = fixture_records()
    binding = frozen(scope, packet, episodes)
    changes = dict(
        episode_id="replaced-source-blue",
        context="B",
        role="final_test",
        transitions=tuple(list(episodes[0].transitions)),
        num_classes=3,
    )
    object.__setattr__(episodes[0], field, changes[field])
    assert_refused(scope, packet, episodes, binding)


@pytest.mark.parametrize("scope", ["schedule", "signature"])
@pytest.mark.parametrize(
    "field",
    ["step", "observation", "action", "reward", "next_observation", "terminal", "num_classes"],
)
def test_should_refuse_mutation_of_every_transition_field(scope, field):
    packet, episodes = fixture_records()
    binding = frozen(scope, packet, episodes)
    step = episodes[0].transitions[0]
    changes = dict(
        step=1,
        observation=tuple(list(step.observation)),
        action=2,
        reward=0.25,
        next_observation=tuple(list(step.next_observation)),
        terminal=True,
        num_classes=3,
    )
    object.__setattr__(step, field, changes[field])
    assert_refused(scope, packet, episodes, binding)


@pytest.mark.parametrize("scope", ["schedule", "signature"])
@pytest.mark.parametrize(
    "field", ["episode_id", "role", "reward", "observation", "next_observation", "transitions"]
)
def test_should_refuse_distinct_equal_native_string_float_and_tuple_replacements(scope, field):
    packet, episodes = fixture_records()
    binding = frozen(scope, packet, episodes)
    owner = (
        episodes[0]
        if field in ("episode_id", "role", "transitions")
        else episodes[0].transitions[0]
    )
    original = getattr(owner, field)
    replacement: Any
    if type(original) is str:
        replacement = original.encode("utf8").decode("utf8")
    elif type(original) is float:
        replacement = float(str(original))
    else:
        replacement = tuple(list(original))
    assert (
        replacement == original
        and type(replacement) is type(original)
        and replacement is not original
    )
    object.__setattr__(owner, field, replacement)
    assert_refused(scope, packet, episodes, binding)


@pytest.mark.parametrize("scope", ["schedule", "signature"])
@pytest.mark.parametrize("field,replacement", [("action", 1.0), ("reward", 0), ("terminal", 0)])
def test_should_refuse_equal_values_with_changed_native_scalar_types(scope, field, replacement):
    packet, episodes = fixture_records()
    binding = frozen(scope, packet, episodes)
    step = episodes[0].transitions[0]
    assert getattr(step, field) == replacement and type(getattr(step, field)) is not type(
        replacement
    )
    object.__setattr__(step, field, replacement)
    assert_refused(scope, packet, episodes, binding)


@pytest.mark.parametrize("owner_kind", ["episode", "transition"])
@pytest.mark.parametrize("change", ["addition", "removal", "order"])
def test_should_preserve_existing_ordered_field_set_contract_for_schedule(owner_kind, change):
    _, episodes = fixture_records()
    binding = paired._binding(episodes)
    owner = episodes[0] if owner_kind == "episode" else episodes[0].transitions[0]
    if change == "addition":
        object.__setattr__(owner, "unobserved", True)
    elif change == "removal":
        object.__delattr__(owner, "role" if owner_kind == "episode" else "reward")
    else:
        values = vars(owner)
        name = next(iter(values))
        value = values.pop(name)
        values[name] = value
    assert paired._same_schedule(episodes, binding) is False


@pytest.mark.parametrize("owner_kind", ["episode", "transition"])
def test_should_bind_noncanonical_initial_field_order_without_losing_leaf_alignment(owner_kind):
    _, episodes = fixture_records()
    owner = episodes[0] if owner_kind == "episode" else episodes[0].transitions[0]
    values = vars(owner)
    reversed_values = list(values.items())[::-1]
    values.clear()
    values.update(reversed_values)
    binding = paired._binding(episodes)
    assert paired._same_schedule(episodes, binding) is True
    field = "role" if owner_kind == "episode" else "reward"
    object.__setattr__(owner, field, "final_test" if owner_kind == "episode" else 0.25)
    assert paired._same_schedule(episodes, binding) is False


@pytest.mark.parametrize("scope", ["schedule", "signature"])
def test_should_refuse_replaced_equal_episode_record(scope):
    packet, episodes = fixture_records()
    binding = frozen(scope, packet, episodes)
    equal = replace(episodes[0])
    assert equal == episodes[0] and equal is not episodes[0]
    if scope == "schedule":
        episodes = (equal, episodes[1])
    else:
        object.__setattr__(packet, "episode", equal)
    assert_refused(scope, packet, episodes, binding)


@pytest.mark.parametrize("change", ["feature_bytes", "target_bytes", "dtype", "shape"])
@pytest.mark.parametrize("has_episode", [True, False])
def test_should_compare_complete_array_bytes_and_geometry_for_legacy_and_paired_packets(
    change, has_episode
):
    packet, _ = fixture_records()
    if not has_episode:
        packet = replace(
            packet,
            episode=None,
            row_ids=tuple(f"paired-row-{i}" for i in range(18)),
            features=np.concatenate((packet.features, packet.features)),
            targets=np.concatenate((packet.targets, packet.targets)),
        )
    binding = shared._signature((packet,))
    if change == "feature_bytes":
        packet.features[-1, -1] += 0.1
    elif change == "target_bytes":
        packet.targets[-1, 0] = 2
    elif change == "dtype":
        object.__setattr__(packet, "features", packet.features.astype(np.float32))
    else:
        object.__setattr__(packet, "features", packet.features.reshape(1, -1))
    assert shared._check_signature((packet,), binding) is False


@pytest.mark.parametrize("has_episode", [True, False])
def test_should_preserve_array_value_ownership_contract_without_inventing_array_identity_requirement(
    has_episode,
):
    packet, _ = fixture_records()
    packet = packet if has_episode else replace(packet, episode=None)
    binding = shared._signature((packet,))
    object.__setattr__(packet, "features", packet.features.copy())
    object.__setattr__(packet, "targets", packet.targets.copy())
    assert shared._check_signature((packet,), binding) is True


@pytest.mark.parametrize("has_episode", [True, False])
def test_should_refuse_adding_or_removing_episode_provenance_after_signature_capture(has_episode):
    packet, episodes = fixture_records()
    packet = packet if has_episode else replace(packet, episode=None)
    binding = shared._signature((packet,))
    object.__setattr__(packet, "episode", None if has_episode else episodes[0])
    assert_refused("signature", packet, episodes, binding)


def test_should_leave_outer_tuple_ownership_to_the_existing_adapter_container_guard():
    packet, episodes = fixture_records()
    equal_outer = tuple(list(episodes))
    assert equal_outer is not episodes
    assert paired._same_schedule(equal_outer, paired._binding(episodes)) is True
    # The batch signature similarly binds actual packet identity, not the
    # transport tuple containing those same packet owners.
    assert shared._check_signature(tuple([packet]), shared._signature((packet,))) is True
