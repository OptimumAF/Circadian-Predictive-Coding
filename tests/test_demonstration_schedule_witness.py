"""Bounded schedule evidence retains every field without recurring serialization.

Pure cases probe ownership and exact native representation. Native cases reuse the
existing bound BP array adapter and never launch game physics or evaluation.
"""

from copy import deepcopy
from typing import Any

import pytest

from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from src.core.demonstration_schedule_witness import DemonstrationScheduleWitness


class IntSubclass(int):
    pass


class FloatSubclass(float):
    pass


class StrSubclass(str):
    pass


class TupleSubclass(tuple):
    pass


class EpisodeSubclass(DemonstrationEpisode):
    pass


class TransitionSubclass(DemonstrationTransition):
    pass


def episode(identity="A", *, dim=2, steps=3):
    observations = tuple(
        tuple(float(index + column) / (steps + dim) for column in range(dim))
        for index in range(steps + 1)
    )
    return DemonstrationEpisode(
        identity,
        "A",
        "train",
        tuple(
            DemonstrationTransition(
                index,
                observations[index],
                index % 2,
                0.0,
                observations[index + 1],
                index == steps - 1,
            )
            for index in range(steps)
        ),
    )


def schedule():
    return (episode(), episode("late"))


def fresh_tuple(value):
    return tuple(list(value))


def subclass_record(value, record_type):
    result = object.__new__(record_type)
    object.__setattr__(result, "__dict__", dict(vars(value)))
    return result


@pytest.mark.parametrize("dim,steps", [(1, 1), (2, 3), (16, 16), (148, 2), (296, 2)])
def test_should_admit_complete_native_train_records_and_recheck_without_changes(dim, steps):
    records = (episode(dim=dim, steps=steps),)
    witness = DemonstrationScheduleWitness(records)
    original_dicts = tuple(vars(item) for item in (records[0], *records[0].transitions))
    witness.check(records)
    witness.check(records)
    assert all(
        vars(item) is original
        for item, original in zip(
            (records[0], *records[0].transitions), original_dicts, strict=True
        )
    )


def test_should_admit_maximum_bounded_schedule_and_check_the_final_record():
    records = tuple(episode(str(index), dim=1, steps=1) for index in range(512))
    witness = DemonstrationScheduleWitness(records)
    witness.check(records)
    object.__setattr__(records[-1].transitions[-1], "reward", 0.5)
    with pytest.raises(ValueError):
        witness.check(records)
    assert records[-1].transitions[-1].reward == 0.5


@pytest.mark.parametrize("fault", ["list", "tuple_subclass", "empty", "too_many", "object"])
def test_should_refuse_unbounded_or_nonexact_schedule_container(fault):
    records: Any = schedule()
    if fault == "list":
        records = list(records)
    elif fault == "tuple_subclass":
        records = TupleSubclass(records)
    elif fault == "empty":
        records = ()
    elif fault == "too_many":
        records = (records[0],) * 513
    else:
        records = object()
    with pytest.raises(ValueError):
        DemonstrationScheduleWitness(records)


@pytest.mark.parametrize(
    "fault",
    [
        "episode_subclass",
        "transition_subclass",
        "episode_extra",
        "transition_extra",
        "role_final",
        "role_probe_train",
        "role_probe_test",
        "role_other",
        "episode_id_blank",
        "context_whitespace",
        "episode_id_surrogate",
        "episode_id_long",
        "context_long",
        "class_bool",
        "class_subclass",
        "transition_class",
        "transition_class_bool",
        "step_bool",
        "step_subclass",
        "step_wrong",
        "action_bool",
        "action_subclass",
        "action_outside",
        "reward_int",
        "reward_bool",
        "reward_subclass",
        "reward_nan",
        "reward_inf",
        "reward_outside",
        "terminal_int",
        "terminal_wrong",
        "early_terminal",
        "observation_list",
        "observation_tuple_subclass",
        "observation_int",
        "observation_bool",
        "observation_subclass",
        "observation_nan",
        "observation_outside",
        "next_int",
        "next_nan",
        "next_width",
        "discontinuous",
        "transitions_list",
        "transitions_empty",
        "transitions_long",
        "id_subclass",
        "context_subclass",
        "role_subclass",
    ],
)
def test_should_validate_all_original_fields_before_capturing_evidence(fault):
    records = schedule()
    packet, transition = records[-1], records[-1].transitions[-1]
    if fault == "episode_subclass":
        records = (records[0], subclass_record(packet, EpisodeSubclass))
    elif fault == "transition_subclass":
        object.__setattr__(
            packet,
            "transitions",
            (*packet.transitions[:-1], subclass_record(transition, TransitionSubclass)),
        )
    elif fault in ("episode_extra", "transition_extra"):
        object.__setattr__(packet if fault == "episode_extra" else transition, "foreign", 1)
    elif fault.startswith("role_"):
        role = {
            "role_final": "final_test",
            "role_probe_train": "probe_train",
            "role_probe_test": "probe_test",
            "role_other": "other",
            "role_subclass": StrSubclass("train"),
        }[fault]
        object.__setattr__(packet, "role", role)
    elif fault.startswith("episode_id_"):
        value = {
            "episode_id_blank": "",
            "episode_id_surrogate": "\ud800",
            "episode_id_long": "é" * 33,
        }[fault]
        object.__setattr__(packet, "episode_id", value)
    elif fault.startswith("context_"):
        value = {
            "context_whitespace": " A",
            "context_long": "界" * 11,
            "context_subclass": StrSubclass("A"),
        }[fault]
        object.__setattr__(packet, "context", value)
    elif fault in ("class_bool", "class_subclass"):
        object.__setattr__(packet, "num_classes", True if fault == "class_bool" else IntSubclass(2))
    elif fault.startswith("transition_class"):
        object.__setattr__(transition, "num_classes", True if fault.endswith("bool") else 7)
    elif fault.startswith("step_"):
        object.__setattr__(
            transition,
            "step",
            {"step_bool": True, "step_subclass": IntSubclass(2), "step_wrong": 0}[fault],
        )
    elif fault.startswith("action_"):
        object.__setattr__(
            transition,
            "action",
            {"action_bool": False, "action_subclass": IntSubclass(0), "action_outside": 2}[fault],
        )
    elif fault.startswith("reward_"):
        object.__setattr__(
            transition,
            "reward",
            {
                "reward_int": 0,
                "reward_bool": False,
                "reward_subclass": FloatSubclass(0.0),
                "reward_nan": float("nan"),
                "reward_inf": float("inf"),
                "reward_outside": 1.1,
            }[fault],
        )
    elif fault in ("terminal_int", "terminal_wrong"):
        object.__setattr__(transition, "terminal", 1 if fault == "terminal_int" else False)
    elif fault == "early_terminal":
        object.__setattr__(packet.transitions[0], "terminal", True)
    elif fault.startswith("observation_"):
        values: Any = list(transition.observation)
        if fault == "observation_list":
            pass
        elif fault == "observation_tuple_subclass":
            values = TupleSubclass(values)
        else:
            values[0] = {
                "observation_int": 0,
                "observation_bool": False,
                "observation_subclass": FloatSubclass(0.0),
                "observation_nan": float("nan"),
                "observation_outside": 1.1,
            }[fault]
            values = tuple(values)
        object.__setattr__(transition, "observation", values)
    elif fault.startswith("next_"):
        values = list(transition.next_observation)
        if fault == "next_width":
            values.pop()
        else:
            values[0] = 0 if fault == "next_int" else float("nan")
        object.__setattr__(transition, "next_observation", tuple(values))
    elif fault == "discontinuous":
        object.__setattr__(transition, "observation", (-0.5, -0.5))
    elif fault.startswith("transitions_"):
        values = {
            "transitions_list": list(packet.transitions),
            "transitions_empty": (),
            "transitions_long": (transition,) * 17,
        }[fault]
        object.__setattr__(packet, "transitions", values)
    else:
        object.__setattr__(packet, "episode_id", StrSubclass(packet.episode_id))
    with pytest.raises(ValueError):
        DemonstrationScheduleWitness(records)


@pytest.mark.parametrize(
    "field,value",
    [("episode_id", "other"), ("context", "B"), ("role", "probe_train"), ("num_classes", 7)],
)
def test_should_refuse_late_episode_scalar_changes_without_repairing_them(field, value):
    records = schedule()
    witness = DemonstrationScheduleWitness(records)
    object.__setattr__(records[-1], field, value)
    with pytest.raises(ValueError):
        witness.check(records)
    assert getattr(records[-1], field) is value
    with pytest.raises(ValueError):
        witness.check(records)


@pytest.mark.parametrize(
    "field,value",
    [("step", 1), ("action", 1), ("reward", 0.5), ("terminal", False), ("num_classes", 7)],
)
def test_should_refuse_every_late_transition_scalar_change_without_repairing_it(field, value):
    records = schedule()
    witness = DemonstrationScheduleWitness(records)
    last = records[-1].transitions[-1]
    object.__setattr__(last, field, value)
    with pytest.raises(ValueError):
        witness.check(records)
    assert getattr(last, field) is value


@pytest.mark.parametrize("start,replacement", [(0.0, -0.0), (-0.0, 0.0)])
def test_should_preserve_reward_sign_bit_even_when_float_equality_cannot_distinguish_it(
    start, replacement
):
    records = schedule()
    last = records[-1].transitions[-1]
    object.__setattr__(last, "reward", start)
    witness = DemonstrationScheduleWitness(records)
    witness.check(records)
    object.__setattr__(last, "reward", replacement)
    assert start == replacement and start.hex() != replacement.hex()
    with pytest.raises(ValueError):
        witness.check(records)
    assert last.reward.hex() == replacement.hex()


@pytest.mark.parametrize(
    "field,value",
    [
        ("step", IntSubclass(2)),
        ("action", False),
        ("reward", FloatSubclass(0.0)),
        ("terminal", 1),
        ("num_classes", IntSubclass(2)),
    ],
)
def test_should_refuse_equal_valued_transition_primitive_substitution(field, value):
    records = schedule()
    witness = DemonstrationScheduleWitness(records)
    last = records[-1].transitions[-1]
    assert getattr(last, field) == value
    object.__setattr__(last, field, value)
    with pytest.raises(ValueError):
        witness.check(records)


@pytest.mark.parametrize("target", ["episode", "transition"])
@pytest.mark.parametrize("fault", ["extra", "dict_replaced", "dict_reordered", "field_deleted"])
def test_should_refuse_record_dictionary_tampering_including_equal_contents(target, fault):
    records = schedule()
    witness = DemonstrationScheduleWitness(records)
    record = records[-1] if target == "episode" else records[-1].transitions[-1]
    original = vars(record)
    if fault == "extra":
        object.__setattr__(record, "foreign", 0)
    elif fault == "dict_replaced":
        object.__setattr__(record, "__dict__", dict(original))
    elif fault == "dict_reordered":
        key = next(iter(original))
        value = original.pop(key)
        original[key] = value
    else:
        original.pop(next(iter(original)))
    with pytest.raises(ValueError):
        witness.check(records)
    if fault == "dict_replaced":
        assert vars(record) is not original
    if fault == "extra":
        assert record.foreign == 0


@pytest.mark.parametrize(
    "fault", ["outer", "episode", "transitions", "transition", "observation", "next_observation"]
)
def test_should_refuse_equal_valued_owned_record_or_immutable_tuple_replacements(fault):
    records = schedule()
    witness = DemonstrationScheduleWitness(records)
    original = records
    packet, last = records[-1], records[-1].transitions[-1]
    if fault == "outer":
        records = fresh_tuple(records)
    elif fault == "episode":
        records = (records[0], deepcopy(packet))
    elif fault == "transitions":
        object.__setattr__(packet, "transitions", fresh_tuple(packet.transitions))
    elif fault == "transition":
        object.__setattr__(packet, "transitions", (*packet.transitions[:-1], deepcopy(last)))
    else:
        object.__setattr__(last, fault, fresh_tuple(getattr(last, fault)))
    assert records == original
    with pytest.raises(ValueError):
        witness.check(records)


def test_should_bind_the_callers_actual_records_and_not_reseal_after_refusal():
    records = schedule()
    witness = DemonstrationScheduleWitness(records)
    detached = deepcopy(records)
    with pytest.raises(ValueError):
        witness.check(detached)
    witness.check(records)
    object.__setattr__(records[-1].transitions[-1], "reward", 0.25)
    with pytest.raises(ValueError):
        witness.check(records)
    assert records[-1].transitions[-1].reward == 0.25
    with pytest.raises(ValueError):
        witness.check(records)
    assert detached[-1].transitions[-1].reward == 0.0


def test_should_validate_and_check_without_calling_a_serializer(monkeypatch):
    import pickle

    records = schedule()
    monkeypatch.setattr(pickle, "dumps", lambda *a, **k: pytest.fail("witness serialized records"))
    witness = DemonstrationScheduleWitness(records)
    witness.check(records)


@pytest.mark.parametrize(
    "fault", [None, "reward_sign", "observation_sign", "context", "length", "string_alias"]
)
def test_should_require_bitwise_complete_copy_values_when_binding_a_new_owner(fault):
    records = schedule()
    witness = DemonstrationScheduleWitness(records)
    copied = deepcopy(records)
    if fault == "reward_sign":
        object.__setattr__(copied[-1].transitions[-1], "reward", -0.0)
    elif fault == "observation_sign":
        # The first view has no preceding transition continuity obligation.
        first = copied[0].transitions[0]
        object.__setattr__(first, "observation", (-0.0, *first.observation[1:]))
    elif fault == "context":
        object.__setattr__(copied[-1], "context", "changed")
    elif fault == "length":
        copied = copied[:-1]
    elif fault == "string_alias":
        object.__setattr__(copied[-1], "role", ("~" + copied[-1].role)[1:])
    other = DemonstrationScheduleWitness(copied)
    if fault is None:
        witness.check_copy(other)
    else:
        with pytest.raises(ValueError):
            witness.check_copy(other)
    witness.check(records)


@pytest.mark.parametrize("target", ["episode", "transition"])
@pytest.mark.parametrize("when", ["admit", "later"])
def test_should_refuse_equal_valued_subclass_dictionary_keys(target, when):
    records = schedule()
    record = records[-1] if target == "episode" else records[-1].transitions[-1]
    if when == "later":
        witness = DemonstrationScheduleWitness(records)
    # Mutate the original dictionary to target key types, not its identity.
    values = vars(record)
    items = tuple(values.items())
    values.clear()
    values.update((StrSubclass(key), value) for key, value in items)
    with pytest.raises(ValueError):
        if when == "admit":
            DemonstrationScheduleWitness(records)
        else:
            witness.check(records)


@pytest.mark.parametrize("field", ["episode_id", "context", "role"])
def test_should_refuse_equal_native_string_substitution_that_can_change_alias_evidence(field):
    records = schedule()
    record = records[-1]
    if field != "role":
        object.__setattr__(record, field, "original-long-value")
    witness = DemonstrationScheduleWitness(records)
    value = getattr(record, field)
    replacement = ("~" + value)[1:]
    assert type(replacement) is str and replacement == value and replacement is not value
    object.__setattr__(record, field, replacement)
    with pytest.raises(ValueError):
        witness.check(records)


def test_should_keep_bound_bp_runtime_functional_when_pickle_is_unavailable_after_admission(
    monkeypatch,
):
    from src.adapters import torch_train_weighted_learner as adapter
    from src.core.demonstration_episode import episode_arrays
    from test_weighted_feature_stream_adapter import learner, schedule as bound_schedule
    from test_torch_feature_learner import assert_tree
    from src.shared.torch_runtime import require_torch

    torch = require_torch()
    source = learner()
    initial = source.snapshot_state()
    monkeypatch.setattr(
        adapter.pickle, "dumps", lambda *a, **k: pytest.fail("bound metadata serialized schedule")
    )
    features, targets = episode_arrays(bound_schedule()[0])
    assert source.predict(features).shape == (3, 7)
    source.train_batch(features, targets)
    saved = source.snapshot_state()
    assert saved.completed_updates == 1
    for child in (source.fork(), deepcopy(source)):
        assert_tree(torch, child.snapshot_state(), saved)
        assert child._schedule_witness is not source._schedule_witness
        assert child._schedule_seal_witness is not source._schedule_seal_witness
        child._schedule_witness.check(child._episodes)
        child._schedule_seal_witness.check(child._schedule_seal)
        with pytest.raises(ValueError):
            child._schedule_witness.check(source._episodes)
    source.restore_state(initial)
    assert_tree(torch, source.snapshot_state(), initial)


@pytest.mark.parametrize("fault", ["different_context", "equal_string"])
def test_should_refuse_a_semantically_valid_but_different_fork_copy(monkeypatch, fault):
    from src.adapters import torch_train_weighted_learner as adapter
    from test_weighted_feature_stream_adapter import learner
    from test_torch_feature_learner import assert_tree
    from src.shared.torch_runtime import require_torch

    source = learner()
    saved = source.snapshot_state()
    original_copy = adapter.copy_episode

    def changed_copy(episode):
        copied = original_copy(episode)
        if fault == "different_context":
            object.__setattr__(copied, "context", "foreign-valid-context")
        else:
            object.__setattr__(copied, "role", ("~" + copied.role)[1:])
        return copied

    monkeypatch.setattr(adapter, "copy_episode", changed_copy)
    with pytest.raises(ValueError, match="copied demonstration"):
        source.fork()
    assert_tree(require_torch(), source.snapshot_state(), saved)


@pytest.mark.parametrize("fault", ["own", "seal", "payload", "witness", "seal_witness"])
def test_should_refuse_bound_schedule_evidence_changes_and_preserve_the_native_donor(fault):
    from test_weighted_feature_stream_adapter import learner
    from test_torch_feature_learner import assert_tree
    from src.shared.torch_runtime import require_torch

    torch = require_torch()
    source = learner()
    donor = source._learner
    saved = donor.snapshot_state()
    if fault in ("own", "seal"):
        records = source._episodes if fault == "own" else source._schedule_seal
        object.__setattr__(records[-1].transitions[-1], "reward", -0.0)
    elif fault == "payload":
        source._schedule_payload = bytes(bytearray(source._schedule_payload))
    elif fault == "witness":
        source._schedule_witness = DemonstrationScheduleWitness(source._episodes)
    else:
        source._schedule_seal_witness = DemonstrationScheduleWitness(source._schedule_seal)
    with pytest.raises(ValueError):
        source.snapshot_state()
    assert source._learner is donor
    assert_tree(torch, donor.snapshot_state(), saved)
    if fault in ("own", "seal"):
        assert records[-1].transitions[-1].reward.hex() == (-0.0).hex()
