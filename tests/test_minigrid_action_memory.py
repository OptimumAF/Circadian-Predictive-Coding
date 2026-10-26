"""Bounded 296/7 count memory and its existing delayed-stream boundary.

Episodes below are explicit engineering arrays, not collected game evidence.
They establish transport/count behavior and make no gameplay-learning claim.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters.action_memory_learner import ActionMemoryLearner
from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.action_memory import (
    ACTION_MEMORY_COUNT_DIAGNOSTIC,
    ACTION_MEMORY_ENTRY_BYTES,
    ActionMemoryEntry,
    ActionMemoryPolicy,
    ActionMemorySnapshot,
    copy_action_memory_snapshot,
    count_action_observations,
    predict_action_memory,
)
from src.core.categorical_probability_scores import probability_scores
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    episode_arrays,
)
from src.core.minigrid_public_history import (
    PublicInitialFrameHistory,
    encode_public_history_episode,
)


def policy(**options: Any) -> ActionMemoryPolicy:
    return ActionMemoryPolicy(feature_dim=296, num_classes=7, **options)


def features(keys: list[float]) -> Any:
    result = np.zeros((len(keys), 296), dtype=np.float64)
    result[:, -1] = keys
    return result


def labels(values: list[int]) -> Any:
    return np.array(values, dtype=np.float64).reshape(-1, 1)


def trained() -> ActionMemoryLearner:
    learner = ActionMemoryLearner(policy())
    learner.train_batch(features([0.0, 0.25]), labels([6, 2]))
    return learner


def test_should_keep_old_defaults_and_charge_the_new_geometry_without_renewing_its_cap():
    old, new = ActionMemoryPolicy(), policy()
    assert (old.feature_dim, old.num_classes, old.max_entries, old.max_payload_bytes) == (
        16,
        5,
        64,
        10752,
    )
    assert ACTION_MEMORY_ENTRY_BYTES == old.entry_payload_bytes == 168
    assert (new.feature_dim, new.num_classes, new.max_entries, new.max_payload_bytes) == (
        296,
        7,
        64,
        10752,
    )
    assert new.entry_payload_bytes == 2424
    assert policy(max_payload_bytes=155136).max_payload_bytes == 64 * 2424
    for bound in (old, new):
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(bound, "entry_payload_bytes", 1)


def test_should_fit_four_default_minigrid_keys_and_atomically_refuse_the_fifth():
    learner = ActionMemoryLearner(policy())
    learner.train_batch(features([0.0, 0.125, 0.25, 0.375]), labels([0, 1, 2, 6]))
    before = learner.snapshot_state()
    assert len(before.entries) * before.policy.entry_payload_bytes == 9696
    with pytest.raises(ValueError, match="capacity"):
        learner.train_batch(features([0.0, 0.5]), labels([6, 5]))
    assert learner.snapshot_state() == before
    assert before.entries[0].counts == (1, 0, 0, 0, 0, 0, 0)
    learner.train_batch(features([0.0] * 128), labels([6] * 128))
    after = learner.snapshot_state()
    assert len(after.entries) == 4 and after.entries[0].counts == (1, 0, 0, 0, 0, 0, 128)
    assert (after.completed_updates, after.training_examples) == (2, 132)


def test_should_require_an_explicit_large_cap_and_never_evict_a_sixty_fifth_key():
    learner = ActionMemoryLearner(policy(max_payload_bytes=155136))
    learner.train_batch(features([index / 128 for index in range(64)]), labels([6] * 64))
    before = learner.snapshot_state()
    assert len(before.entries) == 64 and len(before.entries) * 2424 == 155136
    with pytest.raises(ValueError, match="capacity"):
        learner.train_batch(features([0.0, 0.5]), labels([0, 3]))
    assert learner.snapshot_state() == before
    assert all(entry.counts == (0, 0, 0, 0, 0, 0, 1) for entry in before.entries)


@pytest.mark.parametrize("cap", [1, 2423, 2424])
def test_should_apply_the_declared_byte_cap_to_the_whole_first_batch(cap):
    learner = ActionMemoryLearner(policy(max_payload_bytes=cap))
    before = learner.snapshot_state()
    with pytest.raises(ValueError, match="capacity"):
        learner.train_batch(features([0.0, 0.25]), labels([0, 6]))
    assert learner.snapshot_state() == before


def test_should_return_owned_seven_class_uniform_misses_and_count_conflicting_action_six():
    learner = ActionMemoryLearner(policy())
    empty = learner.snapshot_state()
    prediction = learner.predict(features([0.0, 0.25]))
    np.testing.assert_array_equal(prediction, np.full((2, 7), 1 / 7))
    assert prediction.dtype == np.float64 and learner.snapshot_state() == empty
    prediction.fill(0)
    diagnostic = learner.train_batch(features([0.0, 0.0, 0.0, 0.25]), labels([6, 0, 6, 5]))
    assert diagnostic.definition == ACTION_MEMORY_COUNT_DIAGNOSTIC and diagnostic.value == 4
    assert learner.snapshot_state().entries[0].counts == (1, 0, 0, 0, 0, 0, 2)
    np.testing.assert_array_equal(
        learner.predict(features([0.0, 0.25, 0.5])),
        [[1 / 3, 0, 0, 0, 0, 0, 2 / 3], [0, 0, 0, 0, 0, 1, 0], [1 / 7] * 7],
    )


def test_should_use_the_last_coordinate_exactly_and_keep_first_argmax_ties():
    learner = ActionMemoryLearner(policy())
    learner.train_batch(features([0.0, 0.0]), labels([6, 1]))
    changed = features([np.nextafter(0.0, 1.0)])
    assert np.argmax(learner.predict(features([0.0]))[0]) == 1
    np.testing.assert_array_equal(learner.predict(changed), np.full((1, 7), 1 / 7))
    learner.train_batch(changed, labels([4]))
    assert len(learner.snapshot_state().entries) == 2
    assert np.argmax(learner.predict(changed)[0]) == 4


def test_should_preserve_core_purity_and_detach_arrays_policies_snapshots_restores_and_forks():
    declared = policy()
    learner = ActionMemoryLearner(declared)
    x, y = features([0.0, 0.25]), labels([6, 2])
    empty = learner.snapshot_state()
    expected = count_action_observations(empty, x, y)
    learner.train_batch(x, y)
    x.fill(0.75)
    y.fill(1)
    object.__setattr__(declared, "max_payload_bytes", 2424)
    assert empty.entries == () and learner.snapshot_state() == expected
    child = learner.fork()
    target = ActionMemoryLearner(policy())
    target.restore_state(expected)
    child_state = child.snapshot_state()
    assert child_state == target.snapshot_state() == learner.snapshot_state()
    assert child_state is not expected and child_state.policy is not expected.policy
    assert child_state.entries is not expected.entries
    assert all(
        left is not right for left, right in zip(child_state.entries, expected.entries, strict=True)
    )
    object.__setattr__(expected.entries[0], "counts", (1, 0, 0, 0, 0, 0, 0))
    object.__setattr__(expected.policy, "max_payload_bytes", 2424)
    assert child.snapshot_state() == target.snapshot_state() == learner.snapshot_state()
    child.train_batch(features([0.5]), labels([4]))
    assert (
        child.training_examples == 3 and learner.training_examples == target.training_examples == 2
    )
    assert learner.policy.max_payload_bytes == 10752


@pytest.mark.parametrize("geometry", [(16, 7), (296, 5), (148, 7), (295, 7), (297, 7), (296, 6)])
def test_should_admit_only_the_two_frozen_geometry_pairs(geometry):
    with pytest.raises(ValueError):
        ActionMemoryPolicy(feature_dim=geometry[0], num_classes=geometry[1])
    with pytest.raises(ValueError):
        ActionMemoryEntry((0.0,) * geometry[0], (1,) + (0,) * (geometry[1] - 1))


@pytest.mark.parametrize(
    "field,value",
    [
        ("feature_dim", True),
        ("feature_dim", 296.0),
        ("feature_dim", np.int64(296)),
        ("num_classes", True),
        ("num_classes", 7.0),
        ("num_classes", np.int64(7)),
        ("max_entries", 65),
        ("max_entries", False),
        ("max_payload_bytes", 155137),
        ("max_payload_bytes", True),
        ("max_payload_bytes", 0),
    ],
)
def test_should_refuse_non_native_policy_types_and_geometry_specific_capacity_overflow(
    field, value
):
    options: dict[str, Any] = dict(feature_dim=296, num_classes=7)
    options[field] = value
    with pytest.raises(ValueError):
        ActionMemoryPolicy(**options)


class ArraySubclass(np.ndarray):
    pass


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "subclass",
        "float32",
        "bool",
        "old_width",
        "raw_width",
        "empty",
        "129rows",
        "nan",
        "inf",
        "range",
        "label_nan",
        "label_class7",
        "label_negative",
        "label_fraction",
        "label_float32",
        "label_bool",
        "label_vector",
        "label_row_mismatch",
    ],
)
def test_should_validate_every_minigrid_row_and_hard_label_before_any_commit(fault):
    learner = trained()
    before = learner.snapshot_state()
    x: Any = features([0.0, 0.5])
    y: Any = labels([0, 6])
    if fault == "list":
        x = x.tolist()
    elif fault == "subclass":
        x = x.view(ArraySubclass)
    elif fault in ("float32", "bool"):
        x = x.astype(np.float32 if fault == "float32" else bool)
    elif fault in ("old_width", "raw_width"):
        x = np.zeros((2, 16 if fault == "old_width" else 148), dtype=np.float64)
    elif fault in ("empty", "129rows"):
        rows = 0 if fault == "empty" else 129
        x, y = np.zeros((rows, 296)), np.zeros((rows, 1))
    elif fault in ("nan", "inf", "range"):
        x[1, -1] = {"nan": float("nan"), "inf": float("inf"), "range": 1.01}[fault]
    elif fault in ("label_nan", "label_class7", "label_negative", "label_fraction"):
        y[1, 0] = {
            "label_nan": float("nan"),
            "label_class7": 7,
            "label_negative": -1,
            "label_fraction": 0.5,
        }[fault]
    elif fault in ("label_float32", "label_bool"):
        y = y.astype(np.float32 if fault == "label_float32" else bool)
    elif fault == "label_vector":
        y = y[:, 0]
    else:
        y = y[:1]
    with pytest.raises(ValueError):
        learner.train_batch(x, y)
    assert learner.snapshot_state() == before
    if not fault.startswith("label_"):
        with pytest.raises(ValueError):
            learner.predict(x)
        assert learner.snapshot_state() == before


@pytest.mark.parametrize("old_target", [False, True])
def test_should_refuse_cross_geometry_restores_even_when_the_snapshot_is_empty(old_target):
    target = ActionMemoryLearner(ActionMemoryPolicy() if old_target else policy())
    foreign = ActionMemoryLearner(policy() if old_target else ActionMemoryPolicy())
    before = target.snapshot_state()
    with pytest.raises(ValueError):
        target.restore_state(foreign.snapshot_state())
    assert target.snapshot_state() == before


@pytest.mark.parametrize("foreign_geometry", [False, True])
def test_should_bind_every_snapshot_entry_to_its_policy_not_just_an_admissible_entry_shape(
    foreign_geometry,
):
    geometry = (16, 5) if foreign_geometry else (296, 7)
    entry = ActionMemoryEntry((0.0,) * geometry[0], (1,) + (0,) * (geometry[1] - 1))
    declared = policy() if foreign_geometry else ActionMemoryPolicy()
    invalid = ActionMemorySnapshot(declared, (entry,), 1, 1)
    with pytest.raises(ValueError, match="geometry"):
        copy_action_memory_snapshot(invalid)


@pytest.mark.parametrize(
    "fault",
    [
        "feature_numpy",
        "feature_bool",
        "feature_nan",
        "feature_list",
        "count_bool",
        "count_negative",
        "count_overflow",
        "count_list",
        "duplicate",
        "updates_bool",
        "examples_sum",
        "updates_overflow",
        "policy_extra",
        "entry_extra",
        "snapshot_extra",
    ],
)
def test_should_refuse_corrupted_large_geometry_restore_atomically(fault):
    learner = trained()
    before = learner.snapshot_state()
    invalid = deepcopy(before)
    entry = invalid.entries[0]
    if fault.startswith("feature_"):
        if fault == "feature_list":
            value: Any = list(entry.features)
        else:
            first: Any = {
                "feature_numpy": np.float64(0.0),
                "feature_bool": False,
                "feature_nan": float("nan"),
            }[fault]
            value = (first,) + entry.features[1:]
        object.__setattr__(entry, "features", value)
    elif fault.startswith("count_"):
        counts: Any = {
            "count_bool": (True, 0, 0, 0, 0, 0, 0),
            "count_negative": (-1, 0, 0, 0, 0, 0, 2),
            "count_overflow": (2**63, 0, 0, 0, 0, 0, 0),
            "count_list": list(entry.counts),
        }[fault]
        object.__setattr__(entry, "counts", counts)
    elif fault == "duplicate":
        object.__setattr__(invalid, "entries", (entry, entry))
    elif fault == "updates_bool":
        object.__setattr__(invalid, "completed_updates", True)
    elif fault == "examples_sum":
        object.__setattr__(invalid, "training_examples", 3)
    elif fault == "updates_overflow":
        object.__setattr__(invalid, "completed_updates", 2**53)
    else:
        owner = {"policy_extra": invalid.policy, "entry_extra": entry, "snapshot_extra": invalid}[
            fault
        ]
        object.__setattr__(owner, "extra", 1)
    with pytest.raises(ValueError):
        learner.restore_state(invalid)
    assert learner.snapshot_state() == before


def test_should_refuse_large_geometry_counter_overflow_before_count_or_clock_changes():
    maximum = 2**53 - 1
    snapshot = ActionMemorySnapshot(
        policy(), (ActionMemoryEntry((0.0,) * 296, (0, 0, 0, 0, 0, 0, maximum)),), maximum, maximum
    )
    learner = ActionMemoryLearner(policy())
    learner.restore_state(snapshot)
    before = learner.snapshot_state()
    with pytest.raises(ValueError, match="overflow"):
        learner.train_batch(features([0.0]), labels([6]))
    assert learner.snapshot_state() == before
    np.testing.assert_array_equal(
        predict_action_memory(before, features([0.0])), [[0, 0, 0, 0, 0, 0, 1]]
    )


def raw_frame(initial: float, step: int) -> tuple[float, ...]:
    values = [0.0] * 148
    values[0] = initial if step == 0 else 0.0
    values[-1] = step / 8
    return tuple(values)


def engineering_episode(
    context: str, initial: float, actions: tuple[int, ...]
) -> DemonstrationEpisode:
    """A declared array fixture with continuous transitions; no game or teacher."""
    return DemonstrationEpisode(
        "engineering-" + context,
        context,
        "train",
        tuple(
            DemonstrationTransition(
                step,
                raw_frame(initial, step),
                action,
                0.0,
                raw_frame(initial, step + 1),
                step == len(actions) - 1,
                7,
            )
            for step, action in enumerate(actions)
        ),
        7,
    )


def test_should_compose_public_initial_current_encoding_with_real_delayed_stream_and_owned_readouts():
    raw = (engineering_episode("A", 1.0, (6, 1, 4)), engineering_episode("B", -1.0, (2, 3, 5)))
    raw_before = pickle.dumps(raw)
    encoded = tuple(encode_public_history_episode(episode) for episode in raw)
    training = []
    for original, episode in zip(raw, encoded, strict=True):
        history = PublicInitialFrameHistory()
        history.reset(original.transitions[0].observation)
        for old, new in zip(original.transitions, episode.transitions, strict=True):
            assert new.observation == history.encode(old.observation)
        x, y = episode_arrays(episode)
        training.append(
            AttributionBatch(
                "train",
                episode.context,
                tuple(f"{episode.episode_id}:step:{step.step}" for step in episode.transitions),
                x,
                y,
                episode,
                7,
            )
        )
    data_before = pickle.dumps(training)
    source = ActionMemoryLearner(policy(max_payload_bytes=155136))
    before = source.snapshot_state()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 1.0)
    checkpoints: list[tuple[int, ActionMemorySnapshot]] = []
    prearrival: list[tuple[int, int]] = []

    def observe(point: int, snapshot: ActionMemorySnapshot, batch: AttributionBatch | None) -> None:
        assert (batch is None) == (point == 0)
        checkpoints.append((point, snapshot))

    def observe_features(observation: Any, candidate: Any) -> None:
        assert not hasattr(observation, "targets") and not hasattr(observation, "context")
        prearrival.append((observation.tick, candidate.state.completed_updates))

    stream = AttributionStream(
        source,
        tuple(training),
        (),
        checkpoint_updates=(0, 1, 2),
        label_delay=1,
        budget=budget,
        observe_checkpoint=observe,
        observe_features=observe_features,
        training_only=True,
    )
    assert source.snapshot_state() == before and budget.updates_completed == 0
    stream.run_until(1)
    assert prearrival == [(0, 0), (1, 0)]
    assert [point for point, _ in checkpoints] == [0, 1]
    assert budget.updates_completed == 1 and source.snapshot_state() == before
    stream.run_until(2)
    result = stream.finish()
    assert (result["completed_updates"], result["training_examples"], result["num_classes"]) == (
        2,
        6,
        7,
    )
    assert result["score_forward_calls"] == 0 and result["checkpoints"] == []
    assert result["evaluation_scope"] == "training_only" and budget.updates_completed == 2
    assert [row["observed_at"] for row in result["applied_updates"]] == [0, 1]
    assert [row["arrived_at"] for row in result["applied_updates"]] == [1, 2]
    assert [row["applied_at"] for row in result["applied_updates"]] == [1, 2]
    assert [point for point, _ in checkpoints] == [0, 1, 2]
    assert [len(state.entries) for _, state in checkpoints] == [0, 3, 6]
    readouts = []
    for _, state in checkpoints:
        owned = source.fork()
        owned.restore_state(state)
        readouts.append(
            [
                probability_scores(owned.predict(batch.features.copy()), batch.targets.copy(), 7)
                for batch in training
            ]
        )
        assert owned.snapshot_state() == state
    assert readouts[0][0]["ce"] == pytest.approx(np.log(7))
    assert readouts[1][0] == {"ce": 0.0, "accuracy": 1.0}
    assert readouts[1][1]["ce"] == pytest.approx(np.log(7))
    assert readouts[2] == [{"ce": 0.0, "accuracy": 1.0}] * 2
    assert source.snapshot_state() == before
    assert pickle.dumps(raw) == raw_before and pickle.dumps(training) == data_before


@pytest.mark.parametrize("role", ["final_test", "probe_train", "probe_test"])
def test_should_refuse_reserved_episode_roles_before_fork_or_storage_work(role, monkeypatch):
    raw = engineering_episode("A", 1.0, (6, 1, 4))
    reserved = DemonstrationEpisode(raw.episode_id, raw.context, role, raw.transitions, 7)
    episode = encode_public_history_episode(reserved)
    x, y = episode_arrays(episode)
    packet = AttributionBatch(
        role,
        "A",
        tuple(f"{episode.episode_id}:step:{step.step}" for step in episode.transitions),
        x,
        y,
        episode,
        7,
    )
    source = ActionMemoryLearner(policy())
    before = source.snapshot_state()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 1.0)

    def forbidden_fork():
        pytest.fail("reserved data reached learner construction")

    monkeypatch.setattr(source, "fork", forbidden_fork)
    with pytest.raises(ValueError, match="train"):
        AttributionStream(
            source,
            (packet,),
            (),
            checkpoint_updates=(0, 1),
            label_delay=1,
            budget=budget,
            training_only=True,
        )
    assert source.snapshot_state() == before and budget.updates_completed == 0
