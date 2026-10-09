"""Payload erasure primitives preserve work and consumed identities."""

from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.core.data_erasure import ErasedExperience, ReplayPayloadErasure
from src.core.inbox_cursor import validate_inbox_cursor
from test_experience_inbox import experience, label, setup, make_native_pair
from test_resource_sharing import shared
from test_candidate_checkpoint import controller


@pytest.mark.parametrize("delivery", ["source", "label", "both"])
@pytest.mark.parametrize("applied", [False, True])
def test_should_erase_payloads_preserve_identity_and_keep_work_ledger(delivery, applied):
    inbox, clock, learner, budget = setup(max_experiences=1)
    if delivery in ("source", "both"):
        inbox.record_experience(experience())
    if delivery in ("label", "both"):
        inbox.record_label(label())
    clock.advance_to(3)
    if applied:
        inbox.drain()
    work = budget.updates_completed
    receipts = inbox.applied_updates

    class CannotCopy:
        def __deepcopy__(self, memo):
            raise AssertionError("erasure accessed payload")

    if inbox._experiences:
        object.__setattr__(inbox._experiences[("e1", "s1")], "features", CannotCopy())
    if inbox._labels:
        object.__setattr__(inbox._labels[("e1", "s1")], "targets", CannotCopy())
    (tombstone,) = inbox._erase_payloads((("e1", "s1"),), reason="deleted")
    assert type(tombstone) is ErasedExperience and tombstone.key == ("e1", "s1")
    assert inbox._experiences == inbox._labels == {}
    assert (
        inbox.drain() == ()
        and inbox.applied_updates == receipts
        and budget.updates_completed == work
    )
    assert inbox._erase_payloads((("e1", "s1"),), reason="expired") == (tombstone,)
    cursor = inbox.capture_cursor()
    assert (
        cursor.format_version == 2 and cursor.erased == (tombstone,) and cursor.applied == receipts
    )
    validate_inbox_cursor(pickle.loads(pickle.dumps(cursor)))
    for event, method in ((experience(), inbox.record_experience), (label(), inbox.record_label)):
        with pytest.raises(ValueError, match="duplicate"):
            method(event)
    with pytest.raises(ValueError, match="capacity"):
        inbox.record_experience(experience("new"))
    if delivery in ("label", "both"):
        with pytest.raises(ValueError, match="duplicate"):
            inbox.record_label(label("new", event_id=label().event_id))


@pytest.mark.parametrize("kind", ["unknown", "duplicate", "mutable", "reason", "clock"])
def test_should_refuse_invalid_erasure_before_partial_removal(kind):
    inbox, clock, _, _ = setup()
    inbox.record_experience(experience())
    before = inbox.capture_cursor()
    keys: Any = (("e1", "s1"),)
    reason: Any = "deleted"
    if kind == "unknown":
        keys += (("e1", "unknown"),)
    if kind == "duplicate":
        keys *= 2
    if kind == "mutable":
        keys = list(keys)
    if kind == "reason":
        reason = "unknown"
    if kind == "clock":
        clock.advance_to(3)
        inbox.drain()
        clock._time = 0
    with pytest.raises(ValueError):
        inbox._erase_payloads(keys, reason=reason)
    assert inbox._experiences.keys() == {("e1", "s1")} and inbox._erased == {}
    if kind != "clock":
        assert inbox.capture_cursor() == before


def test_should_allow_quiescent_payload_removal_from_stopped_uncertain_owner():
    inbox, clock, learner, budget = setup()
    inbox.record_experience(experience())
    inbox.record_label(label())
    clock.advance_to(3)
    learner.train_batch = lambda x, y: (_ for _ in ()).throw(RuntimeError("uncertain"))
    with pytest.raises(RuntimeError):
        inbox.drain()
    assert inbox.stopped
    inbox._erase_payloads((("e1", "s1"),), reason="opt_out")
    assert inbox.capture_cursor().stopped and budget.updates_completed == 0
    with pytest.raises(ValueError, match="stopped"):
        inbox.drain()


def test_should_refuse_erasure_during_native_drain_reentrance():
    inbox, clock, learner, _ = setup()
    inbox.record_experience(experience())
    inbox.record_label(label())
    clock.advance_to(3)
    original = learner.train_batch

    def train(features, targets):
        with pytest.raises(ValueError, match="quiescent"):
            inbox._erase_payloads((("e1", "s1"),), reason="deleted")
        return original(features, targets)

    learner.train_batch = train
    assert len(inbox.drain()) == 1 and inbox._erased == {}


def tombstone():
    return ErasedExperience(("e1", "s1"), "actor-0", 1, "event", 3, 3, "deleted")


@pytest.mark.parametrize(
    "changes",
    [
        {"key": ("e1", "")},
        {"key": ["e1", "s1"]},
        {"actor_version": ""},
        {"observed_at": True},
        {"arrived_at": -1},
        {"erased_at": False},
        {"reason": "unknown"},
        {"event_id": None},
        {"arrived_at": None},
        {"observed_at": 4},
        {"observed_at": None, "arrived_at": None, "event_id": None},
    ],
)
def test_should_validate_exact_payload_free_tombstone_metadata(changes):
    with pytest.raises(ValueError):
        replace(tombstone(), **changes)


@pytest.mark.parametrize(
    "kind",
    [
        "legacy",
        "duplicate",
        "live_overlap",
        "event_collision",
        "receipt",
        "future_erasure",
        "missing_source",
        "missing_label",
    ],
)
def test_should_refuse_corrupt_erased_cursor_before_payload_copy(kind):
    inbox, clock, _, _ = setup()
    inbox.record_experience(experience())
    inbox.record_label(label())
    clock.advance_to(3)
    inbox.drain()
    inbox._erase_payloads((("e1", "s1"),), reason="deleted")
    cursor = inbox.capture_cursor()
    changes: dict[str, Any] = {}
    if kind == "legacy":
        changes["format_version"] = 1
    if kind == "duplicate":
        changes["erased"] = cursor.erased * 2
    if kind == "live_overlap":
        changes["experiences"] = (experience(),)
    if kind == "event_collision":
        changes["labels"] = (label("other", event_id=label().event_id),)
    if kind == "receipt":
        changes["applied"] = (replace(cursor.applied[0], event_id="other"),)
    if kind == "future_erasure":
        changes["erased"] = (replace(cursor.erased[0], erased_at=4),)
    if kind == "missing_source":
        changes["erased"] = (replace(cursor.erased[0], observed_at=None),)
    if kind == "missing_label":
        changes["erased"] = (replace(cursor.erased[0], arrived_at=None, event_id=None),)
    with pytest.raises(ValueError):
        replace(cursor, **changes)


def test_should_transfer_tombstone_cursor_with_original_clock_budget_and_hooks():
    from src.app.managed_experience import ManagedExperienceOwner
    from src.core.data_lifecycle import LifecycleLimits
    from test_managed_experience import declaration, record

    owner, old, clock, gate, _, budget = shared()
    managed = ManagedExperienceOwner(owner, limits=LifecycleLimits(4, 4))
    managed.declare(declaration())
    record(managed)
    clock.advance_to(3)
    owner.train_ready()
    with old._exclusive():
        old._inbox._erase_payloads((("e1", "s1"),), reason="deleted")
        old._revision += 1
    gate.pause()
    control = controller(owner)
    new = control.restore(control.capture())
    assert (
        new._inbox._erased == old._inbox._erased
        and new._inbox._experiences == new._inbox._labels == {}
    )
    assert new._inbox._clock is clock and new._budget is budget
    assert new._inbox._registration_guard is old._inbox._registration_guard
    gate.resume()
    assert owner.train_ready().updates == () and budget.updates_completed == 1
    with pytest.raises(ValueError, match="duplicate"):
        record(managed)


def test_should_leave_unerased_legacy_cursor_format_one_unchanged():
    inbox, _, _, _ = setup()
    inbox.record_experience(experience())
    cursor = inbox.capture_cursor()
    assert cursor.format_version == 1 and cursor.erased == ()
    with pytest.raises(ValueError):
        replace(cursor, format_version=2)


@pytest.mark.parametrize(
    "kind", ["source_index", "label_index", "event_index", "erased_index", "receipt"]
)
def test_should_refuse_corrupt_owned_history_before_dropping_payloads(kind):
    inbox, clock, _, _ = setup()
    inbox.record_experience(experience())
    inbox.record_label(label())
    clock.advance_to(3)
    inbox.drain()
    if kind == "source_index":
        inbox._experiences[("e1", "wrong")] = inbox._experiences.pop(("e1", "s1"))
    if kind == "label_index":
        inbox._labels[("e1", "wrong")] = inbox._labels.pop(("e1", "s1"))
    if kind == "event_index":
        inbox._event_ids.add("unknown")
    if kind == "erased_index":
        inbox._erased[("e1", "wrong")] = tombstone()
    if kind == "receipt":
        inbox._applied[("e1", "s1")] = replace(inbox._applied[("e1", "s1")], event_id="unknown")
    before = (dict(inbox._experiences), dict(inbox._labels), dict(inbox._erased))
    with pytest.raises(ValueError):
        inbox._erase_payloads((("e1", "s1"),), reason="deleted")
    assert (inbox._experiences, inbox._labels, inbox._erased) == before


@pytest.mark.parametrize("kind", ["list", "item", "features", "targets"])
def test_should_refuse_corrupt_native_replay_buffer_without_partial_erasure(kind):
    from collections import deque
    from src.core.circadian_predictive_coding import ReplaySnapshot

    native, _ = make_native_pair("circadian")
    item = ReplaySnapshot(np.array([[0.3, -0.2]]), np.array([[1.0]]), 1.0, 1.0)
    if kind == "features":
        item = replace(item, input_batch=np.ones((1, 3)))
    if kind == "targets":
        item = replace(item, target_batch=np.ones((1, 2)))
    native._replay_memory = (
        [item] if kind == "list" else deque([object() if kind == "item" else item])
    )
    before = native._replay_memory
    with pytest.raises(ValueError):
        native.erase_replay_payloads()
    assert native._replay_memory is before and len(before) == 1


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
def test_should_erase_all_supported_real_numeric_replay_dtypes(dtype):
    from src.core.circadian_predictive_coding import ReplaySnapshot

    native, _ = make_native_pair("circadian")
    features = np.array([[1, 0]], dtype=dtype)
    targets = np.array([[1]], dtype=dtype)
    native._replay_memory.append(ReplaySnapshot(features, targets, 1.0, 1.0))
    assert native.erase_replay_payloads() == ReplayPayloadErasure(
        1, 1, features.nbytes + targets.nbytes
    )


@pytest.mark.parametrize("counts", [(-1, 0, 0), (0, True, 0), (0, 0, 1.0)])
def test_should_reject_invalid_native_erasure_counts(counts):
    with pytest.raises(ValueError):
        ReplayPayloadErasure(*counts)


@pytest.mark.parametrize(
    "mode", ["backprop", "default", "budget", "content_hash", "recent_fifo", "seeded_reservoir"]
)
def test_should_erase_native_replay_payloads_preserve_complete_other_state_and_budget(mode):
    from src.app.learner_step import update_learner
    from src.app.toy_execution_budget import (
        ToyBudgetSession,
        ToyExecutionBudget,
        ToyExecutionStopped,
    )
    from src.core.circadian_predictive_coding import ReplayRetentionBudget
    from src.core.replay_retention import ReplayRetentionPolicy

    native, learner = make_native_pair("backprop" if mode == "backprop" else "circadian")
    if mode not in ("backprop", "default"):
        policy = (
            None
            if mode == "budget"
            else ReplayRetentionPolicy(mode, 9012 if mode == "seeded_reservoir" else None)
        )
        native.configure_replay_retention(ReplayRetentionBudget(2, 128), policy=policy)
        learner = type(learner)(
            native, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.2
        )
    source_before = pickle.dumps(native.__dict__)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    update_learner(learner, np.array([[0.3, -0.2], [-0.5, 0.4]]), np.array([[1.0], [0.0]]), budget)
    state_before = learner.snapshot_state()
    raw_before = learner._model.__dict__.copy()
    memory = raw_before.pop("_replay_memory", None)
    other_before = pickle.dumps(raw_before)
    expected = (
        ReplayPayloadErasure(0, 0, 0)
        if mode == "backprop"
        else ReplayPayloadErasure(len(memory), 2, 48)
    )
    assert learner.erase_replay_payloads() == expected
    raw_after = learner._model.__dict__.copy()
    remaining = raw_after.pop("_replay_memory", None)
    assert pickle.dumps(raw_after) == other_before
    assert remaining is None or not remaining
    assert learner.erase_replay_payloads() == ReplayPayloadErasure(0, 0, 0)
    state_after = learner.snapshot_state()
    learner.restore_state(state_after)
    assert pickle.dumps(learner.snapshot_state()) == pickle.dumps(state_after)
    assert pickle.dumps(native.__dict__) == source_before and budget.updates_completed == 1
    with pytest.raises(ToyExecutionStopped):
        update_learner(learner, np.array([[0.3, -0.2]]), np.array([[1.0]]), budget)
    assert budget.updates_completed == 1
    assert state_before is not state_after  # caller-held snapshot is still outside erasure
