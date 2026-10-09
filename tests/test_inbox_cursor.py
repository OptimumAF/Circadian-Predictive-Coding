"""Complete inbox metadata/ownership capture before native checkpoint handoff."""

from contextlib import contextmanager
from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.core.inbox_cursor import InboxCursor, validate_inbox_cursor
from test_experience_inbox import experience, label, make_native_pair, setup


def populated():
    inbox, clock, _, _ = setup()
    for sample in ("s1", "s2"):
        inbox.record_experience(experience(sample))
        inbox.record_label(label(sample))
    inbox.record_label(label("future", arrived_at=5))
    clock.advance_to(3)
    inbox.drain(max_updates=1)
    return inbox, clock


def test_should_capture_full_pending_applied_label_first_and_duplicate_history():
    inbox, _ = populated()
    cursor = inbox.capture_cursor()
    assert type(cursor) is InboxCursor
    assert cursor.capacity == 8 and cursor.last_tick == 3 and not cursor.stopped
    assert cursor.completed_updates == 1 and cursor.learner_version == "candidate-0"
    assert [s.sample_id for s in cursor.experiences] == ["s1", "s2"]
    assert [l.sample_id for l in cursor.labels] == ["future", "s1", "s2"]
    assert [a.sample_id for a in cursor.applied] == ["s1"]
    roundtrip = pickle.loads(pickle.dumps(cursor))
    validate_inbox_cursor(roundtrip)
    assert roundtrip == cursor
    inbox.drain()
    assert cursor.completed_updates == 1 and len(inbox.capture_cursor().applied) == 2
    with pytest.raises(ValueError, match="duplicate"):
        inbox.record_label(label("s1"))


def test_should_detach_all_payloads_without_changing_native_or_budget_state():
    inbox, _ = populated()
    before = inbox._learner.snapshot_state(), inbox._budget.updates_completed
    cursor = inbox.capture_cursor()
    cursor.experiences[0].features.append("mutation")
    cursor.labels[0].targets.append("mutation")
    assert "mutation" not in inbox.capture_cursor().experiences[0].features
    assert "mutation" not in inbox.capture_cursor().labels[0].targets
    assert (inbox._learner.snapshot_state(), inbox._budget.updates_completed) == before


@pytest.mark.parametrize(
    "changes",
    [
        {"format_version": 2},
        {"format_version": True},
        {"capacity": 0},
        {"last_tick": True},
        {"last_tick": -1},
        {"stopped": 1},
        {"completed_updates": -1},
        {"completed_updates": 0},
        {"completed_updates": 2},
        {"experiences": []},
        {"learner_version": " "},
    ],
)
def test_should_refuse_bad_cursor_header_or_observed_work_count(changes):
    inbox, _ = populated()
    with pytest.raises(ValueError):
        replace(inbox.capture_cursor(), **changes)


@pytest.mark.parametrize(
    "kind",
    [
        "source",
        "label",
        "event",
        "applied",
        "missing_source",
        "missing_label",
        "pair_version",
        "pair_time",
        "receipt_version",
        "receipt_time",
        "receipt_number",
        "diagnostic",
    ],
)
def test_should_refuse_corrupt_duplicate_pair_or_applied_metadata(kind):
    inbox, _ = populated()
    cursor = inbox.capture_cursor()
    changes: dict[str, Any] = {}
    if kind == "source":
        changes["experiences"] = cursor.experiences + (cursor.experiences[0],)
    if kind == "label":
        changes["labels"] = cursor.labels + (cursor.labels[0],)
    if kind == "event":
        changes["labels"] = (
            replace(cursor.labels[0], event_id=cursor.labels[1].event_id),
            *cursor.labels[1:],
        )
    if kind == "applied":
        changes["applied"] = cursor.applied * 2
    if kind == "missing_source":
        changes["experiences"] = cursor.experiences[1:]
    if kind == "missing_label":
        changes["labels"] = (cursor.labels[0], cursor.labels[2])
    if kind == "pair_version":
        changes["labels"] = (
            cursor.labels[0],
            replace(cursor.labels[1], model_version="other"),
            cursor.labels[2],
        )
    if kind == "pair_time":
        changes["labels"] = (
            cursor.labels[0],
            replace(cursor.labels[1], arrived_at=0),
            cursor.labels[2],
        )
    if kind == "receipt_version":
        changes["applied"] = (replace(cursor.applied[0], learner_version="other"),)
    if kind == "receipt_time":
        changes["applied"] = (replace(cursor.applied[0], applied_at=9),)
    if kind == "receipt_number":
        changes["applied"] = (replace(cursor.applied[0], update_number=2),)
    if kind == "diagnostic":
        changes["applied"] = (replace(cursor.applied[0], diagnostic=None),)
    with pytest.raises(ValueError):
        replace(cursor, **changes)


def test_should_refuse_heldout_or_unpermitted_metadata_before_copying_poisoned_payload():
    class CannotCopy:
        def __deepcopy__(self, memo):
            raise AssertionError("unauthorized payload copied")

    from src.core.experience import ExperiencePermissions

    inbox, _ = populated()
    original = inbox._experiences[("e1", "s2")]
    inbox._experiences[original.key] = replace(
        original, features=CannotCopy(), permissions=ExperiencePermissions()
    )
    with pytest.raises(ValueError, match="training"):
        inbox.capture_cursor()
    inbox._experiences[original.key] = original
    inbox._labels[("e1", "future")] = replace(
        inbox._labels[("e1", "future")], role="final_test", targets=CannotCopy()
    )
    with pytest.raises(ValueError, match="train"):
        inbox.capture_cursor()


def test_should_refuse_capture_inside_an_active_drain_and_keep_native_cursor():
    inbox, _ = populated()

    @contextmanager
    def admission():
        with pytest.raises(ValueError, match="drain"):
            inbox.capture_cursor()
        yield False

    assert inbox.drain(before_each_update=admission) == ()
    assert inbox.capture_cursor().completed_updates == 1


def test_should_capture_stopped_unknown_native_failure_without_resume_authority():
    inbox, _ = populated()

    def fail(features, targets):
        raise RuntimeError("native uncertain")

    inbox._learner.train_batch = fail
    with pytest.raises(RuntimeError):
        inbox.drain()
    cursor = inbox.capture_cursor()
    assert cursor.stopped and cursor.completed_updates == 1
    assert [a.sample_id for a in cursor.applied] == ["s1"]
    with pytest.raises(ValueError, match="stopped"):
        inbox.drain()


def test_should_preserve_empty_cursor_and_refuse_capacity_excess_or_foreign_type():
    inbox, _, _, _ = setup(max_experiences=1)
    assert inbox.capture_cursor().experiences == ()
    inbox.record_label(label("s1"))
    cursor = inbox.capture_cursor()
    with pytest.raises(ValueError, match="capacity"):
        replace(cursor, labels=(label("s1"), label("s2")))
    with pytest.raises(ValueError):
        validate_inbox_cursor(object())


@pytest.mark.parametrize("index", ["events", "sources", "applied"])
def test_should_refuse_inconsistent_owner_duplicate_indexes(index):
    inbox, _ = populated()
    if index == "events":
        inbox._event_ids.add("orphan")
    if index == "sources":
        inbox._experiences[("wrong", "key")] = inbox._experiences.pop(("e1", "s1"))
    if index == "applied":
        inbox._applied[("wrong", "key")] = inbox._applied.pop(("e1", "s1"))
    with pytest.raises(ValueError, match="indexes"):
        inbox.capture_cursor()


def test_should_revalidate_mutated_permission_flags_without_copying_payload():
    inbox, _ = populated()
    permission = inbox._experiences[("e1", "s2")].permissions
    object.__setattr__(permission, "training", 1)
    with pytest.raises(ValueError, match="boolean"):
        inbox.capture_cursor()


@pytest.mark.parametrize("backwards", [False, True])
def test_should_preserve_equal_applied_ticks_and_refuse_backward_receipt_chronology(backwards):
    inbox, clock = populated()
    clock.advance_to(5)
    inbox.drain()
    cursor = inbox.capture_cursor()
    first, second = cursor.applied
    if backwards:
        # Each tick separately fits its label and cursor, yet sequence5 -> 3 is impossible.
        with pytest.raises(ValueError, match="chronology"):
            replace(cursor, applied=(replace(first, applied_at=5), replace(second, applied_at=3)))
    else:
        equal = replace(
            cursor, applied=(replace(first, applied_at=5), replace(second, applied_at=5))
        )
        validate_inbox_cursor(equal)


@pytest.mark.parametrize("kind", ["backprop", "circadian"])
def test_should_capture_full_native_arrived_cursor_without_extra_native_work(kind):
    _, learner = make_native_pair(kind)
    inbox, clock, _, budget = setup(learner=learner)
    features = np.array([[0.3, -0.2], [-0.5, 0.4]])
    targets = np.array([[1.0], [0.0]])
    inbox.record_experience(replace(experience(), features=features))
    inbox.record_label(replace(label(), targets=targets))
    clock.advance_to(3)
    inbox.drain()  # one new-fixture native wake per kind/capture
    before = pickle.dumps(learner.snapshot_state())
    cursor = inbox.capture_cursor()
    assert cursor.completed_updates == budget.updates_completed == 1
    assert cursor.applied[0] == inbox.applied_updates[0]
    cursor.experiences[0].features[:] = 99
    assert pickle.dumps(learner.snapshot_state()) == before
    np.testing.assert_array_equal(inbox.capture_cursor().labels[0].targets, targets)
