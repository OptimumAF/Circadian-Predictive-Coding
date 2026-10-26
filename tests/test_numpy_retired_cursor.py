"""Historical inbox DATA counts are selected before the real cursor validator."""

from types import SimpleNamespace
from typing import Any

import pytest

from src.adapters.numpy_restored_capture_example import _detached_cursor
from src.core.experience import AppliedExperience, Experience, ExperiencePermissions, LabelArrival
from src.core.inbox_cursor import InboxCursor
from src.core.learner_ports import TrainingDiagnostic


def cursor_data(pair_count: int, historical: int | None):
    experiences: dict[tuple[str, str], Experience[int]] = {}
    labels: dict[tuple[str, str], LabelArrival[int]] = {}
    applied: dict[tuple[str, str], AppliedExperience] = {}
    for number in range(1, pair_count + 1):
        source = Experience(
            f"s{number}",
            "e1",
            number * 2 - 1,
            "actor-0",
            number,
            "train",
            ExperiencePermissions(True, True),
        )
        label = LabelArrival(
            f"label-{number}", source.sample_id, "e1", number * 2, "actor-0", number
        )
        receipt = AppliedExperience(
            source.sample_id,
            "e1",
            label.event_id,
            "actor-0",
            "candidate-0",
            source.observed_at,
            label.arrived_at,
            label.arrived_at,
            number,
            TrainingDiagnostic("engineering_fixture", 0.0),
        )
        experiences[source.key], labels[source.key], applied[source.key] = source, label, receipt
    return {
        "_learner_version": "candidate-0",
        "_capacity": 4,
        "_experiences": experiences,
        "_labels": labels,
        "_applied": applied,
        "_last_time": pair_count * 2,
        "_stopped": False,
        "_erased": {},
        "_historical_completed_updates": historical,
    }


def captured_budget():
    return SimpleNamespace(
        state=SimpleNamespace(budget=SimpleNamespace(fields=(("updates_completed", 3),)))
    )


@pytest.mark.parametrize("historical,pairs,expected", ((1, 1, 1), (None, 3, 3)))
def test_should_validate_retired_and_current_data_against_their_actual_completed_history(
    historical,
    pairs,
    expected,
):
    captured, data = captured_budget(), cursor_data(pairs, historical)
    cursor = _detached_cursor(captured, data)
    assert type(cursor) is InboxCursor
    assert cursor.completed_updates == expected and len(cursor.applied) == pairs
    assert cursor.format_version == 1 and cursor.learner_version == "candidate-0"
    assert cursor.capacity == 4 and cursor.last_tick == pairs * 2 and not cursor.stopped
    assert cursor.experiences == tuple(data["_experiences"].values())
    assert cursor.labels == tuple(data["_labels"].values())
    assert cursor.applied == tuple(data["_applied"].values())
    assert [item.update_number for item in cursor.applied] == list(range(1, pairs + 1))
    assert not cursor.erased
    assert dict(captured.state.budget.fields)["updates_completed"] == 3
    assert data["_historical_completed_updates"] is historical


@pytest.mark.parametrize("historical", (0, True, 2))
def test_should_refuse_inconsistent_or_noninteger_historical_counts_without_repairing_the_cursor(
    historical,
):
    captured, data = captured_budget(), cursor_data(1, historical)
    before: Any = data["_applied"]
    with pytest.raises(ValueError):
        _detached_cursor(captured, data)
    assert data["_applied"] is before and len(before) == 1
    assert data["_historical_completed_updates"] is historical
    assert dict(captured.state.budget.fields)["updates_completed"] == 3
