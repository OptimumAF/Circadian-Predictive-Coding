"""Partial order only: counts and trace positions are not wall-clock release proof."""

import pytest
from typing import Any

from src.core.seed_release_chronology import recorded_release_order


def test_should_place_all_source_reads_before_their_releases_and_all_releases_before_predictions() -> (
    None
):
    nodes = recorded_release_order(2, 3)
    assert [(row.kind, row.record_index) for row in nodes] == [
        ("training_before_release", None),
        ("source_input_read", 0),
        ("source_target_read", 1),
        ("role_release", 0),
        ("source_input_read", 2),
        ("source_target_read", 3),
        ("role_release", 1),
        ("training_after_release", None),
        ("prediction", 0),
        ("prediction", 1),
        ("prediction", 2),
        ("training_after_evaluation", None),
    ]
    assert [row.ordinal for row in nodes] == list(range(12))


def test_should_cover_every_original_fixed_trace_position_without_freshness_or_time_fields() -> (
    None
):
    nodes = recorded_release_order(120, 1680)
    assert len(nodes) == 2043
    assert nodes[360].kind == "role_release" and nodes[360].record_index == 119
    assert nodes[361].kind == "training_after_release"
    assert nodes[-2].kind == "prediction" and nodes[-2].record_index == 1679
    assert nodes[-1].kind == "training_after_evaluation"
    assert all(
        not hasattr(row, "released_utc") and not hasattr(row, "fresh_roles_authorized")
        for row in nodes
    )


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, "1", None])
@pytest.mark.parametrize("position", [0, 1])
def test_should_refuse_boolean_empty_negative_or_inexact_event_counts(
    value: object, position: int
) -> None:
    counts: list[Any] = [1, 1]
    counts[position] = value
    with pytest.raises(ValueError, match="release chronology"):
        recorded_release_order(*counts)
