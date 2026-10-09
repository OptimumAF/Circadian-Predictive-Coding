"""P1: sixteen scalar partition cases; no native work, payloads or admission."""

from dataclasses import replace
from typing import Any

import pytest

from src.app.checkpoint_erased_inbox import require_erased_inbox_records, require_inbox_partition
from src.app.checkpoint_untrained_inbox import require_untrained_inbox_records
from src.app.untrained_inbox_origins import prepare_untrained_inbox_origin
from src.core.data_erasure import ErasedExperience
from src.core.untrained_inbox_origin import UntrainedInboxOriginData
from test_checkpoint_erased_inbox import _case as _trained_case


def _untrained(limits, form="pair", sample="third", event="event3", **changes):
    observed = None if form == "label" else 0
    arrived = None if form == "source" else 1
    data = UntrainedInboxOriginData(
        ("episode", sample),
        "actor",
        "learner",
        "subject",
        "source",
        observed,
        None if form == "source" else event,
        arrived,
        4,
        "deleted",
        1,
    )
    data = replace(data, **changes)
    tombstone = ErasedExperience(
        data.key,
        data.actor_version,
        data.observed_at,
        data.event_id,
        data.arrived_at,
        data.erased_at,
        data.reason,
    )
    return prepare_untrained_inbox_origin(data, tombstone, limits), tombstone


def _case(form="pair"):
    cursor, live, erased, limits = _trained_case()
    record, tombstone = _untrained(limits, form)
    cursor = replace(cursor, erased=cursor.erased + (tombstone,))
    return cursor, live, erased, (record,), limits, (tombstone,)


@pytest.mark.parametrize("form", ["source", "label", "pair"])
def test_should_cover_untrained_tombstone_beside_live_and_erased_trained_history(form):
    cursor, live, erased, untrained, limits, owners = _case(form)
    before = cursor.applied

    require_inbox_partition(cursor, live, erased, limits, 4, untrained=untrained)

    assert cursor.applied is before and len(before) == 2
    assert before[0] is erased[0].receipt() and before[1] is live[0]
    assert untrained[0].data.completed_updates == 1
    assert untrained[0].data.key not in tuple((row.episode_id, row.sample_id) for row in before)
    assert untrained[0].tombstone() is owners[0] is cursor.erased[-1]


@pytest.mark.parametrize(
    "mode",
    [
        "missing_witness",
        "extra_witness",
        "clone_tombstone",
        "live_overlap",
        "trained_overlap",
        "duplicate_event",
        "work_boundary",
        "learner",
        "erase_time",
        "typed_tuple",
        "combined_capacity",
        "limits",
        "aggregate",
    ],
)
def test_should_refuse_incomplete_overlapping_or_unbounded_three_partition_history(mode):
    cursor, live, erased, originals, limits, owners = _case()
    untrained: Any = originals
    if mode == "missing_witness":
        untrained = ()
    elif mode == "extra_witness":
        extra, extra_owner = _untrained(limits, sample="fourth", event="event4")
        untrained = originals + (extra,)
        owners += (extra_owner,)
    elif mode == "clone_tombstone":
        clone = replace(owners[0])
        object.__setattr__(cursor, "erased", cursor.erased[:-1] + (clone,))
        owners += (clone,)
    elif mode in ("live_overlap", "trained_overlap", "duplicate_event"):
        sample = (
            "second"
            if mode == "live_overlap"
            else "first"
            if mode == "trained_overlap"
            else "third"
        )
        event = "event2" if mode == "duplicate_event" else "event3"
        changed, changed_owner = _untrained(limits, sample=sample, event=event)
        untrained = (changed,)
        owners += (changed_owner,)
        # Corrupt current inventory after construction, so the partition gate
        # itself must reject an otherwise sealed scalar witness's overlap.
        object.__setattr__(cursor, "erased", cursor.erased[:-1] + (changed_owner,))
    elif mode in ("work_boundary", "learner", "erase_time"):
        changes = (
            {"completed_updates": 3}
            if mode == "work_boundary"
            else ({"learner_version": "other"} if mode == "learner" else {"erased_at": 9})
        )
        changed, changed_owner = _untrained(limits, **changes)
        untrained = (changed,)
        owners += (changed_owner,)
        object.__setattr__(cursor, "erased", cursor.erased[:-1] + (changed_owner,))
    elif mode == "typed_tuple":
        untrained = list(originals)
    elif mode == "combined_capacity":
        object.__setattr__(cursor, "capacity", 2)
    elif mode == "limits":
        object.__setattr__(limits, "max_depth", True)
    else:
        object.__setattr__(limits, "max_nodes", 128)
        # Each complete witness collection fits on its own. Their combined
        # live/erased-trained/untrained scalar proof must share the same bound.
        require_erased_inbox_records(erased, limits, 4)
        require_untrained_inbox_records(originals, limits, 4)
    assert owners  # Retain every weak original and replacement tombstone.

    with pytest.raises(ValueError):
        require_inbox_partition(cursor, live, erased, limits, 4, untrained=untrained)
