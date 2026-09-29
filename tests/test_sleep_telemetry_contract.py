"""The sleep-event record is typed, JSON-safe, and rejects contradictory facts."""

from __future__ import annotations

from dataclasses import asdict, FrozenInstanceError, replace
import json
from typing import Any

import pytest

from src.core.sleep_telemetry import (
    ChemicalSummary,
    ChemicalSummaries,
    SleepBudgets,
    SleepDurations,
    SleepEventTelemetry,
    SleepGuardMetrics,
    SleepReplayUsage,
    SleepStructuralChanges,
)


def _chemistry(count: int) -> ChemicalSummaries:
    scalar = ChemicalSummary(count=count, minimum=0.1, mean=0.2, maximum=0.3)
    return ChemicalSummaries(primary=scalar, fast=scalar, slow=scalar)


def _accepted() -> SleepEventTelemetry:
    return SleepEventTelemetry(
        format_version=1,
        trigger_reason="periodic",
        outcome="accepted",
        reason="guard_accepted",
        completed_epoch=2,
        wake_batches=4,
        budgets=SleepBudgets(
            split_limit=1, prune_limit=0, replay_update_limit=0, time_limit_seconds=5.0
        ),
        changes=SleepStructuralChanges(
            proposed_split_pairs=((10, 20),),
            applied_split_pairs=((10, 20),),
            proposed_prune_ids=(),
            proposed_scheduled_prune_ids=(),
            proposed_removed_prune_ids=(),
            applied_scheduled_prune_ids=(),
            applied_removed_prune_ids=(),
        ),
        before_width=4,
        proposed_width=5,
        final_width=5,
        guard=SleepGuardMetrics(
            role="inner_guard",
            metric_name="accuracy",
            pre_accuracy=0.5,
            post_accuracy=0.6,
            pre_cross_entropy=1.0,
            post_cross_entropy=0.8,
            delta=-0.1,
            tolerance=0.1,
            examples_scored=8,
        ),
        replay=SleepReplayUsage(
            proposed_examples=0, proposed_updates=0, applied_examples=0, applied_updates=0
        ),
        chemistry_before=_chemistry(4),
        chemistry_proposed=_chemistry(5),
        chemistry_final=_chemistry(5),
        durations=SleepDurations(core_seconds=0.01, attempt_seconds=0.03),
    )


def test_accepted_sleep_record_is_immutable_and_json_safe() -> None:
    event = _accepted()

    encoded = json.dumps(asdict(event), sort_keys=True, allow_nan=False)

    assert json.loads(encoded)["changes"]["applied_split_pairs"] == [[10, 20]]
    assert json.loads(encoded)["guard"]["role"] == "inner_guard"
    with pytest.raises(FrozenInstanceError):
        event.final_width = 4  # type: ignore[misc]


def test_accuracy_guard_can_omit_unmeasured_cross_entropy() -> None:
    accepted = _accepted()
    assert accepted.guard is not None
    accuracy_only = replace(
        accepted.guard,
        pre_cross_entropy=None,
        post_cross_entropy=None,
        role_hash="a" * 64,
    )

    assert replace(accepted, guard=accuracy_only).guard == accuracy_only
    with pytest.raises(ValueError, match="both be present or absent"):
        replace(accuracy_only, post_cross_entropy=0.8)
    with pytest.raises(ValueError, match="requires cross-entropy"):
        replace(accuracy_only, metric_name="cross_entropy")
    with pytest.raises(ValueError, match="role hash"):
        replace(accuracy_only, role_hash="not a digest")


def test_error_guard_can_record_only_completed_pre_score() -> None:
    accepted = _accepted()
    assert accepted.guard is not None
    partial = replace(
        accepted.guard,
        post_accuracy=None,
        pre_cross_entropy=None,
        post_cross_entropy=None,
        delta=None,
        examples_scored=4,
    )
    failed = replace(
        accepted,
        outcome="error",
        reason="inner_guard_post_exception",
        guard=partial,
        changes=replace(accepted.changes, applied_split_pairs=()),
        final_width=accepted.before_width,
        chemistry_final=accepted.chemistry_before,
    )

    assert json.loads(json.dumps(asdict(failed), allow_nan=False))["guard"]["post_accuracy"] is None
    with pytest.raises(ValueError, match="full guard scores"):
        replace(failed, outcome="accepted")
    with pytest.raises(ValueError, match="delta requires a post score"):
        replace(partial, delta=0.1)


def test_cross_entropy_error_guard_can_keep_only_the_completed_pre_pass() -> None:
    accepted = _accepted()
    assert accepted.guard is not None

    partial = replace(
        accepted.guard,
        metric_name="cross_entropy",
        post_accuracy=None,
        post_cross_entropy=None,
        delta=None,
        examples_scored=4,
    )
    failed = replace(
        accepted,
        outcome="error",
        reason="inner_guard_post_exception",
        guard=partial,
        changes=replace(accepted.changes, applied_split_pairs=()),
        final_width=accepted.before_width,
        chemistry_final=accepted.chemistry_before,
    )

    assert failed.guard is not None
    assert failed.guard.pre_cross_entropy == 1.0
    assert failed.guard.post_cross_entropy is None
    assert failed.guard.examples_scored == 4
    assert json.loads(json.dumps(asdict(failed), allow_nan=False))["outcome"] == "error"


def test_rollback_retains_proposal_but_has_no_final_changes() -> None:
    accepted = _accepted()
    assert accepted.guard is not None
    rejected_guard = replace(accepted.guard, pre_accuracy=0.8, post_accuracy=0.4, delta=0.4)

    event = replace(
        accepted,
        outcome="rolled_back",
        reason="guard_rejected",
        changes=replace(accepted.changes, applied_split_pairs=()),
        final_width=accepted.before_width,
        guard=rejected_guard,
        chemistry_final=accepted.chemistry_before,
    )

    assert event.proposed_width == 5
    assert event.final_width == 4
    assert event.changes.proposed_split_pairs == ((10, 20),)
    assert event.changes.applied_split_pairs == ()


def test_skipped_sleep_record_has_explicit_absent_guard() -> None:
    accepted = _accepted()
    event = replace(
        accepted,
        trigger_reason="disabled",
        outcome="skipped",
        reason="sleep_disabled",
        budgets=SleepBudgets(0, 0, 0, None),
        changes=SleepStructuralChanges((), (), (), (), (), (), ()),
        proposed_width=4,
        final_width=4,
        guard=None,
        chemistry_proposed=accepted.chemistry_before,
        chemistry_final=accepted.chemistry_before,
        durations=SleepDurations(0.0, 0.0),
    )

    assert event.guard is None
    assert event.outcome == "skipped"


def test_accepted_no_topology_sleep_can_change_chemistry() -> None:
    accepted = _accepted()
    scalar = ChemicalSummary(count=4, minimum=0.2, mean=0.3, maximum=0.4)
    changed = ChemicalSummaries(primary=scalar, fast=scalar, slow=scalar)
    event = replace(
        accepted,
        changes=SleepStructuralChanges((), (), (), (), (), (), ()),
        proposed_width=4,
        final_width=4,
        chemistry_proposed=changed,
        chemistry_final=changed,
    )

    assert event.changes.applied_split_pairs == ()
    assert event.final_width == 4


def test_chemical_summary_accepts_roundoff_at_a_constant_extremum() -> None:
    summary = ChemicalSummary(count=3, minimum=0.1, mean=0.10000000000000002, maximum=0.1)

    assert summary.count == 3


def test_rollback_retains_proposed_prune_and_replay_facts() -> None:
    accepted = _accepted()
    assert accepted.guard is not None
    event = replace(
        accepted,
        outcome="rolled_back",
        reason="guard_rejected",
        changes=SleepStructuralChanges(
            proposed_split_pairs=(),
            applied_split_pairs=(),
            proposed_prune_ids=(10,),
            proposed_scheduled_prune_ids=(10,),
            proposed_removed_prune_ids=(),
            applied_scheduled_prune_ids=(),
            applied_removed_prune_ids=(),
        ),
        proposed_width=accepted.before_width,
        final_width=accepted.before_width,
        replay=SleepReplayUsage(3, 2, 0, 0),
        chemistry_proposed=accepted.chemistry_before,
        chemistry_final=accepted.chemistry_before,
        guard=replace(accepted.guard, pre_accuracy=0.8, post_accuracy=0.4, delta=0.4),
    )

    assert event.changes.proposed_scheduled_prune_ids == (10,)
    assert event.changes.applied_scheduled_prune_ids == ()
    assert event.replay.proposed_updates == 2
    assert event.replay.applied_updates == 0


def test_error_before_guard_scoring_can_be_recorded_without_a_guard() -> None:
    accepted = _accepted()
    event = replace(
        accepted,
        outcome="error",
        reason="core_exception",
        guard=None,
        changes=replace(accepted.changes, applied_split_pairs=()),
        final_width=accepted.before_width,
        chemistry_final=accepted.chemistry_before,
    )

    assert event.outcome == "error"
    assert event.guard is None
    assert event.changes.proposed_split_pairs == ((10, 20),)


@pytest.mark.parametrize(
    "damage",
    [
        "negative_budget",
        "duplicate_child",
        "wrong_width",
        "missing_guard_rollback",
        "nonfinite_chemical",
        "wrong_guard_delta",
        "replay_overapplied",
        "negative_duration",
        "unknown_trigger",
        "empty_reason",
        "accepted_partial_split",
        "skipped_applied_prune",
    ],
)
def test_sleep_record_rejects_malformed_or_contradictory_fields(damage: str) -> None:
    event = _accepted()
    with pytest.raises((TypeError, ValueError)):
        change: dict[str, Any]
        if damage == "negative_budget":
            change = {"budgets": replace(event.budgets, split_limit=-1)}
        elif damage == "duplicate_child":
            change = {
                "changes": replace(
                    event.changes,
                    proposed_split_pairs=((10, 20), (11, 20)),
                    applied_split_pairs=((10, 20), (11, 20)),
                )
            }
        elif damage == "wrong_width":
            change = {"final_width": 99, "chemistry_final": _chemistry(99)}
        elif damage == "missing_guard_rollback":
            change = {"outcome": "rolled_back", "guard": None}
        elif damage == "nonfinite_chemical":
            change = {
                "chemistry_final": replace(
                    event.chemistry_final,
                    primary=replace(event.chemistry_final.primary, mean=float("nan")),
                )
            }
        elif damage == "wrong_guard_delta":
            assert event.guard is not None
            change = {"guard": replace(event.guard, delta=0.5)}
        elif damage == "replay_overapplied":
            change = {"replay": SleepReplayUsage(1, 1, 2, 2)}
        elif damage == "negative_duration":
            change = {"durations": SleepDurations(-0.1, 0.0)}
        elif damage == "unknown_trigger":
            change = {"trigger_reason": "guess"}
        elif damage == "accepted_partial_split":
            change = {
                "changes": replace(event.changes, applied_split_pairs=()),
                "final_width": 4,
                "chemistry_final": event.chemistry_before,
            }
        elif damage == "skipped_applied_prune":
            change = {
                "outcome": "skipped",
                "trigger_reason": "disabled",
                "changes": SleepStructuralChanges((), (), (), (), (10,), (), (10,)),
                "proposed_width": 3,
                "final_width": 3,
                "chemistry_proposed": _chemistry(3),
                "chemistry_final": _chemistry(3),
            }
        else:
            change = {"reason": ""}
        replace(event, **change)
