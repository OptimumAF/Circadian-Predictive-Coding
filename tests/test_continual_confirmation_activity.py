"""Small fabricated transactions test original signs, scope and preservation."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from src.app import continual_confirmation_activity as activity


def event(outcome: str = "accepted") -> dict[str, Any]:
    committed = outcome == "accepted"
    attempted = outcome != "skipped"
    pre, post = (0.5, 0.75 if committed else 0.25)
    return {
        "outcome": outcome,
        "before_width": 8,
        "final_width": 9 if committed else 8,
        "reason": "original_reason",
        "trigger_reason": "periodic",
        "guard": {"pre_accuracy": pre, "post_accuracy": post, "tolerance": 0.0, "delta": pre - post}
        if attempted
        else None,
        "replay": {
            "proposed_updates": 2 if attempted else 0,
            "applied_updates": 2 if committed else 0,
        },
        "changes": {
            "applied_split_pairs": [[1, 8]] if committed else [],
            "applied_removed_prune_ids": [],
            "applied_scheduled_prune_ids": [],
        },
    }


@pytest.mark.parametrize("outcome", ["accepted", "rolled_back", "skipped"])
def test_should_preserve_original_transaction_and_negative_guard_delta(outcome: str) -> None:
    original = event(outcome)
    before = deepcopy(original)
    activity._check_transaction(original)
    assert original == before
    if outcome == "accepted":
        assert original["guard"]["delta"] == -0.25


@pytest.mark.parametrize(
    "change",
    [
        "delta_sign",
        "acceptance",
        "commit_after_rollback",
        "split_after_rollback",
        "width_after_rollback",
        "guard_for_skip",
        "unknown_outcome",
        "bool_guard",
        "nonfinite_guard",
    ],
)
def test_should_reject_inconsistent_guard_skip_and_rollback_facts(change: str) -> None:
    value = event(
        "rolled_back"
        if "rollback" in change
        else "skipped"
        if change == "guard_for_skip"
        else "accepted"
    )
    if change == "delta_sign":
        value["guard"]["delta"] = 0.25
    elif change == "acceptance":
        value["guard"].update(post_accuracy=0.25, delta=0.25)
    elif change == "commit_after_rollback":
        value["replay"]["applied_updates"] = 2
    elif change == "split_after_rollback":
        value["changes"]["applied_split_pairs"] = [[1, 8]]
    elif change == "width_after_rollback":
        value["final_width"] = 9
    elif change == "guard_for_skip":
        value["guard"] = event()["guard"]
    elif change == "unknown_outcome":
        value["outcome"] = "selected_away"
    elif change == "bool_guard":
        value["guard"]["pre_accuracy"] = True
    else:
        value["guard"]["post_accuracy"] = float("nan")
    with pytest.raises(ValueError):
        activity._check_transaction(value)


def test_should_preserve_every_raw_decision_reason_selector_and_pointer() -> None:
    decisions = [
        {
            "name": "usage_growth",
            "event": event(outcome),
            "before": {"selector": {"rng_sha256": "1" * 64}},
            "after": {"selector": {"rng_sha256": "2" * 64}},
        }
        for outcome in ("accepted", "rolled_back", "skipped")
    ]
    context = {
        "family": "parent",
        "seed": 359,
        "legacy_train_context": {
            "opportunities": [{"phase": "b", "epoch": 4, "decisions": decisions}]
        },
    }
    before = deepcopy(context)
    rows = activity._transaction_rows(context, 50)
    assert [row["raw_record"] for row in rows] == decisions
    assert [row["outcome"] for row in rows] == ["accepted", "rolled_back", "skipped"]
    assert (
        rows[-1]["source_pointer"]
        == "/cost_references/projection/seed_contexts/50/legacy_train_context/opportunities/0/decisions/2"
    )
    rows[0]["raw_record"]["before"]["selector"]["rng_sha256"] = "0" * 64
    assert context == before


def schedule_decision(outcome: str) -> dict[str, Any]:
    accepted, attempted = outcome == "accepted", outcome != "skipped"
    return {
        "policy": "adaptive",
        "outcome": outcome,
        "attempted": attempted,
        "guard_pre_accuracy": 0.5 if attempted else None,
        "guard_post_accuracy": 0.75 if accepted else 0.25 if attempted else None,
        "applied_ids_by_method": {
            name: ["same_original_id"] if accepted else [] for name in ("backprop", "pc", "neutral")
        },
        "reason": "guard_rejected" if outcome == "rolled_back" else "original_reason",
        "trigger_reason": "adaptive",
    }


def test_should_count_one_shared_controller_and_preserve_three_original_appliers() -> None:
    context = {
        "seed": 199,
        "legacy_train_context": {
            "opportunities": [
                {
                    "phase": "a",
                    "epoch": 4,
                    "decisions": [
                        schedule_decision("accepted"),
                        schedule_decision("rolled_back"),
                        schedule_decision("skipped"),
                    ],
                }
            ]
        },
    }
    rows = activity._schedule_rows(context, 30)
    assert len(rows) == 3
    assert "not_three_independent_transactions" in rows[0]["owner_scope"]
    assert rows[0]["raw_record"]["applied_ids_by_method"] == {
        name: ["same_original_id"] for name in ("backprop", "pc", "neutral")
    }
    assert rows[1]["reason"] == "guard_rejected"


@pytest.mark.parametrize("change", ["attempt", "guard", "applied_after_rejection"])
def test_should_reject_inconsistent_shared_controller_decisions(change: str) -> None:
    decision = schedule_decision("rolled_back")
    if change == "attempt":
        decision["attempted"] = False
    elif change == "guard":
        decision["guard_post_accuracy"] = 0.75
    else:
        decision["applied_ids_by_method"]["pc"] = ["uncommitted_id"]
    context = {
        "seed": 199,
        "legacy_train_context": {
            "opportunities": [{"phase": "a", "epoch": 4, "decisions": [decision]}]
        },
    }
    with pytest.raises(ValueError):
        activity._schedule_rows(context, 30)


def cell(family: str, arm: str, attempts: int) -> dict[str, Any]:
    return {
        "family": family,
        "seed": 101,
        "arm": arm,
        "original_cost": {"method_facts": {"original_counter": "retained"}},
        "resource_fields": {
            "sleep_attempts": {"value": attempts},
            "wall_time_seconds": {"value": None, "status": "unmeasured"},
        },
    }


def test_should_keep_unrecorded_replay_commits_unmeasured_even_with_nonzero_attempts() -> None:
    row = cell("replay", "circadian_on", 6)
    result = activity._cell_activity(row, [])
    assert result["recorded_transaction_outcomes"] == {
        "accepted": None,
        "rolled_back": None,
        "skipped": None,
    }
    assert result["original_work_and_capacity_fields"] == row["resource_fields"]
    assert "not_zero_activity" in result["transaction_scope"]


def test_should_reject_transaction_counter_mismatch_and_keep_skips() -> None:
    rows = [
        {
            "family": "combined",
            "seed": 101,
            "owner": "full",
            "outcome": outcome,
            "source_pointer": f"/decision/{index}",
        }
        for index, outcome in enumerate(("accepted", "rolled_back", "skipped"))
    ]
    result = activity._cell_activity(cell("combined", "full", 2), rows)
    assert result["recorded_transaction_outcomes"] == {
        "accepted": 1,
        "rolled_back": 1,
        "skipped": 1,
    }
    assert len(result["activity_references"]) == 3
    with pytest.raises(ValueError):
        activity._cell_activity(cell("combined", "full", 1), rows)


def test_should_attribute_shared_schedule_to_the_neutral_controller_without_triple_counting() -> (
    None
):
    rows = [
        {
            "family": "schedule",
            "seed": 101,
            "owner": "adaptive",
            "outcome": "skipped",
            "source_pointer": "/original_decision",
        }
    ]
    for arm in ("pc_adaptive", "backprop_adaptive"):
        assert (
            activity._cell_activity(cell("schedule", arm, 0), rows)[
                "recorded_transaction_outcomes"
            ]["skipped"]
            is None
        )
    assert activity._cell_activity(cell("schedule", "neutral_adaptive", 0), rows)[
        "recorded_transaction_outcomes"
    ] == {"accepted": 0, "rolled_back": 0, "skipped": 1}


@pytest.mark.parametrize("bad", [None, {}, {"cost_references": {}}, {"value": float("nan")}])
def test_should_refuse_unpinned_or_incomplete_public_bodies_before_projection(
    bad: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any) -> Any:
        pytest.fail("unpinned input reached activity projection")

    monkeypatch.setattr(activity, "_derive_activity", forbidden)
    with pytest.raises(ValueError):
        activity.build_confirmation_activity(bad, {})


def test_should_project_transactions_without_new_model_data_or_file_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core.backprop_mlp import BackpropMLP
    from src.core.predictive_coding import PredictiveCodingNetwork
    from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("activity projection performed science or IO")

    for model in (BackpropMLP, PredictiveCodingNetwork, CircadianPredictiveCodingNetwork):
        for name in ("__init__", "train_epoch", "predict_proba", "compute_accuracy"):
            monkeypatch.setattr(model, name, forbidden)
    for name in ("read_bytes", "read_text", "write_bytes", "write_text", "open"):
        monkeypatch.setattr(Path, name, forbidden)
    rows = activity._transaction_rows(
        {
            "family": "combined",
            "seed": 277,
            "legacy_train_context": {
                "opportunities": [
                    {
                        "phase": "a",
                        "epoch": 4,
                        "decisions": [{"name": "full", "event": event("rolled_back")}],
                    }
                ]
            },
        },
        40,
    )
    assert rows[0]["outcome"] == "rolled_back"
