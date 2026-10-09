"""Declared utility/retention/resource/action gates never change the actor."""

from dataclasses import replace
from typing import Any, cast

import pytest

from src.core.promotion_guard import (
    GuardBatch,
    PromotionEvidence,
    PromotionPolicy,
    decide_promotion,
)


def policy():
    return PromotionPolicy(
        "shared_accuracy_v1", "snapshot_bytes_v1", 0.6, 0.1, 0.05, 0.1, 100, ("allow",)
    )


def evidence():
    return PromotionEvidence(0.5, 0.7, 0.8, 0.8, 0.05, 80, True, True)


def test_should_accept_only_all_declared_gates_with_equal_boundaries():
    observed = replace(
        evidence(),
        actor_new_utility=0.5,
        candidate_new_utility=0.75,
        actor_old_utility=0.875,
        candidate_old_utility=0.75,
        candidate_max_prediction_seconds=0.125,
        candidate_resource_bytes=100,
    )
    declared = replace(
        policy(),
        min_new_utility=0.75,
        min_new_gain=0.25,
        max_old_drop=0.125,
        max_prediction_seconds=0.125,
    )
    assert decide_promotion(declared, observed).accepted


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"candidate_new_utility": 0.55}, "new_task_utility"),
        ({"actor_new_utility": 0.7}, "new_task_gain"),
        ({"candidate_old_utility": 0.7}, "old_task_retention"),
        ({"numerically_valid": False}, "numerical_validity"),
        ({"candidate_new_utility": float("nan")}, "numerical_validity"),
        ({"candidate_max_prediction_seconds": float("inf")}, "numerical_validity"),
        ({"candidate_max_prediction_seconds": 0.2}, "latency"),
        ({"candidate_resource_bytes": 101}, "resource"),
        ({"actions_safe": False}, "action_safety"),
        ({"candidate_old_utility": None}, "missing_measurement"),
    ],
)
def test_should_reject_every_bad_candidate_gate_without_changing_thresholds(changes, reason):
    frozen = policy()
    decision = decide_promotion(frozen, replace(evidence(), **changes))
    assert not decision.accepted and reason in decision.reasons
    assert frozen == policy()


@pytest.mark.parametrize(
    "changes",
    [
        {"metric_id": ""},
        {"resource_metric_id": " "},
        {"min_new_gain": float("nan")},
        {"max_old_drop": -1.0},
        {"max_prediction_seconds": -1.0},
        {"max_resource_bytes": True},
        {"allowed_actions": ()},
        {"allowed_actions": ("allow", "allow")},
    ],
)
def test_should_reject_malformed_prospective_policy(changes):
    with pytest.raises(ValueError):
        replace(policy(), **changes)


def test_should_validate_guard_metadata_without_inspecting_payload_layout():
    batch = GuardBatch("new", (("episode", "sample"),), 3, object(), object(), "inner_guard")
    metadata_cases: list[dict[str, Any]] = [
        {"sample_keys": ()},
        {"sample_keys": (("e", "s"), ("e", "s"))},
        {"labels_arrived_at": True},
        {"role": "unknown"},
        {"task_id": ""},
    ]
    for changes in metadata_cases:
        with pytest.raises(ValueError):
            replace(batch, **changes)


def test_should_reject_wrong_evidence_types_instead_of_coercing_them():
    with pytest.raises(ValueError):
        replace(evidence(), numerically_valid=cast(Any, 1))
    with pytest.raises(ValueError):
        replace(evidence(), candidate_resource_bytes=cast(Any, 1.0))
