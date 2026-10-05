"""All-cell joining and measured-scope hypothesis behavior on small fixtures."""

from copy import deepcopy
from typing import Any

import pytest

from src.app.continual_findings_synthesis import _hypothesis_findings, _join_cells


def _bodies() -> dict[str, Any]:
    key = {"family": "schedule", "seed": 17, "arm": "neutral_adaptive"}
    metrics = {
        "retention_ratio_a": {"value": None, "reason": "zero_a_after_a"},
        "signed_forgetting_a": {"value": -0.025, "reason": None},
    }
    fields = {
        "sleep_attempts": {"value": 0, "status": "derived"},
        "wall_time_seconds": {"value": None, "status": "unmeasured"},
    }
    return {
        "outcome_costs": {
            "rows": [
                {**key, "metrics": metrics, "outcome": {"failure": None}, "resource_fields": fields}
            ]
        },
        "matrix": {"rows": [{**key, "metrics": deepcopy(metrics), "original_cell_failure": None}]},
        "activity": {
            "cells": [
                {
                    **key,
                    "original_work_and_capacity_fields": deepcopy(fields),
                    "recorded_transaction_outcomes": {
                        "accepted": 0,
                        "rolled_back": 0,
                        "skipped": 24,
                    },
                    "transaction_scope": "one_neutral_controller_with_three_appliers",
                }
            ]
        },
    }


def test_should_preserve_negative_null_and_inactive_control_without_benefit_inference():
    bodies = _bodies()
    cells = _join_cells(bodies)
    assert cells[0]["metrics"]["retention_ratio_a"] == {"value": None, "reason": "zero_a_after_a"}
    assert cells[0]["metrics"]["signed_forgetting_a"]["value"] == -0.025
    assert cells[0]["zero_recorded_sleep_attempts"] is True
    assert cells[0]["recorded_transaction_outcomes"]["skipped"] == 24
    cells[0]["metrics"]["signed_forgetting_a"]["value"] = 0
    assert bodies["outcome_costs"]["rows"][0]["metrics"]["signed_forgetting_a"]["value"] == -0.025


@pytest.mark.parametrize("column", ["matrix", "activity"])
def test_should_reject_reordered_or_missing_cell_joins(column):
    bodies = _bodies()
    rows = bodies[column]["rows" if column == "matrix" else "cells"]
    rows[0]["seed"] = 18
    with pytest.raises(ValueError, match="identity"):
        _join_cells(bodies)
    rows.clear()
    with pytest.raises(ValueError):
        _join_cells(bodies)


@pytest.mark.parametrize("field", ["metrics", "original_cell_failure"])
def test_should_reject_changed_original_matrix_metric_or_failure(field):
    bodies = _bodies()
    bodies["matrix"]["rows"][0][field] = {} if field == "metrics" else {"code": "divergence"}
    with pytest.raises(ValueError):
        _join_cells(bodies)


def test_should_reject_invented_zero_per_arm_runtime():
    bodies = _bodies()
    bodies["activity"]["cells"][0]["original_work_and_capacity_fields"]["wall_time_seconds"][
        "value"
    ] = 0
    with pytest.raises(ValueError, match="work/capacity"):
        _join_cells(bodies)


def test_should_preserve_scientific_failure_reasons_instead_of_replacing_them():
    bodies = _bodies()
    failure = {"code": "nonfinite", "error_type": "FloatingPointError"}
    bodies["outcome_costs"]["rows"][0]["outcome"]["failure"] = failure
    bodies["matrix"]["rows"][0]["original_cell_failure"] = deepcopy(failure)
    assert _join_cells(bodies)[0]["failure"] == failure


def test_should_keep_all_hypotheses_unresolved_without_pooling_or_voting():
    primary = {
        "primary_statements": [
            {
                "statement_id": "combined/a/b/mean",
                "classification": "unresolved_zero_included",
                "hypothesis_context": ["H1", "H2", "H3", "H4"],
            },
            {
                "statement_id": "gating/a/b/forgetting",
                "classification": "interval_ineligible",
                "hypothesis_context": ["H1"],
            },
        ]
    }
    cells = [
        {"family": family}
        for family in ("gating", "replay", "sleep", "schedule", "combined", "parent")
    ]
    findings = _hypothesis_findings(primary, cells)
    assert [row["status"] for row in findings] == ["unresolved"] * 4
    assert findings[0]["primary_statement_ids"] == ["combined/a/b/mean", "gating/a/b/forgetting"]
    assert findings[3]["complete_context_cell_indices"] == [3, 4]
    assert all("not_equivalence_or_broad_rejection" in row["interpretation"] for row in findings)
    assert all("vote" not in row for row in findings)


def test_should_not_apply_fixed_unresolved_interpretation_to_changed_directional_evidence():
    with pytest.raises(ValueError, match="original unresolved"):
        _hypothesis_findings(
            {"primary_statements": [{"classification": "directional_left_favorable"}]}, []
        )
