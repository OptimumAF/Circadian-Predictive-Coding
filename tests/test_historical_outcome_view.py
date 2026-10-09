"""Pure presentation controls use tiny fabricated records, never experiment IO."""

from copy import deepcopy
from typing import Any

import pytest

from src.app.historical_outcome_view import build_historical_view, numeric_extent


def sample_body() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed, value in ((23, -0.2), (29, 1.4)):
        rows.append(
            {
                "family": "fixture",
                "seed": seed,
                "arm": "ordinary",
                "outcome": {"failure": None},
                "metrics": {"metric": {"value": value, "reason": None}},
                "resource_fields": {"wall_time_seconds": {"value": None, "reason": "unmeasured"}},
                "owned_retention": {"checkpoints": {"after_b": {"owned_array_bytes": 0}}},
            }
        )
    return {
        "schema_id": "p610_complete_original_outcomes_against_scoped_costs_v1",
        "rows": rows,
        "contexts": [],
        "coverage": {
            "cells": 2,
            "cell_metrics": 2,
            "metric_vectors": 1,
            "primary_contrast_statements": 0,
        },
        "original_report_records": {
            "analysis_contract": {
                "families": [{"name": "fixture", "arms": ["ordinary"], "seeds": [23, 29]}]
            },
            "analysis": {
                "families": [
                    {
                        "name": "fixture",
                        "arms": [
                            {
                                "name": "ordinary",
                                "metrics": [
                                    {
                                        "metric": "metric",
                                        "primary_endpoint": False,
                                        "summary": {
                                            "observations": [
                                                {"seed": r["seed"], **r["metrics"]["metric"]}
                                                for r in rows
                                            ],
                                            "mean": 0.6,
                                            "marginal_interval": {"lower": -0.7, "upper": 1.9},
                                            "simultaneous_interval": None,
                                        },
                                    }
                                ],
                            }
                        ],
                        "contrasts": [],
                    }
                ]
            },
        },
        "stage_storage_totals": {},
        "historical_process_segments": [],
        "presentation_scope": {},
        "original_inventory_gaps": {},
        "work": {},
    }


def test_should_preserve_order_negatives_uncertainty_and_unknown_costs():
    body = sample_body()
    original = deepcopy(body)
    view = build_historical_view(body)
    assert view["rows"] == body["rows"]
    assert view["original_report_records"] == body["original_report_records"]
    assert view["rows"][0]["metrics"]["metric"]["value"] == -0.2
    assert view["rows"][1]["resource_fields"]["wall_time_seconds"]["value"] is None
    view["rows"][0]["metrics"]["metric"]["value"] = 9
    assert body == original


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "reorder",
        "nonfinite",
        "wrong_observation",
        "wrong_coverage",
        "wrong_schema",
    ],
)
def test_should_refuse_incomplete_or_inconsistent_presentations(change):
    body = sample_body()
    if change == "missing":
        body["rows"].pop()
    elif change == "duplicate":
        body["rows"][1] = deepcopy(body["rows"][0])
    elif change == "reorder":
        body["rows"].reverse()
    elif change == "nonfinite":
        body["rows"][0]["metrics"]["metric"]["value"] = float("nan")
    elif change == "wrong_observation":
        body["original_report_records"]["analysis"]["families"][0]["arms"][0]["metrics"][0][
            "summary"
        ]["observations"][0]["value"] = 0
    elif change == "wrong_coverage":
        body["coverage"]["metric_vectors"] = 2
    else:
        body["schema_id"] = "other"
    with pytest.raises(ValueError):
        build_historical_view(body)


def test_should_keep_missing_metric_reason_and_failure_without_imputation():
    body = sample_body()
    body["rows"][0]["metrics"]["metric"] = {"value": None, "reason": "zero denominator"}
    body["rows"][0]["outcome"]["failure"] = "original failure"
    summary = body["original_report_records"]["analysis"]["families"][0]["arms"][0]["metrics"][0][
        "summary"
    ]
    summary["observations"][0].update(value=None, reason="zero denominator")
    summary.update(mean=None, marginal_interval=None)
    view = build_historical_view(body)
    assert view["rows"][0]["metrics"]["metric"]["reason"] == "zero denominator"
    assert view["rows"][0]["outcome"]["failure"] == "original failure"


def test_should_include_full_interval_and_above_one_values_in_plot_extent():
    low, high = numeric_extent([-0.2, 1.4, -0.7, 1.9, None])
    assert low < -0.7 and high > 1.9
    low, high = numeric_extent([0, 0, None])
    assert low < 0 < high


@pytest.mark.parametrize("values", [[None], [float("inf")], [True]])
def test_should_refuse_non_numeric_or_empty_plot_extent(values):
    with pytest.raises(ValueError):
        numeric_extent(values)
