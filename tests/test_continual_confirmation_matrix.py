"""Whole matrix behavior uses fabricated endpoints and unscored cost metadata."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import test_continual_confirmation_report as report_tests
from src.app.continual_confirmation_report import build_confirmation_report
from src.app.continual_confirmation_matrix import build_confirmation_matrix

fabricated_scored_json = report_tests.fabricated_scored_json
original_cost_metadata = report_tests.original_cost_metadata
seal_validation = report_tests.seal_validation


@pytest.fixture(scope="module")
def fabricated_report(
    fabricated_scored_json: dict[str, Any], original_cost_metadata: dict[str, Any]
) -> dict[str, Any]:
    return build_confirmation_report(
        (fabricated_scored_json, deepcopy(fabricated_scored_json)), original_cost_metadata
    )


def test_should_present_all_stage_task_slots_and_role_linked_original_endpoints(
    fabricated_report: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    before = deepcopy(fabricated_report)

    def forbid(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("pure matrix entered IO")

    monkeypatch.setattr(Path, "read_bytes", forbid)
    monkeypatch.setattr(Path, "read_text", forbid)
    result = build_confirmation_matrix(fabricated_report)
    assert fabricated_report == before
    assert result["stage_axis"] == ["after_a", "after_b"]
    assert result["task_axis"] == ["a", "b"]
    assert len(result["rows"]) == 560
    assert result["coverage"]["matrix_slots"] == 2240
    assert result["coverage"]["unmeasured_slots"] == 560
    assert result["coverage"]["successful_endpoints"] == 1680
    assert result["coverage"]["failed_endpoints"] == 0
    for index, row in enumerate(result["rows"]):
        source = fabricated_report["joined_cells"][index]
        assert {k: row[k] for k in ("family", "seed", "arm")} == {
            k: source["outcome"][k] for k in ("family", "seed", "arm")
        }
        assert row["metrics"] == source["metrics"]
        assert row["accuracy_matrix"][0][1] == {
            "value": None,
            "status": "unmeasured",
            "reason": "not_declared_before_task_b_arrival",
            "endpoint_pointer": None,
        }
        assert row["forward_transfer_b"]["value"] is None
        for offset, endpoint in enumerate(row["endpoints"]):
            assert endpoint["pointer"] == f"/endpoint_evaluations/{3 * index + offset}"
            assert (
                endpoint["record"] == fabricated_report["endpoint_evaluations"][3 * index + offset]
            )
    first = result["rows"][0]
    assert first["accuracy_matrix"][0][0]["value"] == 0.025
    assert first["accuracy_matrix"][1][0]["value"] == 0.05
    assert first["accuracy_matrix"][1][1]["value"] == 0.075
    assert first["metrics"]["retention_ratio_a"]["value"] == 2.0
    assert first["backward_transfer_a"]["value"] == 0.025
    assert first["backward_transfer_a"]["direction"] == "positive"
    assert result == build_confirmation_matrix(fabricated_report)


@pytest.mark.parametrize("indices", [[0], [1], [2], [1679], list(range(1680))])
def test_should_keep_successful_raw_endpoints_when_the_original_cell_has_failed(
    fabricated_scored_json: dict[str, Any],
    original_cost_metadata: dict[str, Any],
    indices: list[int],
) -> None:
    payload = deepcopy(fabricated_scored_json)
    report_tests.scored_tests._fail_endpoints(payload, indices, "nonfinite_predictions")
    report = build_confirmation_report((payload, deepcopy(payload)), original_cost_metadata)
    result = build_confirmation_matrix(report)
    assert result["coverage"]["failed_endpoints"] == len(indices)
    assert result["coverage"]["successful_endpoints"] == 1680 - len(indices)
    assert result["coverage"]["unmeasured_slots"] == 560
    for index in {i // 3 for i in indices}:
        row = result["rows"][index]
        assert row["original_cell_failure"] == report["analysis"]["cells"][index]["failure"]
        assert row["metrics"] == report["joined_cells"][index]["metrics"]
        assert row["backward_transfer_a"]["value"] is None
        for offset, coordinates in enumerate(((0, 0), (1, 0), (1, 1))):
            slot = row["accuracy_matrix"][coordinates[0]][coordinates[1]]
            if 3 * index + offset in indices:
                assert slot["status"] == "failed" and slot["value"] is None
                assert "nonfinite_predictions" in slot["reason"]
            else:
                assert slot["status"] == "measured" and slot["value"] is not None


def test_should_preserve_zero_denominator_retention_and_both_transfer_signs(
    fabricated_report: dict[str, Any],
) -> None:
    result = build_confirmation_matrix(fabricated_report)
    zeros = [r for r in result["rows"] if r["metrics"]["a_after_a"]["value"] == 0]
    assert zeros
    assert all(
        r["metrics"]["retention_ratio_a"] == {"value": None, "reason": "zero_a_after_a"}
        for r in zeros
    )
    directions = {r["backward_transfer_a"]["direction"] for r in result["rows"]}
    assert {"positive", "negative"}.issubset(directions)


@pytest.mark.parametrize(
    "change",
    [
        "scope",
        "seed",
        "role",
        "count",
        "checkpoint",
        "endpoint",
        "integer_alias",
        "failure",
        "summary",
        "cost",
        "metric",
        "pair_scope",
        "repeat_replications",
        "contract",
        "unknown",
        "partial",
    ],
)
def test_should_reject_complete_report_corruption_before_publishing_a_matrix(
    fabricated_report: dict[str, Any], change: str
) -> None:
    value = deepcopy(fabricated_report)
    if change == "scope":
        value["endpoint_evaluations"].pop()
    elif change in {"seed", "count", "integer_alias"}:
        key = "seed" if change == "seed" else "example_count"
        value["endpoint_evaluations"][-1][key] = (
            float(value["endpoint_evaluations"][-1][key]) if change == "integer_alias" else -1
        )
    elif change == "role":
        value["endpoint_evaluations"][-1]["role_sha256"] = "f" * 64
    elif change == "checkpoint":
        value["endpoint_evaluations"][-1]["checkpoint"] = "a"
    elif change == "endpoint":
        value["endpoint_evaluations"][-1]["endpoint"] = "b_after_a"
    elif change == "failure":
        value["endpoint_evaluations"][-1]["result"]["failure"] = {
            "code": "unknown",
            "error_type": None,
        }
    elif change == "summary":
        value["analysis"]["families"][-1]["arms"][-1]["metrics"][0]["summary"]["mean"] = 0.123
    elif change == "cost":
        value["joined_cells"][-1]["cost"]["wake_updates"] += 1
    elif change == "metric":
        value["joined_cells"][-1]["metrics"]["signed_forgetting_a"]["value"] = 0.123
    elif change == "pair_scope":
        value["analysis"]["families"][-1]["contrasts"].pop()
    elif change == "repeat_replications":
        value["replication"]["planned_seeds_per_vector"] = 20
    elif change == "contract":
        value["analysis_contract"]["primary_statement_count"] = 105
    elif change == "unknown":
        value["unknown"] = True
    else:
        del value["joined_cells"]
    with pytest.raises(ValueError):
        build_confirmation_matrix(value)
