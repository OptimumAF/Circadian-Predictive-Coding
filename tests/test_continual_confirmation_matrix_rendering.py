"""Exhaustive matrix rendering from fabricated outcomes only."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import test_continual_confirmation_matrix as matrix_tests
from src.app.continual_confirmation_matrix import build_confirmation_matrix
from src.app.continual_confirmation_matrix_rendering import render_confirmation_matrix

fabricated_scored_json = matrix_tests.fabricated_scored_json
original_cost_metadata = matrix_tests.original_cost_metadata
fabricated_report = matrix_tests.fabricated_report
seal_validation = matrix_tests.seal_validation


def test_should_display_all_matrices_metrics_raw_counts_roles_and_unknown_slots(
    fabricated_report: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix = build_confirmation_matrix(fabricated_report)
    before = deepcopy(matrix)

    def forbid(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("pure renderer entered IO")

    monkeypatch.setattr(Path, "read_bytes", forbid)
    monkeypatch.setattr(Path, "read_text", forbid)
    text = render_confirmation_matrix(matrix)
    assert matrix == before
    assert text == render_confirmation_matrix(matrix)
    assert text.count("| Stage / task | A | B |") == 560
    assert text.count("## gating / seed ") == 30
    assert text.count("| Original endpoint | Checkpoint | Task | Correct | Count |") == 560
    for row in matrix["rows"]:
        assert f"seed {row['seed']} / {row['arm']}" in text
        for metric in row["metrics"].values():
            if metric["value"] is not None:
                assert repr(metric["value"]) in text
        for item in row["endpoints"]:
            assert item["pointer"] in text
            assert item["record"]["role_sha256"] in text
    assert "B-after-A is unmeasured" in text
    assert "Forward transfer is unavailable" in text
    assert "Lower forgetting after weak initial A" in text
    assert "ratios above one" in text
    assert "repeats add no seed replications" in text
    assert "confirmation-matrix.result.json" in text
    assert "null (unmeasured: not_declared_before_task_b_arrival)" in text


def test_should_display_failed_endpoints_and_keep_the_other_measured_values(
    fabricated_scored_json: dict[str, Any], original_cost_metadata: dict[str, Any]
) -> None:
    payload = deepcopy(fabricated_scored_json)
    matrix_tests.report_tests.scored_tests._fail_endpoints(
        payload, [2], "numerical_prediction_error"
    )
    report = matrix_tests.build_confirmation_report(
        (payload, deepcopy(payload)), original_cost_metadata
    )
    text = render_confirmation_matrix(build_confirmation_matrix(report))
    assert "numerical_prediction_error:FloatingPointError" in text
    assert "| after A | 0.025 | null (unmeasured:" in text
    assert "| after B | 0.05 | null (failed:" in text


def test_should_escape_free_text_without_mutating_any_matrix_field(
    fabricated_report: dict[str, Any],
) -> None:
    matrix = build_confirmation_matrix(fabricated_report)
    matrix["rows"][0]["original_cell_failure"] = "<script>|a\nb"
    before = deepcopy(matrix)
    text = render_confirmation_matrix(matrix)
    assert "&lt;script&gt;\\|a<br>b" in text
    assert "<script>|" not in text
    assert matrix == before


def test_should_refuse_unknown_renderer_schema() -> None:
    with pytest.raises(ValueError, match="schema"):
        render_confirmation_matrix({"schema_id": "unknown"})
