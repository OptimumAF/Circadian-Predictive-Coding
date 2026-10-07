"""Exhaustive deterministic Markdown; fabricated scores establish no science."""

from copy import deepcopy
from typing import Any

import pytest

import test_continual_confirmation_report as report_tests
from src.app.continual_confirmation_report import build_confirmation_report, METRICS
from src.app.continual_confirmation_report_rendering import render_confirmation_report

fabricated_scored_json = report_tests.fabricated_scored_json
original_cost_metadata = report_tests.original_cost_metadata
seal_validation = report_tests.seal_validation


def test_should_display_every_arm_contrast_vector_seed_and_raw_interval_without_ranking(
    fabricated_scored_json: dict[str, Any], original_cost_metadata: dict[str, Any]
) -> None:
    report = build_confirmation_report(
        (fabricated_scored_json, deepcopy(fabricated_scored_json)), original_cost_metadata
    )
    text = render_confirmation_report(report)
    assert text == render_confirmation_report(report)
    assert text.count("#### ") == 626
    assert text.count("### Arm: ") == 56
    assert text.count("### Contrast: ") == 58
    assert text.count("| Seed | Value | Null reason |") == 626
    assert text.count("Interval status:") == 626
    for family in report["analysis"]["families"]:
        for group in family["arms"] + family["contrasts"]:
            for metric in group["metrics"]:
                for observation in metric["summary"]["observations"]:
                    assert f"| {observation['seed']} |" in text
                for kind in ("marginal", "simultaneous"):
                    interval = metric["summary"][kind + "_interval"]
                    if interval:
                        assert repr(interval["lower"]) in text
                        assert repr(interval["upper"]) in text
    for row in report["joined_cells"]:
        assert row["outcome"]["arm"] in text
    assert "Rejected executed replay" in text
    assert "A-after-A" in text
    assert "Families are analyzed separately" in text
    assert "Lower forgetting after weak initial A" in text
    assert "zero-dispersion" in text
    assert "confirmation-report.result.json" in text
    for name in METRICS:
        assert name in text


def test_should_display_every_failed_seed_and_reason_with_null_intervals(
    fabricated_scored_json: dict[str, Any], original_cost_metadata: dict[str, Any]
) -> None:
    payload = deepcopy(fabricated_scored_json)
    report_tests.scored_tests._fail_endpoints(payload, list(range(1680)), "nonfinite_predictions")
    report = build_confirmation_report((payload, deepcopy(payload)), original_cost_metadata)
    text = render_confirmation_report(report)
    assert text.count("#### ") == 626
    assert "nonfinite_predictions" in text
    assert "null" in text
    assert "| simultaneous | null | null | null | null |" in text
    assert "incomplete_observations" in text


def test_should_escape_table_text_without_changing_the_source_report(
    fabricated_scored_json: dict[str, Any], original_cost_metadata: dict[str, Any]
) -> None:
    report = build_confirmation_report(
        (fabricated_scored_json, deepcopy(fabricated_scored_json)), original_cost_metadata
    )
    report["joined_cells"][0]["outcome"]["failure"] = "<script>|a\nb"
    before = deepcopy(report)
    text = render_confirmation_report(report)
    assert "&lt;script&gt;\\|a<br>b" in text
    assert "<script>|" not in text
    assert report == before


def test_should_refuse_unknown_renderer_schema() -> None:
    with pytest.raises(ValueError, match="schema"):
        render_confirmation_report({"schema_id": "unknown"})
