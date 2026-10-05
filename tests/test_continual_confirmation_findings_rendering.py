"""Exhaustive presentation and resealed-body rejection for fabricated evidence."""

from copy import deepcopy
from typing import Any

import pytest

import test_continual_confirmation_findings as finding_tests
from src.app.continual_confirmation_findings import _derive_findings
from src.app.continual_confirmation_findings_rendering import render_confirmation_findings


fabricated_scored_json = finding_tests.fabricated_scored_json
original_cost_metadata = finding_tests.original_cost_metadata
seal_validation = finding_tests.seal_validation
fabricated_report = finding_tests.fabricated_report


def test_should_render_every_statement_exact_values_endpoints_and_complete_original_evidence(
    fabricated_report: dict[str, Any],
) -> None:
    body = _derive_findings(fabricated_report)
    before = deepcopy(body)
    text = render_confirmation_findings(body)
    assert text == render_confirmation_findings(deepcopy(body))
    assert body == before
    for statement in body["primary_statements"]:
        assert text.count(f"| {statement['statement_id']} |") == 1
        summary = statement["summary"]
        assert repr(summary["mean"]) in text
        assert summary["interval_status"] in text
        assert statement["classification"] in text
        assert repr(summary["observed_sample_standard_deviation"]) in text
    assert "116 simultaneous statements" in text
    assert "Crossing zero is unresolved, not equivalence" in text
    assert "Marginal and secondary intervals cannot replace primary simultaneous intervals" in text
    assert "approximately normal" in text
    assert "P6.12b" in text
    assert "## Complete original report" in text
    assert '"negative_seed_observations"' in text
    assert '"endpoint_evaluations"' in text
    assert '"retention_ratio_a"' in text
    for hypothesis in ("H1", "H2", "H3", "H4"):
        assert f"| {hypothesis} | unresolved |" in text


@pytest.mark.parametrize("change", ["classification", "missing", "reason", "coverage", "note"])
def test_should_reject_changed_or_incomplete_findings_before_rendering(
    fabricated_report: dict[str, Any], change: str
) -> None:
    body = _derive_findings(fabricated_report)
    if change == "classification":
        body["primary_statements"][-1]["classification"] = "supported"
    elif change == "missing":
        body["primary_statements"].pop()
    elif change == "reason":
        body["primary_statements"][-1]["summary"]["observations"][-1]["reason"] = "omitted"
    elif change == "coverage":
        body["coverage"]["original_report"]["seed_observations"] = 1
    else:
        body["hypothesis_conclusions"][0]["status"] = "supported"
    with pytest.raises(ValueError, match="complete findings declarations"):
        render_confirmation_findings(body)


@pytest.mark.parametrize("body", [None, [], {}, {"schema_id": "unknown"}])
def test_should_fail_loudly_for_malformed_presentation_input(body: Any) -> None:
    with pytest.raises(ValueError):
        render_confirmation_findings(body)
