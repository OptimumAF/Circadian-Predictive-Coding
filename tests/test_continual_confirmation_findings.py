"""Fabricated declarations test exhaustive interpretation without experiment authority."""

from copy import deepcopy
from typing import Any

import pytest

import test_continual_confirmation_report as report_tests
import test_continual_confirmation_scoring_validation as scored_tests
from src.app import continual_confirmation_findings as findings
from src.app.continual_confirmation_report import build_confirmation_report


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


@pytest.mark.parametrize(
    ("direction", "bounds", "expected"),
    [
        ("higher", (0.1, 0.3), "directional_left_favorable"),
        ("higher", (-0.3, -0.1), "directional_left_unfavorable"),
        ("lower", (0.1, 0.3), "directional_left_unfavorable"),
        ("lower", (-0.3, -0.1), "directional_left_favorable"),
        ("higher", (-0.1, 0.3), "unresolved_zero_included"),
        ("lower", (-0.1, 0.3), "unresolved_zero_included"),
        ("higher", (0.0, 0.3), "unresolved_zero_included"),
        ("lower", (-0.3, 0.0), "unresolved_zero_included"),
    ],
)
def test_should_use_original_preferred_direction_and_include_zero_boundaries(
    direction: str, bounds: tuple[float, float], expected: str
) -> None:
    metric = {
        "preferred_direction": direction,
        "summary": {
            "interval_status": "estimated",
            "simultaneous_interval": {"lower": bounds[0], "upper": bounds[1]},
            # A favorable marginal interval must never replace the simultaneous one.
            "marginal_interval": {"lower": 0.1, "upper": 0.2},
        },
    }
    assert findings._classify_statement(metric) == expected


@pytest.mark.parametrize("status", ["zero_observed_variance", "incomplete_observations"])
def test_should_keep_ineligible_statements_without_fabricating_an_interval(status: str) -> None:
    metric = {
        "preferred_direction": "higher",
        "summary": {
            "interval_status": status,
            "simultaneous_interval": None,
            "marginal_interval": {"lower": 0.1, "upper": 0.2},
        },
    }
    assert findings._classify_statement(metric) == "interval_ineligible"


def test_should_keep_all_original_vectors_seeds_endpoints_costs_and_statements(
    fabricated_report: dict[str, Any],
) -> None:
    before = deepcopy(fabricated_report)
    result = findings._derive_findings(fabricated_report)

    assert result["original_report"] == before
    assert fabricated_report == before
    assert len(result["primary_statements"]) == 116
    assert len({s["statement_id"] for s in result["primary_statements"]}) == 116
    assert result["coverage"]["original_report"]["metric_vectors"] == 626
    assert result["coverage"]["original_report"]["seed_observations"] == 6260
    assert sum(result["coverage"]["statement_classifications"].values()) == 116
    expected = []
    for family_index, family in enumerate(before["analysis"]["families"]):
        for pair_index, pair in enumerate(family["contrasts"]):
            for metric_index, metric in enumerate(pair["metrics"]):
                if metric["primary_endpoint"]:
                    expected.append((family_index, pair_index, metric_index, family, pair, metric))
    for statement, (fi, pi, mi, family, pair, metric) in zip(
        result["primary_statements"], expected, strict=True
    ):
        assert statement["family"] == family["name"]
        assert (statement["left"], statement["right"]) == (pair["left"], pair["right"])
        assert statement["metric"] == metric["metric"]
        assert statement["summary"] == metric["summary"]
        assert statement["source_pointer"] == f"/analysis/families/{fi}/contrasts/{pi}/metrics/{mi}"
        assert statement["secondary_endpoint_summaries"] == pair["metrics"][:3]
        assert len(statement["summary"]["observations"]) == 10
    assert result["original_p612_acceptance_complete"] is False
    assert result["fresh_official_reader_authority"] is False
    assert result["new_training_scoring_or_final_source_access"] is False


def test_should_keep_every_combined_contrast_as_context_without_a_hypothesis_vote(
    fabricated_report: dict[str, Any],
) -> None:
    result = findings._derive_findings(fabricated_report)
    conclusions = result["hypothesis_conclusions"]
    assert [item["hypothesis_id"] for item in conclusions] == ["H1", "H2", "H3", "H4"]
    combined = {
        s["statement_id"] for s in result["primary_statements"] if s["family"] == "combined"
    }
    for hypothesis in conclusions:
        assert hypothesis["status"] == "unresolved"
        assert (
            hypothesis["decision_scope"]
            == "primary_confirmation_evidence_tradeoff_synthesis_pending"
        )
        assert combined <= set(hypothesis["statement_ids"])
        assert hypothesis["remaining_uncertainty"]
        assert "not_equivalence_or_broad_rejection" in hypothesis["interpretation"]
    for statement in result["primary_statements"]:
        assert statement["hypothesis_context"]
        if statement["family"] == "combined":
            assert (
                statement["attribution_scope"]
                == "full_system_or_control_context_not_isolated_mechanism"
            )


@pytest.mark.parametrize(
    "change", ["last_interval", "last_seed", "missing_statement", "last_cost", "extra", "bool"]
)
def test_should_reject_late_or_resealed_declarations_before_classifying_any_statement(
    fabricated_report: dict[str, Any], monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    report = deepcopy(fabricated_report)
    pair = report["analysis"]["families"][-1]["contrasts"][-1]
    if change == "last_interval":
        pair["metrics"][-1]["summary"]["simultaneous_interval"]["lower"] = -123.0
    elif change == "last_seed":
        pair["metrics"][-1]["summary"]["observations"][-1]["seed"] = 9999
    elif change == "missing_statement":
        pair["metrics"].pop()
    elif change == "last_cost":
        report["joined_cells"][-1]["cost"]["wake_updates"] += 1
    elif change == "extra":
        report["unrecorded_selection"] = True
    else:
        report["coverage"]["primary_contrast_statements"] = True
    report["analysis_repetition"]["identities"] = [
        findings.canonical_body_identity(report["analysis"])
    ] * 2

    def forbidden(*args: Any, **kwargs: Any) -> str:
        pytest.fail("a statement was classified before complete input rejection")

    monkeypatch.setattr(findings, "_classify_statement", forbidden)
    with pytest.raises(ValueError):
        findings._derive_findings(report)


def test_should_preserve_both_failed_sides_all_planned_seeds_and_original_failure_reasons(
    fabricated_scored_json: dict[str, Any], original_cost_metadata: dict[str, Any]
) -> None:
    payload = deepcopy(fabricated_scored_json)
    scored_tests._fail_endpoints(payload, list(range(1680)), "nonfinite_predictions")
    report = build_confirmation_report((payload, deepcopy(payload)), original_cost_metadata)
    result = findings._derive_findings(report)

    assert result["original_report"] == report
    assert result["coverage"]["original_report"]["failed_cells"] == 560
    assert result["coverage"]["statement_classifications"]["interval_ineligible"] == 116
    for statement in result["primary_statements"]:
        summary = statement["summary"]
        assert summary["planned_seed_count"] == 10 and summary["observed_seed_count"] == 0
        assert summary["mean"] is summary["simultaneous_interval"] is None
        assert len(summary["observations"]) == 10
        for observation in summary["observations"]:
            assert observation["value"] is None
            assert (
                "left_failed" in observation["reason"] and "right_failed" in observation["reason"]
            )


def test_should_detach_nested_original_and_statement_records_from_the_input(
    fabricated_report: dict[str, Any],
) -> None:
    report = deepcopy(fabricated_report)
    before = deepcopy(report)
    result = findings._derive_findings(report)
    result["primary_statements"][0]["summary"]["observations"][0]["value"] = -9.0
    result["original_report"]["joined_cells"][-1]["metrics"]["retention_ratio_a"]["value"] = None
    assert report == before
    assert result["original_report"]["analysis"] == before["analysis"]


@pytest.mark.parametrize("value", [None, [], {}, {"analysis": float("nan")}])
def test_should_reject_malformed_or_unpinned_public_inputs(value: Any) -> None:
    with pytest.raises(ValueError):
        findings.build_confirmation_findings(value)


def test_should_reject_fabricated_full_scope_even_when_its_pure_declarations_are_valid(
    fabricated_report: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="whole original confirmation report"):
        findings.build_confirmation_findings(fabricated_report)


def test_should_preserve_exact_repeat_and_call_no_new_scientific_or_io_boundary(
    fabricated_report: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("pure findings called a new scientific or IO boundary")

    from src.core.backprop_mlp import BackpropMLP
    from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
    from src.core.predictive_coding import PredictiveCodingNetwork
    from pathlib import Path

    for model in (BackpropMLP, PredictiveCodingNetwork, CircadianPredictiveCodingNetwork):
        for name in ("__init__", "train_epoch", "predict_proba", "compute_accuracy"):
            monkeypatch.setattr(model, name, forbidden)
    for name in ("read_bytes", "read_text", "write_bytes", "write_text", "open"):
        monkeypatch.setattr(Path, name, forbidden)
    first = findings._derive_findings(fabricated_report)
    second = findings._derive_findings(deepcopy(fabricated_report))
    assert findings.canonical_body_identity(first) == findings.canonical_body_identity(second)
