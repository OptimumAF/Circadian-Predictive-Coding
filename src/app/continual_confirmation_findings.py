"""Preserve and interpret every prospective primary confirmation statement.

Input is the whole pinned original report. Output keeps that entire body,
all original summaries and explicit primary-only H1–H4 uncertainty. Complete
declaration reconstruction is pure; current source/IO proof, development,
cost/activity synthesis, experiment selection and publication belong elsewhere.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_matrix_inputs import verify_matrix_report_input
from src.app.continual_confirmation_report_costs import canonical_body_identity


FINDINGS_SCHEMA = "p612_complete_primary_confirmation_findings_v1"
REPORT_ID = {
    "sha256": "363ed97dba808281d13224526b4b36614ae452188cade256992b530b6ab03088",
    "byte_count": 7_537_678,
}
CLASSIFICATIONS = (
    "unresolved_zero_included",
    "interval_ineligible",
    "directional_left_favorable",
    "directional_left_unfavorable",
)
HYPOTHESES = (
    ("H1", "Plasticity control", "chemical gating retention/adaptation tradeoff"),
    (
        "H2",
        "Structural adaptation",
        "accuracy/capacity or compute tradeoff against fixed and scheduled growth",
    ),
    ("H3", "Consolidation", "replay/homeostasis contribution beyond added exposure and updates"),
    ("H4", "Scheduling", "adaptive versus periodic sleep tradeoff at the declared budget"),
)
REMAINING_UNCERTAINTY = {
    "H1": "Read both A endpoints: weaker A-after-A can reduce signed forgetting without improving retention. Integrate complete treatment/activity and cost evidence.",
    "H2": "Integrate actual initial/A/B and transient capacities, growth controls and costs; a primary sign alone does not establish a capacity/compute tradeoff.",
    "H3": "Integrate matched baseline replay identities, applied/rejected exposure and homeostasis activity; additional updates are not an independent consolidation benefit.",
    "H4": "Integrate actual adaptive/periodic attempts, commits, inactivity and work. An inactive policy does not establish a beneficial scheduling mechanism.",
}


def _classify_statement(metric: dict[str, Any]) -> str:
    """Classify only an already validated simultaneous interval; no marginal fallback."""
    interval = metric["summary"]["simultaneous_interval"]
    if interval is None:
        return "interval_ineligible"
    if interval["lower"] <= 0 <= interval["upper"]:
        return "unresolved_zero_included"
    positive = interval["lower"] > 0
    favorable = positive == (metric["preferred_direction"] == "higher")
    return "directional_left_favorable" if favorable else "directional_left_unfavorable"


def _hypothesis_context(family: str, left: str) -> list[str]:
    # Why this: combined contrasts contextualize all hypotheses, but cannot
    # isolate a mechanism or become extra independent statement replications.
    if family == "combined":
        return ["H1", "H2", "H3", "H4"]
    if family == "sleep":
        return {
            "structure_only": ["H2"],
            "homeostasis_only": ["H3"],
            "gating_reset": ["H1", "H3"],
        }[left].copy()
    return {"gating": ["H1"], "replay": ["H3"], "schedule": ["H4"], "parent": ["H2"]}[family].copy()


def _primary_statements(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    statements = []
    for family_index, family in enumerate(analysis["families"]):
        for pair_index, pair in enumerate(family["contrasts"]):
            for metric_index, metric in enumerate(pair["metrics"]):
                if not metric["primary_endpoint"]:
                    continue
                statements.append(
                    {
                        "statement_id": "/".join(
                            (family["name"], pair["left"], pair["right"], metric["metric"])
                        ),
                        "family": family["name"],
                        "left": pair["left"],
                        "right": pair["right"],
                        "metric": metric["metric"],
                        "preferred_direction": metric["preferred_direction"],
                        "units": metric["units"],
                        "summary": deepcopy(metric["summary"]),
                        "classification": _classify_statement(metric),
                        "hypothesis_context": _hypothesis_context(family["name"], pair["left"]),
                        "attribution_scope": "full_system_or_control_context_not_isolated_mechanism"
                        if family["name"] == "combined"
                        else "declared_family_contrast_only",
                        "secondary_endpoint_summaries": deepcopy(pair["metrics"][:3]),
                        "source_pointer": f"/analysis/families/{family_index}/contrasts/{pair_index}/metrics/{metric_index}",
                    }
                )
    require(len(statements) == 116, "findings require every original primary statement")
    return statements


def _classification_counts(statements: list[dict[str, Any]]) -> dict[str, int]:
    return {
        status: sum(statement["classification"] == status for statement in statements)
        for status in CLASSIFICATIONS
    }


def _hypothesis_conclusions(statements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conclusions = []
    for identifier, title, question in HYPOTHESES:
        context = [s for s in statements if identifier in s["hypothesis_context"]]
        conclusions.append(
            {
                "hypothesis_id": identifier,
                "title": title,
                "question": question,
                "status": "unresolved",
                "decision_scope": "primary_confirmation_evidence_tradeoff_synthesis_pending",
                "statement_ids": [s["statement_id"] for s in context],
                "context_classification_counts": _classification_counts(context),
                "interpretation": "not_equivalence_or_broad_rejection; no automatic vote or isolated mechanism inference from full-system comparisons",
                "remaining_uncertainty": REMAINING_UNCERTAINTY[identifier],
            }
        )
    return conclusions


def _derive_findings(report: dict[str, Any]) -> dict[str, Any]:
    """Development seam: complete pure declarations, never source/execution authority."""
    rebuilt = verify_matrix_report_input(report)
    statements = _primary_statements(rebuilt["analysis"])
    return {
        "schema_id": FINDINGS_SCHEMA,
        "report_identity": canonical_body_identity(report),
        "analysis_contract_sha256": rebuilt["analysis"]["contract_sha256"],
        "comparison_sign": rebuilt["analysis_contract"]["comparison_sign"],
        "primary_statement_count": rebuilt["analysis_contract"]["primary_statement_count"],
        "score_role": rebuilt["analysis"]["evaluation_role"],
        "replication": deepcopy(rebuilt["replication"]),
        "primary_statements": statements,
        "hypothesis_conclusions": _hypothesis_conclusions(statements),
        "coverage": {
            "original_report": deepcopy(rebuilt["coverage"]),
            "statement_classifications": _classification_counts(statements),
        },
        "interpretation_rules": {
            "intervals": "original_student_t_df9_bonferroni_116_model_based",
            "zero_inclusion": "unresolved_not_equivalence_or_broad_rejection",
            "ineligible": "original_status_and_observations_retained_no_interval_fabrication",
            "secondary": "descriptive_not_replacement_for_primary_simultaneous_family",
            "negative_values": "raw_signed_values_not_all_regressions_preferred_direction_matters",
            "hypotheses": "unresolved_primary_evidence_and_pending_tradeoff_synthesis_no_vote",
        },
        "original_report": deepcopy(report),
        "pending_acceptance": [
            "P6.12b development/tuning/activity/cost/failure synthesis and complete IO publication/readback"
        ],
        "validation_scope": "pure_complete_declared_report_and_primary_interpretation_only_not_current_source_or_io_proof",
        "original_p612_acceptance_complete": False,
        "fresh_official_reader_authority": False,
        "new_training_scoring_or_final_source_access": False,
    }


def build_confirmation_findings(report: dict[str, Any]) -> dict[str, Any]:
    """Require the entire original report before any current evidence interpretation."""
    require(type(report) is dict, "findings require the whole original confirmation report")
    same_json(
        canonical_body_identity(report), REPORT_ID, "whole original confirmation report identity"
    )
    return _derive_findings(report)
