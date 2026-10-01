"""Validate every stored report declaration before presenting its matrix.

Input is the complete declared report; output is its rebuilt pure body.
The expected training declaration below is a schema bridge for the existing
public scored validator, not evidence of source/execution. Actual complete
report readback and current artifact/source proof belong to infrastructure.
No IO, model/data, new evaluation, selection or interval rule belongs here.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.app.continual_confirmation_execution import json_value
from src.app.continual_confirmation_json import object_fields, require, same_json
from src.app.continual_confirmation_report import build_confirmation_report
from src.app.continual_confirmation_scoring_manifest import (
    fixed_scoring_manifest,
    scoring_manifest_digest,
)
from src.app.continual_confirmation_scoring_state import (
    TrainingArtifactIdentity,
    TrainingStateProof,
)


REPORT_FIELDS = {
    "schema_id",
    "analysis_contract",
    "analysis",
    "analysis_repetition",
    "replication",
    "joined_cells",
    "endpoint_evaluations",
    "final_role_facts",
    "evaluation_totals",
    "cost_reference",
    "cost_vector_identity",
    "coverage",
    "validation_scope",
    "outer_selection_scored",
}


def _declared_scored_view(report: dict[str, Any]) -> dict[str, Any]:
    manifest = fixed_scoring_manifest()
    families = manifest.train_manifest.families
    rows = sum(len(f.seeds) for f in families)
    cells = sum(len(f.seeds) * len(f.arms) for f in families)
    reference = manifest.training_bundles[0]
    proof = json_value(
        asdict(
            TrainingStateProof(
                TrainingArtifactIdentity(reference.result_sha256, reference.result_bytes),
                rows,
                cells,
                2 * cells,
                scoring_manifest_digest(manifest),
            )
        )
    )
    # Why this: the report exposes endpoint/role/cell declarations, but training
    # proof is owned by its complete reader. This expected schema declaration
    # only lets the public validator rederive all stored outcome links.
    return {
        **{
            name: proof
            for name in (
                "training_before_release",
                "training_after_release",
                "training_after_evaluation",
            )
        },
        "final_roles": report["final_role_facts"],
        "evaluations": report["endpoint_evaluations"],
        "cells": report["analysis"]["cells"],
        "totals": report["evaluation_totals"],
        "scoring_manifest_sha256": scoring_manifest_digest(manifest),
        "validation_scope": "app_training_state_final_role_content_and_endpoint_records_only",
        "external_execution_verified": False,
        "outer_selection_scored": False,
    }


def verify_matrix_report_input(report: Any) -> dict[str, Any]:
    """Rebuild all original outcomes/summaries/costs without source-proof claims."""
    require(type(report) is dict, "matrix report must be a complete object")
    extra = (
        {"provenance", "scored_run_facts"}
        if "provenance" in report or "scored_run_facts" in report
        else set()
    )
    object_fields(report, REPORT_FIELDS | extra, "matrix input complete report")
    require(
        type(report["analysis"]) is dict and "cells" in report["analysis"],
        "matrix input analysis differs",
    )
    require(type(report["cost_reference"]) is dict, "matrix input cost reference differs")
    joined = report["joined_cells"]
    require(type(joined) is list and len(joined) == 560, "matrix input whole joined scope differs")
    for row in joined:
        object_fields(row, {"outcome", "cost", "metrics"}, "matrix input joined cell")
    if extra:
        require(type(report["provenance"]) is dict, "matrix input declared provenance differs")
        require(
            type(report["scored_run_facts"]) is list and len(report["scored_run_facts"]) == 2,
            "matrix input declared run scope differs",
        )
    costs = {**report["cost_reference"], "cells": [row["cost"] for row in joined]}
    declared = _declared_scored_view(report)
    rebuilt = build_confirmation_report((declared, declared), costs)
    same_json(
        {name: report[name] for name in REPORT_FIELDS},
        rebuilt,
        "matrix complete original report declarations",
    )
    return rebuilt
