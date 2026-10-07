"""Bind compact original arm costs to complete preserved cost evidence.

Input is the complete decoded b1 inspection from its actual reader. Output
keeps every raw arm cost and exact pointers to all shared contexts. Pure
fingerprints check declarations; file/source/readback authority stays outside.
No cost estimation, data/model, score, statistic or IO belongs here.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.app.continual_confirmation_execution import digest_json
from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.app.continual_confirmation_scoring_execution import verify_reference_report


COST_INSPECTION_ID = {
    "sha256": "214ce7ad00f61ca1aac8f85ad1381f50e9fc4235baeec59d7daf6043cc7d81dd",
    "byte_count": 112635395,
}
COST_PROJECTION_ID = {
    "sha256": "42bbe9049ddd863cf8daa7cca0dc4d62bd26ba9f1be5f7ae05be300107e27868",
    "byte_count": 103705403,
}
REPORT_COST_ID = {
    "sha256": "865112555b1323aeb873521053e51cc163b142ca82a6aa4a4d72aa08b01a3383",
    "byte_count": 1608771,
}
BASE_SOURCE_MAP_SHA256 = "adc4da3da4a96bd6a9b941137862d34efcccce8751713c916f60fc328b01410f"
COST_FILES = (
    "artifacts/runs/p611-confirmation-costs.json",
    "artifacts/runs/p611-confirmation-costs-repeat.json",
)


def validate_report_costs(value: dict[str, Any]) -> None:
    """Require every original cost value and context link, including nulls."""
    require(type(value) is dict, "report costs must be a complete JSON object")
    same_json(
        canonical_body_identity(value), REPORT_COST_ID, "report complete original cost vector"
    )


def compact_report_costs(inspection: dict[str, Any]) -> dict[str, Any]:
    """Retain complete arm costs; reference large shared proofs without copying them."""
    same_json(canonical_body_identity(inspection), COST_INSPECTION_ID, "report complete b1 body")
    try:
        references = inspection["cost_references"]
        verify_reference_report(references["training_references"])
        same_json(references["projection_identity"], COST_PROJECTION_ID, "report b1 projection")
        same_json(
            digest_json(inspection["source_sha256"]), BASE_SOURCE_MAP_SHA256, "report b1 sources"
        )
        projection = references["projection"]
        value = {
            "schema_id": "p611_confirmation_report_costs_v1",
            "training_result": deepcopy(projection["training_result"]),
            "cells": deepcopy(projection["cells"]),
            "work": deepcopy(projection["work"]),
            "cost_interpretation": deepcopy(projection["cost_interpretation"]),
            "context_references": [
                {
                    "family": row["family"],
                    "seed": row["seed"],
                    "json_pointer": f"/cost_references/projection/seed_contexts/{index}",
                }
                for index, row in enumerate(projection["seed_contexts"])
            ],
            "complete_context_files": [
                {"path": path, "identity": deepcopy(COST_INSPECTION_ID)} for path in COST_FILES
            ],
            "full_projection_identity": deepcopy(COST_PROJECTION_ID),
            "base_source_map_sha256": BASE_SOURCE_MAP_SHA256,
            "validation_scope": "bound_original_cost_values_and_complete_context_references_only_not_file_proof",
        }
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("report complete cost inspection is malformed") from error
    validate_report_costs(value)
    return value
