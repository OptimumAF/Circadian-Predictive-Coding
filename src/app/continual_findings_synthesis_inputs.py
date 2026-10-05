"""Bind full synthesis bodies and preserved records in a pure app boundary.

The fixed catalog identifies complete original bodies, repeats and handoffs.
No filesystem/current source or official-reader authority is supplied here.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

from src.app.continual_confirmation_activity import _derive_activity
from src.app.continual_confirmation_findings import _derive_findings
from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_matrix import build_confirmation_matrix
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.app.continual_findings_development import _derive_development_ledger


CATALOG_ID = {
    "byte_count": 23515,
    "sha256": "55ed0854de4235a6f3daffe8f69b59ccf38a4db796a9baa62861a40424581d6e",
}


def verify_synthesis_identities(inputs: dict[str, Any]) -> None:
    same_json(
        set(inputs), {"catalog", "bodies", "preserved_records"}, "complete synthesis input fields"
    )
    catalog = inputs["catalog"]
    same_json(canonical_body_identity(catalog), CATALOG_ID, "whole fixed synthesis catalog")
    same_json(set(inputs["bodies"]), set(catalog["bodies"]), "all whole synthesis bodies")
    same_json(
        set(inputs["preserved_records"]), set(catalog["records"]), "all preserved whole records"
    )
    for name, declaration in catalog["bodies"].items():
        same_json(
            canonical_body_identity(inputs["bodies"][name]),
            declaration["identity"],
            "whole synthesis " + name,
        )
    for path, expected in catalog["records"].items():
        record = inputs["preserved_records"][path]
        require(type(record["content"]) is str, "preserved record requires complete UTF-8 content")
        raw = record["content"].encode("utf-8")
        actual = {"byte_count": len(raw), "sha256": sha256(raw).hexdigest()}
        same_json(actual, expected, "whole preserved record " + path)
        same_json(record["identity"], expected, "preserved record identity " + path)


def _verify_matrix_declarations(
    matrix: dict[str, Any], report: dict[str, Any], source: dict[str, Any]
) -> None:
    # Why this: the stored matrix retains historical reader provenance, while
    # its pure builder reconstructs declarations only. Preserve and validate
    # both without claiming a fresh reader dispatch from this reconstruction.
    same_json(
        {name: value for name, value in matrix.items() if name != "provenance"},
        build_confirmation_matrix(report),
        "whole original pure matrix declarations",
    )
    same_json(
        matrix["provenance"],
        {
            "validation_scope": "current_source_bound_complete_original_report_readback_and_stored_matrix",
            "source_sha256": source["source_sha256"],
            "source_map_sha256": source["source_map_sha256"],
            "inputs": source["request_template"]["inputs"],
            "new_training_or_final_source_access": False,
        },
        "whole original historical matrix provenance",
    )


def verify_synthesis_declarations(bodies: dict[str, Any], records: dict[str, Any]) -> None:
    """Rebuild complete stored declarations; current IO proof remains external."""
    primary, development = bodies["primary"], bodies["development"]
    report = primary["original_report"]
    same_json(primary, _derive_findings(report), "whole original primary ledger")
    same_json(
        development,
        _derive_development_ledger(development["original_inputs"]),
        "whole original development ledger",
    )
    _verify_matrix_declarations(
        bodies["matrix"],
        report,
        json.loads(records["artifacts/runs/p69-confirmation-matrix-source.json"]["content"]),
    )
    same_json(
        bodies["activity"],
        _derive_activity(bodies["cost_inspection"], bodies["outcome_costs"]),
        "whole original activity ledger",
    )
    for name, value in bodies["outcome_costs"]["original_report_records"].items():
        same_json(value, report[name], "whole outcome-cost original report " + name)
