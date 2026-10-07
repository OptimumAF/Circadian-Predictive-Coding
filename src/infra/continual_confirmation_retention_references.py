"""Project retention while unchanged complete training readers own authority.

Inputs are the fixed original training references, inventory and reader port.
Output retains only complete derived proofs and bound reference metadata;
large decoded training graphs are discarded sequentially. No model/data,
training, scoring, measurement, final release or reader substitute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.app.continual_confirmation_retention_costs import INVENTORY_ID, project_retention_costs
from src.app.continual_confirmation_scoring_execution import verify_reference_report
from src.app.continual_confirmation_scoring_manifest import (
    ConfirmationScoringManifest,
    fixed_scoring_manifest,
)
from src.infra.continual_confirmation_training_references import (
    CompletedTrainingReader,
    read_training_references,
)


RetentionProjector = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
ReferenceReader = Callable[
    [Path, ConfirmationScoringManifest, CompletedTrainingReader], dict[str, Any]
]


def _collect_retention_references(
    root: Path,
    manifest: ConfirmationScoringManifest,
    reader: CompletedTrainingReader,
    inventory: dict[str, Any],
    projector: RetentionProjector,
    *,
    reference_reader: ReferenceReader = read_training_references,
) -> dict[str, Any]:
    """Private IO fixture seam; spies cannot establish original reader authority."""
    projections: list[dict[str, Any]] = []
    captured = 0

    def capture(directory: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        nonlocal captured
        captured += 1
        parts = reader(directory)
        require(
            type(parts) is tuple and len(parts) == 3 and all(type(part) is dict for part in parts),
            "retention complete training reader contract differs",
        )
        projected = projector(parts[1], inventory)
        same_json(
            projected["work"], parts[2]["work"], "retention work versus complete training audit"
        )
        if projections:
            same_json(projected, projections[0], "complete original retention repetition")
        else:
            projections.append(projected)
        # The unchanged reference boundary next verifies every decoded part
        # against complete actual bytes and discards it before the next reader.
        return parts

    references = reference_reader(root, manifest, capture)
    require(
        len(references["bundles"]) == captured == 2 and len(projections) == 1,
        "retention complete two-reader scope differs",
    )
    return {
        "schema_id": "p610_complete_retention_training_references_v1",
        "training_references": references,
        "projection": projections[0],
        "projection_identity": canonical_body_identity(projections[0]),
        "deterministic_retention_repetition": True,
        "validation_scope": "two_unchanged_complete_original_training_readers_and_full_owned_retention_projection",
        "new_measurement_training_or_final_access": False,
    }


def read_retention_references(
    root: Path,
    reader: CompletedTrainingReader,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    """Require whole original inventory and both actual original training readers."""
    same_json(
        canonical_body_identity(inventory), INVENTORY_ID, "retention whole original inventory"
    )
    result = _collect_retention_references(
        root, fixed_scoring_manifest(), reader, inventory, project_retention_costs
    )
    verify_reference_report(result["training_references"])
    return result
