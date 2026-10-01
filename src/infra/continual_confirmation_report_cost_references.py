"""Join original raw costs while sequential complete readers own provenance.

Inputs are the repository root and unchanged complete-training-reader port.
Output keeps all small cost rows/context plus the original verified reference
metadata; neither large checkpoint graph survives to the next reader. No data,
model, training, final release, scoring, statistics or resource estimation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_report_costs import (
    canonical_body_identity,
    project_confirmation_costs,
)
from src.app.continual_confirmation_scoring_execution import verify_reference_report
from src.app.continual_confirmation_scoring_manifest import (
    ConfirmationScoringManifest,
    fixed_scoring_manifest,
)
from src.infra.continual_confirmation_training_references import (
    CompletedTrainingReader,
    read_training_references,
)


CostProjector = Callable[[dict[str, Any]], dict[str, Any]]
ReferenceReader = Callable[
    [Path, ConfirmationScoringManifest, CompletedTrainingReader], dict[str, Any]
]


def _collect_cost_references(
    root: Path,
    manifest: ConfirmationScoringManifest,
    reader: CompletedTrainingReader,
    projector: CostProjector,
    *,
    reference_reader: ReferenceReader = read_training_references,
) -> dict[str, Any]:
    """Private IO seam permits explicit fixtures; the public scope stays fixed."""
    projections: list[dict[str, Any]] = []
    captured_count = 0

    def capture(directory: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        nonlocal captured_count
        captured_count += 1
        parts = reader(directory)
        require(
            type(parts) is tuple and len(parts) == 3 and all(type(part) is dict for part in parts),
            "cost complete reader contract differs",
        )
        projection = projector(parts[1])
        same_json(projection["work"], parts[2]["work"], "cost projected work versus complete audit")
        if projections:
            same_json(projection, projections[0], "repeated original cost projection")
        else:
            projections.append(projection)
        # The complete reference reader next binds every decoded part to the
        # actual file bytes, then discards the large result before its next call.
        return parts

    references = reference_reader(root, manifest, capture)
    require(
        len(references["bundles"]) == 2 and captured_count == 2 and len(projections) == 1,
        "cost reference scope differs",
    )
    projection = projections[0]
    return {
        "schema_id": "p611_confirmation_original_cost_references_v1",
        "training_references": references,
        "projection": projection,
        "projection_identity": canonical_body_identity(projection),
        "deterministic_cost_repetition": True,
        "validation_scope": "two_complete_original_train_readbacks_and_full_raw_cost_join_only",
        "statistical_seed_report_complete": False,
        "scored_bundles_validated": False,
        "final_release_authorized": False,
        "outer_selection_scored": False,
    }


def read_confirmation_cost_references(
    root: Path, reader: CompletedTrainingReader
) -> dict[str, Any]:
    """Require both pinned complete bundles and the exact whole cost projection."""
    result = _collect_cost_references(
        root, fixed_scoring_manifest(), reader, project_confirmation_costs
    )
    verify_reference_report(result["training_references"])
    return result
