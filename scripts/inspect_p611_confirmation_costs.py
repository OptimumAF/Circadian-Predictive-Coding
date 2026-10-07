"""Publish or independently read back the complete original confirmation costs.

The unchanged complete training reader supplies provenance. This local cost
inspection opens no model/data/final source and publishes no statistical result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.run_p67_confirmation_training import read_completed_bundle
from src.app.continual_confirmation_execution import digest_json
from src.app.continual_confirmation_json import same_json
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.infra.continual_confirmation_io import read_json, verify_source_files, write_exclusive
from src.infra.continual_confirmation_report_cost_references import (
    read_confirmation_cost_references,
)
from src.infra.continual_confirmation_scoring_bindings import current_sources
from src.infra.continual_confirmation_training_references import stream_file_identity


REPO_ROOT = Path(__file__).resolve().parents[1]
SCORED_SOURCE_MAP_SHA256 = "e0cfd897867786271974bcc971b647084f7fcf34b28e8db8a140a2d8c3f7ff8c"
COST_COMPONENT_SHA256 = {
    "src/app/continual_confirmation_report_costs.py": "7881c74a462550f407b16e8f2ccbf761f0faa0ed81a46bee14f3d2d099164298",
    "src/infra/continual_confirmation_report_cost_references.py": "7c92d58e1e9c5a168a22afa85e2bfdfdf8ae98ab36f5264c1d121c6bc1a26133",
}


def current_cost_sources() -> dict[str, str]:
    """Extend the exact unchanged 97-source scored closure, never repin it."""
    sources = current_sources(REPO_ROOT)
    same_json(digest_json(sources), SCORED_SOURCE_MAP_SHA256, "cost current scored source closure")
    sources.update(verify_source_files(REPO_ROOT, COST_COMPONENT_SHA256))
    name = "scripts/inspect_p611_confirmation_costs.py"
    sources[name] = stream_file_identity(REPO_ROOT / name)["sha256"]
    return sources


def inspect_confirmation_costs(result_file: Path, *, read_only: bool = False) -> dict[str, Any]:
    """Exclusive inspection output; readback reconstructs every original cost."""
    path = result_file.resolve()
    if read_only:
        if not path.is_file():
            raise ValueError(f"confirmation cost readback missing: {path}")
        before_file = stream_file_identity(path)
    elif path.exists():
        raise FileExistsError(f"confirmation cost output already exists: {path}")
    sources = current_cost_sources()
    body = {
        "schema_id": "p611_confirmation_original_cost_inspection_v1",
        "source_sha256": sources,
        "cost_references": read_confirmation_cost_references(REPO_ROOT, read_completed_bundle),
    }
    expected = canonical_body_identity(body)
    same_json(current_cost_sources(), sources, "cost source changed during complete readback")
    if read_only:
        same_json(read_json(path), body, "cost complete published metadata readback")
        same_json(stream_file_identity(path), before_file, "cost output changed during readback")
        same_json(before_file, expected, "cost canonical published metadata bytes")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_exclusive(path, body)
        same_json(stream_file_identity(path), expected, "cost published metadata bytes")
    same_json(current_cost_sources(), sources, "cost source changed after publication/readback")
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument(
        "--read-only", action="store_true", help="rederive and verify an existing inspection"
    )
    options = parser.parse_args()
    body = inspect_confirmation_costs(options.result_file, read_only=options.read_only)
    report = body["cost_references"]
    print(
        json.dumps(
            {
                "result_file": str(options.result_file.resolve()),
                "read_only": options.read_only,
                "report_identity": stream_file_identity(options.result_file),
                "projection_identity": report["projection_identity"],
                "cells": len(report["projection"]["cells"]),
                "family_seed_rows": len(report["projection"]["seed_contexts"]),
                "statistical_seed_report_complete": False,
                "scored_bundles_validated": False,
                "final_released": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
