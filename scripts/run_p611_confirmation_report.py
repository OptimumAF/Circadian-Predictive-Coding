"""Publish/read back every fixed confirmation seed, interval, contrast and cost.

All original scientific settings and complete readers remain unchanged.
No training, final-source access or statistical/scientific override is exposed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from scripts.inspect_p611_confirmation_costs import inspect_confirmation_costs
from src.app.continual_confirmation_report_cost_binding import COST_FILES
from src.infra.continual_confirmation_report_artifacts import (
    publish_confirmation_report,
    read_completed_confirmation_report,
)
from src.infra.continual_confirmation_scoring_artifacts import read_completed_scored_bundle


REPO_ROOT = Path(__file__).resolve().parents[1]
SCOPE_FILE = REPO_ROOT / "artifacts/runs/p67-confirmation-scope.json"


def _costs() -> dict[str, Any]:
    return inspect_confirmation_costs(REPO_ROOT / COST_FILES[0], read_only=True)


def _scored(
    directory: Path, references: Callable[[], dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return read_completed_scored_bundle(REPO_ROOT, directory, SCOPE_FILE, references)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/runs/p611-confirmation-report")
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--publish", action="store_true")
    mode.add_argument("--read-only", action="store_true")
    options = parser.parse_args()
    if options.publish:
        audit = publish_confirmation_report(
            REPO_ROOT, options.output_dir, SCOPE_FILE, _scored, _costs
        )
    else:
        _, _, audit = read_completed_confirmation_report(
            REPO_ROOT, options.output_dir, SCOPE_FILE, _scored, _costs
        )
    print(
        json.dumps(
            {
                "output_dir": str(options.output_dir.resolve()),
                "mode": "publish" if options.publish else "read-only",
                "status": audit["status"],
                "files": audit["files"],
                "coverage": audit["coverage"],
                "analysis_repetition": audit["analysis_repetition"],
                "new_training_or_final_source_access": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
