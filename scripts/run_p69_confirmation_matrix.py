"""Publish/read back every frozen stored confirmation stage/task matrix.

The fixed CLI supplies the unchanged complete original report reader.
No training/final access, endpoint addition or scientific override is exposed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.run_p611_confirmation_report import REPO_ROOT, SCOPE_FILE, _costs, _scored
from src.infra.continual_confirmation_report_artifacts import read_completed_confirmation_report
from src.infra.continual_confirmation_matrix_artifacts import (
    publish_confirmation_matrix,
    read_completed_confirmation_matrix,
)


def _report() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return read_completed_confirmation_report(
        REPO_ROOT,
        REPO_ROOT / "artifacts/runs/p611-confirmation-report",
        SCOPE_FILE,
        _scored,
        _costs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/runs/p69-confirmation-matrix")
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--publish", action="store_true")
    modes.add_argument("--read-only", action="store_true")
    options = parser.parse_args()
    if options.publish:
        audit = publish_confirmation_matrix(REPO_ROOT, options.output_dir, SCOPE_FILE, _report)
    else:
        _, _, audit = read_completed_confirmation_matrix(
            REPO_ROOT, options.output_dir, SCOPE_FILE, _report
        )
    print(
        json.dumps(
            {
                "directory": str(options.output_dir.resolve()),
                "mode": "publish" if options.publish else "read-only",
                "status": audit["status"],
                "files": audit["files"],
                "coverage": audit["coverage"],
                "report_identity": audit["report_identity"],
                "new_training_or_final_source_access": False,
                "original_fully_measured_matrix_acceptance_complete": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
