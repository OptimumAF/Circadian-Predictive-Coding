"""Publish/read back complete fixed findings through unchanged current readers.

Only the output directory and publication/read-only mode are configurable.
No source/seed/metric/interval/baseline/budget override or new science is exposed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import inspect_p612_development_findings as development
from scripts import run_p610_outcome_costs as outcomes
from scripts import run_p69_confirmation_matrix as matrix
from src.app.continual_findings_readers import FindingsReaders
from src.infra.continual_confirmation_outcome_cost_artifacts import read_completed_outcome_costs
from src.infra.continual_confirmation_matrix_artifacts import read_completed_confirmation_matrix
from src.infra.continual_findings_artifacts import (
    publish_complete_findings,
    read_completed_findings,
)


REPO_ROOT, SCOPE_FILE = outcomes.REPO_ROOT, outcomes.SCOPE_FILE


def _readers() -> FindingsReaders:
    return FindingsReaders(
        outcome_costs=lambda: read_completed_outcome_costs(
            REPO_ROOT,
            REPO_ROOT / "artifacts/runs/p610-outcome-costs-v2",
            SCOPE_FILE,
            outcomes._report,
        ),
        matrix=lambda: read_completed_confirmation_matrix(
            REPO_ROOT,
            REPO_ROOT / "artifacts/runs/p69-confirmation-matrix",
            SCOPE_FILE,
            matrix._report,
        ),
        development=development.inspect_development_findings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/runs/p612-complete-findings-current")
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--publish", action="store_true")
    modes.add_argument("--read-only", action="store_true")
    options = parser.parse_args()
    if options.publish:
        audit = publish_complete_findings(REPO_ROOT, options.output_dir, SCOPE_FILE, _readers())
    else:
        _, _, audit = read_completed_findings(REPO_ROOT, options.output_dir, SCOPE_FILE, _readers())
    print(
        json.dumps(
            {
                "directory": str(options.output_dir.resolve()),
                "mode": "publish" if options.publish else "read-only",
                "status": audit["status"],
                "coverage": audit["coverage"],
                "files": audit["files"],
                "new_scientific_operation_or_final_source_access": False,
                "original_P6_12_acceptance_complete": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
