"""Publish or independently read every original outcome against scoped costs.

The fixed adapter supplies the unchanged whole official report reader. No
scientific source/seed/metric/baseline/cap override or new measurement is exposed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.run_p611_confirmation_report import REPO_ROOT, SCOPE_FILE, _costs, _scored
from src.infra.continual_confirmation_report_artifacts import read_completed_confirmation_report
from src.infra.continual_confirmation_outcome_cost_artifacts import (
    publish_outcome_costs,
    read_completed_outcome_costs,
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
        "--output-dir", type=Path, default=Path("artifacts/runs/p610-outcome-costs-v2")
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--publish", action="store_true")
    modes.add_argument("--read-only", action="store_true")
    options = parser.parse_args()
    if options.publish:
        audit = publish_outcome_costs(REPO_ROOT, options.output_dir, SCOPE_FILE, _report)
    else:
        _, _, audit = read_completed_outcome_costs(
            REPO_ROOT, options.output_dir, SCOPE_FILE, _report
        )
    print(
        json.dumps(
            {
                "directory": str(options.output_dir.resolve()),
                "mode": "publish" if options.publish else "read-only",
                "status": audit["status"],
                "new_measurement_training_or_final_access": False,
                "original_P6_10_acceptance_complete": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
