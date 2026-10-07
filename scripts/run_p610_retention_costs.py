"""Publish or fully reconstruct the fixed original checkpoint retention ledger."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path

from scripts import run_p67_confirmation_training as training_adapter
from src.infra.continual_confirmation_retention_artifacts import (
    publish_retention_costs,
    read_completed_retention_costs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCOPE_FILE = REPO_ROOT / "artifacts/runs/p67-confirmation-scope.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/runs/p610-retention-costs")
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--publish", action="store_true")
    mode.add_argument("--read-only", action="store_true")
    options = parser.parse_args()
    reader = partial(training_adapter.read_completed_bundle, scope_file=SCOPE_FILE)
    if options.read_only:
        _, _, audit = read_completed_retention_costs(
            REPO_ROOT, options.output_dir, SCOPE_FILE, reader
        )
    else:
        audit = publish_retention_costs(REPO_ROOT, options.output_dir, SCOPE_FILE, reader)
    print(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
