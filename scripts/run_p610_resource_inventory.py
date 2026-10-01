"""Publish or fully reconstruct the fixed original resource-field inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.infra.continual_confirmation_resource_artifacts import (
    publish_resource_inventory,
    read_completed_resource_inventory,
)


ROOT = Path(__file__).resolve().parents[1]
SCOPE_FILE = ROOT / "artifacts/runs/p67-confirmation-scope.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--publish", action="store_true")
    mode.add_argument("--read-only", action="store_true")
    options = parser.parse_args()
    if options.read_only:
        _, _, audit = read_completed_resource_inventory(ROOT, options.output_dir, SCOPE_FILE)
    else:
        audit = publish_resource_inventory(ROOT, options.output_dir, SCOPE_FILE)
    print(json.dumps(audit, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
