"""Read both complete unscored bundles before future independent scoring.

The adapter supplies the unchanged pinned public training reader. It writes
exclusive local inspection metadata, never a new scientific run or final score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.run_p67_confirmation_training import read_completed_bundle
from src.app.continual_confirmation_scoring_manifest import fixed_scoring_manifest
from src.infra.continual_confirmation_io import encoded_digest, write_exclusive
from src.infra.continual_confirmation_training_references import (
    read_training_references,
    stream_file_identity,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def inspect_training_references(result_file: Path) -> dict[str, Any]:
    """Refuse occupied output before readback; preserve both scientific bundles."""
    path = result_file.resolve()
    if path.exists():
        raise FileExistsError(f"scoring training reference output already exists: {path}")
    report = read_training_references(REPO_ROOT, fixed_scoring_manifest(), read_completed_bundle)
    expected = encoded_digest(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_exclusive(path, report)
    if stream_file_identity(path)["sha256"] != expected:
        raise ValueError("scoring training reference published metadata bytes differ")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-file", type=Path, required=True)
    options = parser.parse_args()
    report = inspect_training_references(options.result_file)
    print(
        json.dumps(
            {
                "result_file": str(options.result_file.resolve()),
                "report_sha256": stream_file_identity(options.result_file)["sha256"],
                "bundles_read_back": len(report["bundles"]),
                "scoring_manifest_sha256": report["scoring_manifest_sha256"],
                "analysis_contract_sha256": report["analysis_contract_sha256"],
                "final_released": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
