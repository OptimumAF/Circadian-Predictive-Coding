"""Build or verify one artifact-only descriptive table for a completed v14 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.infra.v14_artifact_report_files import (
    verify_v14_artifact_report,
    write_v14_artifact_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", type=Path, help="verified P5.1 v14 bundle to summarize once")
    action.add_argument("--verify-run", type=Path, help="verify an existing derived report")
    arguments = parser.parse_args()
    if arguments.verify_run is not None:
        metadata = verify_v14_artifact_report(arguments.verify_run)
        print(json.dumps({"run": str(arguments.verify_run), "report_id": metadata["report_id"]}))
        return
    destination = write_v14_artifact_report(arguments.run)
    metadata = verify_v14_artifact_report(arguments.run)
    print(json.dumps({"report": str(destination), "report_id": metadata["report_id"]}))


if __name__ == "__main__":
    main()
