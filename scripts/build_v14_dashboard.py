"""Build or verify static v14 plots and dashboard from a checked report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.infra.v14_dashboard_files import verify_v14_dashboard, write_v14_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", type=Path, help="completed v14 bundle with P5.6a report")
    action.add_argument("--verify-run", type=Path, help="verify existing derived dashboard")
    arguments = parser.parse_args()
    if arguments.verify_run is not None:
        metadata = verify_v14_dashboard(arguments.verify_run)
        print(
            json.dumps({"run": str(arguments.verify_run), "dashboard_id": metadata["dashboard_id"]})
        )
        return
    destination = write_v14_dashboard(arguments.run)
    metadata = verify_v14_dashboard(arguments.run)
    print(json.dumps({"dashboard": str(destination), "dashboard_id": metadata["dashboard_id"]}))


if __name__ == "__main__":
    main()
