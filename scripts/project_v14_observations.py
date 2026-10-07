"""Project or verify observed v14 facts from one completed local run bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.infra.observation_projection_files import (
    verify_observation_projection,
    write_observation_projection,
)
from src.infra.measured_observation_files import (
    verify_measured_observation_projection,
    write_measured_observation_projection,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", type=Path, help="completed P5.1 run to project once")
    action.add_argument("--verify-run", type=Path, help="verify an existing projection")
    action.add_argument("--run-measured", type=Path, help="project a completed measured bundle")
    action.add_argument(
        "--verify-measured-run", type=Path, help="verify an existing measured projection"
    )
    arguments = parser.parse_args()
    if arguments.verify_measured_run is not None:
        metadata = verify_measured_observation_projection(arguments.verify_measured_run)
        print(
            json.dumps(
                {
                    "run": str(arguments.verify_measured_run),
                    "projection_id": metadata["projection_id"],
                }
            )
        )
        return
    if arguments.run_measured is not None:
        destination = write_measured_observation_projection(arguments.run_measured)
        metadata = verify_measured_observation_projection(arguments.run_measured)
        print(json.dumps({"projection": str(destination), "records": metadata["files"]}))
        return
    if arguments.verify_run is not None:
        metadata = verify_observation_projection(arguments.verify_run)
        print(
            json.dumps(
                {"run": str(arguments.verify_run), "projection_id": metadata["projection_id"]}
            )
        )
        return
    destination = write_observation_projection(arguments.run)
    metadata = verify_observation_projection(arguments.run)
    print(json.dumps({"projection": str(destination), "records": metadata["files"]}))


if __name__ == "__main__":
    main()
