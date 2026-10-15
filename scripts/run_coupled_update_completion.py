"""Run explicit native-only coordinator completion and validator-failure controls."""

import argparse
import json
import os
from pathlib import Path

from src.adapters.coupled_update_completion import describe_example, run_example


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--describe", action="store_true", help="Preview without IO/models/processes"
    )
    parser.add_argument("--directory", type=Path, help="New dedicated example directory")
    parser.add_argument(
        "--corrupt-checkpoint",
        action="store_true",
        help="Corrupt the owned candidate file and require original-digest refusal",
    )
    args = parser.parse_args()
    if not args.describe and args.directory is None:
        parser.error("--directory is required unless --describe is selected")
    if args.corrupt_checkpoint:
        report = (
            describe_example(corrupt_checkpoint=True)
            if args.describe
            else run_example(args.directory, corrupt_checkpoint=True)
        )
    else:
        report = describe_example() if args.describe else run_example(args.directory)
    encoded = json.dumps(report, allow_nan=False, sort_keys=True, indent=2)
    if len((encoded + "\n").replace("\n", os.linesep).encode("utf-8")) > 2 * 1024**2:
        raise RuntimeError("native completion report exceeds2MiB")
    print(encoded)


if __name__ == "__main__":
    main()
