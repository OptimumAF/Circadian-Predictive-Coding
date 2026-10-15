"""Demonstrate supervised checkpoint-writer interruption with a live parent actor."""

import argparse
import json
import os
from pathlib import Path

from src.adapters.coupled_checkpoint_interruption import describe_example, run_example


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--describe", action="store_true", help="Preview without IO/models/processes"
    )
    parser.add_argument("--directory", type=Path, help="New dedicated example directory")
    args = parser.parse_args()
    if not args.describe and args.directory is None:
        parser.error("--directory is required unless --describe is selected")
    report = describe_example() if args.describe else run_example(args.directory)
    encoded = json.dumps(report, allow_nan=False, sort_keys=True, indent=2)
    if len((encoded + "\n").replace("\n", os.linesep).encode("utf-8")) > 2 * 1024**2:
        raise RuntimeError("checkpoint interruption report exceeds2MiB")
    print(encoded)


if __name__ == "__main__":
    main()
