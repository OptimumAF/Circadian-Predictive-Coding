"""Preview or run the fixed physical Windows coordinator-transfer example."""

import argparse
import json
from pathlib import Path

from src.adapters.windows_transfer_example import describe_example, run_example


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.run and args.directory is None:
        parser.error("--run requires an existing private --directory")
    result = describe_example() if args.describe else run_example(args.directory)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
