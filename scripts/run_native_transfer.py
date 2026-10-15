"""Describe or run the fixed native Windows coordinator-loss continuation."""

import argparse
import json
from pathlib import Path

from src.adapters.native_transfer_example import describe_example, run_example


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--interrupt-after-train", action="store_true")
    args = parser.parse_args()
    if args.run and args.directory is None:
        parser.error("--run requires an existing private --directory")
    result = (
        describe_example(interrupt_after_train=args.interrupt_after_train)
        if args.describe
        else run_example(args.directory, interrupt_after_train=args.interrupt_after_train)
    )
    # Each full-state case already has a canonical byte cap. Compact run output
    # preserves that aggregate bound; the short preview stays easy to read.
    print(
        json.dumps(
            result,
            indent=2 if args.describe else None,
            separators=None if args.describe else (",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
